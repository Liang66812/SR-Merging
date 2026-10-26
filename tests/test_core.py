"""CPU correctness tests; tiny random models, no downloads or accuracy claims."""
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from sr_merging.routing import ERC, SRR, TCSM, ReferenceBank, normalize
from sr_merging.composition import SparseComposition

torch.set_num_threads(2)


def tiny_experts():
    torch.manual_seed(7)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=128,
        pad_token_id=0, eos_token_id=2, bos_token_id=1, attention_dropout=0.0))
    config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, bias="none",
                        target_modules=["gate_proj", "up_proj", "down_proj"],
                        task_type="CAUSAL_LM")
    model = get_peft_model(base, config, adapter_name="expert_0")
    model.add_adapter("expert_1", config)
    model.add_adapter("expert_2", config)
    # Nonzero updates are essential: default-zero LoRA B would hide fusion bugs.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.normal_(0, 0.05)
    return model.eval(), ["expert_0", "expert_1", "expert_2"]


class RoutingTests(unittest.TestCase):
    def test_erc_exactness_and_roundtrip(self):
        torch.manual_seed(3)
        supports = torch.randn(3, 11, 7)
        queries = torch.randn(4, 7)
        c = torch.stack([ERC.compress(s) for s in supports])
        bank = ReferenceBank(("a", "b", "c"), c, {"encoder": "test"})
        expected = torch.stack([(normalize(queries) @ normalize(s).T).mean(-1) for s in supports], -1)
        torch.testing.assert_close(SRR(bank)(queries), expected, atol=1e-6, rtol=1e-6)
        self.assertTrue((c.norm(dim=-1) < 1).all())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"bank.npz"
            bank.save(path)
            torch.testing.assert_close(ReferenceBank.load(path).vectors, c)

    def test_tcsm_temperature_and_ties(self):
        scores = torch.tensor([[3., 2., 1., 0.], [1., 1., 1., 0.]])
        ids, weights = TCSM(3, .8)(scores)
        self.assertEqual(ids.tolist(), [[0, 1, 2], [0, 1, 2]])
        torch.testing.assert_close(weights[0], torch.softmax(scores[0,:3]/.8,0))
        shifted = TCSM(3,.8)(scores+10)[1]
        torch.testing.assert_close(weights, shifted)
        torch.testing.assert_close(TCSM(1,.8)(scores)[1], torch.ones(2,1))
        with self.assertRaises(ValueError):
            TCSM(5,.8)(scores)

    def test_zero_reference_mean_is_valid(self):
        mean = ERC.compress(torch.tensor([[1.,0.],[-1.,0.]]))
        torch.testing.assert_close(mean, torch.zeros(2))
        with self.assertRaises(ValueError):
            ERC.compress(torch.zeros(1,2))


class CompositionTests(unittest.TestCase):
    def test_batched_k1_matches_peft(self):
        peft, names = tiny_experts()
        x = torch.tensor([[1,4,5],[1,6,7]])
        with torch.inference_mode():
            peft.set_adapter(names[0])
            gold_a = peft(input_ids=x).logits.clone()
            peft.set_adapter(names[1])
            gold_b = peft(input_ids=x).logits.clone()
            model = peft.get_base_model()
            fusion = SparseComposition(model,names)
            with fusion.use(torch.tensor([[0],[1]]),torch.ones(2,1),torch.device("cpu")):
                actual = model(input_ids=x).logits
            torch.testing.assert_close(actual,torch.stack([gold_a[0],gold_b[1]]),atol=2e-6,rtol=2e-5)
            self.assertIsNone(fusion.batch_size)

    def test_weighted_updates_match_explicit_dense_layers(self):
        peft, names = tiny_experts()
        original = peft.get_base_model().model.layers[0].mlp.up_proj
        x = torch.randn(2,4,16)
        ids = torch.tensor([[0,2],[1,2]])
        weights = torch.tensor([[.2,.8],[.7,.3]])
        with torch.inference_mode():
            expected=[]
            for row in range(2):
                w=original.base_layer.weight.clone()
                for idx, alpha in zip(ids[row], weights[row]):
                    name=names[idx]
                    w += alpha*original.scaling[name]*(original.lora_B[name].weight @ original.lora_A[name].weight)
                expected.append(torch.nn.functional.linear(x[row],w,original.base_layer.bias))
            model=peft.get_base_model()
            fusion=SparseComposition(model,names)
            with fusion.use(ids,weights,torch.device("cpu")):
                actual=model.model.layers[0].mlp.up_proj(x)
            torch.testing.assert_close(actual,torch.stack(expected),atol=2e-6,rtol=2e-5)

    def test_generation_keeps_per_input_routes(self):
        peft,names=tiny_experts()
        model=peft.get_base_model()
        fusion=SparseComposition(model,names)
        x=torch.tensor([[1,4,5],[1,6,7]])
        ids=torch.tensor([[0,1,2],[2,0,1]])
        weights=torch.tensor([[.6,.3,.1],[.5,.3,.2]])
        with torch.inference_mode():
            with fusion.use(ids,weights,torch.device("cpu")):
                batch=model.generate(x,attention_mask=torch.ones_like(x),max_new_tokens=4,do_sample=False)
            singles=[]
            for row in range(2):
                with fusion.use(ids[row:row+1],weights[row:row+1],torch.device("cpu")):
                    singles.append(model.generate(x[row:row+1],attention_mask=torch.ones_like(x[row:row+1]),max_new_tokens=4,do_sample=False)[0])
            for actual,expected in zip(batch,singles):
                # Batched generation may pad a sequence after EOS.
                torch.testing.assert_close(actual[:len(expected)],expected)


if __name__ == "__main__":
    unittest.main()
