"""Resident, per-input LoRA composition. Inference only; no dense merge."""
from contextlib import contextmanager

import torch
from torch import nn
from peft.tuners.lora.layer import Linear


class SparseComposition:
    """One serial inference worker; route is fixed for the whole generation."""

    def __init__(self, model: nn.Module, names: list[str]):
        self.names = tuple(names)
        self.batch_size = None
        self.groups = ()
        if model.training:
            raise ValueError("Call eval() before installing inference composition")
        replacements = []
        for name, layer in list(model.named_modules()):
            if not isinstance(layer, Linear):
                continue
            if layer.merged or layer.disable_adapters or layer.fan_in_fan_out:
                raise ValueError("Only enabled, unmerged standard Linear LoRA is supported")
            if any(n not in layer.lora_A or n not in layer.lora_B for n in names):
                raise ValueError(f"Experts must adapt identical layers: {name}")
            if any(layer.use_dora.get(n, False) for n in names):
                raise ValueError("DoRA is not additive LoRA composition")
            parent, _, key = name.rpartition(".")
            replacements.append((model.get_submodule(parent), key, _ComposedLinear(layer, self)))
        if not replacements:
            raise ValueError("No compatible LoRA layers found")
        for parent, key, layer in replacements:
            setattr(parent, key, layer)
        model.eval().requires_grad_(False)

    @contextmanager
    def use(self, indices: torch.Tensor, weights: torch.Tensor, device: torch.device):
        if self.batch_size is not None:
            raise RuntimeError("Concurrent/nested calls are unsupported; use one worker per model")
        if (indices.ndim != 2 or weights.shape != indices.shape or indices.shape[0] == 0
                or indices.shape[1] == 0 or indices.dtype != torch.long):
            raise ValueError("Expected nonempty [batch, K] indices and weights")
        if (indices.min() < 0 or indices.max() >= len(self.names)
                or not torch.isfinite(weights).all() or (weights < 0).any()
                or not torch.allclose(weights.sum(-1), torch.ones_like(weights[:, 0]), atol=1e-6)):
            raise ValueError("Invalid expert indices or composition weights")
        sorted_ids = indices.sort(-1).values
        if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
            raise ValueError("Repeated expert indices within one input")
        groups = []
        # Group once per request, not once per layer/token.
        for expert in sorted(indices.unique().tolist()):
            rows, slots = torch.where(indices == expert)
            groups.append((expert, rows.to(device), weights[rows, slots].to(device)))
        self.batch_size, self.groups = len(indices), tuple(groups)
        try:
            yield
        finally:
            self.batch_size, self.groups = None, ()


class _ComposedLinear(nn.Module):
    def __init__(self, original: Linear, controller: SparseComposition):
        super().__init__()
        self.base = original.base_layer
        self.a = nn.ModuleList([original.lora_A[n] for n in controller.names])
        self.b = nn.ModuleList([original.lora_B[n] for n in controller.names])
        self.scales = tuple(float(original.scaling[n]) for n in controller.names)
        self.controller = controller

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != self.controller.batch_size:
            raise RuntimeError("Expected [batch, tokens, hidden] with an active matching route")
        result = self.base(x)
        for expert, rows, weights in self.controller.groups:
            selected = x.index_select(0, rows).to(self.a[expert].weight.dtype)
            delta = self.b[expert](self.a[expert](selected))
            delta = delta * (weights.to(delta.dtype) * self.scales[expert])[:, None, None]
            result = result.index_add(0, rows, delta.to(result.dtype))
        return result
