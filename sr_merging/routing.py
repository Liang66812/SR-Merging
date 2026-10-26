"""ERC: offline means; SRR: cosine scores; TCSM: sparse softmax weights."""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
from torch import Tensor


def normalize(x: Tensor) -> Tensor:
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError("Expected finite [samples, dimension] embeddings")
    norms = x.norm(dim=-1, keepdim=True)
    if (norms <= 0).any():
        raise ValueError("A zero embedding has no defined cosine similarity")
    return x / norms


@dataclass(frozen=True)
class ReferenceBank:
    names: tuple[str, ...]
    vectors: Tensor
    metadata: dict

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Use an .npz output path")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, vectors=self.vectors.cpu().numpy(),
                 names=np.asarray(self.names), metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceBank":
        with np.load(path, allow_pickle=False) as data:
            names = tuple(data["names"].tolist())
            vectors = torch.from_numpy(data["vectors"].copy()).float()
            metadata = json.loads(data["metadata"].item())
        if (not names or len(set(names)) != len(names) or vectors.ndim != 2
                or len(vectors) != len(names) or not torch.isfinite(vectors).all()):
            raise ValueError("Invalid reference bank")
        # A zero mean is valid even though individual embeddings must be nonzero.
        return cls(names, vectors, metadata)


class ERC:
    @staticmethod
    def compress(embeddings: Tensor) -> Tensor:
        if len(embeddings) == 0:
            raise ValueError("Each expert needs at least one reference")
        # Never normalize this mean: its norm is part of average-cosine scoring.
        return normalize(embeddings).mean(dim=0)


class SRR:
    def __init__(self, bank: ReferenceBank):
        self.bank = bank

    def __call__(self, query_embeddings: Tensor) -> Tensor:
        query = normalize(query_embeddings)
        if query.shape[1] != self.bank.vectors.shape[1]:
            raise ValueError("Query and reference dimensions differ")
        return query @ self.bank.vectors.to(query.device).T


class TCSM:
    def __init__(self, k: int = 3, tau: float = 0.8):
        if not isinstance(k, int) or k < 1 or not np.isfinite(tau) or tau <= 0:
            raise ValueError("Require integer K >= 1 and finite tau > 0")
        self.k, self.tau = k, tau

    def __call__(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        if scores.ndim != 2 or self.k > scores.shape[1] or not torch.isfinite(scores).all():
            raise ValueError("Expected finite [batch, experts] scores with K <= N")
        # Stable ordering resolves equal scores by bank/expert order.
        indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :self.k]
        selected = scores.gather(1, indices)
        weights = torch.softmax((selected - selected.amax(1, keepdim=True)) / self.tau, dim=1)
        return indices, weights
