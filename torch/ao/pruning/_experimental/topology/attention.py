from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TopologicalSpectralAttention(nn.Module):
    r"""Scaled dot-product attention with an opt-in clustered approximation gate."""

    def __init__(
        self,
        *,
        min_seq_len: int = 128,
        num_clusters: int = 4,
        spectral_gap_threshold: float = 0.10,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        if min_seq_len <= 0:
            raise ValueError("min_seq_len must be positive")
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if spectral_gap_threshold < 0:
            raise ValueError("spectral_gap_threshold must be non-negative")
        self.min_seq_len = min_seq_len
        self.num_clusters = num_clusters
        self.spectral_gap_threshold = spectral_gap_threshold
        self.dropout_p = dropout_p
        self.is_causal = is_causal

    def _dropout_p(self) -> float:
        return self.dropout_p if self.training else 0.0

    def _cluster_assignments(self, keys: torch.Tensor) -> torch.Tensor | None:
        seq_len = keys.shape[-2]
        if seq_len < self.min_seq_len or seq_len < self.num_clusters:
            return None
        features = keys.detach().mean(dim=tuple(range(keys.ndim - 2))).float()
        centered = features - features.mean(dim=0, keepdim=True)
        affinity = centered @ centered.T
        affinity = affinity - affinity.min()
        affinity.fill_diagonal_(0)
        degree = affinity.sum(dim=1)
        if torch.count_nonzero(degree > 0) < self.num_clusters:
            return None
        inv_sqrt = torch.rsqrt(degree.clamp_min(1e-12))
        laplacian = torch.eye(seq_len, device=keys.device) - inv_sqrt[:, None] * affinity * inv_sqrt[None, :]
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        if eigenvalues.numel() <= self.num_clusters:
            return None
        gap = eigenvalues[self.num_clusters] - eigenvalues[self.num_clusters - 1]
        if float(gap.item()) < self.spectral_gap_threshold:
            return None
        feature = eigenvectors[:, 1 if eigenvectors.shape[1] > 1 else 0]
        order = torch.argsort(feature)
        assignments = torch.empty(seq_len, dtype=torch.long, device=keys.device)
        for cluster_id, chunk in enumerate(torch.tensor_split(order, self.num_clusters)):
            assignments[chunk] = cluster_id
        return assignments

    def _clustered_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        assignments: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        representatives_k = []
        representatives_v = []
        cluster_sizes = []
        for cluster_id in range(self.num_clusters):
            members = assignments == cluster_id
            if not members.any():
                continue
            representatives_k.append(key[..., members, :].mean(dim=-2))
            representatives_v.append(value[..., members, :].mean(dim=-2))
            cluster_sizes.append(members.sum())
        if not representatives_k:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self._dropout_p(),
                is_causal=self.is_causal,
            )
        clustered_key = torch.stack(representatives_k, dim=-2)
        clustered_value = torch.stack(representatives_v, dim=-2)
        cluster_bias = torch.stack(cluster_sizes).to(device=query.device, dtype=query.dtype).log().view(1, -1)
        return F.scaled_dot_product_attention(
            query,
            clustered_key,
            clustered_value,
            attn_mask=cluster_bias,
            dropout_p=self._dropout_p(),
            is_causal=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assignments = None
        if not torch.compiler.is_compiling() and attn_mask is None and not self.is_causal:
            assignments = self._cluster_assignments(key)
        if assignments is None:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self._dropout_p(),
                is_causal=self.is_causal,
            )
        return self._clustered_attention(query, key, value, assignments, attn_mask)
