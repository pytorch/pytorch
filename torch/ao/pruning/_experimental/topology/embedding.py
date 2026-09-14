from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .low_rank import _optimizer_contains_parameter, _replace_optimizer_parameters

if TYPE_CHECKING:
    from torch.optim import Optimizer


class SpectralPrototypeEmbedding(nn.Module):
    r"""Embedding module that compresses token communities into prototypes.

    The module observes token windows, builds a token co-occurrence graph, and
    uses the Fiedler vector of the normalized graph Laplacian to split tokens
    into deterministic communities. Compression is opt-in via
    :meth:`try_compress`; dense embedding lookup remains the default path.
    """

    is_compressed: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        num_prototypes: int,
        padding_idx: int | None = None,
        min_observations: int = 32,
        window_size: int = 2,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_prototypes <= 0 or num_prototypes > num_embeddings:
            raise ValueError("num_prototypes must be in [1, num_embeddings]")
        if padding_idx is not None:
            if padding_idx > 0:
                if padding_idx >= num_embeddings:
                    raise AssertionError("padding_idx must be within num_embeddings")
            elif padding_idx < 0:
                if padding_idx < -num_embeddings:
                    raise AssertionError("padding_idx must be within num_embeddings")
                padding_idx = num_embeddings + padding_idx
        if min_observations <= 0:
            raise ValueError("min_observations must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        self.padding_idx = padding_idx
        self.min_observations = min_observations
        self.window_size = window_size
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        self.prototype_weight: nn.Parameter | None = None
        self.edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        self.edge_weight = torch.empty(0, device=device)
        self.token_counts = torch.zeros(num_embeddings, device=device)
        self.register_buffer("token_to_prototype", torch.zeros(num_embeddings, dtype=torch.long, device=device))
        self.is_compressed = False
        self.reset_parameters()

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        *,
        num_prototypes: int,
        min_observations: int = 32,
        window_size: int = 2,
    ) -> SpectralPrototypeEmbedding:
        module = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            num_prototypes=num_prototypes,
            padding_idx=embedding.padding_idx,
            min_observations=min_observations,
            window_size=window_size,
            max_norm=embedding.max_norm,
            norm_type=embedding.norm_type,
            scale_grad_by_freq=embedding.scale_grad_by_freq,
            sparse=embedding.sparse,
            device=embedding.weight.device,
            dtype=embedding.weight.dtype,
        )
        with torch.no_grad():
            module.weight.copy_(embedding.weight)
        return module

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    @torch.no_grad()
    def observe_tokens(self, tokens: torch.Tensor) -> None:
        flat = tokens.detach().reshape(-1).to(device=self.token_counts.device, dtype=torch.long)
        if flat.numel() == 0:
            return
        valid = (flat >= 0) & (flat < self.num_embeddings)
        if self.padding_idx is not None:
            valid &= flat != self.padding_idx
        flat = flat[valid]
        if flat.numel() == 0:
            return
        self.token_counts.index_add_(0, flat, torch.ones_like(flat, dtype=self.token_counts.dtype))
        rows = tokens.detach().reshape(-1, tokens.shape[-1] if tokens.ndim > 1 else tokens.numel()).to(
            device=self.token_counts.device, dtype=torch.long
        )
        edge_parts = []
        for row in rows:
            row = row[(row >= 0) & (row < self.num_embeddings)]
            if self.padding_idx is not None:
                row = row[row != self.padding_idx]
            for index in range(row.numel()):
                left = max(0, index - self.window_size)
                right = min(row.numel(), index + self.window_size + 1)
                center = row[index]
                neighbors = row[left:right]
                neighbors = neighbors[neighbors != center]
                if neighbors.numel() > 0:
                    centers = center.expand_as(neighbors)
                    edge_parts.append(torch.stack((centers, neighbors), dim=0))
                    edge_parts.append(torch.stack((neighbors, centers), dim=0))
        if not edge_parts:
            return
        new_edges = torch.cat(edge_parts, dim=1)
        new_weights = torch.ones(new_edges.shape[1], dtype=self.token_counts.dtype, device=self.token_counts.device)
        if self.edge_index.numel() > 0:
            new_edges = torch.cat((self.edge_index, new_edges), dim=1)
            new_weights = torch.cat((self.edge_weight, new_weights))
        edge_ids = new_edges[0] * self.num_embeddings + new_edges[1]
        unique_ids, inverse = torch.unique(edge_ids, sorted=True, return_inverse=True)
        coalesced_weights = torch.zeros(unique_ids.numel(), dtype=new_weights.dtype, device=new_weights.device)
        coalesced_weights.index_add_(0, inverse, new_weights)
        self.edge_index = torch.stack((unique_ids // self.num_embeddings, unique_ids % self.num_embeddings), dim=0)
        self.edge_weight = coalesced_weights

    def _spectral_assignments(self) -> torch.Tensor | None:
        observed = self.token_counts > 0
        if int(self.token_counts.sum().item()) < self.min_observations or observed.sum() < self.num_prototypes:
            return None
        if self.edge_index.numel() == 0:
            return None
        adjacency = torch.sparse_coo_tensor(
            self.edge_index,
            self.edge_weight,
            (self.num_embeddings, self.num_embeddings),
            device=self.token_counts.device,
        ).coalesce()
        degree = torch.zeros(self.num_embeddings, dtype=self.edge_weight.dtype, device=self.token_counts.device)
        degree.index_add_(0, adjacency.indices()[0], adjacency.values())
        active = observed & (degree > 0)
        if active.sum() < self.num_prototypes:
            return None
        active_indices = active.nonzero(as_tuple=False).flatten()
        row, col = adjacency.indices()
        active_edges = active[row] & active[col]
        if not active_edges.any():
            return None
        active_lookup = torch.full((self.num_embeddings,), -1, dtype=torch.long, device=self.token_counts.device)
        active_lookup[active_indices] = torch.arange(active_indices.numel(), device=self.token_counts.device)
        sub_indices = torch.stack((active_lookup[row[active_edges]], active_lookup[col[active_edges]]), dim=0)
        sub_adj = torch.sparse_coo_tensor(
            sub_indices,
            adjacency.values()[active_edges],
            (active_indices.numel(), active_indices.numel()),
            device=self.token_counts.device,
        ).to_dense()
        sub_degree = sub_adj.sum(dim=1).clamp_min(1e-12)
        inv_sqrt = torch.rsqrt(sub_degree)
        laplacian = torch.eye(active_indices.numel(), device=sub_adj.device) - inv_sqrt[:, None] * sub_adj * inv_sqrt[None, :]
        _, eigenvectors = torch.linalg.eigh(laplacian)
        feature_index = 1 if eigenvectors.shape[1] > 1 else 0
        feature = eigenvectors[:, feature_index]
        order = torch.argsort(feature)
        assignments = torch.zeros(self.num_embeddings, dtype=torch.long, device=self.token_to_prototype.device)
        for cluster_id, chunk in enumerate(torch.tensor_split(order, self.num_prototypes)):
            if chunk.numel() > 0:
                assignments[active_indices[chunk]] = cluster_id
        inactive = (~active).nonzero(as_tuple=False).flatten()
        if inactive.numel() > 0:
            nearest = torch.argmin(torch.cdist(self.weight.detach()[inactive].float(), self.weight.detach()[active_indices].float()), dim=1)
            assignments[inactive] = assignments[active_indices[nearest]]
        return assignments

    def _apply(self, fn):
        super()._apply(fn)
        self.edge_index = fn(self.edge_index)
        self.edge_weight = fn(self.edge_weight)
        self.token_counts = fn(self.token_counts)
        return self

    @torch.no_grad()
    def try_compress(self, optimizer: Optimizer | None = None) -> bool:
        if self.is_compressed:
            return True
        if optimizer is not None and not _optimizer_contains_parameter(optimizer, self.weight):
            raise ValueError("optimizer does not contain the dense weight parameter")
        assignments = self._spectral_assignments()
        if assignments is None:
            return False
        prototypes = []
        if self.padding_idx is not None:
            prototypes.append(torch.zeros_like(self.weight.detach()[0]))
        for prototype_id in range(self.num_prototypes):
            members = assignments == prototype_id
            if self.padding_idx is not None:
                members[self.padding_idx] = False
            if members.any():
                prototypes.append(self.weight.detach()[members].mean(dim=0))
            else:
                prototypes.append(self.weight.detach()[0])
        prototype_weight = torch.stack(prototypes, dim=0)
        if self.padding_idx is not None:
            assignments = assignments + 1
            assignments[self.padding_idx] = 0
        old_weight = self.weight
        del self._parameters["weight"]
        self.weight = None  # type: ignore[assignment]
        self.prototype_weight = nn.Parameter(prototype_weight.contiguous())
        self.token_to_prototype.copy_(assignments)
        self.edge_index = torch.empty(2, 0, dtype=torch.long, device=self.token_to_prototype.device)
        self.edge_weight = torch.empty(0, dtype=self.token_counts.dtype, device=self.token_to_prototype.device)
        self.token_counts.zero_()
        self.is_compressed = True
        if optimizer is not None:
            _replace_optimizer_parameters(optimizer, old_weight, (self.prototype_weight,))
        return True

    def _checkpoint_factory_kwargs(
        self,
        fallback: torch.Tensor | None = None,
        *,
        assign: bool = False,
    ) -> dict[str, torch.device | torch.dtype]:
        if assign and fallback is not None:
            return {"device": fallback.device, "dtype": fallback.dtype}
        reference = None
        for candidate in (self.weight, self.prototype_weight):
            if isinstance(candidate, torch.Tensor):
                reference = candidate
                break
        if reference is None:
            reference = self.token_to_prototype
        if reference is None:
            reference = fallback
        if reference is None:
            return {}
        dtype = fallback.dtype if fallback is not None and assign else torch.get_default_dtype()
        if reference.is_floating_point():
            dtype = reference.dtype
        elif fallback is not None:
            dtype = fallback.dtype
        return {"device": reference.device, "dtype": dtype}

    def _prepare_dense_parameters(self, weight: torch.Tensor | None = None, *, assign: bool = False) -> None:
        factory_kwargs = self._checkpoint_factory_kwargs(weight, assign=assign)
        if "prototype_weight" in self._parameters:
            del self._parameters["prototype_weight"]
        self.prototype_weight = None
        if "weight" not in self._parameters:
            del self.__dict__["weight"]
            self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, **factory_kwargs))
        self.is_compressed = False

    def _prepare_prototype_parameters(self, prototype_weight: torch.Tensor, *, assign: bool = False) -> None:
        factory_kwargs = self._checkpoint_factory_kwargs(prototype_weight, assign=assign)
        if "weight" in self._parameters:
            del self._parameters["weight"]
        self.weight = None  # type: ignore[assignment]
        self.prototype_weight = nn.Parameter(torch.empty(prototype_weight.shape, **factory_kwargs))
        self.is_compressed = True

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        prototype_weight = state_dict.get(prefix + "prototype_weight")
        if prototype_weight is not None:
            expected_rows = self.num_prototypes + (1 if self.padding_idx is not None else 0)
            if prototype_weight.ndim != 2 or prototype_weight.shape != (expected_rows, self.embedding_dim):
                error_msgs.append(
                    "size mismatch for prototype checkpoint: "
                    f"prototype_weight has shape {tuple(prototype_weight.shape)}, expected "
                    f"({expected_rows}, {self.embedding_dim})"
                )
            else:
                assign = bool(local_metadata.get("assign_to_params_buffers", False))
                self._prepare_prototype_parameters(prototype_weight, assign=assign)
        elif prefix + "weight" in state_dict and self.is_compressed:
            assign = bool(local_metadata.get("assign_to_params_buffers", False))
            self._prepare_dense_parameters(state_dict[prefix + "weight"], assign=assign)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_compressed:
            if self.prototype_weight is None:
                raise RuntimeError("prototype weights are missing")
            mapped = self.token_to_prototype.index_select(0, input.reshape(-1)).reshape(input.shape)
            return F.embedding(
                mapped,
                self.prototype_weight,
                0 if self.padding_idx is not None else None,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
