"""Shape-specialized Helion cross entropy kernel."""

from __future__ import annotations

import torch
from torch._native.instrumentation import instrumented_helion_kernel


try:
    import helion  # pyrefly: ignore[missing-import]
    import helion.language as hl  # pyrefly: ignore[missing-import]
except ModuleNotFoundError as exc:
    if exc.name != "helion":
        raise
    helion = None  # type: ignore[assignment]


if helion is not None:

    @instrumented_helion_kernel(
        "aten::cross_entropy_loss",
        config=helion.Config(block_sizes=[32768], num_warps=8),
        key_fn=lambda labels,
        vocab: f"validate labels={tuple(labels.shape)} vocab={vocab}",
    )
    def validate_labels(labels: torch.Tensor, vocab: int) -> torch.Tensor:
        valid_labels = torch.empty([1], dtype=torch.bool, device=labels.device)
        for tile in hl.tile(labels.shape[0]):
            values = labels[tile]
            valid = (values >= 0) & (values < vocab)
            valid_labels[0] = valid.sum() == labels.shape[0]
        return valid_labels

    @instrumented_helion_kernel(
        "aten::cross_entropy_loss",
        aot=True,
        static_shapes=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
        key_fn=lambda logits, labels: f"cross_entropy logits={tuple(logits.shape)}",
    )
    def cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        n, v = logits.shape
        losses = torch.empty([n], dtype=logits.dtype, device=logits.device)
        logits_flat = logits.view(-1)

        for tile_n in hl.tile(n):
            labels_tile = labels[tile_n]
            base_indices_tile = tile_n.index * v
            flat_indices = base_indices_tile + labels_tile
            logits_at_target = hl.load(logits_flat, [flat_indices])

            logits_rows = logits[tile_n, :]
            max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
            shifted = logits_rows - max_logits
            exp_shifted = torch.exp(shifted)
            sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
            log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))

            losses[tile_n] = log_sum_exp - logits_at_target

        return losses.mean()
