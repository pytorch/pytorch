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
    def validate_labels_and_count(
        labels: torch.Tensor, vocab: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_labels = torch.empty([1], dtype=torch.bool, device=labels.device)
        nonignored_count = torch.empty([1], dtype=torch.int64, device=labels.device)
        for tile in hl.tile(labels.shape[0]):
            values = labels[tile]
            ignored = values == -100
            valid = ((values >= 0) & (values < vocab)) | ignored
            valid_labels[0] = valid.sum() == labels.shape[0]
            nonignored_count[0] = (~ignored).sum()
        return valid_labels, nonignored_count

    @instrumented_helion_kernel(
        "aten::cross_entropy_loss",
        aot=True,
        static_shapes=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
        key_fn=lambda logits,
        labels,
        nonignored_count: f"cross_entropy logits={tuple(logits.shape)}",
    )
    def cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        nonignored_count: torch.Tensor,
    ) -> torch.Tensor:
        n, v = logits.shape
        losses = torch.empty([n], dtype=logits.dtype, device=logits.device)
        logits_flat = logits.view(-1)

        for tile_n in hl.tile(n):
            labels_tile = labels[tile_n]
            ignored = labels_tile == -100
            valid = (labels_tile >= 0) & (labels_tile < v)
            safe_labels = torch.where(valid, labels_tile, 0)
            base_indices_tile = tile_n.index * v
            flat_indices = base_indices_tile + safe_labels
            logits_at_target = hl.load(logits_flat, [flat_indices]).to(torch.float32)

            logits_rows = logits[tile_n, :].to(torch.float32)
            max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
            shifted = logits_rows - max_logits
            exp_shifted = torch.exp(shifted)
            sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
            row_losses = (max_logits.squeeze(-1) - logits_at_target) + torch.log(
                sum_exp.squeeze(-1)
            )
            losses[tile_n] = torch.where(ignored, 0.0, row_losses)

        return (losses.to(torch.float32).sum() / nonignored_count[0]).to(logits.dtype)
