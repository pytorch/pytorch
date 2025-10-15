import torch
import torch.nn.functional as F
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


def view_as_chunked(x, chunk_size, dim=0):
    return x.view(-1, *[chunk_size if i == dim else s for i, s in enumerate(x.size())])


def pad_to_chunk_size(tensor, chunk_size, dim=0, pad_value=0):
    """Pad tensor along specified dimension to make it divisible by chunk_size."""
    size_along_dim = tensor.size(dim)
    remainder = size_along_dim % chunk_size
    if remainder == 0:
        return tensor, 0  # No padding needed

    padding_needed = chunk_size - remainder

    # Create padding tuple for torch.nn.functional.pad
    # pad format is (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back, ...)
    # Pytorch.nn.functional.pad expects padding from last dimension to first
    # For a 3D tensor [B, T, H], padding tuple is [H_left, H_right, T_left, T_right, B_left, B_right]
    pad_tuple = [0] * (2 * tensor.dim())

    # Calculate the correct index for the specified dimension
    # For dim=0 (batch), we want to pad B_right, which is at index -1
    # For dim=1 (sequence), we want to pad T_right, which is at index -3
    # For dim=2 (hidden), we want to pad H_right, which is at index -5
    padding_index = 2 * (tensor.dim() - 1 - dim) + 1
    pad_tuple[padding_index] = padding_needed

    padded_tensor = torch.nn.functional.pad(tensor, pad_tuple, value=pad_value)
    return padded_tensor, padding_needed


class GRPO:
    # Adapted from https://github.com/linkedin/Liger-Kernel/blob/main/test/chunked_loss/test_grpo_loss.py
    class TorchGRPO(torch.nn.Module):
        def __init__(
            self,
            H: int,
            V: int,
            dtype: torch.dtype,
            bias: bool = False,
            beta: float = 0.1,
            epsilon_low: float = 0.2,
            epsilon_high: float = 0.2,
            temperature: float = 1.0,
            use_ref_model: bool = True,
            loss_type: str = "bnpo",
            max_completion_length: int | None = None,
            importance_sampling_level: str = "token",
        ):
            super().__init__()
            self.lin = torch.nn.Linear(
                in_features=H, out_features=V, bias=bias, dtype=dtype
            )
            self.ref_lin = torch.nn.Linear(
                in_features=H, out_features=V, bias=bias, dtype=dtype
            )
            self.beta = beta
            self.epsilon_low = epsilon_low
            self.epsilon_high = epsilon_high
            self.temperature = temperature
            self.use_ref_model = use_ref_model
            self.loss_type = loss_type
            self.max_completion_length = max_completion_length
            self.importance_sampling_level = importance_sampling_level
            if self.loss_type == "dr_grpo":
                assert self.max_completion_length is not None, (
                    "max_completion_length must be provided for dr_grpo"
                )

        def compute_per_token_loss(
            self,
            x,  # Shape: [B, T, H]
            selected_token_ids,  # Shape: [B, T]
            attention_mask,  # Shape: [B, T]
            advantages,  # Shape: [B,]
            ref_input,  # Shape: [B, T, H]
        ):
            """Compute per-token losses and intermediate values needed for metrics.

            Returns:
                per_token_loss: Shape [B, T] for token-level, [B, 1] for sequence-level
                kl_div: Shape [B, T] (only if beta != 0.0, else None)
                coef_1: Shape [B, T] for token-level, [B, 1] for sequence-level
            """
            logits = x @ self.lin.weight.t()  # Shape: [B, T, V]
            if self.lin.bias is not None:
                logits = logits + self.lin.bias.to(x.dtype)
            if self.temperature != 1.0:
                logits = logits / self.temperature
            # Get log probabilities - preserve original dtype for gradient flow
            log_probs = F.log_softmax(logits, dim=-1)  # Shape: [B, T, V]

            # Get chosen token probabilities
            per_token_logps = log_probs.gather(
                dim=-1, index=selected_token_ids.unsqueeze(-1)
            ).squeeze(-1)  # Shape: [B, T]

            # Get reference model probabilities,
            if self.use_ref_model:
                with torch.no_grad():
                    ref_logits = ref_input @ self.ref_lin.weight.t()
                    if self.ref_lin.bias is not None:
                        ref_logits = ref_logits + self.ref_lin.bias.to(ref_input.dtype)
                    if self.temperature != 1.0:
                        ref_logits = ref_logits / self.temperature
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_per_token_logps = ref_log_probs.gather(
                        dim=-1, index=selected_token_ids.unsqueeze(-1)
                    ).squeeze(-1)
            else:
                ref_per_token_logps = per_token_logps.detach()

            # Compute policy gradient loss with importance sampling ratio
            old_per_token_logps = per_token_logps.detach()
            log_ratio = per_token_logps - old_per_token_logps

            if self.importance_sampling_level == "token":
                log_importance_weights = log_ratio
            elif self.importance_sampling_level == "sequence":
                log_importance_weights = (log_ratio * attention_mask).sum(
                    -1
                ) / attention_mask.sum(-1).clamp(min=1.0)
                log_importance_weights = log_importance_weights.unsqueeze(-1)
            else:
                raise ValueError(
                    f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                    "and 'sequence'."
                )

            coef_1 = torch.exp(log_importance_weights)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            kl_div = None
            if self.beta != 0.0:
                # Compute KL divergence between model and reference model - preserve dtype
                kl_div = (
                    torch.exp(ref_per_token_logps - per_token_logps)
                    - (ref_per_token_logps - per_token_logps)
                    - 1.0
                )
                per_token_loss = per_token_loss + self.beta * kl_div

            return per_token_loss, kl_div, coef_1

        def compute_loss_reduction(self, per_token_loss, attention_mask):
            """Compute the final loss by applying the appropriate reduction strategy."""
            if self.loss_type == "grpo":
                # Per-sequence normalization, then batch averaging
                return (
                    (per_token_loss * attention_mask).sum(-1)
                    / torch.clamp(attention_mask.sum(-1), min=1.0)
                ).mean()
            elif self.loss_type == "bnpo":
                # Global normalization by total valid tokens
                return (per_token_loss * attention_mask).sum() / torch.clamp(
                    attention_mask.sum(), min=1.0
                )
            elif self.loss_type == "dr_grpo":
                # Global normalization by total expected tokens
                return (per_token_loss * attention_mask).sum() / (
                    per_token_loss.size(0) * self.max_completion_length
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

        def compute_metrics(self, kl_div, coef_1, attention_mask, advantages):
            """Compute metrics: KL divergence and clipping rate."""
            metrics = []

            # KL divergence metric
            if self.beta != 0.0 and kl_div is not None:
                metrics.append(
                    (kl_div * attention_mask).sum()
                    / torch.clamp(attention_mask.sum(), min=1.0)
                )

            # Clipping metric
            if self.importance_sampling_level == "token":
                is_clipped = (
                    (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
                ) | ((coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0))
            else:  # sequence level
                # For sequence level, coef_1 is shape (B, 1), advantages is shape (B,)
                is_clipped = (
                    (coef_1.squeeze(-1) < 1 - self.epsilon_low) & (advantages < 0)
                ) | ((coef_1.squeeze(-1) > 1 + self.epsilon_high) & (advantages > 0))
                is_clipped = is_clipped.unsqueeze(1).expand_as(attention_mask)

            metrics.append(
                (is_clipped * attention_mask).sum()
                / torch.clamp(attention_mask.sum(), min=1.0)
            )
            return metrics

        def forward(
            self,
            x,  # Shape: [B, T, H]
            selected_token_ids,  # Shape: [B, T]
            attention_mask,  # Shape: [B, T]
            advantages,  # Shape: [B,]
            ref_input,  # Shape: [B, T, H]
        ):
            # Compute per-token losses and intermediate values
            B, T, H = x.size()
            x = x.view(-1, H)  # Flatten batch and sequence dimensions
            selected_token_ids = selected_token_ids.view(-1)
            attention_mask = attention_mask.view(-1)
            advantages = advantages.unsqueeze(-1).expand(B, T).reshape(-1)
            ref_input = ref_input.view(-1, H)
            per_token_loss, kl_div, coef_1 = self.compute_per_token_loss(
                x, selected_token_ids, attention_mask, advantages, ref_input
            )

            # Compute final loss with appropriate reduction
            loss = self.compute_loss_reduction(per_token_loss, attention_mask)

            # Compute metrics
            metrics = self.compute_metrics(kl_div, coef_1, attention_mask, advantages)

            return loss, metrics

    class ScanChunkedGRPO(TorchGRPO):
        def __init__(self, chunk_size=1, **kwargs):
            super().__init__(**kwargs)
            self.chunk_size = chunk_size

        def forward(
            self,
            x,  # Shape: [batch_size, seq_len, hidden_size]
            selected_token_ids,  # Shape: [batch_size, seq_len]
            attention_mask,  # Shape: [batch_size, seq_len]
            advantages,  # Shape: [batch_size,]
            ref_input,  # Shape: [batch_size, seq_len, hidden_size]
        ):
            from torch._higher_order_ops.scan import scan

            # Store original shapes and dtypes for proper restoration
            original_batch_size = x.size(0)
            original_dtype = x.dtype

            # Pad inputs if batch size is not divisible by chunk_size
            x, x_padding = pad_to_chunk_size(x, self.chunk_size, dim=0, pad_value=0)
            selected_token_ids, _ = pad_to_chunk_size(
                selected_token_ids, self.chunk_size, dim=0, pad_value=0
            )
            attention_mask, _ = pad_to_chunk_size(
                attention_mask, self.chunk_size, dim=0, pad_value=0
            )  # Padded attention should be 0
            advantages, _ = pad_to_chunk_size(
                advantages, self.chunk_size, dim=0, pad_value=0
            )
            ref_input, _ = pad_to_chunk_size(
                ref_input, self.chunk_size, dim=0, pad_value=0
            )

            # Store total sequences after padding for correct normalization
            total_sequences = (
                original_batch_size  # Use original batch size for loss computation
            )

            # Chunk the padded inputs along batch dimension
            x = view_as_chunked(x, self.chunk_size)
            selected_token_ids = view_as_chunked(selected_token_ids, self.chunk_size)
            attention_mask = view_as_chunked(attention_mask, self.chunk_size)
            advantages = view_as_chunked(advantages, self.chunk_size)
            ref_input = view_as_chunked(ref_input, self.chunk_size)

            # Initialize carry state: use float32 for all accumulators to ensure numerical stability and scan consistency
            init = (
                # All accumulators use float32 for consistent scan operation
                torch.zeros((), dtype=torch.float32, device=x.device),  # grpo_loss_sum
                torch.zeros(
                    (), dtype=torch.float32, device=x.device
                ),  # total_loss_numerator
                torch.zeros(
                    (), dtype=torch.float32, device=x.device
                ),  # total_valid_tokens
                torch.zeros(
                    (), dtype=torch.float32, device=x.device
                ),  # total_kl_numerator
                torch.zeros(
                    (), dtype=torch.float32, device=x.device
                ),  # total_clipping_numerator
                torch.zeros(
                    (), dtype=torch.float32, device=x.device
                ),  # total_valid_tokens_for_metrics
            )

            # Input tensors to scan over: stacked chunked inputs
            # Shape: [num_chunks, chunk_size, ...]
            xs = (x, selected_token_ids, attention_mask, advantages, ref_input)

            def combine_fn(carry, chunk_inputs):
                """Process one chunk and update carry state."""
                # Unpack carry (accumulators)
                (
                    grpo_loss_sum,
                    total_loss_numerator,
                    total_valid_tokens,
                    total_kl_numerator,
                    total_clipping_numerator,
                    total_valid_tokens_for_metrics,
                ) = carry

                # Unpack chunk inputs
                (
                    x_chunk,
                    selected_token_ids_chunk,
                    attention_mask_chunk,
                    advantages_chunk,
                    ref_input_chunk,
                ) = chunk_inputs

                # Compute per-token losses and intermediate values for this chunk
                per_token_loss_chunk, kl_div_chunk, coef_1_chunk = (
                    self.compute_per_token_loss(
                        x_chunk,
                        selected_token_ids_chunk,
                        attention_mask_chunk,
                        advantages_chunk,
                        ref_input_chunk,
                    )
                )

                # Ensure attention mask matches the dtype for consistent operations
                attention_mask_chunk = attention_mask_chunk.to(original_dtype)

                # Update loss accumulators based on loss type - use float32 for accumulation consistency
                if self.loss_type == "grpo":
                    # Running mean: accumulate per-sequence losses for this chunk
                    per_seq_losses_chunk = (
                        per_token_loss_chunk * attention_mask_chunk
                    ).sum(-1) / torch.clamp(attention_mask_chunk.sum(-1), min=1.0)
                    grpo_loss_sum = grpo_loss_sum + per_seq_losses_chunk.sum().float()
                else:
                    # Accumulate numerator - convert to float32 for consistent accumulation
                    loss_numerator_chunk = (
                        (per_token_loss_chunk * attention_mask_chunk).sum().float()
                    )
                    total_loss_numerator = total_loss_numerator + loss_numerator_chunk

                    if self.loss_type == "bnpo":
                        # Accumulate valid tokens - use float32 for scan consistency
                        valid_tokens_chunk = attention_mask_chunk.sum().float()
                        total_valid_tokens = total_valid_tokens + valid_tokens_chunk

                # Update metric accumulators - use float32 for scan consistency
                valid_tokens_chunk_for_metrics = attention_mask_chunk.sum().float()
                total_valid_tokens_for_metrics = (
                    total_valid_tokens_for_metrics + valid_tokens_chunk_for_metrics
                )

                if self.beta != 0.0 and kl_div_chunk is not None:
                    kl_numerator_chunk = (
                        (kl_div_chunk * attention_mask_chunk).sum().float()
                    )
                    total_kl_numerator = total_kl_numerator + kl_numerator_chunk

                # Compute clipping for this chunk - match the original TorchGRPO logic exactly
                if self.importance_sampling_level == "token":
                    is_clipped_chunk = (
                        (coef_1_chunk < 1 - self.epsilon_low)
                        & (advantages_chunk.unsqueeze(1) < 0)
                    ) | (
                        (coef_1_chunk > 1 + self.epsilon_high)
                        & (advantages_chunk.unsqueeze(1) > 0)
                    )
                else:  # sequence level
                    # For sequence level, coef_1_chunk is shape (chunk_size, 1), advantages_chunk is shape (chunk_size,)
                    is_clipped_chunk = (
                        (coef_1_chunk.squeeze(-1) < 1 - self.epsilon_low)
                        & (advantages_chunk < 0)
                    ) | (
                        (coef_1_chunk.squeeze(-1) > 1 + self.epsilon_high)
                        & (advantages_chunk > 0)
                    )
                    is_clipped_chunk = is_clipped_chunk.unsqueeze(1).expand_as(
                        attention_mask_chunk
                    )

                clipping_numerator_chunk = (
                    (is_clipped_chunk * attention_mask_chunk).sum().float()
                )
                total_clipping_numerator = (
                    total_clipping_numerator + clipping_numerator_chunk
                )

                # Return updated carry and dummy output (not used)
                # Clone tensors to avoid aliasing issues with scan
                next_carry = (
                    grpo_loss_sum.clone(),
                    total_loss_numerator.clone(),
                    total_valid_tokens.clone(),
                    total_kl_numerator.clone(),
                    total_clipping_numerator.clone(),
                    total_valid_tokens_for_metrics.clone(),
                )
                dummy_output = torch.tensor(
                    0.0, device=x.device, dtype=original_dtype
                )  # Match original dtype
                return next_carry, dummy_output

            # Run scan operation
            final_carry, _ = scan(combine_fn, init, xs, dim=0)

            # Unpack final accumulators
            (
                grpo_loss_sum,
                total_loss_numerator,
                total_valid_tokens,
                total_kl_numerator,
                total_clipping_numerator,
                total_valid_tokens_for_metrics,
            ) = final_carry

            # Compute final loss - ensure proper dtype handling
            if self.loss_type == "grpo":
                # Running mean: divide accumulated sum by total sequence count (use original batch size)
                loss = grpo_loss_sum / float(total_sequences)
            elif self.loss_type == "bnpo":
                # Global normalization by total valid tokens
                loss = total_loss_numerator / torch.clamp(total_valid_tokens, min=1.0)
            else:  # dr_grpo
                # Global normalization by total expected tokens (original batch size * max_completion_length)
                total_expected_tokens = float(
                    total_sequences * self.max_completion_length
                )
                loss = total_loss_numerator / total_expected_tokens

            # Compute final metrics
            metrics = []
            if self.beta != 0.0:
                kl_metric = total_kl_numerator / torch.clamp(
                    total_valid_tokens_for_metrics, min=1.0
                )
                metrics.append(kl_metric)

            clipping_metric = total_clipping_numerator / torch.clamp(
                total_valid_tokens_for_metrics, min=1.0
            )
            metrics.append(clipping_metric)

            return loss, metrics


class TestChunkedLosses(TestCase):
    def _assertEqual_with_dtype_tolerance(self, a, b, dtype):
        """Assert equality with dtype-specific tolerances for numerical precision."""
        if dtype == torch.bfloat16:
            # Higher tolerances for bfloat16 due to lower precision
            self.assertEqual(a, b, atol=2e-4, rtol=1e-1)
        else:
            # Standard tolerances for float32 and other dtypes
            self.assertEqual(a, b)

    @parametrize(
        "device",
        ["cpu", "cuda"],
    )
    @parametrize(
        "dtype",
        [torch.bfloat16, torch.float32],
    )
    @parametrize("bias", [True, False])
    @parametrize(
        "beta, epsilon_low, epsilon_high, temperature",
        [
            # Standard settings
            (0.1, 0.2, 0.2, 1.0),
            (0.0, 0.1, 0.1, 2.0),
        ],
    )
    @parametrize("use_ref_model", [True, False])
    @parametrize("loss_type", ["bnpo", "grpo", "dr_grpo"])
    @parametrize("importance_sampling_level", ["token", "sequence"])
    @parametrize("chunk_size", [3])
    def test_grpo_loss(
        self,
        device,
        dtype,
        bias,
        beta,
        epsilon_low,
        epsilon_high,
        temperature,
        use_ref_model,
        loss_type,
        importance_sampling_level,
        chunk_size,
    ):
        B, T, H, V = 3, 47, 31, 72  # random shape
        # B, T, H, V = 3, 1024, 4096, 128256  # random shape
        # importance_sampling_level = "token"
        # dtype = torch.bfloat16
        #         "T": 1024,
        #         "H": 4096,
        #         "V": 128256,
        #         "importance_sampling_level": "token",
        #         "dtype": torch.bfloat16,
        model_kwargs = {
            "H": H,
            "V": V,
            "dtype": dtype,
            "bias": bias,
            "beta": beta,
            "epsilon_low": epsilon_low,
            "epsilon_high": epsilon_high,
            "temperature": temperature,
            "use_ref_model": use_ref_model,
            "loss_type": loss_type,
            "max_completion_length": T if loss_type == "dr_grpo" else None,
            "importance_sampling_level": importance_sampling_level,
        }
        torch_loss_model = GRPO.TorchGRPO(**model_kwargs).to(device)
        torch_loss_model_compile = GRPO.TorchGRPO(**model_kwargs).to(device)
        torch_loss_model_compile.load_state_dict(torch_loss_model.state_dict())
        torch_loss_model_compile.compile()
        torch_loss_model_chunked = GRPO.ScanChunkedGRPO(
            chunk_size=chunk_size, **model_kwargs
        ).to(device)
        torch_loss_model_chunked.load_state_dict(torch_loss_model.state_dict())
        torch_loss_model_chunked_compile = GRPO.ScanChunkedGRPO(
            chunk_size=chunk_size, **model_kwargs
        ).to(device)
        torch_loss_model_chunked_compile.compile(fullgraph=True)
        torch_loss_model_chunked_compile.load_state_dict(
            torch_loss_model_chunked.state_dict()
        )

        _input = torch.randn(B, T, H, requires_grad=True, dtype=dtype, device=device)
        selected_token_ids = torch.randint(
            0, V, (B, T), dtype=torch.long, device=device
        )
        attention_mask = torch.ones(B, T, device=device)
        advantages = torch.randn(B, dtype=dtype, device=device)
        ref_input = torch.randn(B, T, H, dtype=dtype, device=device)

        fw_args = [_input, selected_token_ids, attention_mask, advantages, ref_input]
        fw_args2 = [x.detach().clone().requires_grad_(x.requires_grad) for x in fw_args]
        fw_args3 = [x.detach().clone().requires_grad_(x.requires_grad) for x in fw_args]
        fw_args4 = [x.detach().clone().requires_grad_(x.requires_grad) for x in fw_args]

        loss = torch_loss_model(*fw_args)
        loss_compile = torch_loss_model_compile(*fw_args2)
        loss_chunked = torch_loss_model_chunked(*fw_args3)
        loss_chunked_compile = torch_loss_model_chunked_compile(*fw_args4)

        loss[0].backward()  # loss is a tuple of loss and metrics
        loss_compile[0].backward()
        loss_chunked[0].backward()
        loss_chunked_compile[0].backward()

        def _map_none_grad_to_zero_tensor(t):
            return t.grad if t.grad is not None else torch.zeros_like(t)

        grads = [
            _map_none_grad_to_zero_tensor(x)
            for x in fw_args + list(torch_loss_model.parameters())
        ]
        grads_compile = [
            _map_none_grad_to_zero_tensor(x)
            for x in fw_args2 + list(torch_loss_model_compile.parameters())
        ]
        grads_chunked = [
            _map_none_grad_to_zero_tensor(x)
            for x in fw_args3 + list(torch_loss_model_chunked.parameters())
        ]
        grads_chunked_compile = [
            _map_none_grad_to_zero_tensor(x)
            for x in fw_args4 + list(torch_loss_model_chunked_compile.parameters())
        ]

        self._assertEqual_with_dtype_tolerance(loss, loss_compile, dtype)
        self._assertEqual_with_dtype_tolerance(loss, loss_chunked, dtype)
        self._assertEqual_with_dtype_tolerance(loss, loss_chunked_compile, dtype)
        self._assertEqual_with_dtype_tolerance(grads, grads_compile, dtype)
        self._assertEqual_with_dtype_tolerance(grads, grads_chunked, dtype)
        self._assertEqual_with_dtype_tolerance(grads, grads_chunked_compile, dtype)


instantiate_parametrized_tests(TestChunkedLosses)
