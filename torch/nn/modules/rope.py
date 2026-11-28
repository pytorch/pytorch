import torch
from torch import Tensor
from torch.nn import functional as F

from .module import Module


__all__ = ["RotaryEmbedding"]


class RotaryEmbedding(Module):
    r"""Apply Rotary Position Embedding (RoPE) to query/key tensors.

    Computes and caches cosine/sine tables for standard RoPE frequencies
    (:math:`\theta_i = \text{base}^{-2i/\text{dim}}`) and applies them
    via :func:`~torch.nn.functional.rotary_embedding`.

    The ``cos``/``sin`` tables are stored as non-persistent buffers so they
    travel with the module across devices (``.to("cuda")``) but are not
    included in ``state_dict`` (they can always be recomputed from
    ``dim``, ``max_seq_len``, and ``base``).

    **Extending for long-context variants:** Subclasses can override
    :meth:`build_rope_cache` to apply custom frequency modifications (e.g.,
    YaRN, NTK-aware, LLaMA3-style scaling) before the buffers are
    registered::

        class ScaledRotaryEmbedding(nn.RotaryEmbedding):
            def build_rope_cache(self, max_seq_len: int) -> None:
                self.max_seq_len = max_seq_len
                theta = 1.0 / (
                    self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
                )
                # apply custom scaling to theta here ...
                t = torch.arange(max_seq_len)
                freqs = torch.outer(t, theta)
                self.register_buffer("cos_cache", freqs.cos(), persistent=False)
                self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    Args:
        dim (int): Head dimension (number of features per head). Must be even.
        max_seq_len (int): Maximum sequence length for which to precompute
            the cos/sin cache. Longer sequences require calling
            :meth:`build_rope_cache` before the next :meth:`forward` call.
        base (int): Base for the frequency computation. Default: ``10000``.

    Shape:
        - Input ``x``: :math:`(B, H, S, D)` where :math:`S \leq \text{max\_seq\_len}`
        - Output: :math:`(B, H, S, D)`

    Example::

        >>> rope = nn.RotaryEmbedding(dim=64)
        >>> x = torch.randn(2, 8, 16, 64)
        >>> out = rope(x)
        >>> out.shape
        torch.Size([2, 8, 16, 64])

        >>> # Indexed mode with position_ids
        >>> position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)
        >>> out = rope(x, position_ids=position_ids)
        >>> out.shape
        torch.Size([2, 8, 16, 64])
    """

    cos_cache: Tensor
    sin_cache: Tensor

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.build_rope_cache(max_seq_len)

    def build_rope_cache(self, max_seq_len: int) -> None:
        """Compute and register ``cos_cache`` and ``sin_cache`` buffers.

        Call this to extend the cache when the model needs to process
        sequences longer than the current ``max_seq_len``::

            rope = nn.RotaryEmbedding(dim=64, max_seq_len=512)
            rope.build_rope_cache(1024)  # extend before long-context inference

        Subclasses can override this method to apply custom frequency
        scaling before registering the buffers.

        Args:
            max_seq_len (int): New maximum sequence length.
        """
        self.max_seq_len = max_seq_len
        # Query _buffers directly so we don't go through __getattr__ twice and
        # so a None-valued buffer (register_buffer("cos_cache", None)) doesn't
        # trick hasattr() into returning True and then blow up on .device.
        existing = self._buffers.get("cos_cache")
        device = existing.device if existing is not None else torch.device("cpu")
        theta = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
                / self.dim
            )
        )
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, theta)  # (max_seq_len, dim//2)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        """Apply rotary embeddings to ``x``.

        Args:
            x (Tensor): Shape ``(B, H, S, D)``.
            position_ids (Tensor, optional): Shape ``(B, S)``, integer indices
                into the cos/sin cache. When ``None``, positions
                ``0, 1, ..., S-1`` are used.

        Returns:
            Tensor: Same shape as ``x``.

        Raises:
            RuntimeError: If ``x.shape[2] > self.max_seq_len``. Call
                :meth:`build_rope_cache` with a larger value first.
        """
        seq_len = x.shape[2]
        if seq_len > self.max_seq_len:
            raise RuntimeError(
                f"Input sequence length {seq_len} exceeds max_seq_len "
                f"{self.max_seq_len}. Call build_rope_cache({seq_len}) first."
            )
        if position_ids is None:
            # Sequential positions 0..S-1; (S, dim//2) broadcasts over B and H.
            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
            return F.rotary_embedding(x, cos, sin)
        else:
            # Pass the full cache so the functional handles the indexing.
            return F.rotary_embedding(x, self.cos_cache, self.sin_cache, position_ids)
