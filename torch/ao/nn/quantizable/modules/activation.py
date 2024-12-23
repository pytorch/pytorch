# mypy: allow-untyped-defs
import warnings
from typing import Optional, Tuple

import torch
import torch.jit  # this is needed to avoid a circular import
import torch.nn.functional as F
from torch import nn, Tensor


__all__ = ["MultiheadAttention"]


class MultiheadAttention(nn.MultiheadAttention):
    _FLOAT_MODULE = nn.MultiheadAttention

    r"""Quantizable implementation of the MultiheadAttention.

    Note::
        Please, refer to :class:`~torch.nn.MultiheadAttention` for more
        information

    Allows the model to jointly attend to information from different
    representation subspaces.
    See reference: Attention Is All You Need

    The original MHA module is not quantizable.
    This reimplements it by explicitly instantiating the linear layers.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> multihead_attn = nnqa.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    Note::
        Please, follow the quantization flow to convert the quantizable MHA.
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            **factory_kwargs,
        )
        self.linear_Q = nn.Linear(
            self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs
        )
        self.linear_K = nn.Linear(
            self.kdim, self.embed_dim, bias=bias, **factory_kwargs
        )
        self.linear_V = nn.Linear(
            self.vdim, self.embed_dim, bias=bias, **factory_kwargs
        )
        # for the type: ignore, see https://github.com/pytorch/pytorch/issues/58969
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)  # type: ignore[assignment]

        # Functionals
        self.q_scaling_product = torch.ao.nn.quantized.FloatFunctional()
        # note: importing torch.ao.nn.quantized at top creates a circular import

        # Quant/Dequant
        self.quant_attn_output = torch.ao.quantization.QuantStub()
        self.quant_attn_output_weights = torch.ao.quantization.QuantStub()
        self.dequant_q = torch.ao.quantization.DeQuantStub()
        self.dequant_k = torch.ao.quantization.DeQuantStub()
        self.dequant_v = torch.ao.quantization.DeQuantStub()

    def _get_name(self):
        return "QuantizableMultiheadAttention"

    @classmethod
    def from_float(cls, other):
        assert type(other) == cls._FLOAT_MODULE
        assert hasattr(other, "qconfig"), "The float module must have 'qconfig'"
        # Setting the dropout to 0.0!
        observed = cls(
            other.embed_dim,
            other.num_heads,
            other.dropout,
            (other.in_proj_bias is not None),
            (other.bias_k is not None),
            other.add_zero_attn,
            other.kdim,
            other.vdim,
            other.batch_first,
        )
        observed.bias_k = other.bias_k
        observed.bias_v = other.bias_v
        observed.qconfig = other.qconfig

        # Set the linear weights
        # for the type: ignores, see https://github.com/pytorch/pytorch/issues/58969
        observed.out_proj.weight = other.out_proj.weight
        observed.out_proj.bias = other.out_proj.bias
        if other._qkv_same_embed_dim:
            # Use separate params
            bias = other.in_proj_bias
            _start = 0
            _end = _start + other.embed_dim
            weight = other.in_proj_weight[_start:_end, :]
            if bias is not None:
                bias = torch.nn.Parameter(bias[_start:_end], bias.requires_grad)
            observed.linear_Q.weight = torch.nn.Parameter(weight, weight.requires_grad)
            observed.linear_Q.bias = bias

            bias = other.in_proj_bias
            _start = _end
            _end = _start + other.embed_dim
            weight = other.in_proj_weight[_start:_end, :]
            if bias is not None:
                bias = torch.nn.Parameter(bias[_start:_end], bias.requires_grad)
            observed.linear_K.weight = torch.nn.Parameter(weight, weight.requires_grad)
            observed.linear_K.bias = bias

            bias = other.in_proj_bias
            _start = _end
            weight = other.in_proj_weight[_start:, :]
            if bias is not None:
                bias = torch.nn.Parameter(bias[_start:], bias.requires_grad)
            observed.linear_V.weight = torch.nn.Parameter(weight, weight.requires_grad)
            observed.linear_V.bias = bias
        else:
            observed.linear_Q.weight = nn.Parameter(other.q_proj_weight)
            observed.linear_K.weight = nn.Parameter(other.k_proj_weight)
            observed.linear_V.weight = nn.Parameter(other.v_proj_weight)
            if other.in_proj_bias is None:
                observed.linear_Q.bias = None
                observed.linear_K.bias = None
                observed.linear_V.bias = None
            else:
                observed.linear_Q.bias = nn.Parameter(
                    other.in_proj_bias[0 : other.embed_dim]
                )
                observed.linear_K.bias = nn.Parameter(
                    other.in_proj_bias[other.embed_dim : (other.embed_dim * 2)]
                )
                observed.linear_V.bias = nn.Parameter(
                    other.in_proj_bias[(other.embed_dim * 2) :]
                )
        observed.eval()
        # Explicit prepare
        observed = torch.ao.quantization.prepare(observed, inplace=True)
        return observed

    @torch.jit.unused
    def dequantize(self):
        r"""Utility to convert the quantized MHA back to float.

        The motivation for this is that it is not trivial to conver the weights
        from the format that is used in the quantized version back to the
        float.
        """
        fp = self._FLOAT_MODULE(
            self.embed_dim,
            self.num_heads,
            self.dropout,
            (self.linear_Q._weight_bias()[1] is not None),  # type: ignore[operator]
            (self.bias_k is not None),
            self.add_zero_attn,
            self.kdim,
            self.vdim,
            self.batch_first,
        )
        assert fp._qkv_same_embed_dim == self._qkv_same_embed_dim
        if self.bias_k is not None:
            fp.bias_k = nn.Parameter(self.bias_k.dequantize())
        if self.bias_v is not None:
            fp.bias_v = nn.Parameter(self.bias_v.dequantize())

        # Set the linear weights
        # Note: Because the linear layers are quantized, mypy does not nkow how
        # to deal with them -- might need to ignore the typing checks.
        # for the type: ignore[has-type], see https://github.com/pytorch/pytorch/issues/58969
        w, b = self.out_proj._weight_bias()  # type: ignore[operator, has-type]
        fp.out_proj.weight = nn.Parameter(w.dequantize())
        if b is not None:
            fp.out_proj.bias = nn.Parameter(b)

        wQ, bQ = self.linear_Q._weight_bias()  # type: ignore[operator]
        wQ = wQ.dequantize()
        wK, bK = self.linear_K._weight_bias()  # type: ignore[operator]
        wK = wK.dequantize()
        wV, bV = self.linear_V._weight_bias()  # type: ignore[operator]
        wV = wV.dequantize()
        if fp._qkv_same_embed_dim:
            # Use separate params
            _start = 0
            _end = _start + fp.embed_dim
            fp.in_proj_weight[_start:_end, :] = wQ
            if fp.in_proj_bias is not None:
                assert all(bQ == 0)
                fp.in_proj_bias[_start:_end] = bQ

            _start = _end
            _end = _start + fp.embed_dim
            fp.in_proj_weight[_start:_end, :] = wK
            if fp.in_proj_bias is not None:
                assert all(bK == 0)
                fp.in_proj_bias[_start:_end] = bK

            _start = _end
            fp.in_proj_weight[_start:, :] = wV
            if fp.in_proj_bias is not None:
                assert all(bV == 0)
                fp.in_proj_bias[_start:] = bV
        else:
            fp.q_proj_weight = nn.Parameter(wQ)
            fp.k_proj_weight = nn.Parameter(wK)
            fp.v_proj_weight = nn.Parameter(wV)
            if fp.in_proj_bias is None:
                self.linear_Q.bias = None
                self.linear_K.bias = None
                self.linear_V.bias = None
            else:
                fp.in_proj_bias[0 : fp.embed_dim] = bQ
                fp.in_proj_bias[fp.embed_dim : (fp.embed_dim * 2)] = bK
                fp.in_proj_bias[(fp.embed_dim * 2) :] = bV

        return fp

    @classmethod
    def from_observed(cls, other):
        # The whole flow is float -> observed -> quantized
        # This class does float -> observed only
        # See nn.quantized.MultiheadAttention
        raise NotImplementedError(
            "It looks like you are trying to prepare an "
            "MHA module. Please, see "
            "the examples on quantizable MHAs."
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Note::
            Please, refer to :func:`~torch.nn.MultiheadAttention.forward` for more
            information

        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored (masked out) while the position with the value of ``False`` will be unchanged.
              FloatTensor: specified additive penalties to the attention weights. Typically, large negatves values like -inf mask positions
              by reducing their attention contributions, while 0.0 keeps positions unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
              positions. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.
            - is_causal: If specified, applies a causal mask as attention mask. Mutually exclusive with providing attn_mask.
              Default: ``False``.
            - average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
              heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
              effect when ``need_weights=True.``. Default: True (i.e. average weights across heads)

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - attn_output_weights: If ``average_attn_weights=True``, returns attention weights averaged
              across heads of shape :math:`(N, L, S)`, where N is the batch size, L is the target sequence length,
              S is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(N, num_heads, L, S)`.
        """
        return self._forward_impl(
            query,
            key,
            value,
            key_padding_mask,
            need_weights,
            attn_mask,
            average_attn_weights,
            is_causal,
        )

    def _forward_impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # This version will not deal with the static key/value pairs.
        # Keeping it here for future changes.
        #
        # TODO: This method has some duplicate lines with the
        # `torch.nn.functional.multi_head_attention`. Will need to refactor.
        static_k = None
        static_v = None

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")

        if is_causal:
            raise AssertionError("causal mask not supported by AO MHA module")

        if self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))

        tgt_len, bsz, embed_dim_to_check = query.size()
        assert self.embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = self.embed_dim // self.num_heads
        assert (
            head_dim * self.num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = self.linear_Q(query)
        k = self.linear_K(key)
        v = self.linear_V(value)

        q = self.q_scaling_product.mul_scalar(q, scaling)

        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for `attn_mask` in `nn.MultiheadAttention` is deprecated. "
                    "Use bool tensor instead.",
                    stacklevel=3,
                )
                attn_mask = attn_mask.to(torch.bool)
            assert (
                attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
            ), f"Only float and bool types are supported for attn_mask, not {attn_mask.dtype}"

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * self.num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for `key_padding_mask` in `nn.MultiheadAttention` is deprecated. "
                "Use bool tensor instead.",
                stacklevel=3,
            )
            key_padding_mask = key_padding_mask.to(torch.bool)
        if self.bias_k is not None and self.bias_v is not None:
            if static_k is None and static_v is None:
                # Explicitly assert that bias_k and bias_v are not None
                # in a way that TorchScript can understand.
                bias_k = self.bias_k
                assert bias_k is not None
                bias_v = self.bias_v
                assert bias_v is not None

                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = F.pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = F.pad(key_padding_mask, (0, 1))
            else:
                assert static_k is None, "bias cannot be added to static key."
                assert static_v is None, "bias cannot be added to static value."
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if static_k is not None:
            assert static_k.size(0) == bsz * self.num_heads
            assert static_k.size(2) == head_dim
            k = static_k

        if static_v is not None:
            assert static_v.size(0) == bsz * self.num_heads
            assert static_v.size(2) == head_dim
            v = static_v

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k_zeros = torch.zeros((k.size(0), 1) + k.size()[2:])
            if k.is_quantized:
                k_zeros = torch.quantize_per_tensor(
                    k_zeros, k.q_scale(), k.q_zero_point(), k.dtype
                )
            k = torch.cat([k, k_zeros], dim=1)
            v_zeros = torch.zeros((v.size(0), 1) + k.size()[2:])
            if v.is_quantized:
                v_zeros = torch.quantize_per_tensor(
                    v_zeros, v.q_scale(), v.q_zero_point(), v.dtype
                )
            v = torch.cat([v, v_zeros], dim=1)

            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # Leaving the quantized zone here
        q = self.dequant_q(q)
        k = self.dequant_k(k)
        v = self.dequant_v(v)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        if self.batch_first:
            attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        else:
            attn_output = (
                attn_output.transpose(0, 1)
                .contiguous()
                .view(tgt_len, bsz, self.embed_dim)
            )

        # Reentering the quantized zone
        attn_output = self.quant_attn_output(attn_output)
        # for the type: ignore[has-type], see https://github.com/pytorch/pytorch/issues/58969
        attn_output = self.out_proj(attn_output)  # type: ignore[has-type]
        attn_output_weights = self.quant_attn_output_weights(attn_output_weights)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            return attn_output, attn_output_weights
        else:
            return attn_output, None
