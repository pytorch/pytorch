import dataclasses
from functools import cached_property

import torch


__all__ = []


def _make_empty(shape, dtype, device, when=True):
    # when=False returns a rank-matching empty (shape (0,)*len(shape)).
    return torch.empty(
        shape if when else (0,) * len(shape),
        device=device,
        dtype=dtype,
        requires_grad=False,
    )


def _make_zeros(shape, dtype, device, when=True):
    return torch.zeros(
        shape if when else (0,) * len(shape),
        device=device,
        dtype=dtype,
        requires_grad=False,
    )


def _linear_cross_entropy_batch_chunked_setup_context(ctx, inputs, output):
    *_, allow_retain_graph, compute_input_grad, compute_linear_weight_grad = inputs
    ctx.allow_retain_graph = allow_retain_graph
    ctx.compute_input_grad = compute_input_grad
    ctx.compute_linear_weight_grad = compute_linear_weight_grad
    _, grad_input, grad_linear_weight = output
    ctx._gi = grad_input if compute_input_grad else None
    ctx._gw = grad_linear_weight if compute_linear_weight_grad else None


@dataclasses.dataclass
class _ChunkViews:
    """Per-iteration tensor views; each property picks the right operand
    or scratch from ctx dispatch flags so call sites read raw.
    """

    ctx: "_ChunkContext"
    bchunk_start: int
    bchunk_size: int
    input_chunk: torch.Tensor
    target_chunk: torch.Tensor
    weight_chunk: torch.Tensor
    logits: torch.Tensor
    input_chunk_acc: torch.Tensor

    @property
    def input(self) -> torch.Tensor:
        return (
            self.input_chunk_acc
            if self.ctx.forward_uses_acc_input
            else self.input_chunk
        )

    @property
    def linear_weight(self) -> torch.Tensor:
        ctx = self.ctx
        return (
            ctx.linear_weight
            if ctx.forward_uses_cuda_out_dtype
            else ctx.linear_weight_cast
        )

    @property
    def input_grad_logits(self) -> torch.Tensor:
        if self.ctx.input_grad_uses_logits_lw:
            return self.logits_downcast
        return self.logits

    @property
    def input_grad_linear_weight(self) -> torch.Tensor:
        ctx = self.ctx
        return (
            ctx.linear_weight
            if ctx.input_grad_uses_logits_lw
            else ctx.linear_weight_cast
        )

    @property
    def weight_grad_input(self) -> torch.Tensor:
        ctx = self.ctx
        if ctx.is_cuda and not ctx.weight_grad_mm_same_dtype:
            return self.input_chunk
        return self.input_chunk_acc

    @property
    def logits_upcast(self) -> torch.Tensor:
        # Non-CUDA mixed-dtype: copy into logits_acc_buf for matching-dtype mm.
        ctx = self.ctx
        if ctx.is_cuda or ctx.weight_grad_mm_same_dtype:
            return self.logits
        return ctx.logits_acc_buf.narrow(0, 0, self.bchunk_size).copy_(self.logits)

    @cached_property
    def grad_input_chunk(self) -> torch.Tensor:
        # Buf-and-copy commit happens in ctx.chunks() post-yield.
        ctx = self.ctx
        if ctx.alloc_input_grad_acc_buf:
            return ctx.input_grad_acc_buf.narrow(0, 0, self.bchunk_size)
        return ctx.grad_input.narrow(0, self.bchunk_start, self.bchunk_size)

    @cached_property
    def logits_downcast(self) -> torch.Tensor:
        # Lazy: first access must come AFTER in-place mods to self.logits.
        ctx = self.ctx
        if ctx.loop_caches_logits_downcast:
            return self.logits.to(ctx.linear_weight.dtype)
        return self.logits


@dataclasses.dataclass
class _ChunkContext:
    """Per-call state for the chunked loop, built once via ``build``.
    Methods hide dtype/device/acc_policy dispatch behind single math ops;
    dispatch-free per-iter math is inlined into the loop body. Buffers
    dispatch decided are not needed are rank-matching empty tensors
    (``when=False`` in ``_make_*``) so the dataclass surface stays uniform.
    """

    dtype: torch.dtype
    num_batches: int
    in_features: int
    num_classes: int
    is_cuda: bool
    use_acc_dtype: bool

    acc_dtype: torch.dtype
    weight_chunk_dtype: torch.dtype
    grad_input_dtype: torch.dtype
    linear_weight_cast_dtype: torch.dtype

    input: torch.Tensor
    target: torch.Tensor  # uncorrected; per-iter slice is from corrected_target
    weight: torch.Tensor | None
    ignore_index: int
    reduction: str
    linear_weight: torch.Tensor

    # Optional "when=" buffers (weight_grad_chunk, logits_acc_buf,
    # input_grad_acc_buf, input_chunk_acc_buf, grad_input, grad_linear_weight)
    # are cached_properties below.
    logits_buf: torch.Tensor
    tmp: torch.Tensor
    output: torch.Tensor

    compute_input_grad: bool
    compute_linear_weight_grad: bool
    alloc_weight_grad_chunk: bool
    alloc_input_grad_acc_buf: bool
    alloc_input_chunk_acc_buf: bool
    alloc_logits_acc_buf: bool
    forward_uses_acc_input: bool
    forward_uses_cuda_out_dtype: bool
    weight_grad_mm_same_dtype: bool
    input_grad_uses_logits_lw: bool
    loop_caches_logits_downcast: bool
    weight_grad_uses_logits_buf_temp: bool

    @cached_property
    def _mask(self) -> torch.Tensor:
        return self.target == self.ignore_index

    @cached_property
    def corrected_target(self) -> torch.Tensor:
        # Replace out-of-range ignore_index values with 0 lazily so
        # downstream index_select / index_add_ stay in bounds.
        if self.ignore_index < 0 or self.ignore_index >= self.num_classes:
            return torch.where(self._mask, 0, self.target)
        return self.target

    @cached_property
    def neg_weight_target(self) -> torch.Tensor:
        # Per-row weighting: -(weight[target] if weight else 1) / d on
        # unmasked positions, 0 on masked positions. d = sum of unmasked
        # weights for "mean", 1 for "sum".
        mask = self._mask
        target = self.corrected_target
        weight = self.weight
        weight_chunk_dtype = self.weight_chunk_dtype
        if weight is None:
            neg_weight_target = (~mask).to(weight_chunk_dtype)
        elif target.numel() > weight.numel():
            neg_weight_target = torch.where(
                mask, 0, weight.to(weight_chunk_dtype).index_select(0, target)
            )
        else:
            neg_weight_target = torch.where(
                mask, 0, weight.index_select(0, target).to(weight_chunk_dtype)
            )
        if self.reduction == "mean":
            d = neg_weight_target.sum()
            neg_weight_target.div_(torch.where(d == 0, torch.nan, -d))
        else:  # "sum"
            neg_weight_target.neg_()
        return neg_weight_target

    @cached_property
    def linear_weight_cast(self) -> torch.Tensor:
        if self.linear_weight_cast_dtype != self.dtype:
            return self.linear_weight.to(self.linear_weight_cast_dtype)
        return self.linear_weight

    @cached_property
    def weight_grad_chunk(self) -> torch.Tensor:
        # (V, F) acc_dtype accumulator for keep-path weight-grad mm;
        # rank-empty on the direct (compact) path.
        return _make_empty(
            (self.num_classes, self.in_features),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_weight_grad_chunk,
        )

    @cached_property
    def logits_acc_buf(self) -> torch.Tensor:
        # (B, V) scratch for non-CUDA mixed-dtype logits_upcast.
        return _make_empty(
            (self.logits_buf.shape[0], self.num_classes),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_logits_acc_buf,
        )

    @cached_property
    def input_grad_acc_buf(self) -> torch.Tensor:
        # (B, F) scratch for input-grad addmm on the buf-and-copy path
        # (CPU mixed-dtype, MPS).
        return _make_empty(
            (self.logits_buf.shape[0], self.in_features),
            self.linear_weight_cast_dtype,
            self.input.device,
            when=self.alloc_input_grad_acc_buf,
        )

    @cached_property
    def input_chunk_acc_buf(self) -> torch.Tensor:
        return _make_empty(
            (self.logits_buf.shape[0], self.in_features),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_input_chunk_acc_buf,
        )

    @cached_property
    def grad_input(self) -> torch.Tensor:
        return _make_empty(
            self.input.shape,
            self.grad_input_dtype,
            self.input.device,
            when=self.compute_input_grad,
        )

    @cached_property
    def grad_linear_weight(self) -> torch.Tensor:
        return _make_zeros(
            self.linear_weight.shape,
            self.dtype,
            self.input.device,
            when=self.compute_linear_weight_grad,
        )

    @classmethod
    def build(
        cls,
        input: torch.Tensor,
        linear_weight: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        reduction: str,
        ignore_index: int,
        label_smoothing: float,
        batch_chunk_size: int,
        acc_policy: str,
        acc_dtype: torch.dtype,
        compute_input_grad: bool,
        compute_linear_weight_grad: bool,
    ) -> "_ChunkContext":
        # ===== Validation =====
        if target.dtype != torch.int64:
            raise TypeError(
                f"linear_cross_entropy: target dtype must be torch.int64, got {target.dtype}."
            )
        if label_smoothing > 0.0:
            raise NotImplementedError(
                "linear_cross_entropy does not support label smoothing"
            )
        if reduction not in {"mean", "sum"}:
            raise NotImplementedError(
                f"linear_cross_entropy does not support {reduction=}"
            )

        device = input.device
        dtype = input.dtype
        num_batches, in_features = input.shape
        num_classes, _ = linear_weight.shape
        # CUDA gates the out_dtype= mm fast path; the non-CUDA path
        # (CPU, MPS, XPU, ...) routes mixed-dtype mm through explicit
        # casts until out_dtype= is validated on those backends.
        is_cuda = device.type == "cuda"
        is_mps = device.type == "mps"

        if dtype != acc_dtype and not (
            dtype in {torch.float16, torch.bfloat16} and acc_dtype == torch.float32
        ):
            raise RuntimeError(
                "linear_cross_entropy supports float32 acc_dtype with"
                f" float16/bfloat16 inputs, but got {acc_dtype} acc_dtype and {dtype} inputs."
            )
        use_acc_dtype = dtype != acc_dtype

        # Internal dtype layout. ``compact`` reuses ``balanced``'s layout;
        # its savings come from skipping the weight_grad_chunk scratch, not dtype.
        is_memory_like = acc_policy in {"balanced", "compact"}
        if use_acc_dtype:
            output_dtype = acc_dtype if dtype == torch.float16 else dtype
            grad_input_dtype = dtype if is_memory_like else acc_dtype
            # fp16 + memory-like keeps logits at fp16 (2BC vs 4BC);
            # everything else upcasts for softmax stability.
            logits_buf_dtype = (
                dtype if dtype == torch.float16 and is_memory_like else acc_dtype
            )
            # acc_dtype mandatory: 1/N for reduction="mean" is subnormal
            # in fp16 at N >= 65536.
            weight_chunk_dtype = acc_dtype
        else:
            output_dtype = grad_input_dtype = logits_buf_dtype = weight_chunk_dtype = (
                dtype
            )

        # ===== Dispatch flags =====
        # CUDA + balanced uses cuBLAS out_dtype= directly; no cast.
        needs_linear_weight_cast = use_acc_dtype and (
            not is_cuda or (compute_input_grad and grad_input_dtype == logits_buf_dtype)
        )
        linear_weight_cast_dtype = (
            logits_buf_dtype if needs_linear_weight_cast else dtype
        )
        alloc_weight_grad_chunk = compute_linear_weight_grad and not (
            acc_policy == "compact" and (is_cuda or logits_buf_dtype == dtype)
        )
        alloc_input_grad_acc_buf = (
            compute_input_grad
            and not is_cuda
            and (grad_input_dtype != linear_weight_cast_dtype or is_mps)
        )
        alloc_input_chunk_acc_buf = use_acc_dtype and (
            compute_linear_weight_grad or (not is_cuda and dtype != logits_buf_dtype)
        )
        forward_uses_acc_input = (
            use_acc_dtype and not is_cuda and dtype != logits_buf_dtype
        )
        forward_uses_cuda_out_dtype = is_cuda and use_acc_dtype
        weight_grad_mm_same_dtype = logits_buf_dtype == acc_dtype
        # Storage-trick fires when logits' dtype differs from the input-
        # grad accumulator dtype.
        input_grad_uses_logits_lw = (
            is_cuda and compute_input_grad and logits_buf_dtype != grad_input_dtype
        )
        # Per-iter ``logits.to(linear_weight.dtype)`` shared between the
        # inlined input-grad addmm and the inlined direct weight-grad addmm_.
        loop_caches_logits_downcast = input_grad_uses_logits_lw or (
            compute_linear_weight_grad
            and not alloc_weight_grad_chunk
            and logits_buf_dtype != dtype
        )
        alloc_logits_acc_buf = (
            alloc_weight_grad_chunk and not is_cuda and logits_buf_dtype != acc_dtype
        )
        # Direct weight-grad path: reuse logits_buf storage for the
        # acc_dtype->dtype cast when its bytes/row fits the (B, F) cast.
        weight_grad_uses_logits_buf_temp = (
            not alloc_weight_grad_chunk
            and use_acc_dtype
            and num_classes * logits_buf_dtype.itemsize >= in_features * dtype.itemsize
        )

        return cls(
            dtype=dtype,
            num_batches=num_batches,
            in_features=in_features,
            num_classes=num_classes,
            is_cuda=is_cuda,
            use_acc_dtype=use_acc_dtype,
            acc_dtype=acc_dtype,
            weight_chunk_dtype=weight_chunk_dtype,
            grad_input_dtype=grad_input_dtype,
            linear_weight_cast_dtype=linear_weight_cast_dtype,
            input=input,
            target=target,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            linear_weight=linear_weight,
            logits_buf=torch.empty(
                (batch_chunk_size, num_classes),
                dtype=logits_buf_dtype,
                device=device,
                requires_grad=False,
            ),
            tmp=torch.empty(
                (batch_chunk_size,),
                dtype=logits_buf_dtype,
                device=device,
                requires_grad=False,
            ),
            output=_make_zeros((), output_dtype, device),
            compute_input_grad=compute_input_grad,
            compute_linear_weight_grad=compute_linear_weight_grad,
            alloc_weight_grad_chunk=alloc_weight_grad_chunk,
            alloc_input_grad_acc_buf=alloc_input_grad_acc_buf,
            alloc_input_chunk_acc_buf=alloc_input_chunk_acc_buf,
            alloc_logits_acc_buf=alloc_logits_acc_buf,
            forward_uses_acc_input=forward_uses_acc_input,
            forward_uses_cuda_out_dtype=forward_uses_cuda_out_dtype,
            weight_grad_mm_same_dtype=weight_grad_mm_same_dtype,
            input_grad_uses_logits_lw=input_grad_uses_logits_lw,
            loop_caches_logits_downcast=loop_caches_logits_downcast,
            weight_grad_uses_logits_buf_temp=weight_grad_uses_logits_buf_temp,
        )

    def chunks(self):
        """Yield a ``_ChunkViews`` per iter. Post-yield, commit the buf-and-copy
        input-grad slice into ``grad_input`` (skipped when ``grad_input_chunk``
        aliases ``grad_input``).
        """
        batch_chunk_size = self.logits_buf.shape[0]
        for bchunk_start in range(0, self.num_batches, batch_chunk_size):
            bchunk_size = min(batch_chunk_size, self.num_batches - bchunk_start)
            chunk = self.bind_chunk(bchunk_start, bchunk_size)
            yield chunk
            if self.alloc_input_grad_acc_buf:
                self.grad_input.narrow(0, bchunk_start, bchunk_size).copy_(
                    chunk.grad_input_chunk
                )

    def bind_chunk(self, bchunk_start: int, bchunk_size: int) -> _ChunkViews:
        input_chunk = self.input.narrow(0, bchunk_start, bchunk_size)
        target_chunk = self.corrected_target.narrow(0, bchunk_start, bchunk_size)
        weight_chunk = self.neg_weight_target.narrow(0, bchunk_start, bchunk_size)
        logits = self.logits_buf.narrow(0, 0, bchunk_size)
        input_chunk_acc = (
            self.input_chunk_acc_buf.narrow(0, 0, bchunk_size).copy_(input_chunk)
            if self.alloc_input_chunk_acc_buf
            else input_chunk
        )
        return _ChunkViews(
            ctx=self,
            bchunk_start=bchunk_start,
            bchunk_size=bchunk_size,
            input_chunk=input_chunk,
            target_chunk=target_chunk,
            weight_chunk=weight_chunk,
            logits=logits,
            input_chunk_acc=input_chunk_acc,
        )

    def mm(self, mat1: torch.Tensor, mat2: torch.Tensor, *, out: torch.Tensor) -> None:
        # Mismatched dtype: cuBLAS ``out_dtype=`` upcasts inside the matmul.
        if mat1.dtype == out.dtype:
            torch.mm(mat1, mat2, out=out)
        else:
            torch.mm(mat1, mat2, out_dtype=out.dtype, out=out)

    def amax(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tmp.narrow(0, 0, x.shape[0]).unsqueeze(1)
        torch.amax(x, dim=1, keepdim=True, out=out)
        return out

    def dotgather(
        self, weight: torch.Tensor, x: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        return weight.dot(x.gather(1, indices.unsqueeze(1)).squeeze(1).to(weight.dtype))

    def sumexp_(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # Result always in acc_dtype: fp16 softmax_denom can underflow
        # on rows with widely-spread logits, after which log(0)=-inf
        # poisons the loss and 1/0 the gradient.
        return x.exp_().sum(dim, dtype=self.acc_dtype)

    def div(self, num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        factor = self.tmp.narrow(0, 0, num.shape[0])
        # MPS lacks fused cross-dtype torch.div(out=); copy_ fallback.
        if self.input.device.type == "mps" and (
            num.dtype != factor.dtype or den.dtype != factor.dtype
        ):
            factor.copy_(num / den)
        else:
            torch.div(num, den, out=factor)
        return factor

    def mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Pick a destination from in-flight scratch when possible.
        if self.use_acc_dtype:
            return x.mul_(w.unsqueeze(1))
        if self.num_classes >= self.in_features:
            return torch.mul(
                x,
                w.unsqueeze(1),
                out=self.logits_buf.narrow(0, 0, x.shape[0]).narrow(
                    1, 0, self.in_features
                ),
            )
        return x * w.unsqueeze(1)

    def to(self, x: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        # Reuse logits_buf storage as cast destination when its bytes/row fits.
        if x.dtype == dtype:
            return x
        if self.weight_grad_uses_logits_buf_temp:
            x = (
                self.logits_buf.view(dtype)
                .narrow(0, 0, x.shape[0])
                .narrow(1, 0, x.shape[1])
                .copy_(x)
            )
            return x
        return x.to(dtype)


# Private op. Returns ``(loss, grad_input, grad_linear_weight)`` because
# register_autograd has no save_for_backward: grads flow via return tuple,
# stash on ctx in ``setup_context``, backward mutates them in place (``.mul_()``).
@torch.library.custom_op(
    "torch_nn::_linear_cross_entropy_batch_chunked", mutates_args=()
)
def _linear_cross_entropy_batch_chunked(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    reduction: str,
    ignore_index: int,
    label_smoothing: float,
    batch_chunk_size: int,
    acc_policy: str,
    acc_dtype: torch.dtype,
    allow_retain_graph: bool,
    compute_input_grad: bool,
    compute_linear_weight_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(loss, grad_input, grad_linear_weight)``. Grads precomputed
    in forward and stashed on ctx; backward is a single multiply by the
    upstream grad -- computing in forward is what lets chunking save memory.
    """
    # Direct callers must resolve "auto" / None via _adjust first.
    if acc_policy == "auto" or acc_dtype is None:
        raise RuntimeError(
            f"unresolved acc_policy={acc_policy!r} or acc_dtype={acc_dtype!r};"
            " use F.linear_cross_entropy."
        )
    # AOTAutograd/AOTInductor bake compute_*_grad at trace time; catch
    # the silent-corruption case (False at trace, but grad-enabled at
    # runtime with a requires_grad leaf).
    grad_enabled = torch.is_grad_enabled()
    if not compute_input_grad and input.requires_grad and grad_enabled:
        raise RuntimeError(
            "linear_cross_entropy chunked op: compute_input_grad was False at "
            "trace time but input.requires_grad is True at runtime; recompile "
            "the graph with the desired requires_grad."
        )
    if not compute_linear_weight_grad and linear_weight.requires_grad and grad_enabled:
        raise RuntimeError(
            "linear_cross_entropy chunked op: compute_linear_weight_grad was "
            "False at trace time but linear_weight.requires_grad is True at "
            "runtime; recompile the graph with the desired requires_grad."
        )
    ctx = _ChunkContext.build(
        input,
        linear_weight,
        target,
        weight,
        reduction,
        ignore_index,
        label_smoothing,
        batch_chunk_size,
        acc_policy,
        acc_dtype,
        compute_input_grad,
        compute_linear_weight_grad,
    )
    dtype = ctx.dtype
    output = ctx.output
    grad_input = ctx.grad_input
    grad_linear_weight = ctx.grad_linear_weight
    weight_grad_chunk = ctx.weight_grad_chunk
    linear_weight_cast = ctx.linear_weight_cast

    if reduction == "mean" and ctx.num_batches == 0:
        output.fill_(torch.nan)

    compute_grads = compute_input_grad or compute_linear_weight_grad

    # Do not ``break`` from this loop -- ``ctx.chunks()`` runs a
    # post-yield grad_input commit that the in-flight chunk needs.
    for chunk in ctx.chunks():
        logits = chunk.logits
        weight_chunk = chunk.weight_chunk
        target_chunk = chunk.target_chunk

        ctx.mm(chunk.input, chunk.linear_weight.T, out=logits)
        logits.sub_(ctx.amax(logits))  # softmax stability
        # output += <weight_chunk, logits[:, target_chunk]>
        output.add_(ctx.dotgather(weight_chunk, logits, target_chunk))
        softmax_denom = ctx.sumexp_(logits, dim=1)
        if compute_grads:
            # logits *= weight_chunk / softmax_denom         (= softmax(logits) * w)
            # MPS: in-place mul_ on a narrow view of logits_buf doesn't
            # propagate to parent storage; the out= form does.
            torch.mul(
                logits,
                ctx.div(weight_chunk, softmax_denom).unsqueeze(1),
                out=logits,
            )
        # output -= <weight_chunk, log(softmax_denom)>
        # softmax_denom is in acc_dtype (see ``sumexp_``); promote weight_chunk
        # for the dot rather than rounding the wider denominator down.
        output.sub_(weight_chunk.to(softmax_denom.dtype).dot(softmax_denom.log_()))

        if compute_grads:
            if compute_input_grad:
                grad_input_chunk = chunk.grad_input_chunk
                input_grad_logits = chunk.input_grad_logits
                input_grad_linear_weight = chunk.input_grad_linear_weight

                # grad_input_chunk = linear_weight_cast[target_chunk] * weight_chunk
                torch.mul(
                    torch.index_select(linear_weight_cast, 0, target_chunk),
                    weight_chunk.to(grad_input_chunk.dtype).unsqueeze(1),
                    out=grad_input_chunk,
                )
                # grad_input_chunk -= input_grad_logits @ input_grad_linear_weight
                torch.addmm(
                    grad_input_chunk,
                    input_grad_logits,
                    input_grad_linear_weight,
                    alpha=-1,
                    out=grad_input_chunk,
                )
            if compute_linear_weight_grad:
                input_chunk_acc = chunk.input_chunk_acc

                # grad_linear_weight += per-chunk weight grad
                # (= -logits.T @ input + input*weight scattered at target rows)
                if ctx.alloc_weight_grad_chunk:
                    # Stage per-chunk weight grad in acc_dtype scratch;
                    # one sub_ commit keeps bulk-minus-correction in fp32.
                    logits_upcast = chunk.logits_upcast
                    weight_grad_input = chunk.weight_grad_input
                    ctx.mm(
                        logits_upcast.T,
                        weight_grad_input,
                        out=weight_grad_chunk,
                    )
                    temp = ctx.mul(input_chunk_acc, weight_chunk)
                    weight_grad_chunk.index_add_(0, target_chunk, temp, alpha=-1)
                    grad_linear_weight.sub_(weight_grad_chunk)
                else:
                    # Stream bulk + correction directly into grad_lw without
                    # a (V, F) scratch (the compact path).
                    grad_linear_weight.addmm_(
                        chunk.logits_downcast.T, chunk.input_chunk, alpha=-1
                    )
                    temp = ctx.to(
                        ctx.mul(input_chunk_acc, weight_chunk),
                        dtype=grad_linear_weight.dtype,
                    )
                    grad_linear_weight.index_add_(0, target_chunk, temp, alpha=1)

    return (
        output.to(dtype),
        grad_input.to(dtype),
        grad_linear_weight,
    )


# Op arg count; backward returns this many slots. Hand-maintained because
# CustomOpDef has no public arg-introspection. Update on signature change.
_NUM_OP_INPUTS = 13


@_linear_cross_entropy_batch_chunked.register_fake
def _(
    input,
    linear_weight,
    target,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
    batch_chunk_size,
    acc_policy: str,
    acc_dtype,
    allow_retain_graph,
    compute_input_grad,
    compute_linear_weight_grad,
):
    if reduction in {"mean", "sum"}:
        result = torch.empty((), dtype=input.dtype, device=input.device)
    else:
        raise NotImplementedError(f"linear_cross_entropy does not support {reduction=}")
    grad_input_shape = input.shape if compute_input_grad else (0, 0)
    grad_linear_weight_shape = (
        linear_weight.shape if compute_linear_weight_grad else (0, 0)
    )

    grad_input = torch.empty(
        grad_input_shape,
        dtype=input.dtype,
        device=input.device,
        requires_grad=False,
    )
    grad_linear_weight = torch.empty(
        grad_linear_weight_shape,
        dtype=linear_weight.dtype,
        device=linear_weight.device,
        requires_grad=False,
    )
    return result, grad_input, grad_linear_weight


@_linear_cross_entropy_batch_chunked.register_vmap
def _vmap(info, in_dims, *args):
    """vmap rule (slow path: per-sample Python loop). A fold-into-num_batches
    fast path would need ``reduction="none"`` in the chunked op.
    """
    batch_size = info.batch_size

    moved_args = [
        arg.movedim(in_dim, 0)
        if in_dim is not None and isinstance(arg, torch.Tensor)
        else arg
        for arg, in_dim in zip(args, in_dims)
    ]

    outputs = [
        _linear_cross_entropy_batch_chunked(
            *[
                arg[i] if in_dim is not None and isinstance(arg, torch.Tensor) else arg
                for arg, in_dim in zip(moved_args, in_dims)
            ]
        )
        for i in range(batch_size)
    ]
    losses = torch.stack([o[0] for o in outputs])
    grad_inputs = torch.stack([o[1] for o in outputs])
    grad_linear_weights = torch.stack([o[2] for o in outputs])
    return (losses, grad_inputs, grad_linear_weights), (0, 0, 0)


def _linear_cross_entropy_batch_chunked_backward(ctx, grad_output, _gi_grad, _gw_grad):
    result = [None] * _NUM_OP_INPUTS
    if ctx.compute_input_grad:
        if ctx.allow_retain_graph:
            result[0] = ctx._gi * grad_output
        else:
            gi, ctx._gi = ctx._gi, None
            if gi is None:
                raise RuntimeError(
                    "linear_cross_entropy chunked backward called twice; "
                    "retain_graph=True / double backward are not supported."
                )
            result[0] = gi.mul_(grad_output)

    if ctx.compute_linear_weight_grad:
        if ctx.allow_retain_graph:
            result[1] = ctx._gw * grad_output
        else:
            gw, ctx._gw = ctx._gw, None
            if gw is None:
                raise RuntimeError(
                    "linear_cross_entropy chunked backward called twice; "
                    "retain_graph=True / double backward are not supported."
                )
            result[1] = gw.mul_(grad_output)
    return tuple(result)


_linear_cross_entropy_batch_chunked.register_autograd(
    _linear_cross_entropy_batch_chunked_backward,
    setup_context=_linear_cross_entropy_batch_chunked_setup_context,
)
