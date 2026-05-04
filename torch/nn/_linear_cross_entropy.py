import torch


def chunk_iter(total_size, chunk_size):
    for start in range(0, total_size, chunk_size):
        if start + chunk_size > total_size:
            yield start, total_size - start
        else:
            yield start, chunk_size


_NUM_LINEAR_CROSS_ENTROPY_BATCH_CHUNKED_INPUTS = 13


def _linear_cross_entropy_batch_chunked_setup_context(ctx, inputs, output):
    assert len(inputs) == _NUM_LINEAR_CROSS_ENTROPY_BATCH_CHUNKED_INPUTS  # noqa: S101
    *_, grad_inplace, compute_input_grad, compute_linear_weight_grad = inputs
    ctx.grad_inplace = grad_inplace
    ctx.compute_input_grad = compute_input_grad
    ctx.compute_linear_weight_grad = compute_linear_weight_grad
    _, grad_input, grad_linear_weight = output
    ctx._gi = grad_input if compute_input_grad else None
    ctx._gw = grad_linear_weight if compute_linear_weight_grad else None


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
    grad_inplace: bool,
    compute_input_grad: bool,
    compute_linear_weight_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = input.device
    dtype = input.dtype
    num_batches, in_features = input.shape
    num_classes, _ = linear_weight.shape

    if dtype != acc_dtype and not (
        dtype in {torch.float16, torch.bfloat16} and acc_dtype == torch.float32
    ):
        raise RuntimeError(
            "linear_cross_entropy supports float32 acc_dtype with"
            f" float16/bfloat16 inputs, but got {acc_dtype} acc_dtype and {dtype} inputs."
        )
    use_acc_dtype = dtype != acc_dtype

    if target.dtype != torch.int64:
        raise TypeError(
            f"linear_cross_entropy: target dtype must be torch.int64, got {target.dtype}."
        )

    mask = target == ignore_index
    if ignore_index < 0 or ignore_index >= num_classes:
        # map out-of-range ignore_index to 0:
        target = torch.where(mask, 0, target)
        # The correctness of this mapping is subtle: mask contains
        # the original target that ensures that selected weights
        # are masked from out-of-range ignore_index mapping
        # correctly.
    if weight is None:
        # Uniform weight=1
        neg_weight_target = (~mask).to(dtype)
    else:
        neg_weight_target = torch.where(mask, 0, weight.index_select(0, target))

    if reduction == "mean":
        d = neg_weight_target.sum()
        neg_weight_target.div_(torch.where(d == 0, torch.nan, -d))
    elif reduction == "sum":
        neg_weight_target.neg_()
    else:
        raise NotImplementedError(f"linear_cross_entropy does not support {reduction=}")

    if label_smoothing > 0.0:
        raise NotImplementedError(
            "linear_cross_entropy does not support label smoothing"
        )

    if use_acc_dtype:
        dtypes = dict(
            output=acc_dtype if dtype == torch.float16 else dtype,
            grad_input=dtype if acc_policy == "memory" else acc_dtype,
            grad_linear_weight=dtype,
            X=dtype if dtype == torch.float16 and acc_policy == "memory" else acc_dtype,
            GL=acc_dtype,
        )
    else:
        dtypes = dict(
            output=dtype,
            grad_input=dtype,
            grad_linear_weight=dtype,
            X=dtype,
            GL=dtype,
        )

    # A chunk buffer used to hold logits, softmax of logits:
    X = torch.empty(
        (batch_chunk_size, num_classes),
        device=device,
        dtype=dtypes["X"],
        requires_grad=False,
    )
    X_acc = torch.empty(
        (batch_chunk_size, num_classes)
        if (compute_linear_weight_grad and not input.is_cuda and X.dtype != acc_dtype)
        else (0, 0),
        device=device,
        dtype=acc_dtype,
        requires_grad=False,
    )

    linear_weight_ = (
        linear_weight.to(X.dtype)
        if use_acc_dtype and (compute_input_grad or not input.is_cuda)
        else linear_weight
    )

    grad_input = torch.empty(
        input.shape if compute_input_grad else (0,),
        dtype=dtypes["grad_input"],
        device=device,
        requires_grad=False,
    )
    grad_linear_weight = torch.zeros(
        linear_weight.shape if compute_linear_weight_grad else (0,),
        dtype=dtypes["grad_linear_weight"],
        device=device,
        requires_grad=False,
    )
    GL = torch.empty(
        (num_classes, in_features) if compute_linear_weight_grad else (0, 0),
        device=device,
        dtype=dtypes["GL"],
        requires_grad=False,
    )
    tmp = torch.empty(
        batch_chunk_size, device=device, dtype=dtypes["X"], requires_grad=False
    )

    if reduction in {"mean", "sum"}:
        output = torch.zeros(
            (), device=device, dtype=dtypes["output"], requires_grad=False
        )
    else:
        raise NotImplementedError(f"linear_cross_entropy does not support {reduction=}")
    # chunking along batches dimension:
    for bchunk_start, bchunk_size in chunk_iter(num_batches, batch_chunk_size):
        x = input.narrow(0, bchunk_start, bchunk_size)
        t = target.narrow(0, bchunk_start, bchunk_size)
        neg_weight_t = neg_weight_target.narrow(0, bchunk_start, bchunk_size)
        neg_weight_t_ = neg_weight_t.to(X.dtype)
        X_ = X.narrow(0, 0, bchunk_size) if X.shape[0] != bchunk_size else X
        x_ = x.to(acc_dtype) if use_acc_dtype and compute_linear_weight_grad else x

        # Compute output.

        if use_acc_dtype:
            if input.is_cuda:
                torch.mm(x, linear_weight.T, out_dtype=X.dtype, out=X_)
            else:
                torch.mm(x.to(X.dtype), linear_weight_.T, out=X_)
        else:
            torch.mm(x, linear_weight.T, out=X_)

        Xmax = tmp.narrow(0, 0, bchunk_size).unsqueeze(1)
        torch.amax(X_, dim=1, keepdim=True, out=Xmax)

        X_.sub_(Xmax)

        output.add_(neg_weight_t_.dot(X_.gather(1, t.unsqueeze(1)).squeeze(1)))

        X_.exp_()

        expXsum = X_.sum(dim=1)

        if compute_input_grad or compute_linear_weight_grad:
            tmp_chunk = tmp.narrow(0, 0, bchunk_size)
            torch.div(neg_weight_t_, expXsum, out=tmp_chunk)
            X_.mul_(tmp_chunk.unsqueeze(1))

        expXsum.log_()
        output.sub_(neg_weight_t_.dot(expXsum))

        # Compute gradients.

        if compute_input_grad:
            grad_x = grad_input.narrow(0, bchunk_start, bchunk_size)
            torch.mul(
                torch.index_select(linear_weight_, 0, t),
                neg_weight_t_.unsqueeze(1),
                out=grad_x,
            )
            torch.addmm(grad_x, X_, linear_weight_, alpha=-1, out=grad_x)

        if compute_linear_weight_grad:
            if X.dtype == GL.dtype:
                torch.mm(X_.T, x_, out=GL)
            elif input.is_cuda:
                torch.mm(X_.T, x, out_dtype=GL.dtype, out=GL)
            else:
                X_acc_chunk = X_acc.narrow(0, 0, bchunk_size)
                X_acc_chunk.copy_(X_)
                torch.mm(X_acc_chunk.T, x_, out=GL)
            if use_acc_dtype:
                # x_ is a copy of input slice, so we can
                # change it inplace to reduce memory usage
                x_.mul_(neg_weight_t_.unsqueeze(1))
                GL.index_add_(0, t, x_, alpha=-1)
            else:
                GL.index_add_(0, t, x_ * neg_weight_t_.unsqueeze(1), alpha=-1)
            grad_linear_weight.sub_(GL)

    return output.to(dtype), grad_input.to(dtype), grad_linear_weight


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
    grad_inplace,
    compute_input_grad,
    compute_linear_weight_grad,
):
    if reduction in {"mean", "sum"}:
        result = torch.empty((), dtype=input.dtype, device=input.device)
    else:
        raise NotImplementedError(f"linear_cross_entropy does not support {reduction=}")
    grad_input_shape = input.shape if compute_input_grad else (0,)
    grad_linear_weight_shape = (
        linear_weight.shape if compute_linear_weight_grad else (0,)
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
    """vmap rule: loop over the vmap dimension and stack outputs.

    Each slice runs an independent op invocation, so per-sample input
    and linear_weight gradients flow naturally through autograd — the
    common ``vmap(grad(linear_cross_entropy))`` pattern produces
    per-sample weight gradients without extra plumbing.

    A fold-into-num_batches optimization for the common case (vmap
    over input, shared linear_weight) would require
    ``reduction="none"`` support inside the chunked op, which is not
    currently implemented. Until then, loop.
    """
    batch_size = info.batch_size

    def slice_arg(arg, in_dim, i):
        if in_dim is None or not isinstance(arg, torch.Tensor):
            return arg
        return arg.movedim(in_dim, 0)[i]

    outputs = [
        _linear_cross_entropy_batch_chunked(
            *[slice_arg(arg, in_dim, i) for arg, in_dim in zip(args, in_dims)]
        )
        for i in range(batch_size)
    ]

    losses = torch.stack([o[0] for o in outputs])
    grad_inputs = torch.stack([o[1] for o in outputs])
    grad_linear_weights = torch.stack([o[2] for o in outputs])
    return (losses, grad_inputs, grad_linear_weights), (0, 0, 0)


def _linear_cross_entropy_batch_chunked_backward(ctx, grad_output, _gi_grad, _gw_grad):
    result = [None] * _NUM_LINEAR_CROSS_ENTROPY_BATCH_CHUNKED_INPUTS
    if ctx.compute_input_grad:
        if not ctx.grad_inplace:
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
        if not ctx.grad_inplace:
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
