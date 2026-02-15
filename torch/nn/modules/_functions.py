# mypy: allow-untyped-defs
import warnings

import torch
import torch.distributed as dist
from torch.autograd.function import Function


class SyncBatchNorm(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        self,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        eps,
        momentum,
        process_group,
        world_size,
    ):
        if not (
            input.is_contiguous(memory_format=torch.channels_last)
            or input.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()

        size = int(input.numel() // input.size(1))
        if size == 1 and world_size < 2:
            raise ValueError(
                f"Expected more than 1 value per channel when training, got input size {size}"
            )

        num_channels = input.shape[1]
        if input.numel() > 0:
            # calculate mean/invstd for input.
            mean, invstd = torch.batch_norm_stats(input, eps)

            count = torch.full(
                (1,),
                input.numel() // input.size(1),
                dtype=mean.dtype,
                device=mean.device,
            )

            # C, C, 1 -> (2C + 1)
            combined = torch.cat([mean, invstd, count], dim=0)
        else:
            # for empty input, set stats and the count to zero. The stats with
            # zero count will be filtered out later when computing global mean
            # & invstd, but they still needs to participate the all_gather
            # collective communication to unblock other peer processes.
            combined = torch.zeros(
                2 * num_channels + 1, dtype=input.dtype, device=input.device
            )

        # Use allgather instead of allreduce because count could be different across
        # ranks, simple all reduce op can not give correct results.
        # batch_norm_gather_stats_with_counts calculates global mean & invstd based on
        # all gathered mean, invstd and count.
        # for nccl backend, use the optimized version of all gather.
        # The Gloo backend does not support `all_gather_into_tensor`.
        if process_group._get_backend_name() != "gloo":
            # world_size * (2C + 1)
            combined_size = combined.numel()
            combined_flat = torch.empty(
                1,
                combined_size * world_size,
                dtype=combined.dtype,
                device=combined.device,
            )
            dist.all_gather_into_tensor(
                combined_flat, combined, process_group, async_op=False
            )
            combined = torch.reshape(combined_flat, (world_size, combined_size))
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
        else:
            # world_size * (2C + 1)
            combined_list = [torch.empty_like(combined) for _ in range(world_size)]
            dist.all_gather(combined_list, combined, process_group, async_op=False)
            combined = torch.stack(combined_list, dim=0)
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

        if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            # The lines below force a synchronization between CUDA and CPU, because
            # the shape of the result count_all depends on the values in mask tensor.
            # Such synchronizations break CUDA Graph capturing.
            # See https://github.com/pytorch/pytorch/issues/78549
            # FIXME: https://github.com/pytorch/pytorch/issues/78656 describes
            # a better longer-term solution.

            # remove stats from empty inputs
            mask = count_all.squeeze(-1) >= 1
            count_all = count_all[mask]
            mean_all = mean_all[mask]
            invstd_all = invstd_all[mask]

        # calculate global mean & invstd
        counts = count_all.view(-1)
        if running_mean is not None and counts.dtype != running_mean.dtype:
            counts = counts.to(running_mean.dtype)
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            counts,
        )

        self.save_for_backward(input, weight, mean, invstd, count_all.to(torch.int32))
        self.process_group = process_group

        # apply element-wise normalization
        if input.numel() > 0:
            return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        else:
            return torch.empty_like(input)

    @staticmethod
    def backward(self, grad_output):
        if not (
            grad_output.is_contiguous(memory_format=torch.channels_last)
            or grad_output.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group

        if saved_input.numel() > 0:
            # calculate local stats as well as grad_weight / grad_bias
            (
                sum_dy,
                sum_dy_xmu,
                grad_weight,
                grad_bias,
            ) = torch.batch_norm_backward_reduce(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                self.needs_input_grad[0],
                self.needs_input_grad[1],
                self.needs_input_grad[2],
            )

            if self.needs_input_grad[0]:
                # synchronizing stats used to calculate input gradient.
                num_channels = sum_dy.shape[0]
                combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
                torch.distributed.all_reduce(
                    combined,
                    torch.distributed.ReduceOp.SUM,
                    process_group,
                    async_op=False,
                )
                sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

                # backward pass for gradient calculation
                if weight is not None and weight.dtype != mean.dtype:
                    weight = weight.to(mean.dtype)
                grad_input = torch.batch_norm_backward_elemt(
                    grad_output,
                    saved_input,
                    mean,
                    invstd,
                    weight,
                    sum_dy,
                    sum_dy_xmu,
                    count_tensor,
                )
            # synchronizing of grad_weight / grad_bias is not needed as distributed
            # training would handle all reduce.
            if weight is None or not self.needs_input_grad[1]:
                grad_weight = None

            if weight is None or not self.needs_input_grad[2]:
                grad_bias = None
        else:
            # This process got an empty input tensor in the forward pass.
            # Although this process can directly set grad_input as an empty
            # tensor of zeros, it still needs to participate in the collective
            # communication to unblock its peers, as other peer processes might
            # have received non-empty inputs.
            num_channels = saved_input.shape[1]
            if self.needs_input_grad[0]:
                # launch all_reduce to unblock other peer processes
                combined = torch.zeros(
                    2 * num_channels, dtype=saved_input.dtype, device=saved_input.device
                )
                torch.distributed.all_reduce(
                    combined,
                    torch.distributed.ReduceOp.SUM,
                    process_group,
                    async_op=False,
                )

            # Leave grad_input, grad_weight and grad_bias as None, which will be
            # interpreted by the autograd engine as Tensors full of zeros.

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class CrossMapLRN2d(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, input, size, alpha=1e-4, beta=0.75, k=1):
        ctx.size = size
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.k = k
        ctx.scale = None

        if input.dim() != 4:
            raise ValueError(
                f"CrossMapLRN2d: Expected input to be 4D, got {input.dim()}D instead."
            )

        ctx.scale = ctx.scale or input.new()
        output = input.new()
        channels = input.size(1)

        output.resize_as_(input)
        ctx.scale.resize_as_(input)

        # use output storage as temporary buffer
        input_square = output
        torch.pow(input, 2, out=input_square)

        pre_pad = int((ctx.size - 1) / 2 + 1)
        pre_pad_crop = min(pre_pad, channels)

        scale_first = ctx.scale.select(1, 0)
        scale_first.zero_()
        # compute first feature map normalization
        for c in range(pre_pad_crop):
            scale_first.add_(input_square.select(1, c))

        # reuse computations for next feature maps normalization
        # by adding the next feature map and removing the previous
        for c in range(1, channels):
            scale_previous = ctx.scale.select(1, c - 1)
            scale_current = ctx.scale.select(1, c)
            scale_current.copy_(scale_previous)
            if c < channels - pre_pad + 1:
                square_next = input_square.select(1, c + pre_pad - 1)
                scale_current.add_(square_next, alpha=1)

            if c > pre_pad:
                square_previous = input_square.select(1, c - pre_pad)
                scale_current.add_(square_previous, alpha=-1)

        ctx.scale.mul_(ctx.alpha / ctx.size).add_(ctx.k)

        torch.pow(ctx.scale, -ctx.beta, out=output)
        output.mul_(input)

        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        grad_input = grad_output.new()

        batch_size = input.size(0)
        channels = input.size(1)
        input_height = input.size(2)
        input_width = input.size(3)

        paddded_ratio = input.new(channels + ctx.size - 1, input_height, input_width)
        accum_ratio = input.new(input_height, input_width)

        cache_ratio_value = 2 * ctx.alpha * ctx.beta / ctx.size
        inversePrePad = int(ctx.size - (ctx.size - 1) / 2)

        grad_input.resize_as_(input)
        torch.pow(ctx.scale, -ctx.beta, out=grad_input).mul_(grad_output)

        paddded_ratio.zero_()
        padded_ratio_center = paddded_ratio.narrow(0, inversePrePad, channels)
        for n in range(batch_size):
            torch.mul(grad_output[n], output[n], out=padded_ratio_center)
            padded_ratio_center.div_(ctx.scale[n])
            torch.sum(
                paddded_ratio.narrow(0, 0, ctx.size - 1),
                0,
                keepdim=False,
                out=accum_ratio,
            )
            for c in range(channels):
                accum_ratio.add_(paddded_ratio[c + ctx.size - 1])
                grad_input[n][c].addcmul_(
                    input[n][c], accum_ratio, value=-cache_ratio_value
                )
                accum_ratio.add_(paddded_ratio[c], alpha=-1)

        return grad_input, None, None, None, None


class BackwardHookFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, *args):
        ctx.mark_non_differentiable(*[arg for arg in args if not arg.requires_grad])
        return args

    @staticmethod
    def backward(ctx, *args):
        return args


class LinearCrossEntropyFunction(torch.autograd.Function):
    """Implements linear_cross_entropy operation with chunking along
    batches, features, and classes dimensions.

    Chunking considerably reduces the memory usage in forward and
    backward computations. On the other hand, chunking may reduce
    processing performance, especially when the chunk sizes are set
    too small or chunking along the classes dimension that requires
    extra computations.

    .. warning::
      LinearCrossEntropyFunction chunking is not supported when
      - reduction == "none"
      - label_smoothing > 0
      - target contains probabilities
      - loss is K-dimensional

    In the following we'll provide an optimal chunking strategy to
    reduce the memory usage of the linear_cross_entropy forward and
    backward operations while maximizing the processing performance of
    the operation when using chunking.

    A chunking strategy is defined by the following chunk sizes along
    batches, features, and classes dimensions:

      batches_chunk_size
      features_chunk_size
      classes_chunk_size

    that directly affect the performance of the linear_cross_entropy
    operation: smaller chunk sizes reduce the memory consumption of
    the operation while chunking along classes dimension requires
    extra computations and therefore decreases processing
    performance. We use a parameter, `max_memory_gb`, to find an
    optimal chunking strategy:

      `max_memory_gb` is the upper bound to memory size that
      linear_cross_entropy should use for forward and backward
      computations.

    In addition, a parameter `min_chunk_size` is used to restrict the
    allowed chunk sizes to satisfy the following conditions::

      batches_chunk_size >= min(min_chunk_size, num_batches)
      batches_chunk_size % min_chunk_size == 0
      features_chunk_size >= min(min_chunk_size, in_features)
      features_chunk_size % min_chunk_size == 0
      classes_chunk_size >= min(min_chunk_size, num_classes)
      classes_chunk_size % min_chunk_size == 0

    so that underlying, say, CUDA kernels could perform more
    efficiently.

    To compute the optimal chunking strategy, use::

      opt_options = LinearCrossEntropyFunction.optimal_chunking(
          options,
          num_batches,
          in_features,
          num_classes,
          input_requires_grad,
          linear_weight_requires_grad,
          device,
          dtype,
      )

    where `options` is a dictionary with the following keys:

      max_memory_gb (int, optional) - specifies the upper bound to
        memory usage by the linear_cross_entropy in GB. On a CUDA
        device, the default is 0.75 % of GPU memory, otherwise, 64 GB.

      min_chunk_size (int, optional) - specifies minimal chunk size in
        chunking along batches, features, and classes
        dimensions. Default is 1024.

      grad_inplace (bool, optional) - when True, backward will use
        inplace multiplication to compute the gradients to save extra
        storage space. Warning: when True, torch.autograd.gradcheck
        will fail.  Default is False.

      batches_chunk_size (int, optional) - fix chunk size along
        batches dimension. By default, optimal chunk size along
        batches dimension is computed.

      features_chunk_size (int, optional) - fix chunk size along
        features dimension. By default, optimal chunk size along
        features dimension is computed.

      classes_chunk_size (int, optional) - fix chunk size along
        classes dimension. Note that chunking along classes dimension
        will involve extra computations and will affect processing
        performance. By default, optimal chunk size along classes
        dimension is computed.

    `opt_options` is a copy of `options` dictionary, with computed
    optimal values of batches_chunk_size, features_chunk_size, and
    classes_chunk_size inserted. If `options` already contains these
    keys, the corresponding items are kept constant. If
    `max_memory_gb` is too small or `min_chunk_size` is too large, the
    chunking strategy defined by
    `{batches,features,classes}_chunk_size` will be smallest possible
    under specified constraints except the constraint defined by
    `max_memory_gb` may be not be satisfied.

    """

    @staticmethod
    def optimal_chunking(
        options: dict,
        num_batches: int | None,
        in_features: int,
        num_classes: int,
        input_requires_grad: bool,
        linear_weight_requires_grad: bool,
        device: torch.device,
        dtype: torch.dtype,
        target_dtype: torch.dtype,
    ):
        """Compute optimal chunking strategy. See
        :class:`LinearCrossEntropyFunction` for details.
        """
        opt_options = options.copy()

        if num_batches is None:
            num_batches = 1
            has_batches = False
        else:
            has_batches = True

        if (
            ("batches_chunk_size" in options or not has_batches)
            and "features_chunk_size" in options
            and "classes_chunk_size" in options
        ):
            # all variables are fixed, nothing to optimize
            return opt_options

        grad_inplace = opt_options.get("grad_inplace", False)

        def get_numel_forward(
            batches_chunk_size: int, features_chunk_size: int, classes_chunk_size: int
        ):
            # Keep this function in sync with LinearCrossEntropyFunction.forward method!

            count = 0  # count elements with dtype
            # input:
            count += num_batches * in_features
            # linear_weight:
            count += num_classes * in_features
            if target_dtype.is_floating_point:
                count += num_batches
            else:
                count += num_batches * torch.int64.itemsize // dtype.itemsize
            # weight:
            count += num_classes
            # X:
            count += batches_chunk_size * classes_chunk_size
            # expXsum:
            count += batches_chunk_size * 2
            if classes_chunk_size != num_classes:
                # mask, t_, weight_t_ and related subexpressions, in the 1st for-loop:
                count += batches_chunk_size * 7
            if input_requires_grad or linear_weight_requires_grad:
                # -weight_t / expXsum
                count += batches_chunk_size * 2
                if classes_chunk_size != num_classes:
                    # mask, t_, weight_t_ and related subexpressions, in the 2nd for-loop:
                    count += batches_chunk_size * 7
            if input_requires_grad:
                # grad_input:
                count += num_batches * in_features * (1 if grad_inplace else 2)
            if linear_weight_requires_grad:
                # grad_linear_weight:
                count += num_classes * in_features * (1 if grad_inplace else 2)
                # G:
                count += features_chunk_size * classes_chunk_size
                if classes_chunk_size != num_classes:
                    # x_[mask]:
                    count += batches_chunk_size
            return count

        min_chunk_size: int = opt_options.get("min_chunk_size", 1024)
        max_memory_gb = opt_options.get("max_memory_gb")
        if max_memory_gb is None:
            if device.type == "cuda":
                max_memory_gb = int(torch.cuda.mem_get_info(device)[0] * 0.75 / 1e9)
            else:
                max_memory_gb = 64

        max_total_numel = int(max_memory_gb * 1e9 / dtype.itemsize)

        min_classes_chunk_size: int = min(
            opt_options.get("classes_chunk_size", min_chunk_size), num_classes
        )
        max_classes_chunk_size: int = max(
            opt_options.get("classes_chunk_size", num_classes),
            min_classes_chunk_size,
        )
        min_features_chunk_size: int = min(
            opt_options.get("features_chunk_size", min_chunk_size), in_features
        )
        max_features_chunk_size: int = max(
            opt_options.get("features_chunk_size", in_features),
            min_features_chunk_size,
        )
        min_batches_chunk_size: int = min(
            opt_options.get("batches_chunk_size", min_chunk_size), num_batches
        )
        max_batches_chunk_size: int = max(
            opt_options.get("batches_chunk_size", num_batches),
            min_batches_chunk_size,
        )
        # we start with most efficient chunking strategy (that is, no
        # chunking at all and doing chunking along classes as a last
        # resort) until we'll find one that meets the max_memory_gb
        # criteria:
        for classes_chunk_size in reversed(
            range(min_classes_chunk_size, max_classes_chunk_size + 1, min_chunk_size)
        ):
            minimal_features_batches_nof_chunks: float | None = None
            features_batches_chunk_sizes = None, None
            for features_chunk_size in reversed(
                range(
                    min_features_chunk_size, max_features_chunk_size + 1, min_chunk_size
                )
            ):
                for batches_chunk_size in reversed(
                    range(
                        min_batches_chunk_size,
                        max_batches_chunk_size + 1,
                        min_chunk_size,
                    )
                ):
                    n = get_numel_forward(
                        batches_chunk_size, features_chunk_size, classes_chunk_size
                    )
                    if n <= max_total_numel:
                        features_batches_nof_chunks = (
                            num_batches / batches_chunk_size
                            + in_features / features_chunk_size
                        )
                        if (
                            minimal_features_batches_nof_chunks is None
                            or features_batches_nof_chunks
                            < minimal_features_batches_nof_chunks
                        ):
                            minimal_features_batches_nof_chunks = (
                                features_batches_nof_chunks
                            )
                            features_batches_chunk_sizes = (
                                features_chunk_size,
                                batches_chunk_size,
                            )
            if minimal_features_batches_nof_chunks is not None:
                features_chunk_size, batches_chunk_size = features_batches_chunk_sizes
                opt_options.update(
                    classes_chunk_size=classes_chunk_size,
                    features_chunk_size=features_chunk_size,
                    batches_chunk_size=batches_chunk_size,
                )
                if not has_batches:
                    opt_options.pop("batches_chunk_size")
                return opt_options

        constraints = [f"{num_batches=}, {in_features=}, {num_classes=}"]
        for k, v in opt_options.items():
            if k not in {"max_memory_gb", "min_chunk_size"}:
                constraints.append(f"{k}={v}")
        constraints = ", ".join(constraints)
        warnings.warn(
            "failed to find optimal chunking strategy for linear_cross_entropy: "
            f"{max_memory_gb=} is too small or {min_chunk_size=} is too large. Constraints: {constraints}"
        )

        # Return a strategy that corresponds to minimal chunking size:
        opt_options.update(
            classes_chunk_size=min_classes_chunk_size,
            features_chunk_size=min_features_chunk_size,
            batches_chunk_size=min_batches_chunk_size,
        )
        if not has_batches:
            opt_options.pop("batches_chunk_size")
        return opt_options

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        input: torch.Tensor,
        linear_weight: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        reduction: str,
        label_smoothing: float,
        options: dict,
    ):
        device = input.device
        dtype = input.dtype
        num_batches, in_features = input.shape
        num_classes, _ = linear_weight.shape

        ctx.grad_inplace = options.get("grad_inplace", False)
        batches_chunk_size = options.get("batches_chunk_size", num_batches)
        features_chunk_size = options.get("features_chunk_size", in_features)
        classes_chunk_size = options.get("classes_chunk_size", num_classes)

        def ensure_size(input, dim, size):
            if input.shape[dim] != size:
                return input.narrow(dim, 0, size)
            return input

        def chunk_iter(total_size, chunk_size):
            for start in range(0, total_size, chunk_size):
                if start + chunk_size > total_size:
                    yield start, total_size - start
                else:
                    yield start, chunk_size

        if target.dtype.is_floating_point:
            raise NotImplementedError(
                "LinearCrossEntropyFunction does not support probability targets"
            )
        else:
            weight_target = weight.index_select(0, target)
            if reduction == "mean":
                weight = weight.clone()
                d = weight_target.sum()
                weight.div_(d)
                weight_target.div_(d)
            elif reduction == "sum":
                pass
            else:
                raise NotImplementedError(
                    f"LinearCrossEntropyFunction does not support {reduction=}"
                )

        if label_smoothing > 0.0:
            raise NotImplementedError(
                "LinearCrossEntropyFunction does not support label smoothing"
            )

        # A chunk buffer used to hold logits, softmax of logits:

        X = torch.empty(
            (batches_chunk_size, classes_chunk_size),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        if input.requires_grad:
            grad_input = torch.empty(
                input.shape, device=device, dtype=dtype, requires_grad=False
            )
        else:
            grad_input = None

        if linear_weight.requires_grad:
            grad_linear_weight = torch.zeros(
                linear_weight.shape, device=device, dtype=dtype, requires_grad=False
            )
            # A chunk buffer used in grad_linear_weight computation:
            G = torch.empty(
                (classes_chunk_size, features_chunk_size),
                device=device,
                dtype=dtype,
                requires_grad=False,
            )
        else:
            grad_linear_weight = G = None

        if reduction in {"mean", "sum"}:
            output = torch.zeros((), device=device, dtype=dtype, requires_grad=False)
        else:
            raise NotImplementedError(
                f"LinearCrossEntropyFunction does not support {reduction=}"
            )

        # chunking along batches dimension:
        for bchunk_start, bchunk_size in chunk_iter(num_batches, batches_chunk_size):
            x = input.narrow(0, bchunk_start, bchunk_size)
            t = target.narrow(0, bchunk_start, bchunk_size)
            weight_t = weight_target.narrow(0, bchunk_start, bchunk_size)
            X_ = ensure_size(X, 0, bchunk_size)
            expXsum: torch.Tensor = torch.empty(())
            Xmax: torch.Tensor = torch.empty(())

            # Compute output.

            # chunking along classes dimension:
            for cchunk_start, cchunk_size in chunk_iter(
                num_classes, classes_chunk_size
            ):
                L_ = linear_weight.narrow(0, cchunk_start, cchunk_size)
                X__ = ensure_size(X_, 1, cchunk_size)
                corrXmax: torch.Tensor = torch.empty(())

                torch.mm(x, L_.T, out=X__)  # projection

                if cchunk_start == 0:
                    Xmax = X__.max(dim=1, keepdim=True)[0]
                else:
                    # correct Xmax
                    corrXmax = Xmax
                    Xmax = X__.max(dim=1, keepdim=True)[0].max(corrXmax)
                    corrXmax.sub_(Xmax)

                X__.sub_(Xmax)

                if cchunk_start > 0:
                    # correct output due to possibly under-estimated Xmax
                    total_mask = t < cchunk_start
                    output.sub_(
                        weight_t[total_mask].dot(corrXmax[total_mask].squeeze(1))
                    )

                if cchunk_size == num_classes:
                    output.sub_(weight_t.dot(X__.gather(1, t.unsqueeze(1)).squeeze(1)))
                else:
                    # chunking along classes dimension is expensive!
                    mask = (cchunk_start <= t) & (t < cchunk_start + cchunk_size)
                    t_ = t.masked_select(mask) - cchunk_start
                    weight_t_ = weight_t.masked_select(mask)
                    output.sub_(
                        weight_t_.dot(X__[mask].gather(1, t_.unsqueeze(1)).squeeze(1))
                    )

                X__.exp_()

                if cchunk_start == 0:
                    expXsum = X__.sum(dim=1)
                else:
                    # correct expXsum due to possibly under-estimated Xmax
                    expXsum.add_(corrXmax.squeeze(1))
                    expXsum.exp_()
                    expXsum.add_(X__.sum(dim=1))

                if input.requires_grad or linear_weight.requires_grad:
                    # X__ content will be reused in the classes
                    # chunking for-loop below
                    X__.mul_(-(weight_t / expXsum).unsqueeze(1))

                expXsum.log_()

            output.add_(weight_t.dot(expXsum))

            # Compute gradients.

            if input.requires_grad or linear_weight.requires_grad:
                if grad_input is not None:
                    grad_x = grad_input.narrow(0, bchunk_start, bchunk_size)
                    torch.index_select(linear_weight, 0, t, out=grad_x)
                    grad_x.mul_(-weight_t.unsqueeze(1))  # todo: eliminate neg
                else:
                    grad_x = None

                if linear_weight.requires_grad:
                    if num_classes != classes_chunk_size:
                        # not-trivial chunking along classes dimension
                        # requires recomputing X__
                        expXsum.exp_()

                # chunking along classes dimension:
                for cchunk_start, cchunk_size in chunk_iter(
                    num_classes, classes_chunk_size
                ):
                    X__ = ensure_size(X_, 1, cchunk_size)
                    if num_classes == classes_chunk_size:  # trivial chunking
                        t_ = t
                        L_ = linear_weight
                        weight_ = weight
                        # X__ is computed in the classes chunking
                        # for-loop above
                        mask = None
                    else:
                        # chunking along classes dimension is
                        # expensive!

                        # recompute X__, however, we can reuse Xmax
                        # and expXsum computed from the classes
                        # chunking for-loop above
                        mask = (cchunk_start <= t) & (t < cchunk_start + cchunk_size)
                        t_ = t.masked_select(mask) - cchunk_start
                        L_ = linear_weight.narrow(0, cchunk_start, cchunk_size)
                        weight_ = weight.narrow(0, cchunk_start, cchunk_size)
                        torch.addmm(Xmax, x, L_.T, beta=-1, out=X__)
                        X__.exp_()
                        X__.mul_(
                            -(weight_t / expXsum).unsqueeze(1)
                        )  # todo: eliminate neg

                    if grad_x is not None:
                        grad_x.addmm_(X__, L_, alpha=-1)

                    if grad_linear_weight is not None:
                        G_ = ensure_size(G, 0, cchunk_size)
                        grad_L_ = grad_linear_weight.narrow(
                            0, cchunk_start, cchunk_size
                        )

                        # chunking along features dimension:
                        for fchunk_start, fchunk_size in chunk_iter(
                            in_features, features_chunk_size
                        ):
                            x_ = x.narrow(1, fchunk_start, fchunk_size)
                            G__ = ensure_size(G_, 1, fchunk_size)
                            G__.zero_()
                            if num_classes == classes_chunk_size:
                                G__.index_add_(0, t, x_)
                            else:
                                G__.index_add_(0, t_, x_[mask])
                            G__.mul_(weight_.unsqueeze(1))
                            G__.addmm_(X__.T, x_, alpha=-1, beta=-1)
                            grad_L_.narrow(1, fchunk_start, fchunk_size).add_(G__)

        save_indices: list[int | None] = [None, None]
        saved = []
        if input.requires_grad:
            save_indices[0] = len(saved)
            saved.append(grad_input)
        if linear_weight.requires_grad:
            save_indices[1] = len(saved)
            saved.append(grad_linear_weight)
        if saved:
            ctx.save_indices = save_indices
            ctx.save_for_backward(*saved)

        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        result = [None] * 7

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            saved = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                grad_input = saved[ctx.save_indices[0]]
            else:
                grad_input = None
            if ctx.needs_input_grad[1]:
                grad_linear_weight = saved[ctx.save_indices[1]]
            else:
                grad_linear_weight = None
            if ctx.grad_inplace:
                # With grad_inplace, the memory usage size is reduced
                # 2x when reusing pre-computed grad_input and
                # grad_linear_weight storages. However, gradcheck does
                # not like that.
                if grad_input is not None:
                    grad_input.mul_(grad_output)
                    result[0] = grad_input
                if grad_linear_weight is not None:
                    grad_linear_weight.mul_(grad_output)
                    result[1] = grad_linear_weight
            else:
                # gradcheck-friendly backward:
                if grad_input is not None:
                    # creates a new tensor that increases memory usage size
                    result[0] = grad_input * grad_output
                if grad_linear_weight is not None:
                    # creates a new tensor that increases memory usage size
                    result[1] = grad_linear_weight * grad_output

        return tuple(result)
