import torch
from torch.autograd.function import Function
from torch._thnn import type2backend

from . import _all_functions


class CrossMapLRN2d(Function):

    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(CrossMapLRN2d, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self._backend = None
        self.scale = None

    def forward(self, input):
        assert input.dim() == 4

        self.scale = self.scale or input.new()
        output = input.new()

        backend = type2backend[input.type()]
        if backend is not None:
            try:
                backend.SpatialCrossMapLRN_updateOutput
                self._backend = backend
            except NotImplementedError:
                pass

        if self._backend is not None:
            self._backend.SpatialCrossMapLRN_updateOutput(
                self._backend.library_state,
                input,
                output,
                self.scale,
                self.size,
                self.alpha,
                self.beta,
                self.k
            )
        else:
            batch_size = input.size(0)
            channels = input.size(1)
            input_height = input.size(2)
            input_width = input.size(3)

            output.resize_as_(input)
            self.scale.resize_as_(input)

            # use output storage as temporary buffer
            input_square = output
            torch.pow(input, 2, out=input_square)

            pre_pad = int((self.size - 1) / 2 + 1)
            pre_pad_crop = channels if pre_pad > channels else pre_pad

            scale_first = self.scale.select(1, 0)
            scale_first.zero_()
            # compute first feature map normalization
            for c in range(pre_pad_crop):
                scale_first.add_(input_square.select(1, c))

            # reuse computations for next feature maps normalization
            # by adding the next feature map and removing the previous
            for c in range(1, channels):
                scale_previous = self.scale.select(1, c - 1)
                scale_current = self.scale.select(1, c)
                scale_current.copy_(scale_previous)
                if c < channels - pre_pad + 1:
                    square_next = input_square.select(1, c + pre_pad - 1)
                    scale_current.add_(1, square_next)

                if c > pre_pad:
                    square_previous = input_square.select(1, c - pre_pad)
                    scale_current.add_(-1, square_previous)

            self.scale.mul_(self.alpha / self.size).add_(self.k)

            torch.pow(self.scale, -self.beta, out=output)
            output.mul_(input)

        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = grad_output.new()

        if self._backend is not None:
            self._backend.SpatialCrossMapLRN_updateGradInput(
                self._backend.library_state,
                input,
                grad_output,
                grad_input,
                self.scale,
                output,
                self.size,
                self.alpha,
                self.beta,
                self.k
            )
        else:
            batch_size = input.size(0)
            channels = input.size(1)
            input_height = input.size(2)
            input_width = input.size(3)

            paddded_ratio = input.new(channels + self.size - 1, input_height,
                                      input_width)
            accum_ratio = input.new(input_height, input_width)

            cache_ratio_value = 2 * self.alpha * self.beta / self.size
            inversePrePad = int(self.size - (self.size - 1) / 2)

            grad_input.resize_as_(input)
            torch.pow(self.scale, -self.beta, out=grad_input).mul_(grad_output)

            paddded_ratio.zero_()
            padded_ratio_center = paddded_ratio.narrow(0, inversePrePad,
                                                       channels)
            for n in range(batch_size):
                torch.mul(grad_output[n], output[n], out=padded_ratio_center)
                padded_ratio_center.div_(self.scale[n])
                torch.sum(
                    paddded_ratio.narrow(0, 0, self.size - 1), 0, keepdim=False, out=accum_ratio)
                for c in range(channels):
                    accum_ratio.add_(paddded_ratio[c + self.size - 1])
                    grad_input[n][c].addcmul_(-cache_ratio_value, input[n][c],
                                              accum_ratio)
                    accum_ratio.add_(-1, paddded_ratio[c])

        return grad_input


class SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group=None):
        input = input.contiguous()

        # calcualte mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)
        if torch.distributed.is_initialized():
            world_size = 0
            if process_group:
                world_size = torch.distributed.get_world_size(process_group)
            else:
                process_group = torch.distributed.get_default_group()
                world_size = torch.distributed.get_world_size()
            mean_all = torch.empty(world_size, mean.size(0), dtype=mean.dtype, device=mean.device)
            invstd_all = torch.empty(world_size, invstd.size(0), dtype=invstd.dtype, device=invstd.device)
            mean_l = [mean_all.narrow(0, i, 1) for i in range(world_size)]
            invstd_l = [invstd_all.narrow(0, i, 1) for i in range(world_size)]
            # using all_gather instead of all reduce so we can calculate mean/var in one go
            torch.distributed.all_gather(mean_l, mean, process_group)
            torch.distributed.all_gather(invstd_l, invstd, process_group)
        else:
            mean_all = mean.view(1, -1)
            invstd_all = invstd.view(1, -1)

        # calcualte global mean & invstd
        mean, invstd = torch.batch_norm_update_stats(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            int(input.numel() / input.size(1))
        )

        self.save_for_backward(input, weight, mean, invstd)
        self.process_group = process_group
        self.world_size = world_size

        # apply element-wise normalization
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        return out

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group
        world_size = self.world_size

        # calculate local stats as well as grad_weight / grad_bias
        mean_dy, mean_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2]
        )

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    mean_dy, torch.distributed.ReduceOp.SUM, process_group)
                mean_dy = mean_dy / world_size
                torch.distributed.all_reduce(
                    mean_dy_xmu, torch.distributed.ReduceOp.SUM, process_group)
                mean_dy_xmu = mean_dy_xmu / world_size
            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                mean_dy,
                mean_dy_xmu
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


_all_functions.append(CrossMapLRN2d)
_all_functions.append(SyncBatchNorm)
