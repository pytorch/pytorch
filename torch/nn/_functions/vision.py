import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch._thnn import type2backend
from .thnn.auto import function_by_name
import torch.backends.cudnn as cudnn

MODE_ZEROS = 0
MODE_BORDER = 1


def grid_sampler(input, grid, padding_mode):
    if cudnn.is_acceptable(input.data) and padding_mode == 'zeros' and input.dim() == 4:
        return torch.cudnn_grid_sampler(input, grid)
    else:
        return GridSampler.apply(input, grid, padding_mode)


def affine_grid_generator(theta, size):
    if theta.data.is_cuda and cudnn.enabled and cudnn.is_acceptable(theta.data) and len(size) == 4:
        N, C, H, W = size
        return torch.cudnn_affine_grid_generator(theta, N, C, H, W)
    else:
        return AffineGridGenerator.apply(theta, size)


# TODO: Port these completely into C++


class GridSampler(Function):

    @staticmethod
    def forward(ctx, input, grid, padding_mode='zeros'):
        ctx.save_for_backward(input, grid)

        if padding_mode == 'zeros':
            ctx.padding_mode = MODE_ZEROS
        elif padding_mode == 'border':
            ctx.padding_mode = MODE_BORDER
        else:
            raise ValueError("padding_mode needs to be 'zeros' or 'border', but got {}".format(padding_mode))

        grid_sz = grid.size()
        backend = type2backend[input.type()]
        if input.dim() == 4:
            output = input.new(grid_sz[0], input.size(1), grid_sz[1], grid_sz[2])
            backend.SpatialGridSamplerBilinear_updateOutput(backend.library_state, input, grid,
                                                            output, ctx.padding_mode)
        elif input.dim() == 5:
            output = input.new(grid_sz[0], input.size(1), grid_sz[1], grid_sz[2], grid_sz[3])
            backend.VolumetricGridSamplerBilinear_updateOutput(backend.library_state, input, grid,
                                                               output, ctx.padding_mode)
        else:
            raise ValueError("input has to be 4d or 5d but got input of shape: {}".format(input.shape))
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        padding_mode = ctx.padding_mode

        backend = type2backend[input.type()]
        grad_input = input.new(input.size())
        grad_grid = grid.new(grid.size())
        if input.dim() == 4:
            backend.SpatialGridSamplerBilinear_updateGradInput(
                backend.library_state, input, grad_input,
                grid, grad_grid, grad_output, padding_mode)
        elif input.dim() == 5:
            backend.VolumetricGridSamplerBilinear_updateGradInput(
                backend.library_state, input, grad_input,
                grid, grad_grid, grad_output, padding_mode)
        else:
            raise ValueError("input has to be 4d or 5d but got input of shape: {}".format(input.shape))
        return grad_input, grad_grid, None


class AffineGridGenerator(Function):
    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size

        ctx.size = size
        ctx.is_cuda = theta.is_cuda

        if len(size) == 5:
            N, C, D, H, W = size
            base_grid = theta.new(N, D, H, W, 4)

            base_grid[:, :, :, :, 0] = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
            base_grid[:, :, :, :, 1] = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1]))\
                .unsqueeze(-1)
            base_grid[:, :, :, :, 2] = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1]))\
                .unsqueeze(-1).unsqueeze(-1)
            base_grid[:, :, :, :, 3] = 1

            grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
            grid = grid.view(N, D, H, W, 3)

        elif len(size) == 4:
            N, C, H, W = size
            base_grid = theta.new(N, H, W, 3)

            base_grid[:, :, :, 0] = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
            base_grid[:, :, :, 1] = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1]))\
                .unsqueeze(-1)
            base_grid[:, :, :, 2] = 1

            grid = torch.bmm(base_grid.view(N, H * W, 3), theta.transpose(1, 2))
            grid = grid.view(N, H, W, 2)
        else:
            raise RuntimeError("AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.")

        ctx.base_grid = base_grid

        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        assert ctx.is_cuda == grad_grid.is_cuda
        base_grid = ctx.base_grid

        if len(ctx.size) == 5:
            N, C, D, H, W = ctx.size
            assert grad_grid.size() == torch.Size([N, D, H, W, 3])
            grad_theta = torch.bmm(
                base_grid.view(N, D * H * W, 4).transpose(1, 2),
                grad_grid.view(N, D * H * W, 3))
        elif len(ctx.size) == 4:
            N, C, H, W = ctx.size
            assert grad_grid.size() == torch.Size([N, H, W, 2])
            grad_theta = torch.bmm(
                base_grid.view(N, H * W, 3).transpose(1, 2),
                grad_grid.view(N, H * W, 2))
        else:
            assert False

        grad_theta = grad_theta.transpose(1, 2)

        return grad_theta, None
