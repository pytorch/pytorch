import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch._thnn import type2backend
from .thnn.auto import function_by_name
import torch.backends.cudnn as cudnn


class GridSampler(Function):

    @staticmethod
    def forward(ctx, input, grid):
        ctx.save_for_backward(input, grid)
        grid_sz = grid.size()
        if cudnn.is_acceptable(input):
            output = input.new(grid_sz[0], input.size(1), grid_sz[1], grid_sz[2])
            grid = grid.contiguous()
            if 0 in input.stride():
                input = input.contiguous()
            torch._C._cudnn_grid_sampler_forward(input, grid, output)
        else:
            backend = type2backend[type(input)]
            output = input.new(grid_sz[0], input.size(1), grid_sz[1], grid_sz[2])
            backend.SpatialGridSamplerBilinear_updateOutput(backend.library_state, input, grid, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        if cudnn.is_acceptable(input):
            grad_input = input.new(input.size())
            grad_grid = grid.new(grid.size())
            grid = grid.contiguous()
            if 0 in input.stride():
                input = input.contiguous()
            torch._C._cudnn_grid_sampler_backward(input, grad_input,
                                                  grid, grad_grid,
                                                  grad_output)
        else:
            backend = type2backend[type(input)]
            grad_input = input.new(input.size())
            grad_grid = grid.new(grid.size())
            backend.SpatialGridSamplerBilinear_updateGradInput(
                backend.library_state, input, grad_input,
                grid, grad_grid, grad_output)
        return grad_input, grad_grid


class AffineGridGenerator(Function):

    @staticmethod
    def _enforce_cudnn(input):
        if not cudnn.enabled:
            raise RuntimeError("AffineGridGenerator needs CuDNN for "
                               "processing CUDA inputs, but CuDNN is not enabled")
        assert cudnn.is_acceptable(input)

    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size
        N, C, H, W = size
        ctx.size = size
        if theta.is_cuda:
            ctx.is_cuda = True
            AffineGridGenerator._enforce_cudnn(theta)
            grid = theta.new(N, H, W, 2)
            theta = theta.contiguous()
            torch._C._cudnn_affine_grid_generator_forward(theta, grid, N, C, H, W)
        else:
            ctx.is_cuda = False
            base_grid = theta.new(N, H, W, 3)
            linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
            base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
            linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
            base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
            base_grid[:, :, :, 2] = 1
            ctx.base_grid = base_grid
            grid = torch.bmm(base_grid.view(N, H * W, 3), theta.transpose(1, 2))
            grid = grid.view(N, H, W, 2)
        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        N, C, H, W = ctx.size
        assert grad_grid.size() == torch.Size([N, H, W, 2])
        assert ctx.is_cuda == grad_grid.is_cuda
        if grad_grid.is_cuda:
            AffineGridGenerator._enforce_cudnn(grad_grid)
            grad_theta = grad_grid.new(N, 2, 3)
            grad_grid = grad_grid.contiguous()
            torch._C._cudnn_affine_grid_generator_backward(grad_theta, grad_grid,
                                                           N, C, H, W)
        else:
            base_grid = ctx.base_grid
            grad_theta = torch.bmm(
                base_grid.view(N, H * W, 3).transpose(1, 2),
                grad_grid.view(N, H * W, 2))
            grad_theta = grad_theta.transpose(1, 2)

        return grad_theta, None
