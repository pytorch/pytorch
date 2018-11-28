import torch
import torch.backends.cudnn as cudnn


@torch._jit_internal.weak_script
def affine_grid_generator(theta, size):
    # type: (Tensor, List[int]) -> Tensor
    if theta.is_cuda and cudnn.enabled and cudnn.is_acceptable(theta) and len(size) == 4:
        N, C, H, W = size
        ret = torch.cudnn_affine_grid_generator(theta, N, C, H, W)
    else:
        ret = torch.affine_grid_generator(theta, size)
    return ret
