import torch
import torch.backends.cudnn as cudnn


def affine_grid_generator(theta, size):
    # type: (Tensor, List[int]) -> Tensor
    if theta.is_cuda and cudnn.enabled and cudnn.is_acceptable(theta) and len(size) == 4 and size[0] < 65536:
        N, C, H, W = size
        ret = torch.cudnn_affine_grid_generator(theta, N, C, H, W)
    else:
        ret = torch.affine_grid_generator(theta, size)
    return ret
