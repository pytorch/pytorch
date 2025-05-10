import torch

def safe_lcm_(a: torch.Tensor, b: torch.Tensor):
    if not a.dtype.is_floating_point and not b.dtype.is_floating_point:
        a_64 = a.to(torch.int64)
        b_64 = b.to(torch.int64)

        gcd = torch.gcd(a_64, b_64)
        gcd[gcd == 0] = 1  

        result = torch.abs(a_64 * b_64) // gcd

        a.copy_(result.to(a.dtype)) 
        return a
    else:
        raise TypeError("safe_lcm_ only supports integer tensors.")
