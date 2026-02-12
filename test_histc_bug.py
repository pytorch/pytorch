import torch

def create_input(dtype, hmin, hmax):
    inf = torch.tensor(torch.inf, dtype=torch.float16)
    buffer = torch.tensor([hmin], dtype=torch.float16)
    res = []
    while buffer[0] <= hmax:
        buffer = torch.nextafter(buffer, inf)
        res.append(buffer[0])
    return torch.tensor(res, dtype=dtype)

hbins, hmin, hmax = 20, -5, 5
dtype = torch.float16
tensor = create_input(dtype, hmin, hmax)

# This tensor should be null.
diff = torch.histc(tensor, hbins, hmin, hmax) - (
    torch.histc(tensor[::2], hbins, hmin, hmax) + torch.histc(tensor[1::2], hbins, hmin, hmax)
)
print(f"torch.histc: {tensor.dtype=}, number of differences: {diff.abs().sum()}: {diff}")

