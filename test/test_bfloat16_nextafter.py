import torch

def fn(device):
    print(torch.finfo(torch.bfloat16))
    print(torch.finfo(torch.float32))

    zero = torch.tensor(0, dtype=torch.bfloat16, device=device)
    one = torch.tensor(1, dtype=torch.bfloat16, device=device)

    inf = torch.tensor(float('inf'), dtype=torch.bfloat16, device=device)
    nan = torch.tensor(float('nan'), dtype=torch.bfloat16, device=device)
    print(zero, one, inf, nan)
    torch.abs(inf)
    print(torch.nextafter(zero, -one))
    print(torch.nextafter(zero, one))
    print(torch.nextafter(inf, zero))
    print(torch.nextafter(inf, inf))

fn('cpu')
fn('cuda')
