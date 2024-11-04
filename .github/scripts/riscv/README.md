# Build RISC-V PyTorch From Scratch

Build the image:
```bash
docker build -t riscv-pytorch:latest -f .github/scripts/riscv/Dockerfile .
```

Run the container:
```bash
docker run -it --rm riscv-pytorch:latest
```

Snapshot:
```bash
root@5fc119de244d:/home/ubuntu# python3
Python 3.12.3 (main, Sep 11 2024, 14:17:37) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> import math
>>> dtype = torch.float
>>> device = torch.device("cpu")
>>> x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
>>> y = torch.sin(x)
>>> y
tensor([ 8.7423e-08, -3.1430e-03, -6.2863e-03,  ...,  6.2863e-03,
         3.1430e-03, -8.7423e-08])
>>> exit()
```
