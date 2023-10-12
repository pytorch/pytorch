
import torch
from tqdm import tqdm

device = 'cpu'
mult = 1000

real = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10.1], device=device)
imag = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11.1], device=device)

print(torch.complex(real, imag))

real = torch.hstack([real] * mult)
real = torch.vstack([real] * mult)

imag = torch.hstack([imag] * mult)
imag = torch.vstack([imag] * mult)

a = torch.complex(real, imag)
for i in tqdm(range(50)):
    a += torch.complex(real, imag)
