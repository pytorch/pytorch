import torch


t = torch.randn(3, 3)

torch.linalg.QRResult  # E: Module has no attribute
torch.linalg.logdet(t)  # E: Module has no attribute
torch.linalg.trace(t)  # E: Module has no attribute
torch.linalg.det(input=t)  # E: Unexpected keyword argument
torch.linalg.vander(t, increasing=True)  # E: Unexpected keyword argument
torch.linalg.svd(t, out=t)  # E: incompatible type
torch.fft.fft("invalid")  # E: incompatible type
torch.special.airy_ai(input=t)  # E: Unexpected keyword argument
