import torch
import torch._inductor

model = torch._inductor.aoti_load_package("model.pt2")
print("loaded model.pt2")
print(model(torch.randn(8, 10, device="cuda")))
