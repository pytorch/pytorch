from torchvision import models

import torch

print(torch.version.__version__)

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.eval()
resnet18_traced = torch.jit.trace(resnet18, torch.rand(1, 3, 224, 224)).save(
    "app/src/main/assets/resnet18.pt"
)

resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet50.eval()
torch.jit.trace(resnet50, torch.rand(1, 3, 224, 224)).save(
    "app/src/main/assets/resnet50.pt"
)

mobilenet2q = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
mobilenet2q.eval()
torch.jit.trace(mobilenet2q, torch.rand(1, 3, 224, 224)).save(
    "app/src/main/assets/mobilenet2q.pt"
)
