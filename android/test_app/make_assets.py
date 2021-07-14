import torch
import torchvision

print(torch.version.__version__)

resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.eval()
resnet18_traced = torch.jit.trace(resnet18, torch.rand(1, 3, 224, 224)).save("app/src/main/assets/resnet18.pt")
torch.jit.script(resnet18)._save_for_lite_interpreter("app/src/main/assets/resnet18.ptl")

resnet50 = torchvision.models.resnet50(pretrained=True)
resnet50.eval()
torch.jit.trace(resnet50, torch.rand(1, 3, 224, 224)).save("app/src/main/assets/resnet50.pt")
torch.jit.script(resnet50)._save_for_lite_interpreter("app/src/main/assets/resnet50.ptl")

mobilenet2q = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)
mobilenet2q.eval()
torch.jit.trace(mobilenet2q, torch.rand(1, 3, 224, 224)).save("app/src/main/assets/mnet.pt")
torch.jit.script(mobilenet2q)._save_for_lite_interpreter("app/src/main/assets/mnet.ptl")
