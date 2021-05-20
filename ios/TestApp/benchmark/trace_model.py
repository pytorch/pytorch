import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.script(model, example)
optimized_scripted_module = optimize_for_mobile(traced_script_module)
exported_optimized_scripted_module = optimized_scripted_module._save_for_lite_interpreter("model.ptl")
