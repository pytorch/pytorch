import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v2(pretrained=True)
trace_and_export_model(model, torch.rand(1, 3, 224, 224), 'mobilenet_v2')

model = torchvision.models.mobilenet_v3_small(pretrained=True)
trace_and_export_model(model, torch.rand(1, 3, 224, 224), 'mobilenet_v3_small')

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
trace_and_export_model(model, [torch.rand(3, 300, 400), torch.rand(3, 500, 400)], 'keypointrcnn_resnet50_fpn')


def trace_and_export_model(model, input, name):
    model.eval()
    traced_script_module = torch.jit.trace(model, input)
    optimized_scripted_module = optimize_for_mobile(traced_script_module)
    torch.jit.save(optimized_scripted_module, f'../models/{name}.pt')
    optimized_scripted_module._save_for_lite_interpreter(f'../models/{name}.ptl')
