import torch
import torchvision
import yaml

model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
ops = torch.jit.export_opnames(traced_script_module)
with open('mobilenetv2.yaml', 'w') as output:
    yaml.dump(ops, output)
