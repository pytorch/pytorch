import yaml
from torchvision import models

import torch

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
ops = torch.jit.export_opnames(traced_script_module)
with open("mobilenetv2.yaml", "w") as output:
    yaml.dump(ops, output)
