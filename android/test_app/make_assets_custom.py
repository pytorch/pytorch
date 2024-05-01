"""
This is a script for PyTorch Android custom selective build test. It prepares
MobileNetV2 TorchScript model, and dumps root ops used by the model for custom
build script to create a tailored build which only contains these used ops.
"""

import torch
import yaml
from torchvision import models

# Download and trace the model.
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()
example = torch.rand(1, 3, 224, 224)
# TODO: create script model with `torch.jit.script`
traced_script_module = torch.jit.trace(model, example)

# Save traced TorchScript model.
traced_script_module.save("MobileNetV2.pt")

# Dump root ops used by the model (for custom build optimization).
ops = torch.jit.export_opnames(traced_script_module)

with open("MobileNetV2.yaml", "w") as output:
    yaml.dump(ops, output)
