"""
This is a script for end-to-end mobile custom build test purpose. It prepares
MobileNetV2 TorchScript model, and dumps root ops used by the model for custom
build script to create a tailored build which only contains these used ops.
"""

import yaml
from torchvision import models

import torch


# Download and trace the model.
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)

# Save traced TorchScript model.
traced_script_module.save("MobileNetV2.pt")

# Dump root ops used by the model (for custom build optimization).
ops = torch.jit.export_opnames(traced_script_module)

# Besides the ops used by the model, custom c++ client code might use some extra
# ops, too. For example, the dummy predictor.cpp driver in this test suite calls
# `aten::ones` to create all-one-tensor for testing purpose, which is not used
# by the MobileNetV2 model itself.
#
# This is less a problem for Android, where we expect users to use the limited
# set of Java APIs. To actually solve this problem, we probably need ask users
# to run code analyzer against their client code to dump these extra root ops.
# So it will require more work to switch to custom build with dynamic dispatch -
# in static dispatch case these extra ops will be kept by linker automatically.
#
# For CI purpose this one-off hack is probably fine? :)
EXTRA_CI_ROOT_OPS = ["aten::ones"]

ops.extend(EXTRA_CI_ROOT_OPS)
with open("MobileNetV2.yaml", "w") as output:
    yaml.dump(ops, output)
