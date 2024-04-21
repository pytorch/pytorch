import torch
from torch.utils.bundled_inputs import augment_model_with_bundled_inputs
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision import models


class MobileNetV2Module:
    def getModule(self):
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()
        example = torch.zeros(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model, example)
        optimized_module = optimize_for_mobile(traced_script_module)
        augment_model_with_bundled_inputs(
            optimized_module,
            [
                (example,),
            ],
        )
        optimized_module(example)
        return optimized_module


class MobileNetV2VulkanModule:
    def getModule(self):
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()
        example = torch.zeros(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model, example)
        optimized_module = optimize_for_mobile(traced_script_module, backend="vulkan")
        augment_model_with_bundled_inputs(
            optimized_module,
            [
                (example,),
            ],
        )
        optimized_module(example)
        return optimized_module


class Resnet18Module:
    def getModule(self):
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        example = torch.zeros(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model, example)
        optimized_module = optimize_for_mobile(traced_script_module)
        augment_model_with_bundled_inputs(
            optimized_module,
            [
                (example,),
            ],
        )
        optimized_module(example)
        return optimized_module
