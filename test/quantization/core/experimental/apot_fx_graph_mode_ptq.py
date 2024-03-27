import torch
import torch.nn as nn
import torch.ao.quantization
from torchvision.models.quantization.resnet import resnet18
from torch.ao.quantization.experimental.quantization_helper import (
    evaluate,
    prepare_data_loaders
)

# validation dataset: full ImageNet dataset
data_path = '~/my_imagenet/'

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = resnet18(pretrained=True)
float_model.eval()

# deepcopy the model since we need to keep the original model around
import copy
model_to_quantize = copy.deepcopy(float_model)

model_to_quantize.eval()

"""
Prepare models
"""

# Note that this is temporary, we'll expose these functions to torch.ao.quantization after official releasee
from torch.ao.quantization.quantize_fx import prepare_qat_fx

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

from torch.ao.quantization.experimental.qconfig import (
    uniform_qconfig_8bit,
    apot_weights_qconfig_8bit,
    apot_qconfig_8bit,
    uniform_qconfig_4bit,
    apot_weights_qconfig_4bit,
    apot_qconfig_4bit
)

"""
Prepare full precision model
"""
full_precision_model = float_model

top1, top5 = evaluate(full_precision_model, criterion, data_loader_test)
print(f"Model #0 Evaluation accuracy on test dataset: {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model PTQ for specified qconfig for torch.nn.Linear
"""
def prepare_ptq_linear(qconfig):
    qconfig_dict = {"object_type": [(torch.nn.Linear, qconfig)]}
    prepared_model = prepare_qat_fx(copy.deepcopy(float_model), qconfig_dict)  # fuse modules and insert observers
    calibrate(prepared_model, data_loader_test)  # run calibration on sample data
    return prepared_model

"""
Prepare model with uniform activation, uniform weight
b=8, k=2
"""

prepared_model = prepare_ptq_linear(uniform_qconfig_8bit)
quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model

top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print(f"Model #1 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with uniform activation, uniform weight
b=4, k=2
"""

prepared_model = prepare_ptq_linear(uniform_qconfig_4bit)
quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model

top1, top5 = evaluate(quantized_model1, criterion, data_loader_test)
print(f"Model #1 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with uniform activation, APoT weight
(b=8, k=2)
"""

prepared_model = prepare_ptq_linear(apot_weights_qconfig_8bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #2 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with uniform activation, APoT weight
(b=4, k=2)
"""

prepared_model = prepare_ptq_linear(apot_weights_qconfig_4bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #2 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")


"""
Prepare model with APoT activation and weight
(b=8, k=2)
"""

prepared_model = prepare_ptq_linear(apot_qconfig_8bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #3 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with APoT activation and weight
(b=4, k=2)
"""

prepared_model = prepare_ptq_linear(apot_qconfig_4bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #3 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare eager mode quantized model
"""
eager_quantized_model = resnet18(pretrained=True, quantize=True).eval()
top1, top5 = evaluate(eager_quantized_model, criterion, data_loader_test)
print(f"Eager mode quantized model evaluation accuracy on test dataset: {top1.avg:2.2f}, {top5.avg:2.2f}")
