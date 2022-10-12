from torchvision.models.quantization.resnet import resnet18
from torch.ao.quantization.experimental.quantization_helper import (
    evaluate,
    prepare_data_loaders,
    training_loop
)

# training and validation dataset: full ImageNet dataset
data_path = '~/my_imagenet/'

train_batch_size = 30
eval_batch_size = 50

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = resnet18(pretrained=True)
float_model.eval()

# deepcopy the model since we need to keep the original model around
import copy
model_to_quantize = copy.deepcopy(float_model)

model_to_quantize.eval()

"""
Prepare model QAT for specified qconfig for torch.nn.Linear
"""
def prepare_qat_linear(qconfig):
    qconfig_dict = {"object_type": [(torch.nn.Linear, qconfig)]}
    prepared_model = prepare_fx(copy.deepcopy(float_model), qconfig_dict)  # fuse modules and insert observers
    training_loop(prepared_model, criterion, data_loader)
    prepared_model.eval()
    return prepared_model

"""
Prepare model with uniform activation, uniform weight
b=8, k=2
"""

prepared_model = prepare_qat_linear(uniform_qconfig_8bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print("Model #1 Evaluation accuracy on test dataset (b=8, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))

"""
Prepare model with uniform activation, uniform weight
b=4, k=2
"""

prepared_model = prepare_qat_linear(uniform_qconfig_4bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print("Model #1 Evaluation accuracy on test dataset (b=4, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))

"""
Prepare model with uniform activation, APoT weight
(b=8, k=2)
"""

prepared_model = prepare_qat_linear(apot_weights_qconfig_8bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print("Model #2 Evaluation accuracy on test dataset (b=8, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))

"""
Prepare model with uniform activation, APoT weight
(b=4, k=2)
"""

prepared_model = prepare_qat_linear(apot_weights_qconfig_4bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print("Model #2 Evaluation accuracy on test dataset (b=4, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))


"""
Prepare model with APoT activation and weight
(b=8, k=2)
"""

prepared_model = prepare_qat_linear(apot_qconfig_8bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print("Model #3 Evaluation accuracy on test dataset (b=8, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))

"""
Prepare model with APoT activation and weight
(b=4, k=2)
"""

prepared_model = prepare_qat_linear(apot_qconfig_4bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print("Model #3 Evaluation accuracy on test dataset (b=4, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))
