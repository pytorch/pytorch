import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.transforms as transforms
import os
import torch.quantization

# Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

"""
Define helper functions
"""

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print('')

    return top1, top5

def load_model(model_file):
    model = resnet18(pretrained=False)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to("cpu")
    return model

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")

def prepare_data_loaders(data_path):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(data_path,
                                            split="train",
                                            transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                          transforms.RandomHorizontalFlip(),
                                                                          transforms.ToTensor(),
                                                                          normalize]))
    dataset_test = torchvision.datasets.ImageNet(data_path,
                                                 split="val",
                                                 transform=transforms.Compose([transforms.Resize(256),
                                                                               transforms.CenterCrop(224),
                                                                               transforms.ToTensor(),
                                                                               normalize]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

data_path = '~/my_imagenet/'
saved_model_dir = '/data/home/amandaliu/cluster/pytorch/test/quantization/core/experimental/data/'
float_model_file = 'resnet18_pretrained_float.pth'

train_batch_size = 30
eval_batch_size = 50

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to("cpu")
float_model.eval()

# deepcopy the model since we need to keep the original model around
import copy
model_to_quantize = copy.deepcopy(float_model)

model_to_quantize.eval()

"""
Prepare models
"""

# Note that this is temporary, we'll expose these functions to torch.quantization after official releasee
from torch.quantization.quantize_fx import prepare_fx, convert_fx

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
print("Model #0 Evaluation accuracy on test dataset: %2.2f, %2.2f" % (top1.avg, top5.avg))

"""
Prepare model PTQ for specified qconfig for torch.nn.Linear
"""
def prepare_ptq_linear(qconfig):
    qconfig_dict = {"object_type": [(torch.nn.Linear, qconfig)]}
    prepared_model = prepare_fx(copy.deepcopy(float_model), qconfig_dict)  # fuse modules and insert observers
    calibrate(prepared_model, data_loader_test)  # run calibration on sample data
    return prepared_model

"""
Prepare model with uniform activation, uniform weight
b=8, k=2
"""

prepared_model = prepare_ptq_linear(uniform_qconfig_8bit)
quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model

top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("Model #1 Evaluation accuracy on test dataset (b=8, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))

"""
Prepare model with uniform activation, uniform weight
b=4, k=2
"""

prepared_model = prepare_ptq_linear(uniform_qconfig_4bit)
quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model

top1, top5 = evaluate(quantized_model1, criterion, data_loader_test)
print("Model #1 Evaluation accuracy on test dataset (b=4, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))

"""
Prepare model with uniform activation, APoT weight
(b=8, k=2)
"""

prepared_model = prepare_ptq_linear(apot_weights_qconfig_8bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print("Model #2 Evaluation accuracy on test dataset (b=8, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))

"""
Prepare model with uniform activation, APoT weight
(b=4, k=2)
"""

prepared_model = prepare_ptq_linear(apot_weights_qconfig_4bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print("Model #2 Evaluation accuracy on test dataset (b=4, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))


"""
Prepare model with APoT activation and weight
(b=8, k=2)
"""

prepared_model = prepare_ptq_linear(apot_qconfig_8bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print("Model #3 Evaluation accuracy on test dataset (b=8, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))

"""
Prepare model with APoT activation and weight
(b=4, k=2)
"""

prepared_model = prepare_ptq_linear(apot_qconfig_4bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print("Model #3 Evaluation accuracy on test dataset (b=4, k=2): %2.2f, %2.2f" % (top1.avg, top5.avg))

"""
Prepare eager mode quantized model
"""

from torchvision.models.quantization.resnet import resnet18
eager_quantized_model = resnet18(pretrained=True, quantize=True).eval()
top1, top5 = evaluate(eager_quantized_model, criterion, data_loader_test)
print("Eager mode quantized model evaluation accuracy on test dataset: %2.2f, %2.2f" % (top1.avg, top5.avg))
