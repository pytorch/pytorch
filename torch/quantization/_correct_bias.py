from __future__ import print_function, division, absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
# import torch.nn.quantized.QFunctional as QFunctional
from  torch.nn.quantized.modules import Quantize, DeQuantize
from torch._ops import ops

from torch.quantization import QuantStub, DeQuantStub


import torchvision
import torchvision.transforms as transforms
import os
import torch.quantization
import torch.quantization._numeric_suite as ns
from torchvision.models.quantization.mobilenet import mobilenet_v2
from torch.quantization import (
    default_eval_fn,
    default_qconfig,
    quantize,
)
import copy

from  mobilenet_classes import (
    ConvBNReLU,
    InvertedResidual,
    MobileNetV2,
    ChainModule
)

def get_module(model, name):
    curr = model
    name = name.split('.')
    for subname in name:
        curr = curr._modules[subname]
    return curr

def local_prepare_model_with_stubs(float_module, q_module, module_swap_list, Logger):
    r"""Prepare the model by attaching the float module to its matching quantized
    module as the shadow if the float module type is in module_swap_list.

    Example usage:
        prepare_model_with_stubs(float_model, q_model, module_swap_list, Logger)
        q_model(data)
        ob_dict = get_logger_dict(q_model, Logger)

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        module_swap_list: list of float module types to attach the shadow
        Logger: type of logger to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """

    float_module_children = {}
    for name, mod in float_module.named_children():
        float_module_children[name] = mod

    reassign = {}
    for name, mod in q_module.named_children():
        if name not in float_module_children:
            continue

        float_mod = float_module_children[name]

        if type(float_mod) not in module_swap_list:
            local_prepare_model_with_stubs(float_mod, mod, module_swap_list, Logger)

        if type(float_mod) in module_swap_list:
            reassign[name] = LocalShadow(mod, float_mod, Logger)

    for key, value in reassign.items():
        float_module._modules[key] = value

class LocalShadow(nn.Module):
    r"""Shadow module attaches the float module to its matching quantized module
    as the shadow. Then it uses Logger module to process the outputs of both
    modules.

    Args:
        q_module: module quantized from float_module that we want to shadow
        float_module: float module used to shadow q_module
        Logger: type of logger used to process the outputs of q_module and
            float_module. ShadowLogger or custom loggers can be used.
    """

    def __init__(self, q_module, float_module, Logger):
        super(LocalShadow, self).__init__()
        self.orig_module = q_module
        self.shadow_module = float_module
        self.dequant = nnq.DeQuantize()
        self.logger = Logger()

    def forward(self, *x):
        xl = ns._convert_tuple_to_list(x)
        output = self.orig_module(*xl)
        xl_float = ns._dequantize_tensor_list(xl)
        shadow_output = self.shadow_module(*xl_float)
        self.logger(output, shadow_output)
        return shadow_output

    def add(self, x, y):
        output = self.orig_module.add(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.add(x, y)
        self.logger(output, shadow_output)
        return shadow_output

    def add_scalar(self, x, y):
        output = self.orig_module.add_scalar(x, y)
        x = x.dequantize()
        shadow_output = self.shadow_module.add_scalar(x, y)
        self.logger(output, shadow_output)
        return shadow_output

    def mul(self, x, y):
        output = self.orig_module.mul(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.mul(x, y)
        self.logger(output, shadow_output)
        return shadow_output

    def mul_scalar(self, x, y):
        output = self.orig_module.mul_scalar(x, y)
        x = x.dequantize()
        shadow_output = self.shadow_module.mul_scalar(x, y)
        self.logger(output, shadow_output)
        return shadow_output

    def cat(self, x, dim=0):
        output = self.orig_module.cat(x, dim)
        x = [y.dequantize() for y in x]
        shadow_output = self.shadow_module.cat(x, dim)
        self.logger(output, shadow_output)
        return shadow_output

    def add_relu(self, x, y):
        output = self.orig_module.add_relu(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.add_relu(x, y)
        self.logger(output, shadow_output)
        return shadow_output

class LocalShadowLogger(ns.Logger):
    r"""Class used in Shadow module to record the outputs of the original and
    shadow modules.
    """

    def __init__(self):
        super(LocalShadowLogger, self).__init__()
        self.stats["float"] = None
        self.stats["quantized"] = None

    def forward(self, x, y):
        if len(x) > 1:
            x = x[0]
        if len(y) > 1:
            y = y[0]
        if self.stats["quantized"] is None:
            self.stats["quantized"] = x.detach()
        else:
            self.stats["quantized"] = torch.cat((self.stats["quantized"], x.detach()))

        if self.stats["float"] is None:
            self.stats["float"] = y.detach()
        else:
            self.stats["float"] = torch.cat((self.stats["float"], y.detach()))

module_swap_list = [torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d,
                    torch.nn.Conv2d,
                    ConvBNReLU,
                    torchvision.models.mobilenet.ConvBNReLU,
                    torch.quantization.observer.MinMaxObserver,  # might be a problem
                    torchvision.models.quantization.mobilenet.QuantizableInvertedResidual,
                    torch.nn.modules.batchnorm.BatchNorm2d,
                    torch.nn.modules.linear.Linear,
                    torch.nn.modules.conv.Conv2d,
                    ]

def print_hooks(model):
    for name, module in model.named_modules():
        print(module._forward_hooks)
class OutputLogger(ns.Logger):
    r"""Class used to log the outputs of the module
    """

    def __init__(self):
        super(OutputLogger, self).__init__()
        self.stats["tensor_val"] = None

    def forward(self, x):
        if self.stats["tensor_val"] is None:
            self.stats["tensor_val"] = x
        else:
            self.stats["tensor_val"] = torch.cat((self.stats["tensor_val"], x))
        return x

class MeanLogger(ns.Logger):
    def __init__(self):
        super(MeanLogger, self).__init__()
        self.stats["tensor_val"] = None
        self.count = 0
        self.sum = None
        self.quantized = None

    def forward(self, x):
        if self.stats["tensor_val"] is None:
            self.stats["tensor_val"] = x
            self.sum = x
            self.count = 1
        else:
            if self.quantized is None:
                try:
                    self.sum = self.sum + x
                    self.count += 1
                    self.stats["tensor_val"] = self.sum / self.count
                    self.quantized = False
                except:
                    self.count += 1
                    self.stats["tensor_val"] = ops.quantized.mul_scalar(self.sum, 1/self.count)
                    self.quantized = True
            elif self.quantized:
                self.sum = ops.quantized.add(self.sum, x, 1, 0)
                self.count += 1
                self.stats["tensor_val"] = ops.quantized.mul_scalar(self.sum, 1/self.count)
            else:
                self.sum = self.sum + x
                self.count += 1
                self.stats["tensor_val"] = self.sum / self.count
        return x

def correct_quantized_bias(output_dict, float_model, qmodel):
    ''' Inplace functino to modify the weights of the qmodel based off the
    recorded differences in weights provided by output_dict

    '''
    for key in output_dict:
        q_mod = get_module(qmodel, key[:-6])  #.orig_module
        # print(key[:-6])

        if hasattr(q_mod, 'bias'):
            if (isinstance(q_mod.bias, torch.nn.parameter.Parameter) and (q_mod.bias is not None)) or \
                (q_mod.bias() is not None):
                bias = None
                if isinstance(q_mod.bias, torch.nn.parameter.Parameter):
                    bias = q_mod.bias
                else:
                    bias = q_mod.bias()

                difference = output_dict[key]['float'] - output_dict[key]['quantized'].dequantize()
                dims_to_mean_over = [i for i in range(difference.dim())]
                dims_to_mean_over.remove(1)
                print(dims_to_mean_over)
                print("difference: ", difference.size())
                difference = torch.mean(difference, dims_to_mean_over)
                print("difference: ", difference.size())
                print()

                updated_bias = bias.data - difference

                if isinstance(q_mod.bias, torch.nn.parameter.Parameter):
                    q_mod.bias.data = updated_bias
                else:
                    q_mod.bias().data = updated_bias
    # print(qmodel)
def bias_absorption(float_model, qmodel, paired_modules_list, output_dict):
    #max(0, mean - 3*std)
    pass

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def loading_running_mobilenet_img():
    saved_model_dir = '/home/edmundjr/local/pytorch/torch/quantization/data/'
    float_model_file = 'mobilenet_pretrained_float.pth'
    # float_model = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=False)  # mobilenet_v2, resnet18
    float_model = load_model(saved_model_dir + float_model_file)
    float_model.to('cpu')
    float_model.eval()
    float_model.qconfig = torch.quantization.default_qconfig

    img_data = [(torch.rand(2, 3, 10, 10, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                            for _ in range(30)]
    # qmodel = quantize(float_model, default_eval_fn, img_data, inplace=False)
    # qmodel = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=False)
    qmodel = copy.deepcopy(float_model)
    for data in img_data:
        float_model(data[0])
        qmodel(data[0])
    print(qmodel)
    print('#'*30)
    torch.quantization.prepare(qmodel, inplace=True)
    for data in img_data:
        float_model(data[0])
        qmodel(data[0])
    print(qmodel)
    print('#'*30)
    torch.quantization.convert(qmodel, inplace=True)
    print(qmodel)
    print('#'*30)

    for data in img_data:
        float_model(data[0])
        qmodel(data[0])  # problems

def loading_running_chainmodule_img():
    ''' adding the qconfig allowed the quantization to execute and added the activation_post_process

    '''
    float_model = ChainModule(True)
    float_model.to('cpu')
    float_model.eval()
    float_model.qconfig = torch.quantization.default_qconfig
    img_data = [torch.rand(10, 3, dtype=torch.float) for _ in range(30)]

    qmodel = copy.deepcopy(float_model)
    # qmodel = quantize(float_model, default_eval_fn, img_data, inplace=False)
    for data in img_data:
        float_model(data)
        qmodel(data)
    print(qmodel)
    # print(qmodel.linear1.__dict__)
    # print(qmodel.quant.__dict__)
    # print_hooks(qmodel)
    # print('#'*30)

    torch.quantization.prepare(qmodel, inplace=True)
    for data in img_data:
        float_model(data)
        qmodel(data)
    print(qmodel)
    # print(qmodel.linear1.__dict__)
    # print(qmodel.quant.__dict__)
    # print_hooks(qmodel)
    # print('#'*30)

    torch.quantization.convert(qmodel, inplace=True)
    print(qmodel)
    # print(qmodel.linear1.__dict__)
    # print(qmodel.quant.__dict__)
    # print_hooks(qmodel)
    # print('#'*30)
    for data in img_data:
        float_model(data)
        qmodel(data)


def setup(Logger=ns.OutputLogger):
    float_model = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=False)  # mobilenet_v2, resnet18
    # float_model = nn.Conv2d(3,2,10)
    float_model.to('cpu')
    float_model.eval()
    float_model.fuse_model()
    float_model.qconfig = torch.quantization.default_qconfig
    img_data = [(torch.rand(2, 3, 10, 10, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                            for _ in range(30)]
    print(len(img_data))
    print(img_data[0][0].size())
    print(img_data[0][1].size())

    qmodel = copy.deepcopy(float_model)
    qmodel = quantize(qmodel, default_eval_fn, img_data, inplace=False)
    # for data in img_data:
    #     float_model(data[0])
    #     qmodel(data[0])

    ns.prepare_model_outputs(float_model, qmodel, Logger)  # ns.OutputLogger)
    # print(qmodel)
    for data in img_data:
        with torch.no_grad():
            float_model(data[0])
            qmodel(data[0])
    compare_dict = ns.get_matching_activations(float_model, qmodel)
    '''
    before convert, the activation_post thingy already exists
    '''

    return compare_dict, float_model, qmodel

def compute_error(x,y):
    Ps = torch.norm(x)
    Pn = torch.norm(x-y)
    return 20*torch.log10(Ps/Pn)
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
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
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def main():
    eval_batch_size = 2
    num_eval_batches = 10
    num_calibration_batches = 10
    float_model = ChainModule(True)
    input_img_data = [(torch.rand(10, 3, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                            for _ in range(30)]
    output_img_data = [(torch.rand(10, 3, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                            for _ in range(30)]
    float_model.qconfig = torch.quantization.default_qconfig
    qmodel = quantize(float_model, default_eval_fn, input_img_data, inplace=False)
    criterion = nn.CrossEntropyLoss()

    top1, top5 = evaluate(qmodel, criterion, output_img_data, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))

    ns.prepare_model_outputs(float_model, qmodel, MeanLogger)
    # print(qmodel)
    for data in output_img_data:
        float_model(data[0])
        qmodel(data[0])


    compare_dict = ns.get_matching_activations(float_model, qmodel, MeanLogger)

    correct_quantized_bias(compare_dict, float_model, qmodel)


    top1, top5 = evaluate(qmodel, criterion, output_img_data, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))



if __name__ == "__main__":
    # correct_quantized_bias(*setup())
    compare_dict, float_model, qmodel = setup(ns.OutputLogger)
    for key in compare_dict:
        print(key)
    for key in compare_dict:
        if hasattr(compare_dict[key]['float'], 'size'):
            print(compare_dict[key]['float'].size())
            print(compare_dict[key]['quantized'].size())
    compare_dict, float_model, qmodel = setup(MeanLogger)
    for key in compare_dict:
        if hasattr(compare_dict[key]['float'], 'size'):
            print(compare_dict[key]['float'].size())
            print(compare_dict[key]['quantized'].size())
    correct_quantized_bias(*setup(MeanLogger))
    # main()
    # loading_running_mobilenet_img()
    # loading_running_chainmodule_img()
