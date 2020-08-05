from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.quantization._numeric_suite as ns
from torchvision.models.quantization.mobilenet import mobilenet_v2
import os

# from torchvision.train import train_one_epoch, evaluate, load_data

import torch.quantization
# from torch.nn.quantized import (Quantize, DeQuantize)

from torch.quantization import (
    default_eval_fn,
    default_qconfig,
    quantize,
)
import requests
import copy

import _equalize
# import _correct_bias
import adaround

import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
NON_LEAF_MODULE_TO_ADD_OBSERVER_WHITE_LIST = {
    nnqd.Linear,
    nnq.Linear,
    nnqd.LSTM,
    nn.LSTM,
}

from torch.quantization.adaround import adaround_qconfig

# from  mobilenet_classes import (
#     ConvBNReLU,
#     InvertedResidual,
#     MobileNetV2,
#     ChainModule
# )

# Specify random seed for repeatable results
torch.manual_seed(191009)

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
    print("starting and evaluation")
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
    print("finishing an evaluation")
    return top1, top5

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model


def prepare_data_loaders(data_path):
    train_batch_size = 30
    eval_batch_size = 30

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test


def grab_names(model):
    input = []
    curr_feature = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            curr_feature.append(name)
        if isinstance(module, nn.quantized.modules.functional_modules.FloatFunctional):
            curr_feature = curr_feature[1:]
            input.append(curr_feature)
            curr_feature = []
    input_revised = []
    for feature in input:
        for i in range(len(feature)-1):
            input_revised.append([feature[i], feature[i+1]])
    return input_revised

def downloading_models():
    url = 'https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip'
    PROXIES = {'http': 'fwdproxy:8080', 'https': 'fwdproxy:8080'}

    #filename = '~/Downloads/imagenet_1k_data.zip'
    filename = '/home/edmundjr/local/pytorch/torch/quantization/data/imagenet_1k_data.zip'

    # r = requests.get(url, proxies=PROXIES)
    # with open(filename, 'wb') as f:
    #     f.write(r.content)

    model_urls = {
        'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    }
    url = model_urls['mobilenet_v2']
    filename = '/home/edmundjr/local/pytorch/torch/quantization/data/mobilenet_pretrained_float.pth'

    # r = requests.get(url, proxies=PROXIES)
    # with open(filename, 'wb') as f:
    #     f.write(r.content)

def mobilenet_download():
    data_path = '/home/edmundjr/local/pytorch/torch/quantization/data/imagenet_1k'
    saved_model_dir = '/home/edmundjr/local/pytorch/torch/quantization/data/'
    float_model_file = 'mobilenet_pretrained_float.pth'
    data_loader, data_loader_test = prepare_data_loaders(data_path)
    model = load_model(saved_model_dir + float_model_file)
    for data in data_loader:
        model(data[0])
        break

    return model, data_loader, data_loader_test

def imagenet_download():
    data_path = '/mnt/fair/imagenet_full_size/'
    data_loader, data_loader_test = prepare_data_loaders(data_path)

    model = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=False)
    print("done loading")
    return model, data_loader, data_loader_test

def quantize_model(model, data_loader_test, per_tensor=True):
    print("starting quantization")
    # criterion = nn.CrossEntropyLoss()
    num_calibration_batches = 30
    num_eval_batches = 10

    model = copy.deepcopy(model)
    # if per_tensor:
    #     model.qconfig = torch.quantization.default_qconfig
    # else:
    #     model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model.qconfig = adaround_qconfig

    # unquantized_model = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()
    model = torch.quantization.prepare_qat(model, inplace=False)
    evaluate(model, criterion, data_loader_test, neval_batches=num_eval_batches)
    # evaluate(model,criterion, data_loader, num_calibration_batches)
    # count = 0
    # for image, target in data_loader:
    #     with torch.no_grad():
    #         output = model(image)
    #         print(count)
    #         count += 1
    #         if count >= num_calibration_batches:
    #             break

    # model = torch.quantization.convert(model, inplace=False)

    print("ending quantization")
    return model

def adaround_demo(input_model, data_loader, data_loader_test):
    print("starting adaround")
    train_batch_size = 30
    eval_batch_size = 30
    num_eval_batches = 10
    criterion = nn.CrossEntropyLoss()
    model = copy.deepcopy(input_model)

    # throwing on the equalization
    model.eval()
    # model.fuse_model()

    quantized_tensor_model = quantize_model(model, data_loader_test, True)
    # quantized_tensor_model = copy.deepcopy(model)
    results = []
    print("starting initial evaluation")

    top1, top5 = evaluate(quantized_tensor_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    results.append(str('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg)))
    print('Per tensor quantization accuracy results, no adaround')
    results.append('Per tensor quantization accuracy results, no adaround')

    print("starting adaround quick function")
    # _correct_bias.sequential_bias_correction(quantized_tensor_model, model, data_loader_test)
    adaround.quick_function(quantized_tensor_model, 0, data_loader_test)
    # print(model._modules['features']._modules['9'])

    top1, top5 = evaluate(quantized_tensor_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    results.append(str('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg)))
    print('Per tensor quantization accuracy results, with adaround')
    results.append('Per tensor quantization accuracy results, with adaround')


    for result in results:
        print(result)

def correct_bias_demo(input_model, data_loader, data_loader_test):
    eval_batch_size = 30
    num_eval_batches = 10
    num_calibration_batches = 10
    criterion = nn.CrossEntropyLoss()
    model = copy.deepcopy(input_model)

    count = 0
    for data in data_loader_test:
        with torch.no_grad():
            print(data[0].size())
            print(count)
            count += 1

    # throwing on the equalization
    model.eval()
    model.fuse_model()
    input_revised = grab_names(model)
    _equalize.equalize(model, input_revised, 1e-4)
    results = []

    quantized_tensor_model = quantize_model(model, data_loader, True)
    top1, top5 = evaluate(quantized_tensor_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    results.append(str('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg)))
    print('Per tensor quantization accuracy results, no bias correction')
    results.append('Per tensor quantization accuracy results, no bias correction')

    quantized_channel_model = quantize_model(model, data_loader, False)
    top1, top5 = evaluate(quantized_channel_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    results.append(str('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg)))
    print('Per channel quantization accuracy results, no bias correction')
    results.append('Per channel quantization accuracy results, no bias correction')

    _correct_bias.sequential_bias_correction(quantized_tensor_model, model, data_loader_test)
    top1, top5 = evaluate(quantized_tensor_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    results.append(str('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg)))
    print('Per tensor quantization accuracy results, with bias correction')
    results.append('Per tensor quantization accuracy results, with bias correction')

    _correct_bias.sequential_bias_correction(quantized_channel_model, model, data_loader_test)
    top1, top5 = evaluate(quantized_channel_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    results.append(str('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg)))
    print('Per channel quantization accuracy results, with bias correction')
    results.append('Per channel quantization accuracy results, with bias correction')

    for result in results:
        print(result)


def equalize_accuracy_demo(input_model, data_loader, data_loader_test):
    eval_batch_size = 30
    num_eval_batches = 10
    num_calibration_batches = 10

    criterion = nn.CrossEntropyLoss()

    model = copy.deepcopy(input_model)
    count =0
    for data in data_loader:
        print(data[0].size())
        print(count)
        count+=1
    def eval(quantize = True, per_tensor = True, equalize = False):
        results = []
        model = copy.deepcopy(input_model)
        input_revised = grab_names(model)
        model.eval()
        model.fuse_model()
        if equalize:
            input_revised = grab_names(model)
            _equalize.equalize(model, input_revised, 1e-4)

        unquantized_model = copy.deepcopy(model)
        model = quantize_model(model, data_loader, per_tensor)

        for name, module in model.named_modules():
            if hasattr(module, 'qconfig'):
                del module.qconfig

        top1, top5 = evaluate(model, criterion, data_loader_test, neval_batches=num_eval_batches)
        # print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
        results.append(str('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg)))
        # print('Pre bias correction')
        results.append('Pre bias correction')

        _correct_bias.sequential_bias_correction(unquantized_model, model, data_loader_test)
        # _correct_bias.parallel_bias_correction(unquantized_model, model, data_loader_test)

        top1, top5 = evaluate(model, criterion, data_loader_test, neval_batches=num_eval_batches)
        # print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
        results.append(str('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg)))
        # print('Post bias correction')
        results.append('Post bias correction')
        return results

    results = eval(model, True, True)
    for result in results:
        print(result)
    print("\nper tensor, with equalize")
    results = eval(model, False, True)
    for result in results:
        print(result)
    print("\nper channel, with equalize")
    print("is it christmas :O")


if __name__ == "__main__":
    # linear_playground()
    # equalize_accuracy_demo(*mobilenet_download())
    # imagenet_download()
    # equalize_accuracy_demo(*imagenet_download())
    # correct_bias_demo(*imagenet_download())
    adaround_demo(*imagenet_download())
    # prepare_data_loaders('/mnt/fair/imagenet_full_size/')
    # bias_correction_demo()
    # main()
