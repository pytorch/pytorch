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
import _correct_bias

import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
NON_LEAF_MODULE_TO_ADD_OBSERVER_WHITE_LIST = {
    nnqd.Linear,
    nnq.Linear,
    nnqd.LSTM,
    nn.LSTM,
}

from  mobilenet_classes import (
    ConvBNReLU,
    InvertedResidual,
    MobileNetV2,
    ChainModule
)

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

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

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


###################################
#     loading data set plus custom metrics (sqnr)

def sqnr(og_tensor, quantized_tensor):
    # quantized_tensor.dequantize()
    print(og_tensor)
    print(quantized_tensor)
    a = torch.norm(og_tensor)
    b = torch.norm(og_tensor - quantized_tensor)
    return 20*torch.log10(a/b)

def report_sqnr(og_model, qmodel, names):
    sum = 0
    names = { x for pair in names for x in pair}
    for name in names:
        sum += sqnr(get_module(og_model, name), get_module(qmodel, name))
    print("sum of sqnrs among tensors: ", sum)

def get_module(model, name):
    curr = model
    name = name.split('.')
    for subname in name:
        curr = curr._modules[subname]
    return curr

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

def chain_module_download():
    model = ChainModule(True)
    return


def equalize_accuracy_demo(input_model, data_loader, data_loader_test):
    eval_batch_size = 30
    num_eval_batches = 10
    num_calibration_batches = 10

    criterion = nn.CrossEntropyLoss()

    def eval(quantize = True, per_tensor = True, equalize = False):
        model = copy.deepcopy(input_model)
        input_revised = grab_names(model)
        if quantize:
            model.eval()
            model.fuse_model()
            if equalize:
                input_revised = grab_names(model)
                _equalize.equalize(model, input_revised, 1e-4)

            if per_tensor:
                model.qconfig = torch.quantization.default_qconfig
            else:
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

            unquantized_model = copy.deepcopy(model)

            model = torch.quantization.prepare(model, inplace=False)
            evaluate(model,criterion, data_loader, num_calibration_batches)
            model = torch.quantization.convert(model, inplace=False)

            for name, module in model.named_modules():
                if hasattr(module, 'qconfig'):
                    # print(name)
                    del module.qconfig

            if equalize:
                top1, top5 = evaluate(model, criterion, data_loader_test, neval_batches=num_eval_batches)
                print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
                print("correcting bias?")


                # ns.prepare_model_outputs(unquantized_model, model, _correct_bias.MeanLogger)
                qconfig_debug = torch.quantization.QConfig(activation=_correct_bias.MeanLogger, weight=None)
                unquantized_model.qconfig = qconfig_debug
                model.qconfig = qconfig_debug
                white_list = [nn.Linear, nnq.Linear, nn.Conv2d, nnq.Conv2d]

                torch.quantization.prepare(unquantized_model, inplace=True, white_list=white_list, prehook=_correct_bias.MeanLogger)
                torch.quantization.prepare(model, inplace=True, white_list=white_list, observer_non_leaf_module_list=[nnq.Linear], prehook=_correct_bias.MeanLogger)
                count =0
                for data in data_loader_test:
                    with torch.no_grad():
                        count += 1
                        if count != 34:
                            output = unquantized_model(data[0])
                            q_output = model(data[0])
                            # print("sqnr score: ", _correct_bias.compute_error(output, q_output))
                        else:
                            pass
                print("finished bias calibrating")
                # compare_dict = ns.get_matching_activations(unquantized_model, model)
                output_logger, input_logger = _correct_bias.get_matching_activations(unquantized_model, model)

                _correct_bias.correct_quantized_bias_V2(output_logger, input_logger, unquantized_model, model)


        top1, top5 = evaluate(model, criterion, data_loader_test, neval_batches=num_eval_batches)
        print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
        # report_sqnr(float_model, model, input_revised)

    # eval(False)
    # print("\nno quantize")
    # eval(True, True, False)
    # print("\nper tensor, no equalize")
    # eval(True, False, False)
    # print("\nper channel, no equalize")
    eval(True, True, True)
    print("\nper tensor, with equalize")
    eval(True, False, True)
    print("\nper channel, with equalize")
    print("is it christmas :O")


if __name__ == "__main__":
    # linear_playground()
    equalize_accuracy_demo(*mobilenet_download())
    # bias_correction_demo()
    # main()
