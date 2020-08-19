import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.models.quantization import mobilenet_v2
import os
import requests
import copy

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
    print("starting an evaluation")
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

def imagenet_download():
    data_path = '/mnt/fair/imagenet_full_size/'
    data_loader, data_loader_test = prepare_data_loaders(data_path)

    model = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=False)
    print("done loading")
    return model, data_loader, data_loader_test

class ConvChain(nn.Module):
    def __init__(self):
        super(ConvChain, self).__init__()
        self.conv2d1 = nn.Conv2d(3, 4, 5, 5)
        self.conv2d2 = nn.Conv2d(4, 5, 5, 5)
        self.conv2d3 = nn.Conv2d(5, 6, 5, 5)

    def forward(self, x):
        x1 = self.conv2d1(x)
        x2 = self.conv2d2(x1)
        x3 = self.conv2d3(x2)
        return x3

def load_conv():
    model = ConvChain()
    copy_of_model = copy.deepcopy(model)
    model.train()
    img_data = [(torch.rand(10, 3, 125, 125, dtype=torch.float, requires_grad=True), torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(500)]
    return model, img_data

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
