import numbers
import torch.nn
import torch.nn.functions as functions

def conv2d(input, weight, bias=None, **kwargs):
    return functions.conv.Conv2d(**kwargs)(input, weight, bias)

def conv2d_transpose(input, weight, bias=None, **kwargs):
    return functions.conv.ConvTranspose2d(**kwargs)(input, weight, bias)

def relu(input):
    return torch.nn.ReLU()(input)

def avg_pool(input, **kwargs):
    if input.dim() == 4:
        state = torch.nn.AvgPool2d(**kwargs)
    elif input.dim() == 5:
        state = torch.nn.AvgPool3d(**kwargs)
    return state(input)

def max_pool(input, **kwargs):
    if input.dim() == 3:
        state = functions.thnn.MaxPool1d(**kwargs)
    elif input.dim() == 4:
        state = functions.thnn.MaxPool2d(**kwargs)
    elif input.dim() == 5:
        state = functions.thnn.MaxPool3d(**kwargs)
    return state(input)

def linear(input, weight, bias=None):
    return functions.linear.Linear()(input, weight, bias)

def batch_norm(input, weight, bias, **kwargs):
    return functions.thnn.BatchNorm(**kwargs)(input, weight, bias)

def logsoftmax(input):
    return functions.thnn.LogSoftmax()(input)

def cross_entropy(x, y):
    return torch.nn.CrossEntropyLoss()(x, y)

def dropout(x, **kwargs):
    return functions.dropout.Dropout(**kwargs)(x)
