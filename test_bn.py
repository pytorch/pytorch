import torch
from torch import nn
from torch.autograd import Variable
import time

def conv_test_cpu():

    iters = 200

    inputs = Variable(torch.ones(32, 128, 56, 56), requires_grad=True)
    layer = nn.Conv2d(128, 64, (3, 3), stride=(2, 2), padding=(0, 0))

    out = layer(inputs)
    grad = torch.ones(out.shape)

    # warm up
    for i in range(iters):
        out = layer(inputs)
        out.backward(grad)

    # record forward time without backward
    time_forward = 0
    for i in range(iters):
        start = time.time()
        layer(inputs)
        time_forward += time.time() - start

    # record forward time and backward time with backward
    time_forward_with_backward = 0
    time_backward_with_backward = 0
    for i in range(iters):
        start = time.time()
        out = layer(inputs)
        time_forward_with_backward += time.time() - start
        start = time.time()
        out.backward(grad)
        time_backward_with_backward += time.time() - start

    # print result
    print('--------test cpu----------')
    print('iter number: ' + str(iters))
    print('forward time without backward: ' + str(time_forward))
    print('forward time with backward: ' + str(time_forward_with_backward))
    print('backward time without backward: ' + str(time_backward_with_backward))


def conv_test_mix():

    iters = 200
    layer_cpu = nn.Conv2d(128, 128, (1,1), stride=(1,1), padding=(0,0))
    layer_gpu = nn.Conv2d(128, 64, (3, 3), stride=(2, 2), padding=(0, 0)).cuda()
    inputs = Variable(torch.ones(32, 128, 56, 56), requires_grad=True)
    mid = layer_cpu(inputs).cuda()
    out = layer_gpu(mid)
    grad = torch.ones(out.shape).cuda()

    # warm up
    for i in range(iters):
        mid = layer_cpu(inputs).cuda()
        layer_gpu(mid)

    # record forward time without backward
    time_forward = 0
    for i in range(iters):
        start = time.time()
        mid = layer_cpu(inputs).cuda()
        time_forward += time.time() - start
        layer_gpu(mid)
        torch.cuda.synchronize()

    # record forward time and backward time with backward
    time_forward_with_backward = 0
    time_backward_with_backward = 0
    for i in range(iters):
        start = time.time()
        mid = layer_cpu(inputs).cuda()
        time_forward_with_backward += time.time() - start
        start = time.time()
        out = layer_gpu(mid)
        out.backward(grad)
        time_backward_with_backward += time.time() - start
        torch.cuda.synchronize()

    # print result
    print('--------test cpu and gpu mix----------')
    print('iter number: ' + str(iters))
    print('forward time without backward: ' + str(time_forward))
    print('forward time with backward: ' + str(time_forward_with_backward))
    print('backward time without backward: ' + str(time_backward_with_backward))


if __name__ == '__main__':
    conv_test_cpu()
    conv_test_mix()