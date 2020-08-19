import torch

# Configuration
# torch.set_num_threads(1)
torch.set_grad_enabled(False)

# Tensor Parameters
BATCH = 10
CHANNEL = 16
HEIGHT = 1024
WIDTH = 1024
DEVICE = "cpu"

# Pooling Parameters
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 2
DILATION = 1

# Input Tensor
blob = torch.randn(BATCH, CHANNEL, HEIGHT, WIDTH, device=DEVICE)


def test_max_pool2d(benchmark):
    model = torch.nn.MaxPool2d(KERNEL_SIZE, STRIDE, PADDING, DILATION)
    benchmark(model, blob)


def test_max_pool2d_with_indices(benchmark):
    model = torch.nn.MaxPool2d(KERNEL_SIZE, STRIDE, PADDING, DILATION, True)
    benchmark(model, blob)


def test_mkldnn_max_pool2d(benchmark):
    model = torch.nn.MaxPool2d(KERNEL_SIZE, STRIDE, PADDING, DILATION)
    benchmark(model, blob.to_mkldnn())


def test_correctness():
    model_1 = torch.nn.MaxPool2d(KERNEL_SIZE, STRIDE, PADDING, DILATION)
    model_2 = torch.nn.MaxPool2d(KERNEL_SIZE, STRIDE, PADDING, DILATION, True)
    assert model_1(blob).equal(model_2(blob)[0])
