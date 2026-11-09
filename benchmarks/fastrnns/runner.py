from collections import namedtuple
from functools import partial

import torchvision.models as cnn

import torch

from .factory import (
    imagenet_cnn_creator,
    layernorm_pytorch_lstm_creator,
    lstm_creator,
    pytorch_lstm_creator,
    varlen_lstm_creator,
    varlen_pytorch_lstm_creator,
)


class DisableCuDNN:
    def __enter__(self):
        self.saved = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

    def __exit__(self, *args, **kwargs):
        torch.backends.cudnn.enabled = self.saved


class DummyContext:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class AssertNoJIT:
    def __enter__(self):
        import os

        enabled = os.environ.get("PYTORCH_JIT", 1)
        if enabled:
            raise AssertionError("PYTORCH_JIT must be disabled")

    def __exit__(self, *args, **kwargs):
        pass


RNNRunner = namedtuple(
    "RNNRunner",
    [
        "name",
        "creator",
        "context",
    ],
)


def get_nn_runners(*names):
    return [nn_runners[name] for name in names]


nn_runners = {
    "cudnn": RNNRunner("cudnn", pytorch_lstm_creator, DummyContext),
    "cudnn_dropout": RNNRunner(
        "cudnn_dropout", partial(pytorch_lstm_creator, dropout=0.4), DummyContext
    ),
    "cudnn_layernorm": RNNRunner(
        "cudnn_layernorm", layernorm_pytorch_lstm_creator, DummyContext
    ),
    "vl_cudnn": RNNRunner("vl_cudnn", varlen_pytorch_lstm_creator, DummyContext),
    "vl_py": RNNRunner("vl_py", varlen_lstm_creator, DummyContext),
    "aten": RNNRunner("aten", pytorch_lstm_creator, DisableCuDNN),
    "py": RNNRunner("py", lstm_creator, DummyContext),
    "resnet18": RNNRunner("resnet18", imagenet_cnn_creator(cnn.resnet18), DummyContext),
    "resnet50": RNNRunner("resnet50", imagenet_cnn_creator(cnn.resnet50), DummyContext),
}
