from collections import namedtuple
from functools import partial

import torch
import torchvision.models as cnn

from .factory import (
    dropoutlstm_creator,
    imagenet_cnn_creator,
    layernorm_pytorch_lstm_creator,
    lnlstm_creator,
    lstm_creator,
    lstm_multilayer_creator,
    lstm_premul_bias_creator,
    lstm_premul_creator,
    lstm_simple_creator,
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
        assert not enabled

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
    "vl_jit": RNNRunner(
        "vl_jit", partial(varlen_lstm_creator, script=True), DummyContext
    ),
    "vl_py": RNNRunner("vl_py", varlen_lstm_creator, DummyContext),
    "aten": RNNRunner("aten", pytorch_lstm_creator, DisableCuDNN),
    "jit": RNNRunner("jit", lstm_creator, DummyContext),
    "jit_premul": RNNRunner("jit_premul", lstm_premul_creator, DummyContext),
    "jit_premul_bias": RNNRunner(
        "jit_premul_bias", lstm_premul_bias_creator, DummyContext
    ),
    "jit_simple": RNNRunner("jit_simple", lstm_simple_creator, DummyContext),
    "jit_multilayer": RNNRunner(
        "jit_multilayer", lstm_multilayer_creator, DummyContext
    ),
    "jit_layernorm": RNNRunner("jit_layernorm", lnlstm_creator, DummyContext),
    "jit_layernorm_decom": RNNRunner(
        "jit_layernorm_decom",
        partial(lnlstm_creator, decompose_layernorm=True),
        DummyContext,
    ),
    "jit_dropout": RNNRunner("jit_dropout", dropoutlstm_creator, DummyContext),
    "py": RNNRunner("py", partial(lstm_creator, script=False), DummyContext),
    "resnet18": RNNRunner(
        "resnet18", imagenet_cnn_creator(cnn.resnet18, jit=False), DummyContext
    ),
    "resnet18_jit": RNNRunner(
        "resnet18_jit", imagenet_cnn_creator(cnn.resnet18), DummyContext
    ),
    "resnet50": RNNRunner(
        "resnet50", imagenet_cnn_creator(cnn.resnet50, jit=False), DummyContext
    ),
    "resnet50_jit": RNNRunner(
        "resnet50_jit", imagenet_cnn_creator(cnn.resnet50), DummyContext
    ),
}
