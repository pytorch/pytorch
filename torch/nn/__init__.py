# mypy: allow-untyped-defs
__all__ = [
    # submodules
    "attention",
    "functional",
    "init",
    "modules",
    "parallel",
    "parameter",
    "utils",
    # parameter
    "Parameter",
    "UninitializedBuffer",
    "UninitializedParameter",
    # modules
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveLogSoftmaxWithLoss",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AlphaDropout",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BCELoss",
    "BCEWithLogitsLoss",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Bilinear",
    "CELU",
    "CTCLoss",
    "ChannelShuffle",
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "Container",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "CosineEmbeddingLoss",
    "CosineSimilarity",
    "CrossEntropyLoss",
    "CrossMapLRN2d",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "FeatureAlphaDropout",
    "Flatten",
    "Fold",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "GELU",
    "GLU",
    "GRU",
    "GRUCell",
    "GaussianNLLLoss",
    "GroupNorm",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "Identity",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "KLDivLoss",
    "L1Loss",
    "LPPool1d",
    "LPPool2d",
    "LPPool3d",
    "LSTM",
    "LSTMCell",
    "LayerNorm",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
    "LazyLinear",
    "LeakyReLU",
    "Linear",
    "LocalResponseNorm",
    "LogSigmoid",
    "LogSoftmax",
    "MSELoss",
    "MarginRankingLoss",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "Mish",
    "Module",
    "ModuleDict",
    "ModuleList",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "MultiheadAttention",
    "NLLLoss",
    "NLLLoss2d",
    "PReLU",
    "PairwiseDistance",
    "ParameterDict",
    "ParameterList",
    "PixelShuffle",
    "PixelUnshuffle",
    "PoissonNLLLoss",
    "RMSNorm",
    "RNN",
    "RNNBase",
    "RNNCell",
    "RNNCellBase",
    "RReLU",
    "ReLU",
    "ReLU6",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "SELU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "Softmax",
    "Softmax2d",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "SyncBatchNorm",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
    "Unflatten",
    "Unfold",
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
    # parallel
    "DataParallel",
    # functions
    "factory_kwargs",
]

from torch.nn.parameter import (
    Parameter as Parameter,
    UninitializedBuffer as UninitializedBuffer,
    UninitializedParameter as UninitializedParameter,
)
from torch.nn.modules import *  # noqa: F403
from torch.nn import (
    attention as attention,
    functional as functional,
    init as init,
    modules as modules,
    parallel as parallel,
    parameter as parameter,
    utils as utils,
)
from torch.nn.parallel import DataParallel as DataParallel


def factory_kwargs(kwargs):
    r"""Return a canonicalized dict of factory kwargs.

    Given kwargs, returns a canonicalized dict of factory kwargs that can be directly passed
    to factory functions like torch.empty, or errors if unrecognized kwargs are present.

    This function makes it simple to write code like this::

        class MyModule(nn.Module):
            def __init__(self, **kwargs):
                factory_kwargs = torch.nn.factory_kwargs(kwargs)
                self.weight = Parameter(torch.empty(10, **factory_kwargs))

    Why should you use this function instead of just passing `kwargs` along directly?

    1. This function does error validation, so if there are unexpected kwargs we will
    immediately report an error, instead of deferring it to the factory call
    2. This function supports a special `factory_kwargs` argument, which can be used to
    explicitly specify a kwarg to be used for factory functions, in the event one of the
    factory kwargs conflicts with an already existing argument in the signature (e.g.
    in the signature ``def f(dtype, **kwargs)``, you can specify ``dtype`` for factory
    functions, as distinct from the dtype argument, by saying
    ``f(dtype1, factory_kwargs={"dtype": dtype2})``)
    """
    if kwargs is None:
        return {}
    simple_keys = {"device", "dtype", "memory_format"}
    expected_keys = simple_keys | {"factory_kwargs"}
    if not kwargs.keys() <= expected_keys:
        raise TypeError(f"unexpected kwargs {kwargs.keys() - expected_keys}")

    # guarantee no input kwargs is untouched
    r = dict(kwargs.get("factory_kwargs", {}))
    for k in simple_keys:
        if k in kwargs:
            if k in r:
                raise TypeError(
                    f"{k} specified twice, in **kwargs and in factory_kwargs"
                )
            r[k] = kwargs[k]

    return r
