import inspect
import torch
from functools import wraps
from itertools import chain
from torch.testing import floating_types
from torch.testing._internal.common_device_type import (
    _TestParametrizer, _dtype_test_suffix, _update_param_kwargs, skipIf)
from torch.testing._internal.common_methods_invocations import SkipInfo
from torch.testing._internal.common_nn import nllloss_reference
from torch.testing._internal.common_utils import make_tensor
from unittest.mock import MagicMock


# List of all namespaces containing modules to test.
MODULE_NAMESPACES = [
    torch.nn.modules,
    torch.nn.qat.modules,
    torch.nn.quantizable.modules,
    torch.nn.quantized.modules,
]

# Modules that shouldn't be tested for one reason or another.
MODULES_TO_SKIP = {
    torch.nn.Module,  # abstract base class
    torch.nn.Container,  # deprecated
    torch.nn.NLLLoss2d,  # deprecated
    torch.nn.quantized.modules._ConvNd,  # abstract base class
    torch.nn.quantized.MaxPool2d,  # aliases to nn.MaxPool2d
}

# List of all module classes to test.
MODULE_CLASSES = list(chain(*[[getattr(namespace, module_name) for module_name in namespace.__all__]
                              for namespace in MODULE_NAMESPACES]))
MODULE_CLASSES = [cls for cls in MODULE_CLASSES if cls not in MODULES_TO_SKIP]

# Dict of module class -> common name. Useful for making test names more intuitive.
# Example: torch.nn.modules.linear.Linear -> "nn.Linear"
MODULE_CLASS_NAMES = {}
for namespace in MODULE_NAMESPACES:
    for module_name in namespace.__all__:
        module_cls = getattr(namespace, module_name)
        namespace_name = namespace.__name__.replace('torch.', '').replace('.modules', '')
        MODULE_CLASS_NAMES[module_cls] = f'{namespace_name}.{module_name}'


class modules(_TestParametrizer):
    def __init__(self, module_info_list):
        self.module_info_list = module_info_list

    def _parametrize_test(self, test, generic_cls, device_cls):
        for module_info in self.module_info_list:
            # TODO: Factor a lot of this out since it's similar to OpInfo.
            for dtype in floating_types():
                # Construct the test name.
                test_name = '{}_{}_{}{}'.format(test.__name__,
                                                module_info.name.replace('.', '_'),
                                                device_cls.device_type,
                                                _dtype_test_suffix(dtype))

                # Construct parameter kwargs to pass to the test.
                param_kwargs = {'module_info': module_info}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)

                try:
                    active_decorators = []
                    if module_info.should_skip(generic_cls.__name__, test.__name__, device_cls.device_type, dtype):
                        active_decorators.append(skipIf(True, "Skipped!"))

                    if module_info.decorators is not None:
                        for decorator in module_info.decorators:
                            # Can't use isinstance as it would cause a circular import
                            if decorator.__class__.__name__ == 'DecorateInfo':
                                if decorator.is_active(generic_cls.__name__, test.__name__,
                                                       device_cls.device_type, dtype):
                                    active_decorators += decorator.decorators
                            else:
                                active_decorators.append(decorator)

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    for decorator in active_decorators:
                        test_wrapper = decorator(test_wrapper)

                    yield (test_wrapper, test_name, param_kwargs)
                except Exception as ex:
                    # Provides an error message for debugging before rethrowing the exception
                    print("Failed to instantiate {0} for module {1}!".format(test_name, module_info.name))
                    raise ex


def formatted_module_name(module_cls):
    """ Returns the common name of the module class formatted for use in test names. """
    return MODULE_CLASS_NAMES[module_cls].replace('.', '_')


class FunctionInput(object):
    """ Contains args and kwargs to pass as input to a function. """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class ModuleInput(object):
    __slots__ = ['constructor_input', 'forward_input', 'reference_fn', 'desc']

    def __init__(self, constructor_input, forward_input=None, reference_fn=None, desc=''):
        self.constructor_input = constructor_input
        self.forward_input = forward_input
        self.reference_fn = reference_fn
        self.desc = desc


class ModuleInfo(object):
    """Module information to be used in testing and helper functions for acquiring it."""

    def __init__(self,
                 module_cls,  # Class for the module under test
                 *,
                 module_inputs_func,  # Function to generate module inputs
                 needs_factory_kwargs=False,  # Indicates if factory_kwargs should be passed explicitly to avoid
                                              # conflicts with e.g. a dtype arg (usually for quantized modules)
                 has_inplace_variant=False,  # Indicates if the module can be applied in-place with inplace=True
                 is_pickleable=True,  # Indicates if the module can be pickled
                 has_sparse_gradients = False,  # Indicates whether the module has sparse gradients (e.g. Embedding)
                 skips=[],  # Indicates which tests to skip
                 decorators=None,  # Additional decorators to apply to generated tests
                 ):
        self.module_cls = module_cls
        self.module_inputs_func = module_inputs_func
        self.needs_factory_kwargs = needs_factory_kwargs
        self.has_inplace_variant = has_inplace_variant
        self.is_pickleable = is_pickleable
        self.has_sparse_gradients = has_sparse_gradients
        self.skips = skips
        self.decorators = decorators

    def should_skip(self, cls_name, test_name, device_type, dtype):
        return any(si.is_active(cls_name, test_name, device_type, dtype) for si in self.skips)

    @property
    def name(self):
        return formatted_module_name(self.module_cls)


def mock_wrapper(method):
    """
    Returns a function that calls the real implementation of a method
    in addition to passing args to a mock object.
    """
    mock = MagicMock()

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method(self, *args, **kwargs)
    wrapper.mock = mock
    return wrapper


def module_inputs_torch_nn_BatchNorm1d(module_info, device, dtype, requires_grad, **kwargs):
    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(10),
                    forward_input=FunctionInput(make_tensor((4, 10), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
        ModuleInput(constructor_input=FunctionInput(5),
                    forward_input=FunctionInput(make_tensor((4, 5, 3), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
        ModuleInput(constructor_input=FunctionInput(10, eps=1e-3, momentum=None),
                    forward_input=FunctionInput(make_tensor((4, 10), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
        ModuleInput(constructor_input=FunctionInput(10, eps=1e-3, momentum=0.3, affine=False),
                    forward_input=FunctionInput(make_tensor((4, 10), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
        ModuleInput(constructor_input=FunctionInput(10, eps=1e-3, momentum=0.3, affine=True,
                                                    track_running_stats=False),
                    forward_input=FunctionInput(make_tensor((4, 10), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
        ModuleInput(constructor_input=FunctionInput(5, eps=1e-3, momentum=0.3, affine=False),
                    forward_input=FunctionInput(make_tensor((4, 5, 3), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
        ModuleInput(constructor_input=FunctionInput(5, eps=1e-3, momentum=0.3, affine=False),
                    forward_input=FunctionInput(make_tensor((0, 5, 9), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
    ]

    return module_inputs


def module_inputs_torch_nn_Identity(module_info, device, dtype, requires_grad, **kwargs):
    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_tensor((4, 10), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
    ]

    return module_inputs


def module_inputs_torch_nn_Linear(module_info, device, dtype, requires_grad, **kwargs):
    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(10, 8),
                    forward_input=FunctionInput(make_tensor((4, 10), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
        ModuleInput(constructor_input=FunctionInput(10, 8, bias=False),
                    forward_input=FunctionInput(make_tensor((4, 10), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
    ]

    return module_inputs


def module_inputs_torch_nn_NLLLoss(module_info, device, dtype, requires_grad, **kwargs):
    constructor_kwargs = [
        {},
        {'weight': torch.randn(10, device=device, dtype=dtype)},
        {'weight': torch.randn(10, device=device, dtype=dtype), 'ignore_index': 2},
        {'weight': torch.randn(10, device=device, dtype=dtype), 'ignore_index': -1}
    ]
    module_inputs = []
    for kwargs in constructor_kwargs:

        def reference_fn(i, t, kwargs=kwargs):
            return nllloss_reference(i, t, **kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**kwargs),
                        forward_input=FunctionInput(torch.rand(15, 10, device=device, dtype=dtype).log_softmax(dim=1),
                                                    torch.empty(15, device=device).uniform_().mul(10).floor().long()),
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_ReLU(module_info, device, dtype, requires_grad, **kwargs):
    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(
                        torch.randn(5, 3, device=device, dtype=dtype, requires_grad=requires_grad))),
    ]

    return module_inputs


def module_inputs_torch_nn_TransformerDecoder(module_info, device, dtype, requires_grad, **kwargs):
    decoder_layer = torch.nn.TransformerDecoderLayer(d_model=10, nhead=2, device=device, dtype=dtype)
    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(decoder_layer, 3),
                    forward_input=None),
    ]

    return module_inputs

def module_inputs_torch_nn_TransformerEncoder(module_info, device, dtype, requires_grad, **kwargs):
    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=10, nhead=2, device=device, dtype=dtype)
    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(encoder_layer, 3),
                    forward_input=None),
    ]

    return module_inputs


def module_inputs_torch_nn_quantized_Linear(module_info, device, dtype, requires_grad, **kwargs):
    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(5, 2),
                    forward_input=FunctionInput(make_tensor((3, 3, 3, 3), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
    ]

    return module_inputs


def module_inputs_torch_nn_qat_Conv2d(module_info, device, dtype, requires_grad, **kwargs):
    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(3, 3, 3, qconfig=torch.quantization.default_qconfig),
                    forward_input=FunctionInput(make_tensor((3, 3, 3, 3), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
    ]

    return module_inputs


def module_inputs_torch_nn_quantized_Quantize(module_info, device, dtype, requires_grad, **kwargs):
    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(0.1, 0, dtype=torch.qint8),
                    forward_input=FunctionInput(make_tensor((4, 10), device=device, dtype=dtype,
                                                            requires_grad=requires_grad))),
    ]

    return module_inputs


module_db = [
    ModuleInfo(torch.nn.BatchNorm1d,
               module_inputs_func=module_inputs_torch_nn_BatchNorm1d),
    ModuleInfo(torch.nn.Identity,
               module_inputs_func=module_inputs_torch_nn_Identity),
    ModuleInfo(torch.nn.Linear,
               module_inputs_func=module_inputs_torch_nn_Linear),
    ModuleInfo(torch.nn.NLLLoss,
               module_inputs_func=module_inputs_torch_nn_NLLLoss,
               skips=[
                   # doesn't need to support device / dtype kwargs
                   SkipInfo('TestModule', 'test_factory_kwargs')
               ]),
    ModuleInfo(torch.nn.ReLU,
               module_inputs_func=module_inputs_torch_nn_ReLU,
               has_inplace_variant=True),
    ModuleInfo(torch.nn.TransformerDecoder,
               module_inputs_func=module_inputs_torch_nn_TransformerDecoder,
               skips=[
                   # doesn't need to support device / dtype kwargs
                   SkipInfo('TestModule', 'test_factory_kwargs')
               ]),
    ModuleInfo(torch.nn.TransformerEncoder,
               module_inputs_func=module_inputs_torch_nn_TransformerEncoder,
               skips=[
                   # doesn't need to support device / dtype kwargs
                   SkipInfo('TestModule', 'test_factory_kwargs')
               ]),
    ModuleInfo(torch.nn.qat.Conv2d,
               module_inputs_func=module_inputs_torch_nn_qat_Conv2d),
    ModuleInfo(torch.nn.quantized.Linear,
               module_inputs_func=module_inputs_torch_nn_quantized_Linear),
    ModuleInfo(torch.nn.quantized.Quantize,
               module_inputs_func=module_inputs_torch_nn_quantized_Quantize,
               needs_factory_kwargs=True),
]


# TODO: Remove these as ModuleInfo entries are added.
# This is used as a stop-gap so we still get full coverage for certain tests
# that depend on instantiating every module. Skeleton ModuleInfo entries are
# generated from this list for the time being.
def build_constructor_arg_db():
    return {
        torch.nn.AdaptiveAvgPool1d: ((5,), {}),
        torch.nn.AdaptiveAvgPool2d: ((5,), {}),
        torch.nn.AdaptiveAvgPool3d: ((5,), {}),
        torch.nn.AdaptiveLogSoftmaxWithLoss: ((100, 20, [5, 10, 15]), {}),
        torch.nn.AdaptiveMaxPool1d: ((5,), {}),
        torch.nn.AdaptiveMaxPool2d: ((5,), {}),
        torch.nn.AdaptiveMaxPool3d: ((5,), {}),
        torch.nn.AlphaDropout: ((), {}),
        torch.nn.AvgPool1d: ((3,), {}),
        torch.nn.AvgPool2d: ((3,), {}),
        torch.nn.AvgPool3d: ((3,), {}),
        torch.nn.BCELoss: ((), {}),
        torch.nn.BCEWithLogitsLoss: ((), {}),
        #torch.nn.BatchNorm1d: ((5,), {}),
        torch.nn.BatchNorm2d: ((5,), {}),
        torch.nn.BatchNorm3d: ((5,), {}),
        torch.nn.Bilinear: ((2, 3, 4), {}),
        torch.nn.CELU: ((), {}),
        torch.nn.CTCLoss: ((), {}),
        torch.nn.ChannelShuffle: ((4,), {}),
        torch.nn.ConstantPad1d: ((2, 3.5), {}),
        torch.nn.ConstantPad2d: ((2, 3.5), {}),
        torch.nn.ConstantPad3d: ((2, 3.5), {}),
        torch.nn.Conv1d: ((3, 3, 3), {}),
        #torch.nn.Conv2d: ((3, 3, 3), {}),
        torch.nn.Conv3d: ((3, 3, 3), {}),
        torch.nn.ConvTranspose1d: ((3, 3, 3), {}),
        torch.nn.ConvTranspose2d: ((3, 3, 3), {}),
        torch.nn.ConvTranspose3d: ((3, 3, 3), {}),
        torch.nn.CosineEmbeddingLoss: ((), {}),
        torch.nn.CosineSimilarity: ((), {}),
        torch.nn.CrossEntropyLoss: ((), {}),
        torch.nn.CrossMapLRN2d: ((5,), {}),
        torch.nn.Dropout2d: ((), {}),
        torch.nn.Dropout3d: ((), {}),
        torch.nn.Dropout: ((), {}),
        torch.nn.ELU: ((), {}),
        torch.nn.Embedding: ((10, 5), {}),
        torch.nn.EmbeddingBag: ((10, 5), {}),
        torch.nn.FeatureAlphaDropout: ((), {}),
        torch.nn.Flatten: ((), {}),
        torch.nn.Fold: ((5, 2), {}),
        torch.nn.FractionalMaxPool2d: ((5, 2), {}),
        torch.nn.FractionalMaxPool3d: ((5, 2), {}),
        torch.nn.GELU: ((), {}),
        torch.nn.GLU: ((), {}),
        torch.nn.GRU: ((5, 10), {}),
        torch.nn.GRUCell: ((5, 10), {}),
        torch.nn.GaussianNLLLoss: ((), {}),
        torch.nn.GroupNorm: ((3, 6, 1e-5, True), {}),
        torch.nn.Hardshrink: ((), {}),
        torch.nn.Hardsigmoid: ((), {}),
        torch.nn.Hardswish: ((), {}),
        torch.nn.Hardtanh: ((), {}),
        torch.nn.HingeEmbeddingLoss: ((), {}),
        torch.nn.HuberLoss: ((), {}),
        #torch.nn.Identity: ((), {}),
        torch.nn.InstanceNorm1d: ((5, 1e-5, 0.1, True), {}),
        torch.nn.InstanceNorm2d: ((5, 1e-5, 0.1, True), {}),
        torch.nn.InstanceNorm3d: ((5, 1e-5, 0.1, True), {}),
        torch.nn.KLDivLoss: ((), {}),
        torch.nn.L1Loss: ((), {}),
        torch.nn.LPPool1d: ((2, 3), {}),
        torch.nn.LPPool2d: ((2, 3), {}),
        torch.nn.LSTM: ((5, 10), {}),
        torch.nn.LSTMCell: ((5, 10), {}),
        torch.nn.LayerNorm: ((2,), {}),
        torch.nn.LazyBatchNorm1d: ((), {}),
        torch.nn.LazyBatchNorm2d: ((), {}),
        torch.nn.LazyBatchNorm3d: ((), {}),
        torch.nn.LazyConv1d: ((5, 2), {}),
        torch.nn.LazyConv2d: ((5, 2), {}),
        torch.nn.LazyConv3d: ((5, 2), {}),
        torch.nn.LazyConvTranspose1d: ((5, 2), {}),
        torch.nn.LazyConvTranspose2d: ((5, 2), {}),
        torch.nn.LazyConvTranspose3d: ((5, 2), {}),
        torch.nn.LazyLinear: ((5,), {}),
        torch.nn.LeakyReLU: ((), {}),
        #torch.nn.Linear: ((10, 5), {}),
        torch.nn.LocalResponseNorm: ((2,), {}),
        torch.nn.LogSigmoid: ((), {}),
        torch.nn.LogSoftmax: ((), {}),
        torch.nn.MSELoss: ((), {}),
        torch.nn.MarginRankingLoss: ((), {}),
        torch.nn.MaxPool1d: ((3,), {}),
        torch.nn.MaxPool2d: ((3,), {}),
        torch.nn.MaxPool3d: ((3,), {}),
        torch.nn.MaxUnpool1d: ((5,), {}),
        torch.nn.MaxUnpool2d: ((5,), {}),
        torch.nn.MaxUnpool3d: ((5,), {}),
        torch.nn.ModuleDict: ((), {}),
        torch.nn.ModuleList: ((), {}),
        torch.nn.MultiLabelMarginLoss: ((), {}),
        torch.nn.MultiLabelSoftMarginLoss: ((), {}),
        torch.nn.MultiMarginLoss: ((), {}),
        torch.nn.MultiheadAttention: ((100, 2), {}),
        #torch.nn.NLLLoss2d: ((), {}),
        #torch.nn.NLLLoss: ((), {}),
        torch.nn.PReLU: ((), {}),
        torch.nn.PairwiseDistance: ((), {}),
        torch.nn.ParameterDict: ((), {}),
        torch.nn.ParameterList: ((), {}),
        torch.nn.PixelShuffle: ((2,), {}),
        torch.nn.PixelUnshuffle: ((2,), {}),
        torch.nn.PoissonNLLLoss: ((), {}),
        torch.nn.RNN: ((5, 10), {}),
        torch.nn.RNNBase: (('LSTM', 5, 10), {}),
        torch.nn.RNNCell: ((5, 10), {}),
        torch.nn.RNNCellBase: ((5, 10, True, 2), {}),
        torch.nn.RReLU: ((), {}),
        torch.nn.ReLU6: ((), {}),
        #torch.nn.ReLU: ((), {}),
        torch.nn.ReflectionPad1d: ((2,), {}),
        torch.nn.ReflectionPad2d: ((2,), {}),
        torch.nn.ReplicationPad1d: ((2,), {}),
        torch.nn.ReplicationPad2d: ((2,), {}),
        torch.nn.ReplicationPad3d: ((2,), {}),
        torch.nn.SELU: ((), {}),
        torch.nn.Sequential: ((), {}),
        torch.nn.SiLU: ((), {}),
        torch.nn.Sigmoid: ((), {}),
        torch.nn.SmoothL1Loss: ((), {}),
        torch.nn.SoftMarginLoss: ((), {}),
        torch.nn.Softmax2d: ((), {}),
        torch.nn.Softmax: ((), {}),
        torch.nn.Softmin: ((), {}),
        torch.nn.Softplus: ((), {}),
        torch.nn.Softshrink: ((), {}),
        torch.nn.Softsign: ((), {}),
        torch.nn.SyncBatchNorm: ((5,), {}),
        torch.nn.Tanh: ((), {}),
        torch.nn.Tanhshrink: ((), {}),
        torch.nn.Threshold: ((0.1, 20), {}),
        torch.nn.Transformer: ((), {}),
        torch.nn.TransformerDecoderLayer: ((10, 2), {}),
        torch.nn.TransformerEncoderLayer: ((10, 2), {}),
        torch.nn.TripletMarginLoss: ((), {}),
        torch.nn.TripletMarginWithDistanceLoss: ((), {}),
        torch.nn.Unflatten: ((1, (2, 5, 5)), {}),
        torch.nn.Unfold: ((3,), {}),
        torch.nn.Upsample: ((), {}),
        torch.nn.UpsamplingBilinear2d: ((), {}),
        torch.nn.UpsamplingNearest2d: ((), {}),
        torch.nn.ZeroPad2d: ((0,), {}),
        #torch.nn.qat.Conv2d: ((3, 3, 3), {
        #    'qconfig': torch.quantization.default_qconfig,
        #}),
        torch.nn.qat.Conv3d: ((3, 3, 3), {
            'qconfig': torch.quantization.default_qconfig,
        }),
        torch.nn.qat.Linear: ((5, 2), {
            'qconfig': torch.quantization.default_qconfig,
        }),
        torch.nn.quantizable.LSTM: ((5, 6), {}),
        torch.nn.quantizable.LSTMCell: ((5, 6), {}),
        torch.nn.quantizable.MultiheadAttention: ((10, 2), {}),
        torch.nn.quantized.BatchNorm2d: ((2,), {}),
        torch.nn.quantized.BatchNorm3d: ((2,), {}),
        torch.nn.quantized.Conv1d: ((3, 3, 3), {}),
        torch.nn.quantized.Conv2d: ((3, 3, 3), {}),
        torch.nn.quantized.Conv3d: ((3, 3, 3), {}),
        torch.nn.quantized.ConvTranspose1d: ((3, 3, 3), {}),
        torch.nn.quantized.ConvTranspose2d: ((3, 3, 3), {}),
        torch.nn.quantized.ConvTranspose3d: ((16, 33, (3, 3, 5)), {
            'stride': (2, 1, 1),
            'padding': (4, 2, 2),
            'output_padding': (2, 2, 2),
            'dilation': (1, 1, 1),
        }),
        torch.nn.quantized.DeQuantize: ((), {}),
        torch.nn.quantized.ELU: ((0.01, 0), {}),
        torch.nn.quantized.Embedding: ((10, 3), {}),
        torch.nn.quantized.EmbeddingBag: ((10, 3), {}),
        torch.nn.quantized.GroupNorm: ((2, 3, torch.nn.Parameter(torch.tensor(2.)),
                                        torch.nn.Parameter(torch.tensor(2.)), 0.1, 0), {}),
        torch.nn.quantized.Hardswish: ((0.1, 0,), {}),
        torch.nn.quantized.InstanceNorm1d: ((2, torch.nn.Parameter(torch.tensor(2.)),
                                             torch.nn.Parameter(torch.tensor(2.)), 0.1, 0), {}),
        torch.nn.quantized.InstanceNorm2d: ((2, torch.nn.Parameter(torch.tensor(2.)),
                                             torch.nn.Parameter(torch.tensor(2.)), 0.1, 0), {}),
        torch.nn.quantized.InstanceNorm3d: ((2, torch.nn.Parameter(torch.tensor(2.)),
                                             torch.nn.Parameter(torch.tensor(2.)), 0.1, 0), {}),
        torch.nn.quantized.LayerNorm: ((2, torch.nn.Parameter(torch.tensor(2.)),
                                        torch.nn.Parameter(torch.tensor(2.)), 0.1, 0), {}),
        torch.nn.quantized.LeakyReLU: ((0.01, 0), {}),
        #torch.nn.quantized.Linear: ((5, 2), {}),
        torch.nn.quantized.MaxPool2d: ((3,), {}),
        #torch.nn.quantized.Quantize: ((0.1, 0), {
        #    'dtype': torch.int16,
        #    'factory_kwargs': {},
        #}),
        torch.nn.quantized.ReLU6: ((), {}),
        torch.nn.quantized.Sigmoid: ((0.1, 0), {}),
        torch.nn.quantized.FloatFunctional: ((), {}),
        torch.nn.quantized.FXFloatFunctional: ((), {}),
        torch.nn.quantized.QFunctional: ((), {}),
    }


# Temporary set of modules that don't support device / dtype kwargs
# for generated ModuleInfo entries. This should be removed once
# the full ModuleInfo entries have been created for these modules.
TEMP_MODULES_WITHOUT_DEVICE_DTYPE_KWARGS = {
    torch.nn.BCELoss,
    torch.nn.BCEWithLogitsLoss,
    torch.nn.CrossEntropyLoss,
    torch.nn.FractionalMaxPool2d,
    torch.nn.FractionalMaxPool3d,
    torch.nn.MultiLabelSoftMarginLoss,
    torch.nn.MultiMarginLoss,
    torch.nn.NLLLoss,
    torch.nn.NLLLoss2d,
    torch.nn.TransformerDecoder,
    torch.nn.TransformerEncoder,
}


for module_cls, (mod_args, mod_kwargs) in build_constructor_arg_db().items():

    def module_inputs_func(module_info, device, dtype, requires_grad, mod_args=mod_args, mod_kwargs=mod_kwargs,
                           **kwargs):
        mod_args = [torch.nn.Parameter(torch.tensor(arg.clone().detach(), device=device, dtype=dtype))
                    if isinstance(arg, torch.nn.Parameter) else arg for arg in mod_args]
        mod_kwargs = {
            k: torch.nn.Parameter(torch.tensor(v.clone().detach(), device=device, dtype=dtype))
            if isinstance(v, torch.nn.Parameter) else v for k, v in mod_kwargs.items()
        }

        return [
            ModuleInput(constructor_input=FunctionInput(*mod_args, **mod_kwargs))
        ]

    needs_factory_kwargs = 'factory_kwargs' in mod_kwargs
    skip_factory_kwargs_test = module_cls in TEMP_MODULES_WITHOUT_DEVICE_DTYPE_KWARGS
    skips = [SkipInfo('TestModule', 'test_factory_kwargs')] if skip_factory_kwargs_test else []
    module_db.append(ModuleInfo(module_cls, module_inputs_func=module_inputs_func,
                                needs_factory_kwargs=needs_factory_kwargs,
                                skips=skips))
