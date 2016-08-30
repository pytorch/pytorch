from .backend import FunctionBackend

class THNNFunctionBackend(FunctionBackend):
    pass


def _initialize_backend():
    from ..functions.thnn import _generated_functions
    from ..functions.linear import LinearFunction

    backend.register_function('Linear', LinearFunction)
    name_remap = {
        'SpatialAveragePoolingFunction': 'AvgPool2dFunction',
        'SpatialConvolutionMMFunction': 'Conv2dFunction',
        'SpatialMaxPoolingFunction': 'MaxPool2dFunction',
        'SoftMaxFunction': 'SoftmaxFunction',
        'LogSoftMaxFunction': 'LogSoftmaxFunction',
        'BatchNormalizationFunction': 'BatchNormFunction',
    }
    for cls in _generated_functions:
        name = cls.__name__
        new_name = name_remap.get(name, name)
        backend.register_function(new_name.replace('Function', ''), cls)


backend = THNNFunctionBackend()
_initialize_backend()
