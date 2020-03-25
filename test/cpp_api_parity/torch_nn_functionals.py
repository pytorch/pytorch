from collections import namedtuple

TorchNNFunctionalMetadata = namedtuple(
    'TorchNNFunctionalMetadata',
    [
        'cpp_sources',
    ]
)
TorchNNFunctionalMetadata.__new__.__defaults__ = ('',)

functional_cpp_sources = {}
