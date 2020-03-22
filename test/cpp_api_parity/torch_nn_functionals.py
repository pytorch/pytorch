from collections import namedtuple
import torch

TorchNNFunctionalMetadata = namedtuple(
    'TorchNNFunctionalMetadata',
    [
        'cpp_sources',
    ]
)
TorchNNFunctionalMetadata.__new__.__defaults__ = ('',)

functional_metadata_map = {}
