from collections import namedtuple
import torch

TorchNNModuleMetadata = namedtuple(
    'TorchNNModuleMetadata',
    [
        'cpp_sources',
    ]
)
TorchNNModuleMetadata.__new__.__defaults__ = ('',)

module_metadata_map = {}
