from dataclasses import dataclass, field
from enum import Enum
from typing import List

import torch
from torch.distributed._shard.sharding_spec import ShardMetadata


class MEM_FORMAT_ENCODING(Enum):
    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2

@dataclass
class TensorProperties(object):
    """ Properties used to create :class:`Tensor` """

    # Regular tensor fields
    dtype: torch.dtype = field(default=torch.get_default_dtype())
    layout: torch.layout = field(default=torch.strided)
    requires_grad: bool = False
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    pin_memory: bool = False

@dataclass
class ShardedTensorMetadata(object):
    """
    Represents metadata for :class:`ShardedTensor`
    """

    # Metadata about each shard of the Tensor
    shards_metadata: List[ShardMetadata] = field(default_factory=list)

    # Size of each dim of the overall Tensor.
    size: torch.Size = field(default=torch.Size([]))

    tensor_properties: TensorProperties = field(
        default=TensorProperties(dtype=torch.get_default_dtype(),
                                 layout=torch.strided,
                                 requires_grad=False,
                                 memory_format=torch.contiguous_format,
                                 pin_memory=False))

    def __getstate__(self):
        # Since torch.memory_format cannot be pickled!
        memory_format = self.tensor_properties.memory_format
        if memory_format == torch.contiguous_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        elif memory_format == torch.channels_last:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        elif memory_format == torch.preserve_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
        else:
            raise RuntimeError(f'Invalid torch.memory_format: {memory_format}')

        # Keep old serialization to ensure backward compatibility
        return (
            self.shards_metadata,
            self.size,
            self.tensor_properties.dtype,
            self.tensor_properties.layout,
            self.tensor_properties.requires_grad,
            mem_format_encoding,
            self.tensor_properties.pin_memory,
        )

    def __setstate__(
        self,
        state,
    ):
        (self.shards_metadata, self.size, dtype, layout, requires_grad, mem_format_encoding, pin_memory) = state

        if mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
            memory_format = torch.contiguous_format
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
            memory_format = torch.channels_last
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
            memory_format = torch.preserve_format
        else:
            raise RuntimeError(f'Invalid torch.memory_format encoding: {mem_format_encoding}')

        self.tensor_properties = TensorProperties(
            dtype=dtype, layout=layout, requires_grad=requires_grad,
            memory_format=memory_format, pin_memory=pin_memory, )
