from abc import ABC
import torch
from typing import List, Union

from ._internals import is_valid_device

Device = Union[torch.device, int, str]

class PlacementSpec(ABC):
    """
    Base class representing the placement of an entity. Subclasses of this
    class can be used to specify customized placements which might not be
    covered by existing APIs.
    """
    pass

class DevicePlacementSpec(PlacementSpec):
    """
    Associates placement of an entity with a single device. The device can be a
    local device or a remote device specified by one of the following remote
    formats:

        1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
        2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").

    Args:
        device(str, :class:`torch.device`): The device to place the entity on.
    """
    def __init__(self, device: Device):
        super(DevicePlacementSpec, self).__init__()
        if not is_valid_device(device):
            raise ValueError(f'{device} is not a valid device')
        self._device = device

    @property
    def device(self) -> Device:
        """
        Retrieves the device for placement.
        """
        return self._device

class ChunkShardingSpec(PlacementSpec):
    """
    This is type of PlacementSpec that defines the placement as being sharded
    across multiple devices. In particular, it represents sharding a Tensor
    along a single dimension into equal chunks (similar to :meth:`torch.chunk`).

    Args:
        dim (int or str):
            The dimension to shard on, could be an integer representing the
            dimension or a string in case of named tensors where dimensions are
            named.
        placement(List[Device] or List[PlacementSpec]):
            Specifies the placement of each shard of the Tensor. The size of
            the list represents the number of shards to be created. This
            parameter can be a list of devices
            (ex: ["rank:0/cuda:0", "rank:1/cuda:1"]) or a list of custom
            placement specs.

            The device can be a local device or a remote device specified by one
            of the following remote formats:

                1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
                2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").
    """

    ShardPlacements = List[Union[Device, PlacementSpec]]
    ShardingDim = Union[int, str]

    def __init__(self, dim: ShardingDim, placements: ShardPlacements):
        super(ChunkShardingSpec, self).__init__()
        self._verify_dim(dim)
        self._verify_devices(placements)
        self._dim = dim
        self._placements = placements

    @staticmethod
    def _verify_devices(placements):
        for dev in placements:
            if not isinstance(dev, PlacementSpec) and not is_valid_device(dev):
                raise ValueError(f'{dev} is not a valid device')

    @staticmethod
    def _verify_dim(dim):
        if not (isinstance(dim, int) or isinstance(dim, str)):
            raise ValueError(f'{dim} needs to either be an int or str')

    @property
    def dim(self) -> ShardingDim:
        """
        Retrieves the dimension to shard on.
        """
        return self._dim

    @property
    def placements(self) -> ShardPlacements:
        """
        Retrieves the shard placements.
        """
        return self._placements
