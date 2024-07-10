# mypy: allow-untyped-defs
from typing import (
    Dict,
    Sized,
    Tuple,
)

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from enum import IntEnum

__all__ = [
    "SHARDING_PRIORITIES",
    "ShardingFilterIterDataPipe",
]


class SHARDING_PRIORITIES(IntEnum):
    DEFAULT = 1
    DISTRIBUTED = 2
    MULTIPROCESSING = 3


class _ShardingIterDataPipe(IterDataPipe):
    def apply_sharding(self, num_of_instances: int, instance_id: int, sharding_group: SHARDING_PRIORITIES):
        raise NotImplementedError


@functional_datapipe('sharding_filter')
class ShardingFilterIterDataPipe(_ShardingIterDataPipe):
    r"""
    Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``).

    After ``apply_sharding`` is called, each instance of the DataPipe (on different workers) will have every `n`-th element of the
    original DataPipe, where `n` equals to the number of instances.

    Args:
        source_datapipe: Iterable DataPipe that will be sharded
    """

    def __init__(self, source_datapipe: IterDataPipe, sharding_group_filter=None):
        self.source_datapipe = source_datapipe
        self.sharding_group_filter = sharding_group_filter
        self.groups: Dict[int, Tuple[int, int]] = {}
        self.num_of_instances = 1
        self.instance_id = 0
        self._update_num_of_instances()

    def apply_sharding(self, num_of_instances, instance_id, sharding_group=SHARDING_PRIORITIES.DEFAULT):
        if instance_id >= num_of_instances:
            raise ValueError(f"instance_id({instance_id}) should be smaller than num_of_instances({num_of_instances})")
        if sharding_group == SHARDING_PRIORITIES.DEFAULT:
            if len(self.groups) and SHARDING_PRIORITIES.DEFAULT not in self.groups:
                raise Exception('ShardingFilter cannot mix DEFAULT and non DEFAULT groups')  # noqa: TRY002
        else:
            if SHARDING_PRIORITIES.DEFAULT in self.groups:
                raise Exception('ShardingFilter cannot mix DEFAULT and non DEFAULT groups')  # noqa: TRY002
        self.groups[sharding_group] = (num_of_instances, instance_id)
        self._update_num_of_instances()

    def _update_num_of_instances(self):
        sorted_sharding_groups = []
        for key in sorted(self.groups.keys()):
            if self.sharding_group_filter is None or key == self.sharding_group_filter:
                sorted_sharding_groups.append(self.groups[key])

        sorted_sharding_groups.reverse()

        self.num_of_instances = 1
        self.instance_id = 0

        for group_num_of_instances, group_instance_id in sorted_sharding_groups:
            self.instance_id += self.num_of_instances * group_instance_id
            self.num_of_instances *= group_num_of_instances

    def __iter__(self):
        for i, item in enumerate(self.source_datapipe):
            if i % self.num_of_instances == self.instance_id:
                yield item

    def __len__(self):
        if isinstance(self.source_datapipe, Sized):
            return len(self.source_datapipe) // self.num_of_instances +\
                (1 if (self.instance_id < len(self.source_datapipe) % self.num_of_instances) else 0)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
