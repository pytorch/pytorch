import os
import io
from typing import (
    List,
    Callable,
    Optional,
    Union,
    TypeVar,
    Dict,
    Any,
    cast,
    Sequence,
)
import torch.distributed as dist
from .api import (
    CheckpointException,
    _wrap_exception,
    _is_wrapped_exception,
    WRAPPED_EXCEPTION,
)

import torch

from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._tensor import DTensor

from .metadata import (
    STATE_DICT_TYPE,
    MetadataIndex,
)

__all__ = ["find_tensor_shard", "find_state_dict_object"]

T = TypeVar("T")
R = TypeVar("R")


def _get_failure_dict(
    results: List[Union[T, WRAPPED_EXCEPTION]]
) -> Dict[int, WRAPPED_EXCEPTION]:
    return cast(
        Dict[int, WRAPPED_EXCEPTION],
        {i: err for i, err in enumerate(results) if _is_wrapped_exception(err)},
    )


class _DistWrapper:
    """
    This is a wrapper around PG that provides a series of features around object collectives.

    It works without distributed initialized, where most collectives turns into nops.

    All variants that take functions are exception robust, meaning that if one or more
    ranks raise errors, all ranks will observe those.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup],
        use_dist: bool,
        coordinator_rank: int,
    ):
        self.group = group
        self.use_dist = use_dist
        self.coordinator_rank = coordinator_rank
        if self.use_dist:
            self.rank = dist.get_rank(group)
            self.is_coordinator = self.rank == coordinator_rank
        else:
            self.rank = 0
            self.is_coordinator = True

    def get_rank(self) -> int:
        return self.rank

    def get_world_size(self) -> int:
        if self.use_dist:
            return dist.get_world_size(self.group)
        return 1

    def broadcast_object(self, object: Optional[T]) -> T:
        """
        Same as c10d::broadcast_object_list but works without distributed enabled.
        """
        object_list = [object]
        if self.use_dist:
            dist.broadcast_object_list(
                object_list=object_list,
                group=self.group,
                src=self.coordinator_rank,
            )
        return cast(T, object_list[0])

    def gather_object(self, object: T) -> Optional[List[T]]:
        """
        Same as c10d::gather_object but works without distributed enabled.
        """
        if self.use_dist:
            gather_objs = (
                cast(List[T], [None] * dist.get_world_size(self.group))
                if self.is_coordinator
                else None
            )

            dist.gather_object(
                obj=object,
                object_gather_list=gather_objs if self.is_coordinator else None,
                dst=self.coordinator_rank,
                group=self.group,
            )
            result = gather_objs
        else:
            result = [object]
        return result

    def all_gather_object(self, object: T) -> List[T]:
        """
        Same as c10d::all_gather_object but works without distributed enabled.
        """
        if self.use_dist:
            gather_objs = cast(
                List[T], [None] * dist.get_world_size(self.group)
            )

            dist.all_gather_object(
                object_list=gather_objs, obj=object, group=self.group
            )
        else:
            gather_objs = [object]
        return gather_objs

    def scatter_object(self, object_list: Optional[List[T]]) -> T:
        """
        Same as c10d::scatter_object but works without distributed enabled.
        """
        if self.use_dist:
            gather_result = cast(List[T], [None])
            dist.scatter_object_list(
                scatter_object_output_list=gather_result,
                scatter_object_input_list=object_list
                if self.is_coordinator
                else None,
                src=self.coordinator_rank,
                group=self.group,
            )

            local_reply = gather_result[0]
        else:
            assert object_list is not None
            local_reply = object_list[0]
        return local_reply

    def reduce_scatter(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[List[T]], List[R]],
    ) -> R:
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a scatter.

        This method operates in the following way:
            Run ``map_fun`` on all ranks
            Gather results on rank 0
            Call ``reduce_fun`` on all those values
            Scatter to each rank part of the result.
        """
        local_data: Union[WRAPPED_EXCEPTION, T]
        try:
            local_data = map_fun()
        except BaseException as e:
            local_data = _wrap_exception(e)

        all_data = self.gather_object(local_data)
        all_results: Optional[List[Union[R, CheckpointException]]] = None
        if self.is_coordinator:
            assert all_data is not None
            node_failures = _get_failure_dict(all_data)

            if len(node_failures) == 0:
                try:
                    # N.B. why can't mypy cast List[R] to List[Union[R, WRAPPED_EXCEPTION]]?
                    all_results = cast(
                        List[Union[R, CheckpointException]],
                        reduce_fun(cast(List[T], all_data)),
                    )
                except BaseException as e:
                    node_failures[self.rank] = _wrap_exception(e)

            if len(node_failures) > 0:
                all_results = [
                    CheckpointException(step, node_failures)
                ] * self.get_world_size()

        result = self.scatter_object(all_results)
        if isinstance(result, CheckpointException):
            raise result
        return result

    def all_reduce(
        self,
        step: str,
        map_fun: Callable[[], T],
        reduce_fun: Callable[[List[T]], R],
    ) -> R:
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a broadcast.

        This method operates in the following way:
            Run ``map_fun`` on all ranks
            Gather results on rank 0
            Call ``reduce_fun`` on all those values
            Broadcast the reduced value to all ranks.
        """
        local_data: Union[T, WRAPPED_EXCEPTION]
        try:
            local_data = map_fun()
        except BaseException as e:
            local_data = _wrap_exception(e)

        all_data = self.gather_object(local_data)
        result: Optional[Union[R, CheckpointException]] = None
        if self.is_coordinator:
            assert all_data is not None
            node_failures = _get_failure_dict(all_data)
            if len(node_failures) == 0:
                try:
                    result = reduce_fun(cast(List[T], all_data))
                except BaseException as e:
                    node_failures[self.rank] = _wrap_exception(e)

            if len(node_failures) > 0:
                result = CheckpointException(step, node_failures)

        final_result = self.broadcast_object(result)
        if isinstance(final_result, CheckpointException):
            raise final_result
        return cast(R, final_result)

    def all_gather(
        self,
        step: str,
        map_fun: Callable[[], T],
    ) -> List[T]:
        """
        Compute a value on each rank, then all_gather them.

        This method operates in the following way:
            Run ``map_cp`` on all ranks
            all_gather the values to all ranks
        """
        result: Union[T, WRAPPED_EXCEPTION]
        try:
            result = map_fun()
        except BaseException as e:
            result = _wrap_exception(e)

        all_results = self.all_gather_object(result)

        node_failures = _get_failure_dict(all_results)
        if len(node_failures) > 0:
            raise CheckpointException(step, node_failures)
        return cast(List[T], all_results)

    def broadcast(
        self,
        step: str,
        map_fun: Callable[[], T],
    ) -> T:
        """
        Compute a value on rank 0 and broadcast it.

        This method operates in the following way:
            Run ``map_cp`` on rank 0
            broadcast the value
        """
        result: Optional[Union[T, CheckpointException]] = None
        if self.is_coordinator:
            try:
                result = map_fun()
            except BaseException as e:
                result = CheckpointException(
                    step, {self.rank: _wrap_exception(e)}
                )
        final_result = self.broadcast_object(result)
        if isinstance(final_result, CheckpointException):
            raise final_result
        return cast(T, final_result)


def _find_shard(tensor: ShardedTensor, index: MetadataIndex) -> Shard:
    if index.offset is None:
        raise ValueError(
            f"Cannot lookup {index.fqn} since its a ShardedTensor and no offset was provided"
        )

    shards = tensor.local_shards()
    # index fast path
    if index.index is not None:
        if (
            len(shards) > index.index
            and torch.Size(shards[index.index].metadata.shard_offsets)
            == index.offset
        ):
            return shards[index.index]

    for shard in shards:
        if torch.Size(shard.metadata.shard_offsets) == index.offset:
            return shard
    raise ValueError(
        f"Could not find shard at '{index.offset}' for FQN: '{index.fqn}'"
    )


def find_tensor_shard(
    tensor: torch.Tensor, index: MetadataIndex
) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    if isinstance(tensor, ShardedTensor):
        return _find_shard(tensor, index).tensor
    if index.offset is not None:
        # special case looking up a tensor by origin
        if index.offset == torch.Size([0] * len(tensor.size())):
            return tensor
        raise ValueError(
            f"FQN: '{index.fqn}' is not a ShardedTensor, can't find by offset: '{index.offset}'"
        )
    return tensor


def find_state_dict_object(
    state_dict: STATE_DICT_TYPE, index: MetadataIndex
) -> Any:
    if index.fqn not in state_dict:
        raise ValueError(f"Could not find FQN: '{index.fqn}'")
    obj = state_dict[index.fqn]

    if isinstance(obj, torch.Tensor):
        return find_tensor_shard(obj, index)
    elif index.offset is not None:
        raise ValueError(
            f"FQN: '{index.fqn}' is not a ShardedTensor, can't find by offset: '{index.offset}'"
        )
    return obj


def _element_wise_add(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [i_a + i_b for i_a, i_b in zip(a, b)]


def _element_wise_sub(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [i_a - i_b for i_a, i_b in zip(a, b)]

class _ReaderView(io.IOBase):
    def __init__(self, base_stream: io.IOBase, offset: int, len: int):
        super().__init__()
        self.offset = offset
        self.len = len
        self.base_stream = base_stream
        self.seek(0)

    def seek(self, __offset: int, __whence: int = os.SEEK_SET) -> int:
        if __whence == os.SEEK_SET:
            __offset = self.offset + __offset
        elif __whence == os.SEEK_END:
            __whence = os.SEEK_SET
            __offset = (self.offset + self.len) - __offset
        return self.base_stream.seek(__offset, __whence)

    def tell(self) -> int:
        return self.base_stream.tell() - self.offset

    def readable(self) -> bool:
        return self.base_stream.readable()

    def seekable(self) -> bool:
        return self.base_stream.seekable()

    def readinto(self, b):
        return self.base_stream.readinto(b)  # type: ignore[attr-defined]

    def read(self, size=-1):
        return self.base_stream.read(size)

def _create_file_view(file: io.IOBase, offset: int, length: int) -> io.IOBase:
    # FIXME (kumpera) torch.load fails if we wrap with io.BufferedReader
    return _ReaderView(file, offset, length)
