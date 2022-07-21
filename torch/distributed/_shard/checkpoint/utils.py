from typing import Any, List, Union, Callable, Tuple, Optional
import torch.distributed as dist
from .api import CheckpointException
import torch

def tensor_narrow_n(tensor: torch.Tensor, offsets: Tuple[int, ...], lengths: Tuple[int, ...]) -> torch.Tensor:
    for dim, (start, length) in enumerate(zip(offsets, lengths)):
        tensor = torch.narrow(tensor, dim, start, length)
    return tensor

class _DistWrapper:
    """
    This is a wrapper around PG that provides a series of features around object collectives.

    It works without distributed initialized, where most collectives turns into nops.

    All variants that take functions are exception robust, meaning that if one or more
    ranks raise errors, all ranks will observe those.
    """
    def __init__(self, group: Optional[dist.ProcessGroup], use_dist: bool, coordinator_rank: int):
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

    def broadcast_object(self, object: Any) -> Any:
        """
        Same as c10d::broadcast_object_list but works without distributed enabled.
        """
        object_list = [object]
        if self.use_dist:
            dist.broadcast_object_list(
                object_list=object_list,
                group=self.group,
                src=self.coordinator_rank)
        return object_list[0]

    def gather_object(self, object: Any) -> Union[List[Any], None]:
        """
        Same as c10d::gather_object but works without distributed enabled.
        """
        if self.use_dist:
            gather_objs = [None] * dist.get_world_size(self.group) if self.is_coordinator else None

            dist.gather_object(
                obj=object,
                object_gather_list=gather_objs if self.is_coordinator else None,
                dst=self.coordinator_rank,
                group=self.group
            )
            result = gather_objs
        else:
            result = [object]
        return result

    def all_gather_object(self, object: Any) -> List[Any]:
        """
        Same as c10d::all_gather_object but works without distributed enabled.
        """
        if self.use_dist:
            gather_objs = [None] * dist.get_world_size(self.group)

            dist.all_gather_object(
                object_list=gather_objs,
                obj=object,
                group=self.group
            )
        else:
            gather_objs = [object]
        return gather_objs

    def scatter_object(self, object_list: Optional[List[Any]]) -> Any:
        """
        Same as c10d::scatter_object but works without distributed enabled.
        """
        if self.use_dist:
            gather_result = [None]
            dist.scatter_object_list(
                scatter_object_output_list=gather_result,
                scatter_object_input_list=object_list if self.is_coordinator else None,
                src=self.coordinator_rank,
                group=self.group
            )

            local_reply = gather_result[0]
        else:
            assert object_list is not None
            local_reply = object_list[0]
        return local_reply

    def reduce_scatter(
        self,
        step: str,
        map_fun: Callable[[], Any],
        reduce_fun: Callable[[List[Any]], List[Any]]
    ) -> Any:
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a scatter.

        This method operates in the following way:
            Run ``map_cp`` on all ranks
            Gather results on rank 0
            Call ``reduce_cb`` on all those values
            Scatter to each rank part of the result.
        """
        try:
            local_data = map_fun()
        except BaseException as e:
            local_data = e

        all_data = self.gather_object(local_data)
        all_results = None
        if self.is_coordinator:
            assert all_data is not None
            node_failures = {i: err for i, err in enumerate(all_data) if isinstance(err, BaseException)}

            if len(node_failures) == 0:
                try:
                    all_results = reduce_fun(all_data)
                except BaseException as e:
                    node_failures[self.rank] = e

            if len(node_failures) > 0:
                all_results = [CheckpointException(step, node_failures)] * self.get_world_size()

        result = self.scatter_object(all_results)
        if isinstance(result, CheckpointException):
            raise result
        return result

    def all_reduce(
        self,
        step: str,
        map_cb: Callable[[], Any],
        reduce_cb: Callable[[List[Any]], Any]
    ) -> Tuple[Any, Any]:
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a broadcast.

        This method operates in the following way:
            Run ``map_cp`` on all ranks
            Gather results on rank 0
            Call ``reduce_cb`` on all those values
            Broadcast the reduced value to all ranks
        """
        try:
            local_data = map_cb()
        except BaseException as e:
            local_data = e

        all_data = self.gather_object(local_data)
        result = None
        if self.is_coordinator:
            assert all_data is not None
            node_failures = {i: err for i, err in enumerate(all_data) if isinstance(err, BaseException)}

            if len(node_failures) == 0:
                try:
                    result = reduce_cb(all_data)
                except BaseException as e:
                    node_failures[self.rank] = e

            if len(node_failures) > 0:
                result = CheckpointException(step, node_failures)

        result = self.broadcast_object(result)
        if isinstance(result, CheckpointException):
            raise result
        return result

    def all_gather(
        self,
        step: str,
        map_fun: Callable[[], Any],
    ) -> List[Any]:
        """
        Compute a value on each rank, then all_gather them.

        This method operates in the following way:
            Run ``map_cp`` on all ranks
            all_gather the values to all ranks
        """
        try:
            result = map_fun()
        except BaseException as e:
            result = e

        all_results = self.all_gather_object(result)

        node_failures = {i: err for i, err in enumerate(all_results) if isinstance(err, BaseException)}
        if len(node_failures) > 0:
            raise CheckpointException(step, node_failures)
        return all_results

    def broadcast(
        self,
        step: str,
        map_cb: Callable[[], Any],
    ) -> Any:
        """
        Compute a value on rank 0 and broadcast it.

        This method operates in the following way:
            Run ``map_cp`` on rank 0
            broadcast the value
        """
        result = None
        if self.is_coordinator:
            try:
                result = map_cb()
            except BaseException as e:
                result = CheckpointException(step, {self.rank: e})
        result = self.broadcast_object(result)
        if isinstance(result, CheckpointException):
            raise result
        return result
