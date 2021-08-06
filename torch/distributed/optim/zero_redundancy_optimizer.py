# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import collections
import copy
import enum
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Type, Union

import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim import functional_optim_map
from torch.optim import Optimizer

__all__ = ["ZeroRedundancyOptimizer"]


# Credits:  classy_vision/generic/distributed_util.py
def _recursive_copy_to_device(
    value: Any,
    non_blocking: bool,
    device: torch.device,
) -> Any:
    r"""
    Recursively searches lists, tuples, dicts and copies tensors to device if
    possible. Non-tensor values are passed as-is in the result.

    .. note:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the device.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)

    if isinstance(value, (list, tuple)):
        values = [_recursive_copy_to_device(val, non_blocking=non_blocking, device=device) for val in value]
        return values if isinstance(value, list) else tuple(values)

    if isinstance(value, collections.abc.Mapping):
        return {
            key: _recursive_copy_to_device(val, non_blocking=non_blocking, device=device) for key, val in value.items()
        }

    return value


def _is_trainable(param: torch.Tensor) -> bool:
    r"""
    Returns if a parameter is trainable, where trainability is equivalent to
    requiring a gradient.
    """
    return param.requires_grad


def _broadcast_object(
    obj: Any, src_rank: int,
    group: object = dist.group.WORLD,
    device: torch.device = torch.device("cpu")
) -> Any:
    r"""
    Broadcasts an object to the given group, sending the object if called from
    the source rank and receiving the object otherwise.

    Arguments:
        obj: object to broadcast; only used if called on the source rank.
        src_rank (int): source rank.
        group (``ProcessGroup``, optional): group used for the broadcast
            (default: ``dist.group.WORLD``).
        device (``torch.device``, optional): device to send from or receive
            to (default: ``torch.device("cpu")``).

    Returns:
        The broadcasted object.
    """
    if dist.get_rank() == src_rank:
        # Send the object
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)]).to(device)
        data_send_tensor = torch.ByteTensor(data).to(device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        dist.broadcast(data_send_tensor, src=src_rank, group=group, async_op=False)
    else:
        # Receive the object
        length_tensor = torch.LongTensor([0]).to(device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        data_recv_tensor = torch.empty([int(length_tensor.item())], dtype=torch.uint8, device=device)
        dist.broadcast(data_recv_tensor, src=src_rank, group=group, async_op=False)
        buffer = io.BytesIO(data_recv_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location=device)
    return obj


def _get_global_rank(group: Any, rank: int) -> int:
    r"""
    Returns the global rank for the given group and rank.
    """
    return (rank if group is dist.group.WORLD
            else dist.distributed_c10d._get_global_rank(group, rank))


class _ZeROJoinHook(JoinHook):
    def __init__(self, zero):
        assert isinstance(zero, ZeroRedundancyOptimizer), \
            "ZeRO join hook requires passing in a ZeroRedundancyOptimizer " \
            "instance as the state"
        self.zero = zero
        super().__init__()

    def main_hook(self):
        """
        Performs an optimizer step, which updates the joined process's shard of
        the parameters and broadcasts those parameters.
        """
        self.zero.step()


class _DDPBucket(NamedTuple):
    r"""
    This contains the model parameters corresponding to a
    :class:`DistributedDataParallel` gradient bucket.

        bucket_index (int): index of the bucket determined by the DDP gradient
            bucket all-reduce order.
        params (List[torch.Tensor]): model parameters in the bucket.
    """
    bucket_index: int
    params: List[torch.Tensor]


class _OverlapStatus(enum.IntEnum):
    r"""
    This defines the three possible statuses that
    :class:`ZeroRedundancyOptimizer` can be in when overlapping with
    :class:`DistributedDataParallel`.

        ``UNINITIALIZED``: The ZeRO instance is effectively uninitialized and
            is waiting for DDP to finalize its bucketing.
        ``DDP_HAS_REBUILT_BUCKETS``: DDP has rebuilt its buckets, meaning that
            its bucketing is finalized. The ZeRO instance can now collect the
            necessary information about the DDP bucketing.
        ``INITIALIZED``: The ZeRO instance is fully initialized and can now
            optimize parameters.
    """
    UNINITIALIZED = 0
    DDP_HAS_REBUILT_BUCKETS = 1
    INITIALIZED = 2


class _OverlapInfo():
    r"""
    This contains the information needed by :class:`ZeroRedundancyOptimizer`
    to overlap with :class:`DistributedDataParallel`.

        status (_OverlapStatus): current status; see :class:`_OverlapStatus`
            for more information.
        params_per_bucket (List[List[torch.Tensor]]): ``params_per_bucket[i]``
            gives the model parameters in the ``i``th bucket.
        params_per_rank (List[List[torch.Tensor]]): ``params_per_rank[i]``
            gives the model parameters assigned to the ``i``th rank, where the
            parameters are grouped by increasing bucket indices.
        offsets (Dict[int, int]): maps from bucket index to the offset in
            ``self.params_per_rank[rank]`` giving the index of the first
            parameter in that bucket, where ``rank`` is this process's own
            rank; the keys of this :class:`dict` are the bucket indices
            assigned to this rank.
        broadcast_handles (List[Work]): :class:`list` of async work handles for
            the parameter broadcasts.
        bucket_index_to_future (Dict[int, torch.futures.Future]):
            :class:`dict` mapping bucket index to the corresponding all-reduce
            future.
        bucket_index_to_bucket (Dict[int, dist.GradBucket]): :class:`dict`
            mapping bucket index to the corresponding bucket.
        bucket_indices_seen (List[int]): :class:`list` of the bucket indices
            seen on this iteration.
    """
    def __init__(self) -> None:
        self.status: _OverlapStatus = _OverlapStatus.UNINITIALIZED

        # Modified per bucket reconstruction
        self.params_per_bucket: List[List[torch.Tensor]] = []
        self.params_per_rank: List[List[torch.Tensor]] = \
            [[] for _ in range(dist.get_world_size())]
        self.offsets: Dict[int, int] = {}

        # Modified per iteration
        self.broadcast_handles: List[Any] = []
        self.bucket_indices_seen: List[int] = []
        # Used by `hook_with_zero_step()`
        self.bucket_index_to_future: Dict[int, torch.futures.Future] = {}
        self.bucket_index_to_bucket: Dict[int, dist.GradBucket] = {}

    def wait_for_broadcasts(self, num_buckets, rank) -> None:
        r"""
        Waits for all parameter broadcasts. This should be called once all
        broadcasts have been scheduled, meaning ``self.broadcast_handles`` is
        filled. This clears ``self.broadcast_handles`` in preparation for the
        next iteration.

        Arguments:
            num_buckets (int): total number of buckets.
            rank (int): the calling process's rank.
        """
        assert len(self.broadcast_handles) == num_buckets, \
            f"Missing at least one broadcast handle on rank {rank}"
        _ = list(map(lambda x: x.wait(), self.broadcast_handles))
        self.broadcast_handles.clear()

    def clear_per_iter_info(self) -> None:
        r"""
        Clears the data structures that are modified per-iteration. This should
        be called at the end of an iteration.
        """
        self.bucket_indices_seen.clear()
        self.bucket_index_to_future.clear()
        self.bucket_index_to_bucket.clear()


class ZeroRedundancyOptimizer(Optimizer, Joinable):
    r"""
    This class wraps an arbitrary :class:`optim.Optimizer
    <torch.optim.Optimizer>` and shards its states across ranks in the group as
    described by ZeRO_. The local optimizer instance in each rank is only
    responsible for updating approximately ``1 / world_size`` parameters and
    hence only needs to keep ``1 / world_size`` optimizer states. After
    parameters are updated locally, each rank will broadcast its parameters to
    all other peers to keep all model replicas in the same state.
    ``ZeroRedundancyOptimizer`` can be used in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel` to reduce per-rank peak
    memory consumption.

    ``ZeroRedundancyOptimizer`` uses a sorted-greedy algorithm to pack a number
    of parameters at each rank. Each parameter belongs to a single rank and is
    not divided among ranks. The partition is arbitrary and might not match the
    the parameter registration or usage order.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            giving all parameters, which will be sharded across ranks.

    Keyword Args:
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
        process_group (``ProcessGroup``, optional): ``torch.distributed``
            ``ProcessGroup`` (default: ``dist.group.WORLD`` initialized by
            :meth:`torch.distributed.init_process_group`).
        parameters_as_bucket_view (bool, optional): if ``True``, parameters are
            packed into buckets to speed up communication, and ``param.data``
            fields point to bucket views at different offsets; if ``False``,
            each individual parameter is communicated separately, and each
            ``params.data`` stays intact (default: ``False``).
        overlap_with_ddp (bool, optional): if ``True``, :meth:`step` is
            overlapped with :class:`DistributedDataParallel` 's gradient
            synchronization; this requires (1) either a functional optimizer
            for the ``optimizer_class`` argument or one with a functional
            equivalent and (2) registering a DDP communication hook
            constructed from one of the functions in ``ddp_zero_hook.py``;
            parameters are packed into buckets matching those in
            :class:`DistributedDataParallel`, meaning that the
            ``parameters_as_bucket_view`` argument is ignored.
            If ``False``, :meth:`step` runs disjointly after the backward pass
            (per normal).
            (default: ``False``)
        **defaults: any trailing arguments, which are forwarded to the local
            optimizer.

    Example::

        >>> import torch.nn as nn
        >>> from torch.distributed.optim import ZeroRedundancyOptimizer
        >>> from torch.nn.parallel import DistributedDataParallel as DDP

        >>> model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
        >>> ddp = DDP(model, device_ids=[rank])
        >>> opt = ZeroRedundancyOptimizer(
        >>>     ddp.parameters(),
        >>>     optimizer_class=torch.optim.Adam,
        >>>     lr=0.01
        >>> )
        >>> ddp(inputs).sum().backward()
        >>> opt.step()

    .. warning::
        Currently, ``ZeroRedundancyOptimizer`` requires that all of the
        passed-in parameters are the same dense type.

    .. warning::
        If you pass ``overlap_with_ddp=True``, be wary of the following: Given
        the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two training iterations do not perform parameter updates in the
        optimizer step. This is because it needs information about the gradient
        bucketing strategy used by :class:`DistributedDataParallel`, which is
        not finalized until the second forward pass if ``static_graph=False``
        or until the third forward pass if ``static_graph=True``. To adjust
        for this, one option is to prepend dummy inputs. Note, however, that it
        is important to still include ``ZeroRedundancyOptimizer.step()`` in the
        training loop.

    .. warning:: ZeroRedundancyOptimizer is experimental and subject to change.

    .. _ZeRO: https://arxiv.org/abs/1910.02054

    """

    def __init__(
        self,
        params,
        optimizer_class: Type[Optimizer],
        process_group: Optional[Any] = None,
        parameters_as_bucket_view: bool = False,
        overlap_with_ddp: bool = False,
        **defaults: Any,
    ):
        # Perform type and assumption checks on the input parameters
        self._verify_and_init_params(params)
        self._verify_same_dense_param_type()

        # NOTE: The parent constructor uses `add_param_group()` which is
        # partially overloaded in ZeroRedundancyOptimizer, so we use the
        # `initialized` flag to dissociate the behaviour of `add_param_group()`
        # between the parent and child.
        self.initialized = False

        Optimizer.__init__(self, self._all_params, defaults)
        Joinable.__init__(self)
        # Now, all parameters are held in both `self._all_params` and
        # `self.param_groups`

        # Internal data structures (`_cache` indicates lazily evaluated)
        self._param_to_rank_cache: Dict[torch.Tensor, int] = {}
        self._param_to_index_cache: Dict[torch.Tensor, int] = {}
        self._partition_parameters_cache: List[List[Dict]] = []
        self._index_to_param_cache: List[torch.Tensor] = []
        self._device_to_params_per_rank_cache: Dict[torch.device, List[List[torch.Tensor]]] = {}
        self._device_to_buckets_cache: Dict[torch.device, List[List[_DDPBucket]]] = {}
        self._device_to_device_index: Dict[torch.device, int] = {}
        self._is_trainable_mask = self._get_is_trainable_mask()

        # Default device for collective communication and buckets
        self._default_device = self._all_params[0].device

        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.world_size: int = dist.get_world_size(self.process_group)
        self.rank: int = dist.get_rank(self.process_group)
        self.global_rank: int = _get_global_rank(self.process_group, self.rank)

        self._overlap_with_ddp: bool = overlap_with_ddp
        self._optim_defaults = defaults
        self._optim_constructor = self._get_optimizer_constructor(optimizer_class)

        # If `overlap_with_ddp=True`, local optimizer initialization is delayed
        # to run time after the necessary information has been collected
        if not overlap_with_ddp:
            self._init_local_optimizer()
        else:
            self._overlap_info: _OverlapInfo = _OverlapInfo()
            if parameters_as_bucket_view:
                logging.warning(
                    "`parameters_as_bucket_view=True` will be ignored since "
                    "`overlap_with_ddp=True`; instead, a different bucketing "
                    "strategy will be used"
                )

        # `self._buckets` is used if `parameters_as_bucket_view=True` or
        # `overlap_with_ddp=True`, in which case parameter data is flattened
        # into buckets (i.e. contiguous tensors)
        # If `overlap_with_ddp=True`, the bucketing requires an additional
        # dimension to match the DDP gradient bucketing
        self.parameters_as_bucket_view = parameters_as_bucket_view
        self._buckets: Union[
            List[List[torch.Tensor]],
            List[List[Dict[int, torch.Tensor]]]
        ] = []  # type: ignore[assignment]
        self._build_param_buckets()

        # Optional consolidated optimizer state, only populated if this rank
        # is the target in `consolidate_state_dict()`
        self._all_state_dicts: List[Dict[str, Any]] = []

        self.initialized = True

    def _clear_cache(self) -> None:
        r"""
        Clears the cached data structures giving partition information.
        """
        self._partition_parameters_cache.clear()
        self._param_to_rank_cache.clear()
        self._index_to_param_cache.clear()
        self._param_to_index_cache.clear()
        self._device_to_params_per_rank_cache.clear()
        self._device_to_buckets_cache.clear()

    def add_param_group(self, param_group: dict) -> None:
        r"""
        Add a parameter group to the :class:`Optimizer` 's ``param_groups``.

        This can be useful when fine tuning a pre-trained network, as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.

        Arguments:
            param_group (dict): specifies the parameters to be optimized and
                group-specific optimization options.

        .. warning:: This method handles updating the shards on all partitions
            but needs to be called on all ranks. Calling this on a subset of
            the ranks will cause the training to hang because communication
            primitives are called depending on the managed parameters and
            expect all the ranks to participate on the same set of parameters.
        """
        if self.initialized and self._overlap_with_ddp:
            raise RuntimeError(
                "ZeroRedundancyOptimizer with `overlap_with_ddp=True` only "
                "supports a single parameter group"
            )

        super().add_param_group(param_group)
        # NOTE: The rest of the method assumes that the call to the parent's
        # `add_param_group()` appends the new parameter group and preserves
        # the previous parameter-group ordering

        if self.initialized:
            # Force a re-partitioning of the parameters
            self._clear_cache()
            param_groups = self._partition_parameters()[self.rank]
            # NOTE: All parameters in the old parameter groups should be
            # assigned to the same ranks so that the local optimizers do not
            # need to be reinitialized

            # Add the parameters assigned to this rank from the new parameter
            # group to the local optimizer, if any
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])

            # Update the bucketing strategy accordingly
            if self.parameters_as_bucket_view:
                self._build_param_buckets()

    def consolidate_state_dict(self, to: int = 0) -> None:
        r"""
        Consolidate a list of ``state_dict`` s (one per rank) on the target
        rank.

        Arguments:
            to (int): the rank that receives the optimizer states (default: 0).

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt.

        .. warning:: This needs to be called on all ranks.
        """
        self._check_overlap_initialized()

        # Sync the exposed `param_groups` attributes to the local optimizer in
        # case they have been updated
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        # Pull the sharded state from all ranks and store them in rank order
        empty_messenger = torch.tensor([0], dtype=torch.uint8, device=self._default_device)

        # NOTE: We wastefully use `broadcast()` (e.g. instead of `gather()`)
        # due to compatibility issues with NCCL backend; a possible follow-up
        # is to move all sharded state management to RPC RRef
        self._all_state_dicts = []
        for rank in range(self.world_size):
            global_rank = _get_global_rank(self.process_group, rank)
            if self.rank == to:
                # Consolidate all local `state_dict`s on this rank, storing on
                # CPU to save GPU memory
                if rank == self.rank:
                    # Directly append own optimizer state
                    self._all_state_dicts.append(
                        _recursive_copy_to_device(self.optim.state_dict(), non_blocking=True, device=torch.device("cpu"),)
                    )
                else:
                    # Receive the optimizer state from the source rank
                    local_state_dict = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )
                    self._all_state_dicts.append(
                        _recursive_copy_to_device(local_state_dict, non_blocking=True, device=torch.device("cpu"))
                    )
            else:
                if rank == self.rank:
                    # Send the optimizer state to the target rank
                    _ = _broadcast_object(
                        self.optim.state_dict(),
                        src_rank=self.global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )
                elif rank != to:
                    # Discard the received object; `broadcast()` is used for
                    # compatibility reasons
                    _ = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )

    def _verify_params_per_rank(
        self,
        params_per_rank: List[List[torch.Tensor]],
    ) -> None:
        r"""
        Verifies ``params_per_rank`` for :meth:`_partition_parameters`,
        checking that ``params_per_rank`` has length equal to the world size
        and that it does not contain any parameters not passed into the
        :class:`ZeroRedundancyOptimizer` constructor.

        The parameters in ``params_per_rank`` being a strict subset of those
        passed into the constructor is valid since some parameters may be
        frozen.

        Raises:
            ValueError: if ``params_per_rank`` does not have length equal to
                the world size or if it contains a parameter that was not
                passed into the :class:`ZeroRedundancyOptimizer` constructor.
        """
        if len(params_per_rank) != self.world_size:
            raise ValueError(
                "`params_per_rank` must have length equal to the world size"
            )
        all_params_set = set(self._all_params)
        for params in params_per_rank:
            for param in params:
                if param not in all_params_set:
                    raise ValueError(
                        "Passing a new parameter in `params_per_rank` that "
                        "was not passed into the ZeroRedundancyOptimizer "
                        "constructor"
                    )

    def _partition_param_group(
        self,
        param_group: Dict[str, Any],
        params_per_rank: List[List[torch.Tensor]]
    ) -> None:
        r"""
        Partitions the parameter group ``param_group`` according to
        ``params_per_rank`` by modifying ``self._partition_parameters_cache``.

        This method should only be used as a subroutine for
        :meth:`_partition_parameters`.

        Arguments:
            param_group (dict[str, Any]): a parameter group as normally defined
                in an optimizer state.
            params_per_rank (list[list[torch.Tensor]]): a :class:`list` of
                length world size containing :class:`list` s of parameters to
                assign to each rank.
        """
        for rank, params in enumerate(params_per_rank):
            rank_param_group = copy.copy(param_group)
            rank_param_group["params"] = params
            self._partition_parameters_cache[rank].append(rank_param_group)

    def _partition_parameters(
        self,
        params_per_rank: Optional[List[List[torch.Tensor]]] = None,
    ) -> List[List[Dict]]:
        r"""
        Partitions parameters across distributed data parallel ranks.

        Arguments:
            params_per_rank (list[list[torch.Tensor]], optional): a
                :class:`list` of length world size containing :class:`list` s
                of parameters to assign to each rank; this provides a way to
                specify a partition manually.
                If ``None``, the parameters are partitioned according to an
                internal algorithm.
                (default: ``None``)

        Returns:
            A :class:`list` where each element of the list contains the
            ``param_groups`` for a rank (which itself is a :class:`list` of
            :class:`dict`); element 0 corresponds to rank 0, etc.; each rank
            stores the ``param_groups`` for all ranks for the collective
            communication in :meth:`step`.

        Raises:
            ValueError: see :meth:`_validate_params_per_rank`.
            RuntimeError: if ``params_per_rank`` is not ``None`` and this
                :class:`ZeroRedundancyOptimizer` instance is using more than
                one parameter group.
        """
        if params_per_rank is None:
            # Partition the parameters optimizing for uniformity
            if len(self._partition_parameters_cache) == 0:
                self._partition_parameters_cache = [[] for _ in range(self.world_size)]
                sizes = [0] * self.world_size
                for param_group in self.param_groups:
                    param_group_params_per_rank: List[List] = [[] for _ in range(self.world_size)]
                    # Sort the parameters by size (largest first)
                    params_sorted = sorted(param_group["params"], key=lambda t: t.numel(), reverse=True)
                    for param in params_sorted:
                        # Greedily add the parameter to rank with smallest size so far
                        rank = sizes.index(min(sizes))
                        param_group_params_per_rank[rank].append(param)
                        sizes[rank] += param.numel()
                    # Apply the constructed partition of the parameter group
                    self._partition_param_group(param_group, param_group_params_per_rank)

            return self._partition_parameters_cache

        # Partition the parameters according to `params_per_rank`
        assert len(self._partition_parameters_cache) == 0, \
            "Specifying `params_per_rank` should only be done when the " \
            "parameters have not been partitioned yet"
        if len(self.param_groups) != 1:
            raise RuntimeError(
                "Specifying `params_per_rank` only supports a single "
                "parameter group"
            )
        self._verify_params_per_rank(params_per_rank)
        self._partition_parameters_cache = [[] for _ in range(self.world_size)]

        # Apply the passed-in partition of the parameter group
        param_group = self.param_groups[0]
        self._partition_param_group(param_group, params_per_rank)

        return self._partition_parameters_cache

    @property
    def _param_to_rank(self) -> Dict[torch.Tensor, int]:
        r"""
        :class:`dict` mapping parameters to their assigned data parallel rank
        in the partition.
        """
        if len(self._param_to_rank_cache) == 0:
            for rank, param_groups in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        self._param_to_rank_cache[param] = rank
        return self._param_to_rank_cache

    @property
    def _param_to_index(self) -> Dict[torch.Tensor, int]:
        r"""
        :class:`dict` mapping parameters to their indices in the global
        optimizer state.

        NOTE: This assumes that the global optimizer state's indexing (in
        ``state_dict``) follows a linear ordering over the parameter groups.
        """
        if len(self._param_to_index_cache) == 0:
            self._param_to_index_cache = {
                p: i for i, p in enumerate(chain(*(g["params"] for g in self.param_groups)))
            }
        return self._param_to_index_cache

    @property
    def _index_to_param(self) -> List[torch.Tensor]:
        r"""
        List mapping parameter indices in the global optimizer scheme to the
        actual params.
        """
        if len(self._index_to_param_cache) == 0:
            self._index_to_param_cache = list(chain(*(g["params"] for g in self.param_groups)))
        return self._index_to_param_cache

    def _broadcast_params_from_rank(self, rank: int):
        r"""
        Broadcasts the shard of parameters from a given rank to all other
        ranks asynchronously.

        Arguments:
            rank (int): the source rank.

        Returns:
            A :class:`list` of async work handles for the ``broadcast()`` s
            performed to synchronize the parameters.
        """
        assert not self._overlap_with_ddp, \
            "`_broadcast_params_from_rank()` should not be used if " \
            "`overlap_with_ddp=True`; instead, the broadcasting should " \
            "happen in the DDP communication hook"
        handles = []
        if self.parameters_as_bucket_view:
            for dev_i_buckets in self._buckets:
                bucket = dev_i_buckets[rank]
                global_rank = _get_global_rank(self.process_group, rank)
                handles.append(
                    dist.broadcast(tensor=bucket, src=global_rank,
                                   group=self.process_group, async_op=True)
                )
        else:
            param_groups = self._partition_parameters()[rank]
            global_rank = _get_global_rank(self.process_group, rank)
            for param_group in param_groups:
                for param in param_group["params"]:
                    handles.append(
                        dist.broadcast(tensor=param.data, src=global_rank,
                                       group=self.process_group, async_op=True)
                    )
        return handles

    def _sync_params(self):
        r"""
        Syncs all parameter shards across the ranks.

        This rank sends its shard of the parameters to all other ranks and
        receives a shard from each other rank. This is done using
        ``broadcast()``. Parameters are sent bucket-by-bucket if
        ``parameters_as_bucket_view=True``and sent parameter-by-parameter
        otherwise.
        """
        handles = []
        for rank in range(self.world_size):
            handles.extend(self._broadcast_params_from_rank(rank))
        _ = list(map(lambda x: x.wait(), handles))

    @property
    def _device_to_params_per_rank(
        self
    ) -> Dict[torch.device, List[List[torch.Tensor]]]:
        r"""
        :class:`dict` mapping each device to a :class:`list` of the per-rank parameter
        lists filtered to only include the parameters stored on that device.
        Each per-rank parameter list gives the parameters assigned to that rank
        to update.

        This is used for constructing the parameter buckets if
        ``parameters_as_bucket_view=True``.

        Let ``dev_i`` denote the ``i``th device for this rank. Then:
        ``dev_0`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_0``,
            rank 1's assigned parameters stored on ``dev_0``,
            ...
        ``dev_1`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_1``,
            rank 1's assigned parameters stored on ``dev_1``,
            ...
        ...
        """
        assert self.parameters_as_bucket_view, \
            "`_device_to_params_per_rank` should only be used if " \
            "`parameters_as_bucket_view=True`"
        if len(self._device_to_params_per_rank_cache) == 0:
            for rank, param_groups in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        device = param.device
                        if device not in self._device_to_params_per_rank_cache:
                            self._device_to_params_per_rank_cache[device] = [[] for _ in range(self.world_size)]
                        self._device_to_params_per_rank_cache[device][rank].append(param)
        return self._device_to_params_per_rank_cache

    @property
    def _device_to_buckets(
        self
    ) -> Dict[torch.device, List[List[_DDPBucket]]]:
        r"""
        :class:`dict` mapping each device to a :class:`list` of :class:`list`
        of :class:`_DDPBucket` s.

        ``_device_to_buckets[d][r][i]`` gives the ``i``th bucket
        assigned to rank ``r`` stored on device ``d``, where each bucket
        contains a list of the model parameters associated with the
        corresponding logical :class:`DistributedDataParallel` gradient bucket.

        This is used for constructing the parameter buckets if
        ``overlap_with_ddp=True``.
        """
        assert self._overlap_with_ddp, \
            "`_device_to_buckets()` should only be used if " \
            "`overlap_with_ddp=True`"
        if len(self._device_to_buckets_cache) > 0:
            return self._device_to_buckets_cache

        overlap_info = self._overlap_info
        assert overlap_info.status == _OverlapStatus.INITIALIZED, \
            "Accessing `_device_to_buckets` before the necessary " \
            "information has been collected"

        params_per_bucket = overlap_info.params_per_bucket
        for bucket_idx, bucket_params in enumerate(params_per_bucket):
            assert len(bucket_params) > 0, "Empty bucket"
            rank = self._ddp_bucket_index_to_rank(bucket_idx)
            bucket = _DDPBucket(bucket_idx, bucket_params)
            device = bucket_params[0].device  # assume same device per bucket
            if device not in self._device_to_buckets_cache:
                self._device_to_buckets_cache[device] = [[] for _ in range(self.world_size)]
            self._device_to_buckets_cache[device][rank].append(bucket)

        return self._device_to_buckets_cache

    def _local_step(
        self,
        gradients: Optional[List[Optional[torch.Tensor]]] = None,
        closure: Optional[Callable[[], float]] = None,
        **kwargs: Any,
    ) -> Optional[float]:
        r"""
        Performs a single optimizer step without syncing parameters across
        ranks.

        Arguments:
            gradients (list[Optional[torch.Tensor]], optional): a :class:`list`
                of length equal to the number of parameters assigned to this
                rank containing gradient tensors or ``None`` as its elements;
                a ``None`` in the :class:`list` indicates that the
                corresponding parameter should not be updated.
                If the argument itself is ``None``, then all parameters are
                updated, and the gradients are assumed to be already populated.
                (default: ``None``)
            closure (callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers and should be
                ``None`` if ``gradients`` is not ``None``; (default: ``None``)
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. warning::
            The argument ``gradients`` should only be specified (i.e. not
            ``None``) if ``overlap_with_ddp=True``, in which case
            :class:`ZeroRedundancyOptimizer` wraps a functional optimizer.
        """
        Join.notify_join_context(self)
        # Check if the model trainability has changed
        is_trainable_mask = self._get_is_trainable_mask()
        if is_trainable_mask != self._is_trainable_mask:
            if self._overlap_with_ddp:
                raise RuntimeError(
                    "ZeroRedundancyOptimizer with `overlap_with_ddp=True` "
                    "does not support changing parameter trainability at run "
                    "time"
                )
            logging.warning(
                "ZeroRedundancyOptimizer detected that the trainable "
                "parameters changed; rebuilding the parameter buckets if "
                "enabled"
            )
            self._build_param_buckets()
            self._is_trainable_mask = is_trainable_mask

        # Sync the exposed `param_groups` attributes to the local optimizer in
        # case they have been updated
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        # Run the optimizer step on this shard only
        if gradients is None:
            loss = self.optim.step(**kwargs) if closure is None \
                else self.optim.step(closure=closure, **kwargs)
        else:
            assert self._overlap_with_ddp, "Specifying `gradients` should not " \
                "be used when `overlap_with_ddp=False`"
            assert closure is None, "`closure` is not supported when using " \
                "a local functional optimizer"
            loss = self.optim.step(gradients=gradients)

        # Sync any updated attributes in the local optimizer to the exposed
        # `param_groups`
        self._sync_param_groups(self.optim.param_groups, self.param_groups)

        return loss

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        **kwargs: Any,
    ) -> Optional[float]:
        r"""
        Performs a single optimizer step and syncs parameters across all ranks.

        Arguments:
            closure (callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers.
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. note: Any extra parameters are passed to the base optimizer as-is.
        """
        if self._overlap_with_ddp:
            # If DDP buckets have been rebuilt, calling `step()` indicates that
            # the backward pass has fully completed and all information has
            # been collected; hence, this ZeRO instance can be initialized
            if self._overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
                self._overlap_info.status = _OverlapStatus.INITIALIZED
                # Since all information has been collected, perform the delayed
                # initialization of the local optimizer and supporting state
                self._init_zero_for_overlap()

            # `step()` does not actually perform any parameter updates and is
            # only used for bookkeeping when `overlap_with_ddp=True`
            return None

        # Perform the local optimizer step
        loss = self._local_step(closure=closure, **kwargs)

        # Sync all of the updated parameter shards across the ranks
        self._sync_params()

        return loss

    def join_hook(self, **kwargs):
        r"""
        Returns the ZeRO join hook, which enables training on uneven inputs by
        shadowing the collective communications in the optimizer step.

        Gradients must be properly set before this hook is called.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.

        This hook does not support any keyword arguments; i.e. ``kwargs`` is
        unused.
        """
        return _ZeROJoinHook(self)

    @property
    def join_device(self) -> torch.device:
        return self._default_device

    @property
    def join_process_group(self) -> Any:
        return self.process_group

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Load the state pertaining to the given rank from the input
        ``state_dict``, updating the local optimizer as needed.

        Arguments:
            state_dict (dict): optimizer state; should be an object returned
                from a call to :meth:`state_dict`.

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt.
        """
        self._check_overlap_initialized()

        for index, value in state_dict["state"].items():
            param = self._index_to_param[index]
            if self._param_to_rank[param] != self.rank:
                # Clear any state irrelevant to this rank
                state_dict["state"][index] = None
            else:
                # Load the parameter state to the local optimizer
                self.optim.state[param] = _recursive_copy_to_device(value, non_blocking=True, device=param.device)

        super().load_state_dict(state_dict)

        # Sync the input state with the exposed and local optimizer states
        self._sync_param_groups(state_dict["param_groups"], self.param_groups)
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

    def state_dict(self) -> Dict[str, Any]:
        r"""
        Returns the last global optimizer state known to this rank.

        .. warning:
            If the state has not been consolidated to this rank, this raises a
            runtime error, and even if it has, the state may not be up-to-date,
            depending on when :meth:`consolidate_state_dict` was last called.

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt; or if this method is called without a preceding call
                to :meth:`consolidate_state_dict`.
        """
        self._check_overlap_initialized()

        if len(self._all_state_dicts) == 0:
            raise RuntimeError(
                "Optimizer state has not been consolidated on this rank. "
                f"Please call `consolidate_state_dict(to={self.rank})` on "
                "all ranks beforehand if you meant to save the global state."
            )

        # Get the possibly-stale global optimizer state that uses global
        # parameter indexing
        state_dict = super().state_dict()

        # Update the global optimizer state with local state information,
        # factoring in the translation from local to global indexing
        for rank, local_state_dict in enumerate(self._all_state_dicts):
            local_param_groups = local_state_dict["param_groups"]
            global_param_groups = self._partition_parameters()[rank]
            assert len(local_param_groups) == len(global_param_groups), \
                "Mismatch between number of local and global parameter groups"

            for local_param_group, global_param_group in zip(local_param_groups, global_param_groups):
                # `local_param_group` stores local indices, while
                # `global_param_group` stores the tensors directly
                local_param_indices = local_param_group["params"]
                global_params = global_param_group["params"]

                assert len(local_param_indices) == len(global_params), \
                    "Mismatch between number of local and global parameters in parameter group"
                for local_param_index, global_param in zip(local_param_indices, global_params):
                    # Update the global parameter state, if any
                    if local_param_index in local_state_dict["state"]:
                        global_param_index = self._param_to_index[global_param]
                        state_dict["state"][global_param_index] = local_state_dict["state"][local_param_index]

        # Sort the parameters in the state
        state_dict["state"] = dict(sorted(state_dict["state"].items()))
        return state_dict

    @staticmethod
    def _sync_param_groups(
        src_param_groups: List[Dict[Any, Any]],
        dst_param_groups: List[Dict[Any, Any]],
    ) -> None:
        r"""
        Syncs the attributes from the source parameter groups to the
        destination parameter groups.

        Example attributes include learning rate or scheduler attributes. The
        two parameter groups should have the same length (i.e. same number of
        parameter groups).

        Arguments:
            src_param_groups (list[dict]): parameter groups giving the
                attribute settings to copy.
            dst_param_groups (list[dict]): parameter groups giving the
                attribute settings to set.
        """
        assert len(src_param_groups) == len(dst_param_groups), \
            "Mismatch between number of source and destination parameter groups"
        for src_param_group, dst_param_group in zip(src_param_groups, dst_param_groups):
            # Sync all attributes except the parameters
            for attr in filter(lambda x: x != "params", src_param_group.keys()):
                dst_param_group[attr] = src_param_group[attr]

    def _build_param_buckets(self) -> None:
        r"""
        Builds parameter buckets if ``parameters_as_bucket_view=True`` so
        that for each device that stores this rank's parameters, there is a
        bucket (represented as a tensor) containing all of the parameters on
        that device that are assigned to a given rank in the parameter update
        partition.

        This method is called in the constructor and any time parameter
        trainability is changed.

        .. warning::
            The current implementation assumes that all of the parameters in a
            bucket are of the same dense type when allocating the bucket's
            tensor.

        .. warning::
            If the model parameters are stored across more than one device,
            then the storage partitioning must be the same across all
            processes in order for parameter synchronization to work.
        """
        if not self.parameters_as_bucket_view or self._overlap_with_ddp:
            return

        # `self._buckets[i][j]` are the parameters stored on device i and
        # assigned to rank j
        num_devices = len(self._device_to_params_per_rank)
        self._buckets = [[] for _ in range(num_devices)]  # type: ignore[assignment]

        for dev_i, (device, params_per_rank) in enumerate(self._device_to_params_per_rank.items()):
            for params in params_per_rank:
                bucket_size = 0
                dtype = None
                trainable_params = []
                for param in params:
                    if not _is_trainable(param):
                        # Clone in case the parameter was previously part of
                        # a bucket to avoid the data from being destroyed
                        param.data = param.data.detach().clone()
                    else:
                        bucket_size += param.numel()
                        trainable_params.append(param)
                    dtype = param.dtype  # assumes all same dtype

                if bucket_size == 0:
                    # Create a dummy bucket if there are no parameters
                    bucket = torch.zeros(1, device=device)
                else:
                    # Construct the bucket (assuming all dense and same dtype)
                    bucket = torch.empty(bucket_size, dtype=dtype, device=device)
                    offset = 0
                    for param in trainable_params:
                        offset_next = offset + param.numel()
                        bucket[offset:offset_next].copy_(param.data.flatten())
                        param.data = bucket[offset:offset_next].view_as(param.data)
                        offset = offset_next
                self._buckets[dev_i].append(bucket)  # type: ignore[arg-type]

    def _build_ddp_param_buckets(self) -> None:
        r"""
        Builds parameter buckets if ``overlap_with_ddp`` so that for each
        device that stores this rank's parameters, there is a :class:`list` of
        buckets (represented as tensors) containing the parameters on that
        device that are assigned to the rank in the parameter update
        partition and grouped following the :class:`DistributedDataParallel`
        gradient buckets.

        This method should only be called during the delayed initialization
        when ``overlap_with_ddp=True``.

        .. warning::
            The current implementation assumes that all of the parameters in a
            bucket are of the same dense type when allocating the bucket's
            tensor.

        .. warning::
            If the model parameters are stored across more than one device,
            then the storage partitioning must be the same across all
            processes in order for parameter synchronization to work.
        """
        assert self._overlap_with_ddp, \
            "`_build_ddp_param_buckets()` should only be called when " \
            "`overlap_with_ddp=True`"

        num_devices = len(self._device_to_buckets)
        self._buckets = [[{} for _ in range(self.world_size)] for _ in range(num_devices)]  # type: ignore[assignment]

        for dev_idx, (device, ddp_buckets_per_rank) in enumerate(self._device_to_buckets.items()):
            self._device_to_device_index[device] = dev_idx
            for rank, ddp_buckets in enumerate(ddp_buckets_per_rank):
                for ddp_bucket in ddp_buckets:
                    bucket_index = ddp_bucket.bucket_index  # type: ignore[attr-defined]
                    params = ddp_bucket.params              # type: ignore[attr-defined]
                    bucket_size = 0
                    dtype = None
                    for param in params:
                        assert _is_trainable(param), \
                            "Model parameter corresponding to a gradient in " \
                            "a DDP bucket should require a gradient"
                        bucket_size += param.numel()
                        dtype = param.dtype  # assumes all same dtype
                    assert bucket_size > 0
                    bucket = torch.empty(bucket_size, dtype=dtype, device=device)
                    offset = 0
                    # Construct the bucket (assuming all dense and same dtype)
                    for param in params:
                        offset_next = offset + param.numel()
                        bucket[offset:offset_next].copy_(param.data.flatten())
                        param.data = bucket[offset:offset_next].view_as(param.data)
                        offset = offset_next
                    self._buckets[dev_idx][rank][bucket_index] = bucket

    def _verify_and_init_params(self, params: Any) -> None:
        r"""
        Verifies the type of ``params`` and initializes ``self._all_params``
        if ``params`` is valid.

        While :class:`optim.Optimizer <torch.optim.Optimizer>` allows
        ``params`` to be an iterable of :class:`dict` s, currently
        ``ZeroRedundancyOptimizer`` strictly requires ``params`` to be an
        iterable of :class:`torch.Tensor` s.

        Raises:
            TypeError: ``params`` has an invalid type.
            ValueError: ``params`` is empty.
        """
        if isinstance(params, torch.Tensor):
            raise TypeError("params argument should be an iterable of "
                            f"Tensors, but got {torch.typename(params)}")
        try:
            self._all_params = list(params)
        except TypeError:
            raise TypeError("params argument should be an iterable of "
                            f"Tensors, but got {torch.typename(params)}")
        if len(self._all_params) == 0:
            raise ValueError("ZeroRedundancyOptimizer got an empty parameter "
                             "list")
        for param in self._all_params:
            if not isinstance(param, torch.Tensor):
                raise TypeError("params argument should be an iterable of "
                                "Tensors, but got an iterable containing "
                                f"{torch.typename(param)}")

    def _verify_same_dense_param_type(self) -> None:
        r"""
        Verifies that all parameters are of the same dense type.

        The method assumes that ``self._all_params`` has been initialized
        and is non-empty.

        Raises:
            ValueError: ``params`` contains sparse parameters or parameters
            of varying dense types.

        NOTE: This method can be removed once support for sparse parameters
        and varying parameter types is added.
        """
        typename = torch.typename(self._all_params[0])
        if self._all_params[0].is_sparse:
            raise ValueError("ZeroRedundancyOptimizer only supports using "
                             "the same dense type for all parameters but got "
                             f"{typename}")
        for param in self._all_params[1:]:
            other_typename = torch.typename(param)
            if other_typename != typename:
                raise ValueError("ZeroRedundancyOptimizer only supports "
                                 "using the same dense type for all "
                                 f"parameters but got both {typename} and "
                                 f"{other_typename}")

    def _get_is_trainable_mask(self) -> List[bool]:
        r"""
        Returns a boolean mask indicating if each parameter is trainable
        (``requires_grad``) or not.
        """
        return list(map(_is_trainable, self._all_params))

    def _init_local_optimizer(self) -> None:
        r"""
        Initializes this rank's local optimizer, responsible for its subset of
        the parameters.

        The local optimizer is saved in ``self.optim``.
        """
        assert self._optim_constructor is not None, \
            "The local optimizer class has not been set"

        param_groups = self._partition_parameters()[self.rank]
        # `overlap_with_ddp=True` requires a local functional optimizer
        if self._overlap_with_ddp:
            # Functional optimizers only support a single parameter group and
            # require passing in the parameters as a list
            assert len(param_groups) == 1, "Initializing the local " \
                "functional optimizer with more than one parameter group"
            params = param_groups[0]["params"]
            self.optim: Any = self._optim_constructor(params, **self._optim_defaults)
        else:
            # NOTE: Passing `param_groups` into the local optimizer constructor
            # bypasses the empty parameter list check
            self.optim: Optimizer = self._optim_constructor(param_groups, **self._optim_defaults)  # type: ignore[no-redef]

        # TODO: Manually add `self.param_groups` if using a functional
        # optimizer; remove this if/when the functional optimizers support
        # multiple parameter groups
        if self._overlap_with_ddp and not hasattr(self.optim, "param_groups"):
            assert hasattr(self.optim, "param_group"), \
                "The functional optimizer should set at least one of the " \
                "attributes `param_group` or `param_groups`"
            self.optim.param_groups = [self.optim.param_group]  # type: ignore[attr-defined]

        self._sync_param_groups(self.optim.param_groups, self.param_groups)

    def _init_zero_for_overlap(self) -> None:
        r"""
        Performs a delayed initialization of the local optimizer and the
        supporting data structures.
        """
        assert self._overlap_with_ddp, \
            "`_init_zero_for_overlap()` should only be called when " \
            "`overlap_with_ddp=True`"
        self._clear_cache()
        self._partition_parameters(self._overlap_info.params_per_rank)
        self._build_ddp_param_buckets()
        self._init_local_optimizer()

    def _ddp_bucket_index_to_rank(self, bucket_index: int) -> int:
        r"""Assigns a rank to a given DDP gradient bucket index."""
        return bucket_index % self.world_size

    def _get_assigned_ddp_bucket_indices(self) -> List[int]:
        r"""
        Returns a list of the DDP gradient bucket indices assigned to this rank
        to update.
        """
        assert self._overlap_info.status == _OverlapStatus.INITIALIZED
        num_buckets = len(self._overlap_info.params_per_bucket)
        assigned_indices = [
            bucket_index for bucket_index in range(num_buckets)
            if self._ddp_bucket_index_to_rank(bucket_index) == self.global_rank
        ]
        return assigned_indices

    def _check_overlap_initialized(self):
        r"""
        Checks that the delayed initialization has occurred (see
        :meth:`_init_zero_for_overlap`) if ``overlap_with_ddp=True``, and
        raises a ``RuntimeError`` if not. This should preface methods that
        should not be run before that delayed initialization.

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and
                :meth:`_init_zero_for_overlap` has not been called.
        """
        if self._overlap_with_ddp \
                and self._overlap_info.status != _OverlapStatus.INITIALIZED:
            raise RuntimeError(
                "This method should not be called until this "
                "ZeroRedundancyOptimizer instance has been fully "
                "initialized"
            )

    def _get_optimizer_constructor(self, optimizer_class: Any) -> Any:
        r"""
        Returns the proper optimizer constructor, performing the necessary
        validation and transformation depending on ``overlap_with_ddp``.

        Returns:

            - ``optimizer_class`` if ``overlap_with_ddp=False`` and
                ``optimizer_class`` is not a functional optimizer.
            - ``optimizer_class`` if ``overlap_with_ddp=True`` and
                ``optimizer_class`` is already a functional optimizer.
            - The functional equivalent of ``optimizer_class`` if
                ``overlap_with_ddp=True`` and ``optimizer_class`` is not
                already a functional optimizer (assuming the equivalent
                exists).

        Raises:
            ValueError:

                - if ``overlap_with_ddp=True`` but ``optimizer_class`` is
                    neither a functional optimizer nor translatable to a
                    functional optimizer.
                - if ``overlap_with_ddp=False`` and ``optimizer_class`` is a
                    functional optimizer.
        """
        functional_optims = functional_optim_map.values()
        if not self._overlap_with_ddp:
            if optimizer_class in functional_optims:
                # Using a functional optimizer is only supported when
                # `overlap_with_ddp=True`
                raise ValueError(
                    f"Passing in a functional optimizer {optimizer_class} "
                    "when `overlap_with_ddp=False`"
                )
            else:
                return optimizer_class
        else:
            if optimizer_class in functional_optims:
                # Already a functional optimizer
                return optimizer_class
            elif optimizer_class in functional_optim_map:
                # Translate the passed-in optimizer class to its functional
                # equivalent if `overlap_with_ddp=True`
                optim_constructor = functional_optim_map[optimizer_class]
                logging.info(
                    f"Using the functional optimizer {optim_constructor} "
                    f"instead of {optimizer_class} since "
                    "`overlap_with_ddp=True`"
                )
                return optim_constructor
            else:
                raise ValueError(
                    "Using `ddp_with_overlap=True` requires using a "
                    "functional optimizer, but there is no supported functional "
                    f"optimizer equivalent for {optimizer_class}"
                )
