# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import collections
import copy
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import _Join, _Joinable, _JoinHook
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


class _ZeROJoinHook(_JoinHook):
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


class ZeroRedundancyOptimizer(Optimizer, _Joinable):
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
        parameters_as_bucket_view (bool): when enabled, parameters are packed
            into larger buckets to speed up communication, and ``param.data``
            fields point to bucket views at different offsets; when disabled,
            each individual parameter is communicated separately, but each
            ``params.data`` stays intact.
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

    .. warning:
        Currently, ``ZeroRedundancyOptimizer`` requires that all of the
        passed-in parameters are the same dense type.

    .. warning: ZeroRedundancyOptimizer is experimental and subject to change.

    .. _ZeRO: https://arxiv.org/abs/1910.02054

    """

    def __init__(
        self,
        params,
        optimizer_class: Type[Optimizer],
        process_group: Optional[Any] = None,
        parameters_as_bucket_view: bool = False,
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
        _Joinable.__init__(self)
        # Now, all parameters are held in both `self._all_params` and
        # `self.param_groups`

        # Partition information (evaluated lazily)
        self._param_to_rank_cache: Dict[torch.Tensor, int] = {}
        self._param_to_index_cache: Dict[torch.Tensor, int] = {}
        self._partition_parameters_cache: List[List[Dict]] = []
        self._index_to_param_cache: List[torch.Tensor] = []
        self._device_to_per_rank_params_cache: Dict[torch.device, List[List[torch.Tensor]]] = {}

        # Default device for collective communication and buckets
        self._default_device = self._all_params[0].device

        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)
        self.global_rank = _get_global_rank(self.process_group, self.rank)

        self._optim_defaults = defaults
        self._optim_constructor = optimizer_class
        self._init_local_optimizer()

        self.parameters_as_bucket_view = parameters_as_bucket_view
        self._is_trainable_mask = self._get_is_trainable_mask()
        self._buckets: List[List[torch.Tensor]] = []
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
        self._device_to_per_rank_params_cache.clear()

    def add_param_group(self, param_group: dict) -> None:
        r"""
        Add a parameter group to the :class:`Optimizer` s ``param_groups``.

        This can be useful when fine tuning a pre-trained network, as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.

        Arguments:
            param_group (dict): specifies the parameters to be optimized and
                group-specific optimization options.

        .. warning: This method handles updating the shards on all partitions
            but needs to be called on all ranks. Calling this on a subset of
            the ranks will cause the training to hang because communication
            primitives are called depending on the managed parameters and
            expect all the ranks to participate on the same set of parameters.
        """
        super().add_param_group(param_group)
        # NOTE: The rest of the function assumes that the call to the parent's
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

        .. warning: This needs to be called on all ranks.
        """
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

    def _partition_parameters(self) -> List[List[Dict]]:
        r"""
        Partitions parameters across distributed data parallel ranks.

        Returns:
            A :class:`list` of ``param_groups`` (which is a :class:`list` of
            :class:`dict`) where each element of the list contains the
            ``param_groups`` for a rank. Element 0 corresponds to rank 0, etc.
            Each rank stores the ``param_groups`` for all of the ranks for the
            collective communication in :meth:`step`.
        """
        if len(self._partition_parameters_cache) == 0:
            self._partition_parameters_cache = [list() for _ in range(self.world_size)]
            sizes = [0] * self.world_size
            for param_group in self.param_groups:
                param_lists: List[List] = [list() for _ in range(self.world_size)]
                # Sort the parameters by size (largest first)
                params_sorted = sorted(param_group["params"], key=lambda t: t.numel(), reverse=True)
                for param in params_sorted:
                    # Greedily add the parameter to rank with smallest size so far
                    rank = sizes.index(min(sizes))
                    param_lists[rank].append(param)
                    sizes[rank] += param.numel()

                for rank, params in enumerate(param_lists):
                    param_group_rank = copy.copy(param_group)
                    param_group_rank["params"] = params
                    self._partition_parameters_cache[rank].append(param_group_rank)

        return self._partition_parameters_cache

    @property
    def _param_to_rank(self) -> Dict[torch.Tensor, int]:
        r"""
        Dict mapping parameters to their assigned data parallel rank in
        the partition.
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
        Dict mapping parameters to their indices in the global optimizer
        state.

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

    def _sync_parameters(self):
        r"""
        Syncs all parameter shards across the ranks.

        The rank sends its shard to all other ranks and receives a shard from
        each other rank using ``broadcast()``. Parameters are sent bucket-by-
        bucket if ``parameters_as_bucket_view`` is enabled and sent parameter-
        by-parameter otherwise.
        """
        handles = []
        if self.parameters_as_bucket_view:
            for dev_i_buckets in self._buckets:
                for rank, bucket in enumerate(dev_i_buckets):
                    global_rank = _get_global_rank(self.process_group, rank)
                    handles.append(
                        dist.broadcast(tensor=bucket, src=global_rank,
                                       group=self.process_group, async_op=True)
                    )
        else:
            for rank, param_groups in enumerate(self._partition_parameters()):
                global_rank = _get_global_rank(self.process_group, rank)
                for param_group in param_groups:
                    for param in param_group["params"]:
                        handles.append(
                            dist.broadcast(tensor=param.data, src=global_rank,
                                           group=self.process_group, async_op=True)
                        )
        _ = list(map(lambda x: x.wait(), handles))

    @property
    def _device_to_per_rank_params(self) -> Dict[torch.device, List[List[torch.Tensor]]]:
        r"""
        Dict mapping device to a list of the per-rank parameter lists
        containing the parameters stored on the device.

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
        if len(self._device_to_per_rank_params_cache) == 0:
            for rank, param_groups in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        device = param.device
                        if device not in self._device_to_per_rank_params_cache:
                            self._device_to_per_rank_params_cache[device] = [[] for _ in range(self.world_size)]
                        self._device_to_per_rank_params_cache[device][rank].append(param)
        return self._device_to_per_rank_params_cache

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        **kwargs: Any,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers.
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. note: Any extra parameters are passed to the base optimizer as-is.
        """
        _Join.notify_join_context(self)
        # Check if the model trainability has changed
        is_trainable_mask = self._get_is_trainable_mask()
        if is_trainable_mask != self._is_trainable_mask:
            logging.warning(
                "ZeroRedundancyOptimizer detected that the trainable params "
                "changed, updating the partitioning"
            )
            self._build_param_buckets()
            self._is_trainable_mask = is_trainable_mask

        # Sync the exposed `param_groups` attributes to the local optimizer in
        # case they have been updated
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        # Run the optimizer step on this shard only
        if closure is not None:
            loss = self.optim.step(closure=closure, **kwargs)  # type: ignore[call-arg]
        else:
            loss = self.optim.step(**kwargs)

        # Sync all of the updated parameter shards across the ranks
        self._sync_parameters()

        # Sync any updated attributes in the local optimizer to the exposed
        # `param_groups`
        self._sync_param_groups(self.optim.param_groups, self.param_groups)

        return loss

    def _join_hook(self, **kwargs):
        r"""
        Returns the ZeRO join hook, which enables training on uneven inputs by
        shadowing the collective communications in the optimizer step.

        Gradients must be properly set before this hook is called.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`_Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.
        """
        return _ZeROJoinHook(self)

    @property
    def _join_device(self) -> torch.device:
        return self._default_device

    @property
    def _join_process_group(self) -> Any:
        return self.process_group

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Load the state pertaining to the given rank from the input
        ``state_dict``, updating the local optimizer as needed.

        Arguments:
            state_dict (dict): optimizer state; should be an object returned
                from a call to :meth:`state_dict`.
        """
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
        """

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
        Builds parameter buckets if ``parameters_as_bucket_view`` is enabled so
        that for each device that stores this rank's parameters, there is a
        bucket (represented as a tensor) containing all of the parameters on
        that device that are assigned to a given rank in the parameter update
        partition.

        This function is called in the constructor and any time parameter
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
        if not self.parameters_as_bucket_view:
            return

        # Bucket B_{i,j}: parameters stored on dev_i assigned to rank j
        num_devices = len(self._device_to_per_rank_params)
        self._buckets = [[] for _ in range(num_devices)]

        for dev_i, (device, param_lists) in enumerate(self._device_to_per_rank_params.items()):
            for params in param_lists:
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
                self._buckets[dev_i].append(bucket)

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

        The function assumes that ``self._all_params`` has been initialized
        and is non-empty.

        Raises:
            ValueError: ``params`` contains sparse parameters or parameters
            of varying dense types.

        NOTE: This function can be removed once support for sparse parameters
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

    def _init_local_optimizer(self) -> None:
        r"""
        Initializes this rank's local optimizer, responsible for its subset of
        the parameters.

        The local optimizer is saved in ``self.optim``.
        """
        assert self._optim_constructor is not None
        self._clear_cache()
        self.optim: Optimizer = self._optim_constructor(self._partition_parameters()[self.rank], **self._optim_defaults)
        self._sync_param_groups(self.optim.param_groups, self.param_groups)

    def _get_is_trainable_mask(self) -> List[bool]:
        r"""
        Returns a boolean mask indicating if each parameter is trainable
        (``requires_grad``) or not.
        """
        return list(map(_is_trainable, self._all_params))
