# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import collections
import copy
import io
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer
import logging

__all__ = ["ZeroRedundancyOptimizer"]


# Credits:  classy_vision/generic/distributed_util.py
def _recursive_copy_to_device(value: Any, non_blocking: bool, device: torch.device) -> Any:
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
    return param.requires_grad


def _broadcast_object(
    obj: Any,
    src_rank: int,
    group: object = dist.group.WORLD,
    dist_device: torch.device = torch.device("cpu"),
) -> Any:
    r"""
    Either broadcast from master to the fleet (default),
    or use the src setting as the original rank.
    """

    if dist.get_rank() == src_rank:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)]).to(dist_device)
        data_send_tensor = torch.ByteTensor(data).to(dist_device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        dist.broadcast(data_send_tensor, src=src_rank, group=group, async_op=False)
    else:
        # Fetch from the source
        length_tensor = torch.LongTensor([0]).to(dist_device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        data_recv_tensor = torch.empty([int(length_tensor.item())], dtype=torch.uint8, device=dist_device)
        dist.broadcast(data_recv_tensor, src=src_rank, group=group, async_op=False)
        buffer = io.BytesIO(data_recv_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location=dist_device)
    return obj


def _get_global_rank(group: Any, rank: int) -> int:
    return rank if group is dist.group.WORLD else dist.distributed_c10d._get_global_rank(group, rank)


class ZeroRedundancyOptimizer(Optimizer):
    r"""
    This class wraps an arbitrary :class:`optim.Optimizer <torch.optim.Optimizer>`
    and shards its states across ranks in the group as described by
    ZeRO_. The optimizer instance in each rank is only responsible for
    updating ``1 / world_size`` parameters and hence only needs to keep
    ``1 / world_size`` optimizer states. After parameters are updated locally,
    each rank will broadcast its parameters to all other peers to keep all
    model replicas in the same state. ``ZeroRedundancyOptimizer`` can be used
    in conjunction with :class:`torch.nn.parallel.DistributedDataparallel` to
    reduce per-rank peak memory consumption.

    ``ZeroRedundancyOptimizer`` uses a sorted-greedy algorithm to pack a number of
    parameters at each rank. Each parameter belongs to a single rank and is not
    divided among ranks. The partition is arbitrary and might not match the
    the parameter registration or usage order.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s

    Keyword Args:
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
        group (``ProcessGroup``, optional): ``torch.distributed``
            ``ProcessGroup`` (default: ``group.WORLD`` initialized by
            :meth:`torch.distributed.init_process_group`).
        parameters_as_bucket_view (bool): when enabled, parameters will
            be packed into larger buckets to speed up communication and
            ``param.data`` fields will point to bucket views at different
            offsets. When disabled, each individual parameter will be
            communicated separately, but ``params.data`` will stay intact.
        **default: all trailing arguments will be forwarded to the given optimizer.

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

    .. note: Currently, ``ZeroRedundancyOptimizer`` requires that all of the
        passed-in parameters are on the same device and that they are the same
        dense type.

    .. warning: ZeroRedundancyOptimizer is experimental and subject to change.

    .. _ZeRO: https://arxiv.org/abs/1910.02054

    """

    def __init__(
        self,
        params,
        optimizer_class: Type[Optimizer],
        group: Optional[Any] = None,
        parameters_as_bucket_view: bool = False,
        **default: Any,
    ):
        # Perform type and assumption checks on the input parameters
        self._verify_and_init_params(params)
        self._verify_same_param_device()
        self._verify_same_dense_param_type()
        self._device = self._all_params[0].device

        # Hold all the model params in the root .param_groups
        # NOTE: the default constructor uses `add_param_group` which is partially overloaded here
        # we introduce the `initialized` flag for be able to dissociate the behaviour of
        # `add_param_group` in between super() and ZeroRedundancyOptimizer
        self.initialized = False
        super().__init__(self._all_params, default)

        # Partition information (evaluated lazily)
        self._param_to_rank_cache: Dict[torch.Tensor, int] = {}
        self._param_to_index_cache: Dict[int, int] = {}
        self._partition_parameters_cache: List[List[Dict]] = []
        self._index_to_param_cache: List[torch.Tensor] = []

        # Build the wrapped optimizer, responsible for a shard of the params
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.global_rank = _get_global_rank(self.group, self.rank)
        self.parameters_as_bucket_view = parameters_as_bucket_view

        self._optim_defaults = default
        self._optim_constructor = optimizer_class

        # Optional consolidated optimizer state
        self._all_states: List[Dict[str, Any]] = []

        self._reference_is_trainable_mask = list(map(_is_trainable, self._all_params))
        self.buckets: List[torch.Tensor] = []

        self._update_trainable()
        self.initialized = True

    def _clear_cache(self) -> None:
        r"""
        Clears the cached data structures giving partition information.
        """
        self._partition_parameters_cache.clear()
        self._param_to_rank_cache.clear()
        self._index_to_param_cache.clear()
        self._param_to_index_cache.clear()

    def add_param_group(self, param_group: dict) -> None:
        r"""
        Add a param group to the :class:`Optimizer` s ``param_groups``.

        This can be useful when fine tuning a pre-trained network, as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized
                along with group specific optimization options.

        .. warning: This method handles updating the shards on all partitions,
            but needs to be called on all ranks. Calling this on a subset of the
            ranks will cause the training to hang, because communication
            primitives are called depending on the managed parameters, and
            expect all the ranks to participate on the sane set of parameters.
        """

        super().add_param_group(param_group)
        if self.initialized:
            # Force a re-partitioning
            self._clear_cache()

            param_groups = self.partition_parameters()[self.rank]
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])

            # Update the bucketing strategy accordingly
            if self.parameters_as_bucket_view:
                self._setup_flat_buffers()

    def consolidate_state_dict(self, to: int = 0) -> None:
        r"""
        Update the consolidated state_dict list, one per rank.

        Arguments:
            to (int): the rank that receives the global states. (default: 0)

        .. warning: This needs to be called on all replicas
        """

        # Sync lr and other attributes in case its been updated
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        empty_messenger = torch.tensor([0], dtype=torch.uint8, device=self._device)

        # Pull the sharded state from all the other replicas
        # Store all the states in order, rank by rank

        # NOTE: In practice, `broadcast` is used, which is wasteful (gather would have been appropriate)
        # compatibility issues with some backends make the use of broadcast mandatory for now.
        # a possible follow up would be to move all sharded state management to RPC RRef

        self._all_states = []
        for rank in range(self.world_size):
            global_rank = _get_global_rank(self.group, rank)

            # This rank collects the whole state
            if self.rank == to:
                if rank == self.rank:
                    self._all_states.append(
                        _recursive_copy_to_device(
                            self.local_state_dict(),
                            non_blocking=True,
                            device=torch.device("cpu"),
                        )
                    )
                else:
                    # Fetch the optim state from the other replicas
                    replica_state = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.group,
                        dist_device=self._device,
                    )

                    self._all_states.append(
                        _recursive_copy_to_device(replica_state, non_blocking=True, device=torch.device("cpu"))
                    )
            else:
                # Acknowledge broadcasts, and send this rank's shard when needed
                # Default to CPU space to gain some memory headroom
                if rank == self.rank:
                    # Send the state to the reference replica
                    _ = _broadcast_object(
                        self.local_state_dict(),
                        src_rank=self.global_rank,
                        group=self.group,
                        dist_device=self._device,
                    )

                elif rank != to:
                    # Discard this tensor/rank, broadcast was being use for compatibility reasons
                    _ = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.group,
                        dist_device=self._device,
                    )

    def partition_parameters(self) -> List[List[Dict]]:
        r"""
        Partitions parameters across distributed data parallel ranks.

        Returns:
            a list of ``param_groups`` (which is a list of dict) where each
            element of the list contains the param_groups for a rank. Element 0
            corresponds to rank 0, etc. We need all the ranks for the broadcast
            inside ``step()``.

        NOTE: `test_sharding()` and `test_add_param_group()` rely on this
        function using the sorted-greedy algorithm for partitioning. If the
        algorithm is changed, please re-examine those two tests in
        `test_zero_redundancy_optimizer.py` accordingly.
        """
        if len(self._partition_parameters_cache) == 0:
            self._partition_parameters_cache = [list() for _ in range(self.world_size)]
            sizes = [0] * self.world_size
            for param_group in self.param_groups:
                param_lists: List[List] = [list() for _ in range(self.world_size)]
                # Sort the params by size (largest first)
                params_sorted = sorted(param_group["params"], key=lambda t: t.size()[0], reverse=True)
                for param in params_sorted:
                    # Add this param to rank with smallest size.
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
        Hash table mapping parameters to their assigned data parallel rank in
        the partition.
        """
        if len(self._param_to_rank_cache) == 0:
            for rank, param_groups in enumerate(self.partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        self._param_to_rank_cache[param] = rank
        return self._param_to_rank_cache

    @property
    def _param_to_index(self) -> Dict[int, int]:
        r"""
        Hash table mapping parameters to their indices in the global optimizer
        scheme.
        """
        if len(self._param_to_index_cache) == 0:
            self._param_to_index_cache = {
                id(p): i for i, p in enumerate(chain(*(g["params"] for g in self.param_groups)))
            }
        return self._param_to_index_cache

    @property
    def _index_to_param(self) -> Dict[int, torch.Tensor]:
        r"""
        List mapping parameter indices in the global optimizer scheme to the
        actual params.
        """
        if len(self._index_to_param_cache) == 0:
            self._index_to_param_cache = list(chain(*(g["params"] for g in self.param_groups)))
        return self._index_to_param_cache

    def step(self, closure: Optional[Callable[[], float]] = None, **kwargs: Any) -> Optional[float]:
        r"""
        Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        Returns:
            optional loss, depends on the underlying optimizer

        .. note: Any extra parameter is passed to the base optimizer as-is
        """

        # Check whether the model trainability graph changed
        trainable_mask = list(map(_is_trainable, self._all_params))
        if trainable_mask != self._reference_is_trainable_mask:
            logging.warning(
                "ZeroRedundancyOptimizer detected that the trainable params "
                "changed, updating the partitioning"
            )
            self._update_trainable()
            self._reference_is_trainable_mask = trainable_mask

        # Sync oss param_groups attributes in case they've been updated by a scheduler.
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        # Run the optimizer step on this shard only:
        if closure is not None:
            loss = self.optim.step(closure=closure, **kwargs)  # type: ignore[call-arg]
        else:
            loss = self.optim.step(**kwargs)

        # Sync all the updated shards in between the ranks
        handles = []
        if self.parameters_as_bucket_view:
            for rank, bucket in enumerate(self.buckets):
                global_rank = _get_global_rank(self.group, rank)
                handles.append(
                    dist.broadcast(tensor=bucket, src=global_rank,
                                   group=self.group, async_op=True)
                )
        else:
            for rank, param_groups in enumerate(self.partition_parameters()):
                global_rank = _get_global_rank(self.group, rank)
                for param_group in param_groups:
                    for param in param_group["params"]:
                        handles.append(
                            dist.broadcast(tensor=param.data, src=global_rank,
                                           group=self.group, async_op=True)
                        )
        _ = list(map(lambda x: x.wait(), handles))

        # Sync hypothethical new results from the wrapped optimizer to the exposed param_groups
        self._sync_param_groups(self.optim.param_groups, self.param_groups)

        return loss

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Restore the global parameter groups as well as the shard.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`
        """

        for key, value in state_dict["state"].items():
            param = self._index_to_param[key]

            # Populate the sharded optimizer state on the fly
            if self._param_to_rank[param] != self.rank:
                state_dict["state"][key] = None
            else:
                self.optim.state[param] = _recursive_copy_to_device(value, non_blocking=True, device=param.device)

        super().load_state_dict(state_dict)

        # Sync with the optimizer param groups
        self._sync_param_groups(state_dict["param_groups"], self.param_groups)
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

    def local_state_dict(self) -> Dict:
        r"""
        Gets this rank's ``state_dict``.

        Returns:
            The state of the optimizer as a :class:`dict`.
            It contains two entries:

            * state - a dict holding current optimization state. Its content
                differs between optimizer classes.
            * param_groups - a dict containing all parameter groups
        """
        return self.optim.state_dict()

    def state_dict(self) -> Dict[str, Any]:
        r"""
        Returns:
            the last known global optimizer state, which consist of a list of
            the shards.

        .. warning:
            If the state has not been consolidated, this returns a shard's worth,
            not the global state.

        .. warning:
            Returning the global state is limited to the replica which was
            responsible for the consolidation. The state may also not be up to
            date, depending on when :meth:`consolidate_state_dict` was last called.
        """

        if len(self._all_states) == 0:
            raise RuntimeError(
                "Optimizer state has not been consolidated on this rank. \
                Please call `consolidate_state_dict()` on all ranks beforehand if you meant to save the global state"
            )

        # Unify the shard states and the state that pytorch would expect, given the model.
        # Indexation needs several redirections, since each shard only knows a limited scope of the model
        # - get the pytorch compliant parameter indexing
        state_dict = super().state_dict()

        # - go through the per-shard states, which are all indexed locally
        for rank, s in enumerate(self._all_states):
            # -- match the local indexing and the global partition, update the corresponding saved state globally
            for local_pg, global_pg in zip(s["param_groups"], self.partition_parameters()[rank]):
                local_index_to_param_id = {
                    i_param: id(global_pg["params"][i]) for i, i_param in enumerate(local_pg["params"])
                }

                for local_param_index in local_pg["params"]:
                    # Update the state, if any
                    if local_param_index in s["state"].keys():
                        global_id = self._param_to_index[local_index_to_param_id[local_param_index]]
                        state_dict["state"][global_id] = s["state"][local_param_index]

        # Make sure that the parameters are sorted in the state, as expected
        state_dict["state"] = dict(sorted(state_dict["state"].items()))
        return state_dict

    @staticmethod
    def rank_local_state_dict(rank: int, state_dict: dict) -> dict:
        r"""
        Returns the local_state_dict for a given rank.

        Arguments:
            rank (int): rank to get ``local_state_dict`` for
            state_dict (dict): global ``state_dict``
        """
        param_groups = state_dict["param_groups"][state_dict["partition"][rank][0] : state_dict["partition"][rank][1]]
        return {"state": state_dict["state"][rank], "param_groups": param_groups}

    @staticmethod
    def _sync_param_groups(source: List[Dict[Any, Any]], destination: List[Dict[Any, Any]]) -> None:
        r"""Sync learning rate and other optimizer attributes (needed to support schedulers)."""

        for source_group, destination_group in zip(source, destination):
            # Sync everything but the parameters
            for k in filter(lambda x: x != "params", source_group.keys()):
                destination_group[k] = source_group[k]

    def _setup_flat_buffers(self) -> None:
        r"""
        Make all params which are on the same device and tied to the same rank
        views of a single buffer. This is used at construction time, and
        anytime parameter trainability is changed (frozen or unfrozen) and
        ``_update_trainable`` is called.
        """
        for rank, param_groups in enumerate(self.partition_parameters()):
            # Clone the non-trainable params, find the buffer size and dtype
            # for the trainable params' bucket, and compile a list of the
            # trainable params
            buffer_size = 0
            dtype = None
            trainable_params = []
            for param_group in param_groups:
                for param in param_group["params"]:
                    if not _is_trainable(param):
                        param.data = param.data.detach().clone()
                    else:
                        buffer_size += param.numel()
                        trainable_params.append(param)
                    dtype = param.dtype  # assumes all dense and same dtype

            # Create a dummy bucket if there are no params
            if buffer_size == 0:
                self.buckets.append(torch.zeros(1, device=self._device))
                continue

            # Otherwise, construct the bucket
            bucket = torch.empty(buffer_size, dtype=dtype, device=self._device)
            offset = 0
            for param in trainable_params:
                offset_next = offset + param.numel()
                bucket[offset:offset_next].copy_(param.data.flatten())
                param.data = bucket[offset:offset_next].view_as(param.data)
                offset = offset_next

            # Either replace the existing bucket or create it
            if len(self.buckets) != rank:
                self.buckets[rank] = bucket
            else:
                self.buckets.append(bucket)

    def _update_trainable(self) -> None:
        r"""
        Updates the partitioning and communication patterns if the trainability
        (``requires_grad``) of some parameters changed.
        """

        # Create the optim which will work on the param shard
        if not hasattr(self, "optim"):
            self._clear_cache()
            self.optim = self._optim_constructor(self.partition_parameters()[self.rank], **self._optim_defaults)
            self._sync_param_groups(self.optim.param_groups, self.param_groups)

        if self.parameters_as_bucket_view:
            self._setup_flat_buffers()

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

    def _verify_same_param_device(self) -> None:
        r"""
        Verifies that ZeRO is being used under the single-process single-
        device regime where a process operates exclusively on a full model
        replica on a single device.

        The function assumes that ``self._all_params`` has been initialized
        and is non-empty.

        Raises:
            ValueError: ``params`` contains parameters across multiple
                devices.

        NOTE: This function can be removed once support for sharding a rank's
        model parameters across multiple devices is added.
        """
        device = self._all_params[0].device
        for param in self._all_params[1:]:
            if param.device != device:
                raise ValueError("ZeroRedundancyOptimizer assumes that each "
                                 "rank's model parameters are on the same "
                                 f"device but got both {device} and "
                                 f"{param.device}")

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
