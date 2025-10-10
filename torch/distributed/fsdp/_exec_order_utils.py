# mypy: allow-untyped-defs
import itertools
import warnings
from enum import auto, Enum
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _get_param_to_fqns
from torch.distributed.fsdp._flat_param import FlatParamHandle


class _ExecOrderWarnStatus(Enum):
    """Used internally for execution order validation."""

    NONE = auto()  # no deviation yet
    WARNING = auto()  # deviated this iteration; currently issuing warnings
    WARNED = auto()  # deviated in a previous iteration


class _ExecOrderData:
    """
    This contains the data structures to track the execution order. We track
    the pre-forward order on the *first* iteration for forward prefetching
    (which thus assumes static graph) and the post-forward order on *every*
    iteration for backward prefetching (which thus does not assume static
    graph but may be provide an incorrect order).
    """

    def __init__(
        self,
        debug_level: dist.DebugLevel,
        backward_prefetch_limit: int,
        forward_prefetch_limit: int,
    ) -> None:
        # Tracks the (static) pre-forward order for execution order validation
        # and forward prefetching
        self.handles_pre_forward_order: list[FlatParamHandle] = []
        # Tracks the post-forward order for pre-backward prefetching
        self.handles_post_forward_order: list[Optional[FlatParamHandle]] = []
        self._iter = 0

        # Gives the max number of backward/forward prefetched all-gathers by a
        # single module
        self._backward_prefetch_limit = backward_prefetch_limit
        self._forward_prefetch_limit = forward_prefetch_limit

        # Data structures for execution order validation
        self._checking_order: bool = debug_level == dist.DebugLevel.DETAIL
        self.process_group: Optional[dist.ProcessGroup] = None
        self.world_size: Optional[int] = None
        self.all_handles: list[FlatParamHandle] = []
        # Names are prefixed from the root module
        self.param_to_fqn: dict[nn.Parameter, list[str]] = {}
        # Current index in the pre-forward execution order
        self.current_order_index = 0
        self.warn_status = _ExecOrderWarnStatus.NONE

    def init(
        self,
        state: _FSDPState,
        root_module: nn.Module,
        process_group: dist.ProcessGroup,
    ) -> None:
        """
        Initializes the data structures needed for checking the forward order.
        This should be called after a root FSDP instance has been set during
        lazy initialization.
        """
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        # Fix an order over the handles, which should be the same across ranks
        for handle in traversal_utils._get_fsdp_handles(root_module):
            index = len(self.all_handles)
            self.all_handles.append(handle)
            handle._handle_index = index
        self.param_to_fqn = _get_param_to_fqns(root_module)
        # TODO (awgu): We can broadcast the metadata of rank 0's `all_handles`
        # to check that all ranks have the same handles in the same order.
        # https://github.com/pytorch/pytorch/issues/79620

    @property
    def is_first_iter(self) -> bool:
        return self._iter == 0

    def get_handle_to_backward_prefetch(
        self,
        current_handle: FlatParamHandle,
    ) -> Optional[FlatParamHandle]:
        """
        Returns a :class:`list` of the handles keys of the handles to backward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._post_forward_index
        if current_index is None:
            return None
        target_index = current_index - 1
        target_handle: Optional[FlatParamHandle] = None
        for _ in range(self._backward_prefetch_limit):
            if target_index < 0:
                break
            target_handle = self.handles_post_forward_order[target_index]
            target_index -= 1
        return target_handle

    def get_handle_to_forward_prefetch(
        self,
        current_handle: FlatParamHandle,
    ) -> Optional[FlatParamHandle]:
        """
        Returns a :class:`list` of the handles keys of the handles to forward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._pre_forward_order_index
        if current_index is None:
            return None
        target_index = current_index + 1
        target_handle: Optional[FlatParamHandle] = None
        for _ in range(self._forward_prefetch_limit):
            if target_index >= len(self.handles_pre_forward_order):
                break
            target_handle = self.handles_pre_forward_order[target_index]
            target_index += 1
        return target_handle

    def record_post_forward(self, handle: Optional[FlatParamHandle]) -> None:
        """
        Records ``handles`` in the post-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        Unlike :meth:`record_pre_forward`, this records the order *every*
        iteration with the expectation that the recorded order is reset in
        :meth:`next_iter`.
        """
        if not handle:
            return
        # Only record the first usage of a handles key
        if handle._post_forward_index:
            self.handles_post_forward_order.append(handle)
            return
        index = len(self.handles_post_forward_order)
        handle._post_forward_index = index
        self.handles_post_forward_order.append(handle)

    def record_pre_forward(
        self, handle: Optional[FlatParamHandle], is_training: bool
    ) -> None:
        """
        Records ``handles`` in the pre-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        On the first iteration, this checks the execution order across ranks.
        See :meth:`_check_order` for details.
        """
        if not handle:
            return
        self._check_order(handle, is_training)
        # Fix the order after the first iteration and only record the first
        # usage of a handles key
        if not self.is_first_iter or handle._pre_forward_order_index is not None:
            return
        index = len(self.handles_pre_forward_order)
        handle._pre_forward_order_index = index
        self.handles_pre_forward_order.append(handle)

    def _check_order(self, handle: FlatParamHandle, is_training: bool) -> None:
        """
        Checks the forward execution order as long as ``is_training`` is
        ``True`` since checking in eval mode is not supported. This only checks
        if the distributed debug level is DETAIL.

        - On the first iteration, this uses all-gathers to check that all ranks
        are all-gathering the same handles and hence ``FlatParameter`` s,
        raising an error if not.
        - On subsequent iterations, this checks that each rank is locally
        consistent with its own forward order from the first iteration, issuing
        a warning if not. This issues a warning on the first deviating
        iteration and stops warning thereafter.
        """
        # Do not check order in eval mode since the post-backward callback does
        # not run so it cannot be used to mark the end of an iteration
        if not is_training or not self._checking_order:
            return
        if self.is_first_iter:
            msg_prefix = "Forward order differs across ranks:"
            optional_local_indices: tuple[Optional[int], ...] = (
                self._get_handle_indices(handle)
            )
            device = handle.device  # guaranteed to be non-CPU
            num_valid_indices = sum(
                (index is not None) for index in optional_local_indices
            )
            tensor_kwargs: dict[str, Union[torch.dtype, torch.device]] = {
                "dtype": torch.int32,
                "device": device,
            }
            world_num_valid_indices = torch.zeros(self.world_size, **tensor_kwargs)  # type: ignore[arg-type, call-overload]
            local_num_valid_indices = torch.tensor([num_valid_indices], **tensor_kwargs)  # type: ignore[arg-type, call-overload]
            dist.all_gather_into_tensor(
                world_num_valid_indices,
                local_num_valid_indices,
                group=self.process_group,
            )
            # Copy entire tensor from D2H once to avoid per element D2H copies
            world_num_valid_indices = world_num_valid_indices.cpu()
            # Check that all ranks plan to all-gather the same number of
            # parameters
            # TODO (awgu): Since every module has at most one handle in the
            # current implementation, this should never raise the error.
            assert self.world_size is not None  # mypy
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                # TODO(voz): Don't graph break on this - dynamo hates the n1 != n2
                # tensor comparison control flow.
                # https://github.com/pytorch/pytorch/issues/107055
                for (r1, n1), (r2, n2) in itertools.combinations(
                    (
                        (rank, world_num_valid_indices[rank])
                        for rank in range(self.world_size)
                    ),
                    2,
                ):
                    if n1 != n2:
                        raise RuntimeError(
                            f"{msg_prefix} rank {r1} is all-gathering {n1} parameters "
                            f"while rank {r2} is all-gathering {n2} parameters"
                        )
            world_indices = torch.zeros(  # type: ignore[call-overload]
                self.world_size * num_valid_indices, **tensor_kwargs
            )
            local_indices = torch.tensor(optional_local_indices, **tensor_kwargs)  # type: ignore[arg-type]
            dist.all_gather_into_tensor(
                world_indices, local_indices, group=self.process_group
            )
            # Copy entire tensor from D2H once to avoid per element D2H copies
            world_indices = world_indices.cpu()
            # Check that all ranks plan to all-gather the same index parameters
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                # TODO(voz): Don't graph break on this - dynamo hates the i1 != i2
                # tensor comparison control flow.
                # https://github.com/pytorch/pytorch/issues/107055
                for (r1, i1), (r2, i2) in itertools.combinations(
                    (
                        (
                            rank,
                            world_indices[
                                rank * num_valid_indices : (rank + 1)
                                * num_valid_indices
                            ],
                        )
                        for rank in range(self.world_size)
                    ),
                    2,
                ):
                    if i1 != i2:
                        r1_param_names = self._get_names_from_handle_indices(i1)
                        r2_param_names = self._get_names_from_handle_indices(i2)
                        raise RuntimeError(
                            f"{msg_prefix} rank {r1} is all-gathering parameters "
                            f"for {r1_param_names} while rank {r2} is all-gathering "
                            f"parameters for {r2_param_names}"
                        )
        else:
            # Only issue warnings on the first deviating iteration and stop
            # checking thereafter to avoid flooding the console
            if self.warn_status == _ExecOrderWarnStatus.WARNED:
                return
            msg_prefix = None  # non-`None` means we should warn
            if self.current_order_index >= len(self.handles_pre_forward_order):
                # This iteration sees extra all-gather(s) compared to the first
                msg_prefix = (
                    "Expected to not all-gather any more parameters in the "
                    "forward but trying to all-gather parameters for "
                )
            else:
                expected_handle = self.handles_pre_forward_order[
                    self.current_order_index
                ]
                if expected_handle != handle:
                    expected_param_names = self._get_names_from_handles(expected_handle)
                    msg_prefix = (
                        f"Expected to all-gather for {expected_param_names} "
                        "but trying to all-gather parameters for "
                    )
            if msg_prefix is not None:
                param_names = self._get_names_from_handles(handle)
                msg_suffix = (
                    f"{param_names}"
                    if param_names
                    else "a newly-added parameter since construction time"
                )
                warnings.warn(
                    "Forward order differs from that of the first iteration "
                    f"on rank {self.rank}. Collectives are unchecked and may "
                    f"give incorrect results or hang.\n{msg_prefix}{msg_suffix}"
                )
                self.warn_status = _ExecOrderWarnStatus.WARNING
            self.current_order_index += 1

    def _get_handle_indices(
        self,
        handle: FlatParamHandle,
    ) -> tuple[Optional[int], ...]:
        """
        Returns the handle indices (i.e. indices into ``self.all_handles``)
        corresponding to the handles in ``handle``. An entry in the
        returned tuple is ``None`` if the handle is invalid.
        """
        indices: list[Optional[int]] = []
        if handle:
            indices.append(handle._handle_index)
        return tuple(indices)

    def _get_names_from_handle_indices(
        self,
        handle_indices: tuple[int, ...],
    ) -> list[list[str]]:
        """
        Returns a list of FQNs for each handle in ``handle_indices``. If a
        handle index is invalid, then its FQNs are omitted from the returned
        list.
        """
        fqns: list[list[str]] = []
        for index in handle_indices:
            if index is None or index < 0 or index >= len(self.all_handles):
                continue
            handle = self.all_handles[index]
            flat_param = handle.flat_param
            fqns.append(self.param_to_fqn[flat_param])
        return fqns

    def _get_names_from_handles(
        self,
        handle: FlatParamHandle,
    ) -> list[list[str]]:
        """
        Returns a list of FQNs for each handle in ``handles_key``. If a handle
        is invalid, then its FQNs are omitted from the returned list.
        """
        fqns: list[list[str]] = []
        if handle:
            flat_param = handle.flat_param
            if flat_param in self.param_to_fqn:
                fqns.append(self.param_to_fqn[flat_param])
        return fqns

    def next_iter(self):
        """
        Advances the internal data structures per iteration. This should be
        called in the post-backward callback since that marks the true end of
        an iteration.
        """
        self._iter += 1
        self.handles_post_forward_order.clear()
        if self._checking_order:
            self.current_order_index = 0
            if self.warn_status == _ExecOrderWarnStatus.WARNING:
                self.warn_status = _ExecOrderWarnStatus.WARNED
