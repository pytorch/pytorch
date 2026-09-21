"""Experimental, private contract for FSDP all-gather backend authors.

Preparation selects a layout before output allocation. Returning None selects
the native rank-major input packing and output finalizer. Custom layouts pack
into independent input storage by default; the internal copy-in callable keeps
the native operator's signature, including its rank argument.

Finalization receives tensor-only parameter metadata on the compute stream,
after collective completion. Existing parameter objects and saved aliases keep
their storage when layouts change. Backend-owned storage stays allocated across
reshard, but its lease may be shared after AllGather.release_output(). The
backend, not FSDP, must order reuse after all local and remote consumers and
restore each parameter's original storage region before its next use.

These authoring types are not exported from torch.distributed.fsdp. Out-of-tree
backends must target a matching revision of this private interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from ._fsdp_param import FSDPParam


AllGatherCopyIn = Callable[
    [list[torch.Tensor], torch.Tensor, list[int], int, int],
    tuple[torch.Tensor, torch.Tensor],
]


@dataclass(frozen=True, kw_only=True)
class AllGatherInputMetadata:
    """Description of this call's flattened local input and output eligibility."""

    input_split_sizes: list[int]
    input_numel: int
    world_size: int
    dtype: torch.dtype
    device: torch.device
    can_use_param_contiguous_output: bool


@dataclass
class AllGatherParamMetadata:
    """Tensor-only description of one parameter's collective outputs.

    ``outputs`` contains the existing destinations, if any. Their storage and
    tensor identities must survive while autograd can hold saved aliases.
    ``padded_sharded_size`` describes this call's shard before all-gather.
    """

    input_numels: list[int]
    input_dtypes: list[torch.dtype]
    shard_dim: int
    padded_sharded_size: torch.Size
    outputs: list[torch.Tensor]
    backend_owned: bool


@dataclass
class AllGatherOutputs:
    """Outputs and whether FSDP must leave their storage to the backend.

    Backend-owned storage must remain allocated, including across reshard,
    until the parameter group is destroyed. After ``AllGather.release_output``
    it may be shared with other groups, provided overwrites are ordered after
    consumers and the same parameter regions are restored before the next use.
    """

    tensors: list[list[torch.Tensor]]
    backend_owned: bool = False


class AllGatherLayout(ABC):
    """Input packing and output handling for an all-gather backend.

    FSDP orders collective completion before finalization. The backend owns
    communication stream lifetimes and any persistent registered storage.
    Create a separate stateful backend/layout instance per parameter group;
    share storage through the backend's pool, not by sharing the layout instance.
    The stateless default layout is exempt from this ownership restriction.
    """

    _owner: object | None = None

    def _bind_owner(self, owner: object) -> None:
        if self._owner is None:
            self._owner = owner
        elif self._owner is not owner:
            raise ValueError(
                "an all-gather layout instance cannot be shared across FSDP "
                "parameter groups; create a separate stateful backend/layout "
                "instance for each group (shared storage may use a backend pool)"
            )

    def prepare(
        self,
        input_metadata: AllGatherInputMetadata,
    ) -> tuple[AllGatherCopyIn, AllGatherLayout, object | None]:
        """Select input packing and metadata before allocating the output."""
        metadata = self.prepare_output(input_metadata)
        if metadata is None:
            return torch.ops.fsdp.all_gather_copy_in, DEFAULT_ALL_GATHER_LAYOUT, None
        return self.copy_in, self, metadata

    @abstractmethod
    def prepare_output(
        self,
        input_metadata: AllGatherInputMetadata,
    ) -> object | None:
        """Return per-call metadata, or None to use rank-major input and output.

        The backend must produce the selected layout for this collective.
        Metadata must remain valid until its result is finalized.
        """
        ...

    def copy_in(
        self,
        all_gather_inputs: list[torch.Tensor],
        all_gather_output: torch.Tensor,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack into independent input storage without modifying the output.

        The signature matches the native rank-major copy-in operator. Layouts
        that opt into rank-local packing may use rank; this default does not.
        """
        all_gather_input = torch.empty(
            (all_gather_input_numel,),
            dtype=all_gather_output.dtype,
            device=all_gather_output.device,
        )
        torch._foreach_copy_(
            torch.split(all_gather_input, all_gather_input_split_sizes),
            all_gather_inputs,
        )
        return all_gather_input, all_gather_output

    @abstractmethod
    def finalize_outputs(
        self,
        all_gather_output: torch.Tensor,
        param_metadata: list[AllGatherParamMetadata],
        world_size: int,
        output_metadata: object | None,
    ) -> AllGatherOutputs:
        """Copy or view outputs after collective completion on the compute stream.

        Existing parameter storage is preserved by FSDP when returned views
        differ from it. Such calls use copy-out instead of replacing aliases.
        """
        ...

    def param_contiguous_output_views(
        self,
        all_gather_output: torch.Tensor,
        param_input_numels: list[list[int]],
        world_size: int,
    ) -> list[list[torch.Tensor]]:
        """Carve a parameter-contiguous output into per-parameter views."""
        output_offset = 0
        outputs: list[list[torch.Tensor]] = []
        for input_numels in param_input_numels:
            output_numel = input_numels[0] * world_size
            param_output = all_gather_output.narrow(0, output_offset, output_numel)
            outputs.append([param_output])
            output_offset += output_numel
        if output_offset != all_gather_output.numel():
            raise AssertionError(
                "parameter-contiguous all-gather output covered "
                f"{output_offset} of {all_gather_output.numel()} elements"
            )
        return outputs


class DefaultAllGatherLayout(AllGatherLayout):
    """Rank-major collective output copied into stable parameter storage."""

    def _bind_owner(self, owner: object) -> None:
        # Stateless layouts may be shared by existing collective backends.
        pass

    def prepare(
        self, input_metadata: AllGatherInputMetadata
    ) -> tuple[AllGatherCopyIn, AllGatherLayout, object | None]:
        return torch.ops.fsdp.all_gather_copy_in, self, None

    def prepare_output(self, input_metadata: AllGatherInputMetadata) -> None:
        return None

    def copy_in(
        self,
        all_gather_inputs: list[torch.Tensor],
        all_gather_output: torch.Tensor,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,
            all_gather_output,
            all_gather_input_split_sizes,
            all_gather_input_numel,
            rank,
        )

    def finalize_outputs(
        self,
        all_gather_output: torch.Tensor,
        param_metadata: list[AllGatherParamMetadata],
        world_size: int,
        output_metadata: object | None,
    ) -> AllGatherOutputs:
        from ._fsdp_collectives import _copy_all_gather_outputs
        from ._fsdp_param import alloc_storage

        outputs = []
        copy_outputs = []
        reorder_infos = []
        split_sizes = []
        for param in param_metadata:
            param_outputs = param.outputs or [
                torch.empty(
                    numel * world_size, dtype=dtype, device=all_gather_output.device
                )
                for numel, dtype in zip(param.input_numels, param.input_dtypes)
            ]
            if not param.backend_owned:
                for tensor in param_outputs:
                    alloc_storage(tensor)
            outputs.append(param_outputs)
            split_sizes.extend(
                numel * tensor.element_size() // all_gather_output.element_size()
                for numel, tensor in zip(param.input_numels, param_outputs)
            )
            if param.shard_dim != 0:
                temporary_outputs = [torch.empty_like(t) for t in param_outputs]
                reorder_infos.append((param, temporary_outputs, param_outputs))
                copy_outputs.extend(temporary_outputs)
            else:
                copy_outputs.extend(param_outputs)

        # Fallback may receive the same persistent buffer that parameters alias.
        if any(
            param.backend_owned
            and any(
                tensor.untyped_storage().data_ptr()
                == all_gather_output.untyped_storage().data_ptr()
                for tensor in param.outputs
            )
            for param in param_metadata
        ):
            all_gather_output = all_gather_output.clone()
        _copy_all_gather_outputs(
            all_gather_output, split_sizes, copy_outputs, world_size
        )
        for param, temporary_outputs, param_outputs in reorder_infos:
            pre_param_size = list(param.padded_sharded_size)
            pre_param_size[0] *= world_size
            post_param_size = list(param.padded_sharded_size)
            post_param_size[param.shard_dim] *= world_size
            with torch.autograd._unsafe_preserve_version_counter(
                tuple(t for t in param_outputs if not t.is_inference())
            ):
                for source, target in zip(temporary_outputs, param_outputs):
                    chunks = torch.chunk(source.view(pre_param_size), world_size, dim=0)
                    torch.cat(
                        chunks, dim=param.shard_dim, out=target.view(post_param_size)
                    )
        return AllGatherOutputs(outputs)


DEFAULT_ALL_GATHER_LAYOUT = DefaultAllGatherLayout()


def _can_use_param_contiguous_output(
    fsdp_params: list[FSDPParam],
    param_input_dtypes: list[list[torch.dtype]],
    param_input_numels: list[list[int]],
    output_dtype: torch.dtype,
) -> bool:
    from torch._dynamo.compiled_autograd import (
        compiled_autograd_enabled,
        in_compiled_autograd_region,
    )

    from ._fsdp_param import ShardedState

    if compiled_autograd_enabled or in_compiled_autograd_region:
        return False
    if not (len(fsdp_params) == len(param_input_dtypes) == len(param_input_numels)):
        return False
    for fsdp_param, input_dtypes, input_numels in zip(
        fsdp_params, param_input_dtypes, param_input_numels
    ):
        if (
            len(input_dtypes) != 1
            or len(input_numels) != 1
            or input_dtypes[0] != output_dtype
            or fsdp_param.fsdp_placement.dim != 0
            or fsdp_param.is_dtensor
            or hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
            or hasattr(fsdp_param._sharded_local_tensor, "fsdp_post_all_gather")
            or fsdp_param.sharded_state == ShardedState.SHARDED_POST_FORWARD
        ):
            return False
    return True


def _init_layout_outputs(
    fsdp_params: list[FSDPParam],
    result: AllGatherOutputs,
) -> None:
    from ._fsdp_param import alloc_storage

    if len(fsdp_params) != len(result.tensors):
        raise AssertionError(
            f"all-gather layout returned {len(result.tensors)} parameter outputs "
            f"for {len(fsdp_params)} parameters"
        )
    for fsdp_param, outputs in zip(fsdp_params, result.tensors):
        if not outputs:
            raise AssertionError("all-gather layout returned no output for a parameter")
        previous = fsdp_param.all_gather_outputs
        if not previous:
            fsdp_param.all_gather_outputs = outputs
            fsdp_param._keep_all_gather_output_storage = result.backend_owned
        elif outputs is not previous:
            if len(previous) != len(outputs):
                raise AssertionError("all-gather layout changed the number of outputs")
            # Saved tensors may alias this storage even after reshard. Preserve
            # it when switching layouts or receiving a new backend buffer.
            for target, source in zip(previous, outputs):
                if target.shape != source.shape or target.dtype != source.dtype:
                    raise AssertionError(
                        "all-gather layout changed output shape or dtype"
                    )
                if not fsdp_param._keep_all_gather_output_storage:
                    alloc_storage(target)
                if target.data_ptr() != source.data_ptr():
                    with torch.autograd._unsafe_preserve_version_counter(
                        () if target.is_inference() else (target,)
                    ):
                        target.copy_(source)
