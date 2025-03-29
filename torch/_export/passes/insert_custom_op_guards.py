import functools
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch._export.passes._node_metadata_hook import (
    _node_metadata_hook,
    _set_node_metadata_hook,
)


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorMetadata:
    rank: int
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout

    @staticmethod
    def maybe_from_tensor(t: Any) -> Optional["TensorMetadata"]:
        if not isinstance(t, torch.Tensor):
            return None
        return TensorMetadata(t.dim(), t.dtype, t.device, t.layout)


@dataclass(frozen=True)
class OpProfile:
    args_profile: tuple[Optional[TensorMetadata]]
    out_profile: Union[TensorMetadata, tuple[TensorMetadata]]


def get_op_profile(node: torch.fx.Node) -> OpProfile:
    args_profile = tuple(
        [
            TensorMetadata.maybe_from_tensor(arg.meta.get("val"))
            if isinstance(arg, torch.fx.Node)
            else None
            for arg in (*node.args, *node.kwargs.values())
        ]
    )

    out_profile = None
    meta = node.meta.get("val")
    assert meta is not None
    if isinstance(meta, torch.Tensor):
        out_profile = TensorMetadata.maybe_from_tensor(meta)
    elif isinstance(meta, (list, tuple)):
        out_profile = tuple([TensorMetadata.maybe_from_tensor(m) for m in meta])  # type: ignore[assignment]
    assert out_profile is not None

    return OpProfile(args_profile, out_profile)  # type: ignore[arg-type]


def insert_custom_op_guards(gm: torch.fx.GraphModule, ops_to_guard: set[str]) -> None:
    """
    This is used by draft_export to insert guards in front of calls to custom
    operators which have a generated fake kernel.
    """
    for node in gm.graph.nodes:
        if node.op == "call_function" and str(node.target) in ops_to_guard:
            with _set_node_metadata_hook(
                gm,
                functools.partial(
                    _node_metadata_hook, stack_trace=node.meta.get("stack_trace")
                ),
            ), gm.graph.inserting_before(node):
                for arg in (*node.args, *node.kwargs.values()):
                    if isinstance(arg, torch.fx.Node) and isinstance(
                        arg.meta.get("val"), torch.Tensor
                    ):
                        val = arg.meta["val"]
                        gm.graph.call_function(
                            torch.ops.aten._assert_tensor_metadata.default,
                            args=(arg,),
                            kwargs={
                                "dtype": val.dtype,
                                "device": val.device,
                                "layout": val.layout,
                            },
                        )

    gm.recompile()


def get_custom_op_profiles(
    gm: torch.fx.GraphModule, ops_to_guard: set[str]
) -> dict[str, set[OpProfile]]:
    """
    This is used by draft_export to get a list of custom operator profiles so
    that we can generate fake kernels.
    """

    custom_op_profiles: dict[str, set[OpProfile]] = defaultdict(set)

    for node in gm.graph.nodes:
        if node.op == "call_function" and str(node.target) in ops_to_guard:
            custom_op_profiles[str(node.target)].add(get_op_profile(node))

    return custom_op_profiles


def generate_and_register_fake_kernels(op_profiles: dict[str, set[OpProfile]]) -> None:
    """
    Given the op profiles, we will generate a fake kernel which generates fake
    tensors with unbacked shapes based on the profiles we recorded beforehand.
    """

    def _match_args(args_profile: tuple[Optional[TensorMetadata]], args: Any) -> bool:
        return all(
            TensorMetadata.maybe_from_tensor(arg) == args_profile[i]
            for i, arg in enumerate(args)
        )

    def _generate_res(
        out_profile: Union[TensorMetadata, tuple[TensorMetadata]],
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        ctx = torch.library.get_ctx()

        def _generate_tensor_out(t: TensorMetadata) -> torch.Tensor:
            fake_shape = [ctx.new_dynamic_size() for _ in range(t.rank)]
            fake_strides = [-1] * t.rank
            expected = 1
            fake_stride = expected
            for i in range(t.rank):
                fake_strides[i] = fake_stride  # type: ignore[assignment]
                fake_stride = fake_stride * fake_shape[i]  # type: ignore[assignment]

            return torch.empty_strided(
                fake_shape,
                fake_strides,
                device=t.device,
                dtype=t.dtype,
                layout=t.layout,
            )

        if isinstance(out_profile, TensorMetadata):
            return _generate_tensor_out(out_profile)
        else:
            return [_generate_tensor_out(t) for t in out_profile]

    for op_name, profiles in op_profiles.items():
        log.warning("Generating fake kernel for %s", op_name)

        op_name_split = op_name.split(".")
        op_str = f"{op_name_split[0]}::{op_name_split[1]}"

        @torch.library.register_fake(op_str)
        def _(*args, **kwargs):  # type: ignore[no-untyped-def]
            for profile in profiles:
                if _match_args(profile.args_profile, (*args, *kwargs.values())):
                    return _generate_res(profile.out_profile)

            # No operator profiles match the existing inputs
            # If we are in the mode where we want to generate fake kernels if
            # there are any meta kernel mismatches, then we can return None and
            # have maybe_infer_fake update the result.
            if torch._functorch.config.generate_fake_kernels_from_real_mismatches:
                return None

            raise RuntimeError(
                f"No fake kernel was found for {op_name}, and although we have "
                "previously registered some profiles to generate a fake kernel, "
                f"no profiles match the given inputs: {args, kwargs}."
            )
