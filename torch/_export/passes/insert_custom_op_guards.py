import functools
from collections import defaultdict

import torch
from torch._export.passes._node_metadata_hook import (
    _node_metadata_hook,
    _set_node_metadata_hook,
)
from torch._library.fake_profile import OpProfile, TensorMetadata


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


def get_op_profiles(
    gm: torch.fx.GraphModule, ops_to_guard: set[str]
) -> dict[str, set[OpProfile]]:
    """
    This is used by draft_export to get a list of custom operator profiles so
    that we can generate fake kernels.
    """

    def _get_op_profile(node: torch.fx.Node) -> OpProfile:
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

    op_profiles: dict[str, set[OpProfile]] = defaultdict(set)

    for node in gm.graph.nodes:
        if node.op == "call_function" and str(node.target) in ops_to_guard:
            op_profiles[str(node.target)].add(_get_op_profile(node))

    return op_profiles
