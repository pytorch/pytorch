# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this file may be removed once we move to a dynamo frontend

import functools
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    clone_outputs_aliasing_inputs,
    redirect_to_mode,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


# Proxy the HOP instead of inlining into it
_DEFER_INLINING = False


@contextmanager
def defer_inlining() -> Generator[None, None, None]:
    global _DEFER_INLINING
    prior = _DEFER_INLINING
    try:
        _DEFER_INLINING = True
        yield
    finally:
        _DEFER_INLINING = prior


class LocalMapHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("local_map_hop")

    def __call__(self, fw_gm: GraphModule, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(fw_gm, *args, **kwargs)


local_map_hop = LocalMapHOP()

# Registers dispatches for SAC
redirect_to_mode(local_map_hop, _CachingTorchDispatchMode)
redirect_to_mode(local_map_hop, _CachedTorchDispatchMode)


def create_hop_fw_bw(
    fw_gm: GraphModule,
    *_args: Any,
) -> tuple[GraphModule, GraphModule, int, int, set[int]]:
    """
    Traces a joint, applies passes and partitions it
    """
    # Keeping these imports here
    # Avoid circular dependencies once we upstream with dynamo frontend
    from torch._dispatch.python import suspend_functionalization
    from torch._functorch.aot_autograd import AOTConfig, create_joint
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing, make_fx

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # create a tensor (fake) from a compiler wrapped FunctionalTensor
            def _from_fun(t: Any) -> Any:
                if isinstance(t, torch.Tensor):
                    return torch.empty_strided(
                        t.size(),
                        t.stride(),
                        device=t.device,
                        dtype=t.dtype,
                        requires_grad=t.requires_grad,
                    )
                return t

            # If someone runs this hop under the default compiler backend ("eager")
            # Then this path will be run with the actual user inputs. We convert them
            # to fake tensors in order to not perform any actual compute.

            fake_mode = detect_fake_mode(_args)
            if fake_mode is None:
                fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

            with fake_mode:
                fw_inputs = pytree.tree_map(_from_fun, _args)

            assert all(
                isinstance(t, (FakeTensor, int, torch.SymInt)) for t in fw_inputs
            ), f"Unexpected element in {fw_inputs=}"

            example_grads = pytree.tree_map(
                _from_fun,
                fw_gm(*fw_inputs),
            )
            if not isinstance(example_grads, (list, tuple)):
                example_grads = [example_grads]

            num_fw_inputs = len(fw_inputs)
            num_fw_outputs = len(example_grads)

        def joint_f(
            *primals_and_tangents: list[torch.Tensor],
        ) -> Any:
            primals = primals_and_tangents[:num_fw_inputs]
            tangents = primals_and_tangents[num_fw_inputs:]

            def prepare_fw_with_masks(fn: Callable[..., Any]) -> Callable[..., Any]:
                def fw_with_masks(*args: Any) -> tuple[tuple[Any], list[bool]]:
                    fw_out = fn(*args)
                    assert isinstance(fw_out, tuple), (
                        "Dynamo traced submodule should return tuple"
                    )
                    return fw_out, [
                        bool(isinstance(ret, torch.Tensor) and ret.requires_grad)
                        for ret in fw_out
                    ]

                return fw_with_masks

            fw_outs, grads = create_joint(
                prepare_fw_with_masks(fw_gm), aot_config=dummy_aot_config
            )(primals, tangents)

            maybe_clone = clone_outputs_aliasing_inputs(primals_and_tangents)
            # put grads first to work with existing hop utils
            return pytree.tree_map(maybe_clone, (*grads, *fw_outs))

        filtered_grads_idx = set()
        for i, example_grad in enumerate(example_grads):
            # Filter out grads that are None or do not require_grad.
            # The AOTAutograd utils we rely on force this assumption.
            # We must also filter the runtime tangents too.
            if example_grad is not None and (
                isinstance(example_grad, torch.Tensor) and example_grad.requires_grad
            ):
                filtered_grads_idx.add(i)

        primals_and_tangents = [
            *fw_inputs,
            *[example_grads[i] for i in filtered_grads_idx],
        ]
        joint_hop_gm = make_fx(joint_f)(*primals_and_tangents)

        from torch._functorch._aot_autograd.graph_compile import prepare_for_partitioner
        from torch._inductor.compile_fx import partition_fn

        # Match partitioner convention
        prepped_joint_hop_gm = prepare_for_partitioner(
            joint_hop_gm, num_fw_inputs, num_fw_outputs
        )
        # Also runs joint passes
        new_fw_gm, new_bw_gm = partition_fn(
            prepped_joint_hop_gm,
            [],
            num_fwd_outputs=num_fw_outputs,
            static_lifetime_input_indices=[],
        )

        # Propagate meta onto fw/bw graphs, later will be set on proxied nodes
        local_map_kwargs = fw_gm.meta["local_map_kwargs"]  # type: ignore[attr-defined]

        new_fw_gm.meta["local_map_kwargs"] = local_map_kwargs
        new_bw_gm.meta["local_map_kwargs"] = {**local_map_kwargs}
        # Okay because Autoparallel assumes same sharding between param and grads
        new_bw_gm.meta["local_map_kwargs"]["in_placements"] = local_map_kwargs[
            "out_placements"
        ]
        new_bw_gm.meta["local_map_kwargs"]["out_placements"] = local_map_kwargs[
            "in_placements"
        ]

        return new_fw_gm, new_bw_gm, num_fw_inputs, num_fw_outputs, filtered_grads_idx


class LocalMapAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore  # bad-override
    def forward(
        ctx: Any,
        fw_gm: GraphModule,
        bw_gm: GraphModule,
        num_fw_ins: int,
        num_fw_outs: int,
        filtered_grads_idx: set[int],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Optional[torch.Tensor], ...]:
        from torch._functorch._aot_autograd.schemas import MemoryFormatMeta

        ctx.bw_gm = bw_gm
        ctx.num_fw_ins = num_fw_ins
        ctx.filtered_grads_idx = filtered_grads_idx

        with torch._C._AutoDispatchBelowAutograd():
            fw_outs_with_saved_activations = local_map_hop(fw_gm, *args, **kwargs)

        fw_outs = fw_outs_with_saved_activations[:num_fw_outs]
        saved_activations = fw_outs_with_saved_activations[num_fw_outs:]
        save_tensors_and_symints_for_backward(ctx, saved_activations)

        ctx.expected_tangent_metadata = {
            i: MemoryFormatMeta.from_tensor(fw_outs[i]) for i in filtered_grads_idx
        }
        return fw_outs

    @staticmethod
    def backward(
        ctx: Any, *_grads: tuple[torch.Tensor]
    ) -> tuple[Optional[torch.Tensor], ...]:
        from torch._functorch._aot_autograd.runtime_wrappers import (
            coerce_to_expected_memory_format,
        )

        saved_activations = saved_tensors_and_symints(ctx)
        with torch._C._AutoDispatchBelowAutograd():
            # Filter out grads that are None or do not require_grad.
            # The AOTAutograd utils we rely on force this assumption.
            grads = [_grads[i] for i in ctx.filtered_grads_idx]
            assert len(grads) == len(ctx.expected_tangent_metadata), (
                f"{len(grads)=} vs {len(ctx.expected_tangent_metadata)}"
            )

            for i, meta in ctx.expected_tangent_metadata.items():
                # pyrefly: ignore  # bad-argument-type
                grads[i] = coerce_to_expected_memory_format(grads[i], meta)

            grad_ins = local_map_hop(ctx.bw_gm, *saved_activations, *grads)
            if len(grad_ins) != ctx.num_fw_ins:
                raise RuntimeError(
                    f"Expected {ctx.num_fw_ins} grad_ins, got {len(grad_ins)}"
                )
        return None, None, None, None, None, *grad_ins


@local_map_hop.py_impl(torch._C.DispatchKey.Autograd)
def autograd_key(
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if _DEFER_INLINING:
        fw_gm, bw_gm, num_fw_ins, num_fw_outs, filtered_grads_idx = create_hop_fw_bw(
            fw_gm, *args
        )
        return LocalMapAutogradOp.apply(
            fw_gm, bw_gm, num_fw_ins, num_fw_outs, filtered_grads_idx, *args, **kwargs
        )

    return fw_gm(*args, **kwargs)


@local_map_hop.py_functionalize_impl
def functional_mode_key(
    ctx: Any, fw_gm: GraphModule, *args: Any, **kwargs: Any
) -> tuple[torch.Tensor]:
    assert not kwargs

    unwrapped_inputs = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        out = local_map_hop(fw_gm, *unwrapped_inputs)
        return ctx.wrap_tensors(out)


@local_map_hop.py_impl(FakeTensorMode)
def fake_mode_key(
    mode: FakeTensorMode,
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    with mode:
        return fw_gm(*args, **kwargs)


def proxy_mode_key_common(
    call_hop: Callable[..., Any],
    proxy_mode: ProxyTorchDispatchMode,
    gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    assert proxy_mode is not None, (
        "Mode should always be enabled for python fallback key"
    )
    assert len(kwargs) == 0

    example_out = call_hop(*args, **kwargs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)  # type: ignore[union-attr]

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", call_hop, proxy_args, {}
    )

    # extract local_map args, post-dispatch operates on GraphModules
    assert gm.meta["local_map_kwargs"]
    local_map_kwargs = gm.meta["local_map_kwargs"]

    # propagate local_map args to the call_function node
    out_proxy.node.meta["local_map_kwargs"] = local_map_kwargs
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@local_map_hop.py_impl(ProxyTorchDispatchMode)
def proxy_mode_key(
    proxy_mode: ProxyTorchDispatchMode,
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    # TODO: get rid of this when we can install as a subgraph
    def call_local_map(*_args: Any, **_kwargs: Any) -> Any:
        return functools.partial(local_map_hop, fw_gm)(*_args, **_kwargs)

    return proxy_mode_key_common(call_local_map, proxy_mode, fw_gm, *args, **kwargs)


# Running HOP in eager with real tensors
@local_map_hop.py_impl(DispatchKey.CompositeExplicitAutograd)
def real_impl(
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    return fw_gm(*args, **kwargs)
