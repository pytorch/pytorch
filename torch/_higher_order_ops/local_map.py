# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this file may be removed once we move to a dynamo frontend

import functools
from typing import Any, Callable, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._functorch._aot_autograd.graph_compile import prepare_for_partitioner
from torch._higher_order_ops.utils import (
    clone_outputs_aliasing_inputs,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
)
from torch._inductor.compile_fx import partition_fn
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tensor.experimental import local_map
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


class LocalMapHOP(HigherOrderOperator):
    """
    A HOP that integrates with autoparallel's current frontend (aot_export_module).
    This HOP exists starting the pre-solver graph and lives until we apply sharding.
    During which, orig_fwd will be inlined into the post-solver graph.
    """

    def __init__(self):
        super().__init__("local_map_hop")

    def __call__(self, fwd: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return super().__call__(fwd, *args, **kwargs)


local_map_hop = LocalMapHOP()


class LocalMapBackwardAOTExportModule(HigherOrderOperator):
    """
    For the backward of local_map_hop. To override the proxy key,
    and trace the whole bwd as a single node in the pre-solver graph.
    """

    def __init__(self):
        super().__init__("local_map_hop_backward")

    def __call__(self, bwd: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return super().__call__(bwd, *args, **kwargs)


local_map_hop_backward = LocalMapBackwardAOTExportModule()


def create_hop_fw_bw(
    orig_fwd: Callable[..., Any],
    *_args: Any,
) -> tuple[GraphModule, GraphModule, int]:
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
            def _from_fun(t):
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

            # redundant? we already _from_fun'd the inputs
            example_grads = pytree.tree_map(
                _from_fun,
                orig_fwd(*fw_inputs),
            )
            if not isinstance(example_grads, (list, tuple)):
                example_grads = [example_grads]

            num_fw_inputs = len(fw_inputs)
            num_fw_outputs = len(example_grads)

        def joint_f(
            *primals_and_tangents,
        ):
            primals = primals_and_tangents[:num_fw_inputs]
            tangents = primals_and_tangents[num_fw_inputs:]

            optional_grads = []
            for example_grad in example_grads:
                if example_grad.requires_grad:
                    optional_grads.append(example_grad)

            def prepare_fw_with_masks(fn):
                def fw_with_masks(*args):
                    fw_out = fn(*args)
                    assert isinstance(
                        fw_out, tuple
                    ), "apply_local_map'd functions must return tuples! This will be relaxed with a dynamo frontend."
                    return fw_out, [
                        True
                        if isinstance(ret, torch.Tensor) and ret.requires_grad
                        else False
                        for ret in fw_out
                    ]

                return fw_with_masks

            fw_outs, grads = create_joint(
                prepare_fw_with_masks(orig_fwd), aot_config=dummy_aot_config
            )(primals, tangents)

            maybe_clone = clone_outputs_aliasing_inputs(primals_and_tangents)
            # put grads first to work with existing hop utils
            return pytree.tree_map(maybe_clone, tuple([*grads, *fw_outs]))

        primals_and_tangents = [*fw_inputs, *example_grads]
        joint_hop_gm = make_fx(joint_f)(*primals_and_tangents)
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
        if "custom" not in new_fw_gm.meta:
            new_fw_gm.meta["custom"] = {}
        if "custom" not in new_bw_gm.meta:
            new_bw_gm.meta["custom"] = {}
        local_map_kwargs = orig_fwd.local_map_kwargs  # type: ignore[attr-defined]
        new_fw_gm.meta["custom"]["local_map_kwargs"] = local_map_kwargs
        new_bw_gm.meta["custom"]["local_map_kwargs"] = {
            # Okay because Autoparallel assumes same sharding between param and grads
            "in_placements": local_map_kwargs["out_placements"],
            "out_placements": local_map_kwargs["in_placements"],
        }

        return new_fw_gm, new_bw_gm, num_fw_outputs


class LocalMapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        fw_gm: GraphModule,
        bw_gm: GraphModule,
        num_fw_outs: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Optional[torch.Tensor], ...]:
        ctx.bw_gm = bw_gm

        with torch._C._AutoDispatchBelowAutograd():
            fw_outs_with_saved_activations = local_map_hop(fw_gm, *args, **kwargs)

        fw_outs = fw_outs_with_saved_activations[:num_fw_outs]
        saved_activations = fw_outs_with_saved_activations[num_fw_outs:]
        save_tensors_and_symints_for_backward(ctx, saved_activations)

        return fw_outs

    @staticmethod
    def backward(
        ctx: Any, *grads: tuple[torch.Tensor]
    ) -> tuple[Optional[torch.Tensor], ...]:
        # TODO: prevent double backward
        saved_activations = saved_tensors_and_symints(ctx)
        with torch._C._AutoDispatchBelowAutograd():
            grad_ins = local_map_hop_backward(ctx.bw_gm, *saved_activations, *grads)
        return None, None, None, *grad_ins


@local_map_hop.py_impl(torch._C.DispatchKey.Autograd)
def autograd_key(
    orig_fwd: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    if "_inline" in kwargs:
        # Solver pass adds a _inline kwarg, which tells this hop to desugar on the next trace
        del kwargs["_inline"]
        return orig_fwd(*args, **kwargs)

    fw_gm, bw_gm, num_fw_outs = create_hop_fw_bw(orig_fwd, *args)
    return LocalMapAutogradOp.apply(fw_gm, bw_gm, num_fw_outs, *args, **kwargs)


@local_map_hop.py_functionalize_impl
def functional_mode_key(
    ctx: Any, fw_gm: GraphModule, *args: Any, **kwargs: Any
) -> tuple[torch.Tensor]:
    assert not kwargs

    unwrapped_inputs = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        # TODO: dynamo safety checks on fw_gm
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
    hop: Union[LocalMapHOP, LocalMapBackwardAOTExportModule],
    call_hop: Callable[..., Any],
    proxy_mode: ProxyTorchDispatchMode,
    gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    assert (
        proxy_mode is not None
    ), "Mode should always be enabled for python fallback key"
    assert len(kwargs) == 0

    example_out = hop(gm, *args, **kwargs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)  # type: ignore[union-attr]

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", call_hop, proxy_args, {}
    )
    # Propagate meta for proxied node
    if proxy_mode.pre_dispatch:
        # autograd key didn't run yet, gm is still a func
        assert gm.local_map_kwargs
        local_map_kwargs = gm.local_map_kwargs
    else:
        # post-dispatch, gm is already a graph module
        assert gm.meta["custom"]["local_map_kwargs"]
        local_map_kwargs = gm.meta["custom"][
            "local_map_kwargs"
        ]
    if "custom" not in out_proxy.node.meta:
        out_proxy.node.meta["custom"] = {}
    out_proxy.node.meta["custom"]["local_map_kwargs"] = local_map_kwargs
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
    def call_local_map(*another_args, **another_kwargs):
        return functools.partial(local_map_hop, fw_gm)(*another_args, **another_kwargs)

    return proxy_mode_key_common(
        local_map_hop, call_local_map, proxy_mode, fw_gm, *args, **kwargs
    )


# Running HOP in eager with real tensors
@local_map_hop.py_impl(torch._C.DispatchKey.CPU)
@local_map_hop.py_impl(torch._C.DispatchKey.CUDA)
def real_impl(
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    return fw_gm(*args, **kwargs)


@local_map_hop_backward.py_impl(torch._C.DispatchKey.Autograd)
def bw_autograd_key(
    bw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    if "_inline" in kwargs:
        # Solver pass adds a _inline kwarg, which tells this hop to desugar on the next trace
        del kwargs["_inline"]
        # TODO: prevent double backward
        return bw_gm(*args, **kwargs)

    assert False, "Shouldn't get here"


@local_map_hop_backward.py_functionalize_impl
def bw_functional_mode_key(
    ctx: Any, bw_gm: GraphModule, *args: Any, **kwargs: Any
) -> tuple[torch.Tensor]:
    assert not kwargs

    unwrapped_inputs = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        # TODO: dynamo safety checks on bw_gm
        out = local_map_hop_backward(bw_gm, *unwrapped_inputs)
        return ctx.wrap_tensors(out)


@local_map_hop_backward.py_impl(FakeTensorMode)
def bw_fake_mode_key(
    mode: FakeTensorMode,
    bw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    with mode:
        return bw_gm(*args, **kwargs)


@local_map_hop_backward.py_impl(ProxyTorchDispatchMode)
def bw_proxy_mode_key(
    proxy_mode: ProxyTorchDispatchMode,
    bw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    def call_local_map_backward(*another_args, **another_kwargs):
        return functools.partial(local_map_hop_backward, bw_gm)(
            *another_args, **another_kwargs
        )

    return proxy_mode_key_common(
        local_map_hop_backward,
        call_local_map_backward,
        proxy_mode,
        bw_gm,
        *args,
        **kwargs,
    )


# TODO: figure out what's going on with the error here
# @local_map_hop.py_impl(torch._C.DispatchKey.CPU)
# @local_map_hop.py_impl(torch._C.DispatchKey.CUDA)
# def bw_real_impl(
#     bw_gm: GraphModule,
#     *args: Any,
#     **kwargs: Any,
# ) -> tuple[torch.Tensor]:
#     return bw_gm(*args, **kwargs)


def apply_local_map(*local_map_args, **local_map_kwargs):
    # NOTE: We manually issue the hop, which will not be not necessary with a dynamo frontend.
    # 1. Same as local_map, must be applied on a function, not a method.
    # 2. the local_map'd function must be make_fx traceable. Otherwise, we may
    # inline the wrong graph. In a dynamo frontend, speculate_subgraph will handle this.
    # 3. All inputs to the local_map'd function must be Tensor types. Otherwise, we won't
    # know which tensors to apply _from_fun to. For instance, don't pass nn.Modules to local_map.
    # In dynamo frontend, tensors will be lifted, and will modify the wrapped function's signature.

    assert len(local_map_args) == 0, "Please pass as kwargs, no Dynamo frontend"
    assert local_map_kwargs[
        "redistribute_inputs"
    ], "Autoparallel should always be allowed to redistribute inputs"
    assert local_map_kwargs["in_grad_placements"] is None, "Not yet implemented"
    assert local_map_kwargs["device_mesh"] is None, "Must be provided by Autoparallel"
    assert local_map_kwargs["in_placements"] is not None
    assert local_map_kwargs["out_placements"] is not None
    assert len(local_map_kwargs) == 5, "Got unexpected kwargs, no Dynamo frontend"

    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            def orig_fwd(*runtime_args, **runtime_kwargs):
                # wrap the functools.partial for hop utils to work out of box
                return local_map(
                    fn,
                    *local_map_args,
                    **local_map_kwargs,
                )(*runtime_args, **runtime_kwargs)

            orig_fwd.local_map_kwargs = local_map_kwargs
            return local_map_hop(orig_fwd, *args, **kwargs)

        return wrapped

    return decorator
