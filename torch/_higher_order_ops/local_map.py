# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this file may be removed once we move to a dynamo frontend

import functools
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable, Optional

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

# Used to unwrap tensors classes like FunctionalTensor and Parameter
def _clone(t: Any) -> Any:
    if isinstance(t, torch.Tensor):
        return torch.empty_strided(
            t.size(),
            t.stride(),
            device=t.device,
            dtype=t.dtype,
            requires_grad=t.requires_grad,
        )
    return t

# Autoparallel specific, we want to treat plain tensors as DTensors
def redistribute_inputs(args: Any, local_map_kwargs: dict[str, Any]) -> tuple[torch.Tensor, int, torch.SymInt]:
    from torch._guards import detect_fake_mode
    from torch._dispatch.python import suspend_functionalization
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    with suspend_functionalization(), disable_functional_mode(), disable_proxy_modes_tracing():
        # If someone runs this hop under the default compiler backend ("eager")
        # Then this path will be run with the actual user inputs. We convert them
        # to fake tensors in order to not perform any actual compute.

        fake_mode = detect_fake_mode(args)
        if fake_mode is None:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        with fake_mode:
            fw_inputs = list(pytree.tree_map(_clone, args))
            assert len(fw_inputs) == len(local_map_kwargs["in_placements"])
            for i, (global_tensor, placements) in enumerate(zip(fw_inputs, local_map_kwargs["in_placements"])):
                # why can't i use distribute_tensor on replicate()?
                if not any([not p.is_replicate() for p in placements]):
                    continue

                from torch.distributed._tensor import distribute_tensor
                temp = distribute_tensor(
                    global_tensor.detach().requires_grad_(global_tensor.requires_grad),
                    local_map_kwargs["device_mesh"],
                    placements,
                    src_data_rank=None,
                )
                fw_inputs[i] = temp._local_tensor

            fw_inputs = tuple(fw_inputs)
            assert all(
                isinstance(t, (FakeTensor, int, torch.SymInt)) for t in fw_inputs
            ), f"Unexpected element in {args=}"

    return fw_inputs

def redistribute_outputs(args: Any, local_map_kwargs: dict[str, Any]) -> tuple[torch.Tensor, int, torch.SymInt]:
    from torch._guards import detect_fake_mode
    from torch._dispatch.python import suspend_functionalization
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.distributed.tensor._utils import compute_global_tensor_info

    with suspend_functionalization(), disable_functional_mode(), disable_proxy_modes_tracing():
        # If someone runs this hop under the default compiler backend ("eager")
        # Then this path will be run with the actual user inputs. We convert them
        # to fake tensors in order to not perform any actual compute.

        fake_mode = detect_fake_mode(args)
        if fake_mode is None:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        with fake_mode:
            fw_outs = list(pytree.tree_map(_clone, args))
            assert len(fw_outs) == len(local_map_kwargs["out_placements"])
            for i, (local_tensor, placements) in enumerate(zip(fw_outs, local_map_kwargs["out_placements"])):
                if not any([not p.is_replicate() for p in placements]):
                    continue

                global_shape, global_stride = compute_global_tensor_info(
                    local_tensor, local_map_kwargs["device_mesh"], placements
                )

                global_tensor = torch.empty_strided(
                    global_shape,
                    global_stride,
                    dtype=local_tensor.dtype,
                    device=local_tensor.device,
                    requires_grad=local_tensor.requires_grad,
                )

                fw_outs[i] = global_tensor

    return fw_outs

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
    *fw_inputs: Any,
) -> tuple[GraphModule, GraphModule, int, int, set[int]]:
    """
    Traces a joint, applies passes and partitions it
    """
    # Keeping these imports here
    # Avoid circular dependencies once we upstream with dynamo frontend
    from torch._dispatch.python import suspend_functionalization
    from torch._functorch.aot_autograd import AOTConfig, create_joint
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
            example_grads = pytree.tree_map(
                _clone,
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
                        True
                        if isinstance(ret, torch.Tensor) and ret.requires_grad
                        else False
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
        new_bw_gm.meta["filtered_grads_idx"] = filtered_grads_idx

        # Propagate meta onto fw/bw graphs, later will be set on proxied nodes
        local_map_kwargs = fw_gm.meta["local_map_kwargs"]
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


# class LocalMapAutogradOp(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx: Any,
#         fw_gm: GraphModule,
#         bw_gm: GraphModule,
#         num_fw_ins: int,
#         num_fw_outs: int,
#         filtered_grads_idx: set[int],
#         *args: Any,
#         **kwargs: Any,
#     ) -> tuple[Optional[torch.Tensor], ...]:
#         from torch._functorch._aot_autograd.schemas import MemoryFormatMeta

#         ctx.bw_gm = bw_gm
#         ctx.num_fw_ins = num_fw_ins
#         ctx.filtered_grads_idx = filtered_grads_idx

#         with torch._C._AutoDispatchBelowAutograd():
#             fw_outs_with_saved_activations = local_map_hop(fw_gm, *args, **kwargs)

#         fw_outs = fw_outs_with_saved_activations[:num_fw_outs]
#         saved_activations = fw_outs_with_saved_activations[num_fw_outs:]
#         save_tensors_and_symints_for_backward(ctx, saved_activations)

#         ctx.expected_tangent_metadata = {
#             i: MemoryFormatMeta.from_tensor(fw_outs[i]) for i in filtered_grads_idx
#         }
#         return fw_outs

#     @staticmethod
#     def backward(
#         ctx: Any, *_grads: tuple[torch.Tensor]
#     ) -> tuple[Optional[torch.Tensor], ...]:
#         from torch._functorch._aot_autograd.runtime_wrappers import (
#             coerce_to_expected_memory_format,
#         )

#         saved_activations = saved_tensors_and_symints(ctx)
#         with torch._C._AutoDispatchBelowAutograd():
#             # Filter out grads that are None or do not require_grad.
#             # The AOTAutograd utils we rely on force this assumption.
#             grads = [_grads[i] for i in ctx.filtered_grads_idx]
#             assert len(grads) == len(ctx.expected_tangent_metadata), (
#                 f"{len(grads)=} vs {len(ctx.expected_tangent_metadata)}"
#             )

#             for i, meta in ctx.expected_tangent_metadata.items():
#                 grads[i] = coerce_to_expected_memory_format(grads[i], meta)

#             grad_ins = local_map_hop(ctx.bw_gm, *saved_activations, *grads)
#             if len(grad_ins) != ctx.num_fw_ins:
#                 raise RuntimeError(
#                     f"Expected {ctx.num_fw_ins} grad_ins, got {len(grad_ins)}"
#                 )
#         return None, None, None, None, None, *grad_ins


@local_map_hop.py_impl(torch._C.DispatchKey.Autograd)
def autograd_key(
    fw_gm: GraphModule,
    *fw_inputs: Any,
    **kwargs: Any,
) -> Any:
    assert isinstance(fw_gm, GraphModule)
    if "proxied_fake_tensors" in fw_gm.meta:
        del fw_gm.meta["proxied_fake_tensors"]

    if not _DEFER_INLINING:
        # if fw_gm.meta["is_backward"]:
        return fw_gm(*fw_inputs, **kwargs)

    from torch._guards import detect_fake_mode
    from torch._dispatch.python import suspend_functionalization
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    class LocalMapAutogradOpWithRedistribute(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fw_gm, *fw_inputs, **kwargs):
            print("KEY: AUTOGRAD")
            # if fw_gm.meta.get("is_backward", False):
            #     torch.distributed.breakpoint()
            # TODO: assert is flat inputs
            global_inputs = fw_inputs
            local_inputs_with_grad = redistribute_inputs(fw_inputs, fw_gm.meta["local_map_kwargs"])
            assert len(global_inputs) == len(local_inputs_with_grad)
            for a, b in zip(global_inputs, local_inputs_with_grad):
                assert a.requires_grad == b.requires_grad
            # for t in local_inputs_with_grad:
            #     assert t.grad_fn is not None
            fw_inputs = local_inputs_with_grad
            fw_gm, bw_gm, num_fw_ins, num_fw_outs, filtered_grads_idx = create_hop_fw_bw(
                fw_gm, *fw_inputs
            )
            # for t in fw_inputs:
            #     assert t.grad_fn is not None
            fw_gm.meta["proxied_fake_tensors"] = global_inputs
            assert fw_gm.meta["proxied_fake_tensors"][0].shape[0] == 64

            fw_gm.meta["is_backward"] = False
            bw_gm.meta["is_backward"] = True
            bw_gm.meta["num_fw_outs"] = num_fw_outs

            from torch._logging import trace_structured
            assert isinstance(fw_gm, torch.fx.GraphModule)
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "hop_fw_gm",
                    "encoding": "string",
                },
                payload_fn=lambda: fw_gm.print_readable(
                    print_output=False,
                    include_stride=True,
                    include_device=True,
                    expanded_def=True,
                ),
            )
            assert isinstance(bw_gm, torch.fx.GraphModule)
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "hop_bw_gm",
                    "encoding": "string",
                },
                payload_fn=lambda: bw_gm.print_readable(
                    print_output=False,
                    include_stride=True,
                    include_device=True,
                    expanded_def=True,
                ),
            )

            # out = LocalMapAutogradOp.apply(
            #     fw_gm, bw_gm, num_fw_ins, num_fw_outs, filtered_grads_idx, *fw_inputs, **kwargs
            # )
            # LocalMapAutogradOp
            from torch._functorch._aot_autograd.schemas import MemoryFormatMeta

            ctx.bw_gm = bw_gm
            ctx.num_fw_ins = num_fw_ins
            ctx.filtered_grads_idx = filtered_grads_idx

            with torch._C._AutoDispatchBelowAutograd():
                fw_outs_with_saved_activations = local_map_hop(fw_gm, *fw_inputs, **kwargs)

            fw_outs = fw_outs_with_saved_activations[:num_fw_outs]
            # fw_outs is expected to have requires_grad properly...
            saved_activations = fw_outs_with_saved_activations[num_fw_outs:]
            print(f"NUMBER OF SAVED ACTIVATIONS: {len(saved_activations)}")
            save_tensors_and_symints_for_backward(ctx, saved_activations)

            ctx.expected_tangent_metadata = {
                i: MemoryFormatMeta.from_tensor(fw_outs[i]) for i in filtered_grads_idx
            }
            # LocalMapAutogradOp
            out = fw_outs

            # assert any([t.grad_fn is not None for t in out])
            global_outputs_no_grad = redistribute_outputs(out, fw_gm.meta["local_map_kwargs"])
            out = global_outputs_no_grad
            del fw_gm.meta["proxied_fake_tensors"]
            if "override_tracked_output_fn" in fw_gm.meta:
                out = fw_gm.meta["override_tracked_output_fn"](out)
                del fw_gm.meta["override_tracked_output_fn"]
            return tuple(out)

        @staticmethod
        def backward(ctx, *global_grad_outs):
            # convert
            bw_gm = ctx.bw_gm
            bw_gm.meta["proxied_fake_tensors"] = [global_grad_outs[i] for i in ctx.filtered_grads_idx]
            # [rank0]:(Pdb++) [rank0]:(FunctionalTensor(_to_functional_tensor(FakeTensor(..., device='cuda:0', size=(64, 2048, 256)),
            # [rank0]:       device='cuda:0')),
            # [rank0]: FunctionalTensor(_to_functional_tensor(FakeTensor(..., device='cuda:0', size=(8,), dtype=torch.int64),
            # [rank0]:       device='cuda:0')))

            # torch.distributed.breakpoint()
            local_inputs = redistribute_inputs(global_grad_outs, bw_gm.meta["local_map_kwargs"])
            assert len(global_grad_outs) == len(local_inputs)
            _grads = local_inputs

            # local shapes region
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
                    grads[i] = coerce_to_expected_memory_format(grads[i], meta)

                bw_gm.meta["num_saved_activations"] = len(saved_activations)
                # torch.distributed.breakpoint()
                grad_ins = local_map_hop(ctx.bw_gm, *saved_activations, *grads)
                if len(grad_ins) != ctx.num_fw_ins:
                    raise RuntimeError(
                        f"Expected {ctx.num_fw_ins} grad_ins, got {len(grad_ins)}"
                    )
            # local shapes region end
            grad_ins = redistribute_outputs(grad_ins, bw_gm.meta["local_map_kwargs"])
            del bw_gm.meta["proxied_fake_tensors"]
            if "override_tracked_output_fn" in bw_gm.meta:
                grad_ins = bw_gm.meta["override_tracked_output_fn"](grad_ins)
                del bw_gm.meta["override_tracked_output_fn"]

            return None, *grad_ins

    out = LocalMapAutogradOpWithRedistribute.apply(fw_gm, *fw_inputs, **kwargs)
    return out


@local_map_hop.py_functionalize_impl
def functional_mode_key(
    ctx: Any, fw_gm: GraphModule, *args: Any, **kwargs: Any
) -> tuple[torch.Tensor]:
    fw_inputs = ctx.unwrap_tensors(args)
    if _DEFER_INLINING:
        print("KEY: FUNC")
        assert "local_map_kwargs" in fw_gm.meta
        assert "proxied_fake_tensors" in fw_gm.meta
        unwrapped_proxied_fake_tensors = []
        for func_tensor in fw_gm.meta["proxied_fake_tensors"]:
            unwrapped_proxied_fake_tensors.append(ctx.unwrap_tensors(func_tensor))
        fw_gm.meta["proxied_fake_tensors"] = tuple(unwrapped_proxied_fake_tensors)

    assert not kwargs
    with ctx.redispatch_to_next():
        out = local_map_hop(fw_gm, *fw_inputs)
        return ctx.wrap_tensors(out)


@local_map_hop.py_impl(FakeTensorMode)
def fake_mode_key(
    mode: FakeTensorMode,
    fw_gm: GraphModule,
    *fw_inputs: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    if _DEFER_INLINING:
        print("KEY: FAKE")
        # assert fw_gm.meta["proxied_fake_tensors"][0].shape[0] == 64
        assert "proxied_fake_tensors" in fw_gm.meta
    with mode:
        return fw_gm(*fw_inputs, **kwargs)


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
    if proxy_mode.pre_dispatch:
        for proxy in proxy_args:
            assert not isinstance(proxy, torch.Tensor), f"Found tensor untracked by active pre-dispatch proxy mode: {proxy}"
    else:
        assert "proxied_fake_tensors" in gm.meta
        # assert gm.meta["proxied_fake_tensors"][0].shape[0] == 64
        # So because we traced with local shapes, we created new fake tensors during the Autograd key.
        # But the original tensors were bound with proxies. Since this local shape shenanigan is temporary
        # and lasts for the duration of the Autograd dispatch, we just swap the proxies, tricking the tracer
        # to continue tracing normally, despite this HOP body using different shapes.

        # this step is no longer true! we have proxies due to grad_fn=<SwapGlobalInputsForLocalInputsBackward>
        # for proxy in pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args):  # type: ignore[union-attr]
        #     assert not isinstance(proxy, torch.fx.Proxy), f"Unexpectedly found a tracked tensor. They should have been cloned at the Autograd key."
        assert "proxied_fake_tensors" in gm.meta
        # skips the hacky nodes we added in Autograd key for grad propagation purposes

        global_proxies = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, gm.meta["proxied_fake_tensors"])  # type: ignore[union-attr]
        if "num_fw_outs" in gm.meta:
            # is backward
            # bw has activations as inputs too, and they are at the front
            # num_fw_outs = gm.meta["num_fw_outs"]
            assert (gm.meta["num_saved_activations"] + len(global_proxies)) == len(args)
            proxy_args = (*proxy_args[:gm.meta["num_saved_activations"]], *global_proxies,)
        else:
            # is forward
            assert len(proxy_args) == len(args)
            proxy_args = global_proxies
        for proxy in proxy_args:
            assert not isinstance(proxy, torch.Tensor), f"Found tensor untracked by active post-dispatch proxy mode: {proxy}"

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", call_hop, proxy_args, {}
    )

    # extract local_map args, post-dispatch operates on GraphModules
    assert gm.meta["local_map_kwargs"]
    local_map_kwargs = gm.meta["local_map_kwargs"]

    # propagate local_map args to the call_function node
    out_proxy.node.meta["local_map_kwargs"] = local_map_kwargs
    out_proxy.node.meta["debug_gm"] = gm
    out_proxy.node.meta["debug_bwd_hop"] = "num_fw_outs" in gm.meta
    out_proxy.node.meta["num_activation_inputs"] = gm.meta.get("num_saved_activations", 0)
    out_proxy.node.meta["filtered_grads_idx"] = gm.meta.get("filtered_grads_idx", None)

    if not proxy_mode.pre_dispatch:
        def override_tracked_output_fn(new_out):
            return track_tensor_tree(
                new_out, out_proxy, constant=None, tracer=proxy_mode.tracer
            )

        gm.meta["override_tracked_output_fn"] = override_tracked_output_fn
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
    if _DEFER_INLINING:
        print("KEY: PROXY")

    # TODO: get rid of this when we can install as a subgraph
    def call_local_map(*_args: Any, **_kwargs: Any) -> Any:
        return functools.partial(local_map_hop, fw_gm)(*_args, **_kwargs)

    call_local_map.gm = fw_gm

    out = proxy_mode_key_common(call_local_map, proxy_mode, fw_gm, *args, **kwargs)
    return out


# Running HOP in eager with real tensors
@local_map_hop.py_impl(DispatchKey.CompositeExplicitAutograd)
def real_impl(
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    # do i need something here?
    return fw_gm(*args, **kwargs)
