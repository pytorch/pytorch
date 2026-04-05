from typing import TYPE_CHECKING

from .autograd import WrapHigherOrderVariable
from .common import (
    _assert_tensors_nonaliasing,
    _call_function_with_auto_output_flattening,
    Any,
    contextlib,
    Generator,
    get_fake_value,
    make_attr,
    pytree,
    Sequence,
    torch,
    TorchHigherOrderOperatorVariable,
    TupleVariable,
    unimplemented,
    variables,
    VariableTracker,
)


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


class ExecutorchCallDelegateHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.executorch_call_delegate"

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..builder import wrap_fx_proxy

        # This is operator for delegation within Executorch which calls a
        # specific function in the given lowered module with the given
        # operators. The actual operator is defined in the Executorch codebase.
        # This is a bad hierarchical violation since
        # executorch_call_delegate sits at a higher level than dynamo, but
        # there's no real solution to this issue yet.
        if len(kwargs) > 0:
            unimplemented(
                gb_type="executorch_call_delegate: kwargs not supported",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"executorch_call_delegate expects no keyword arguments (got {len(kwargs)})",
                hints=[],
            )
        lowered_module, lowered_node = None, None
        if isinstance(args[0], variables.NNModuleVariable):
            lowered_module = tx.output.get_submodule(args[0].module_key)
            lowered_node = make_attr(tx, args[0].module_key)
        elif isinstance(args[0], variables.UnspecializedNNModuleVariable):
            # This nn module is special sa delegated by executorch. Just
            # install it as a attr in the graph.
            lowered_module = args[0].value
            lowered_node = tx.output.register_static_attr_and_return_proxy(
                "delegate", lowered_module
            )
        else:
            unimplemented(
                gb_type="executorch_call_delegate: first arg not supported",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"executorch_call_delegate expects the first argument to be a nn.Module (got {args[0]})",
                hints=[],
            )

        p_args = tuple(arg.as_proxy() for arg in args[1:])
        real_sub_args = pytree.tree_map_only(
            torch.fx.Proxy, lambda a: get_fake_value(a.node, tx), p_args
        )
        assert tx.fake_mode is not None
        with tx.fake_mode:
            example_value = lowered_module.original_module.module()(*real_sub_args)  # type: ignore[attr-defined]

        # NOTE [Guaranteeing the 1-1 correspondence of FakeTensors and real tensors]:
        # executorch modules promise not to alias inputs and outputs.
        # Thus, output FakeTensors will correctly not alias input FakeTensors.
        _assert_tensors_nonaliasing(real_sub_args, example_value)

        p_args = (lowered_node,) + p_args

        # Store the invocation as a call
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs={},
            ),
            example_value=example_value,
        )


class LocalMapWrappedHigherOrderVariable(WrapHigherOrderVariable):
    _HOP_NAME = "torch.ops.higher_order.local_map_hop"
    supports_input_mutation = False
    supports_aliasing = False

    # Subclasses aren't supported by speculate_subgraph yet
    # So this HOP is only usable with plain tensors
    _enabled = False

    @classmethod
    @contextlib.contextmanager
    def enable(cls) -> Generator[None, None, None]:
        """Context manager to temporarily enable local map wrapping.
        Will be removed when speculate_subgraph supports subclass inputs:
        https://github.com/pytorch/pytorch/issues/161456.

        Usage:
            with LocalMapWrappedHigherOrderVariable.enable_wrapping():
                # Code where should_wrap_in_hop will return True
                pass
        """
        old_value = cls._enabled
        cls._enabled = True
        try:
            yield
        finally:
            cls._enabled = old_value

    @classmethod
    def should_wrap_in_hop(cls, value: Any) -> bool:
        if not torch.distributed.is_available():
            return False

        from torch.distributed.tensor.experimental._func_map import _local_map_wrapped

        # check is important to avoid subclass dispatch
        if type(value) is not type(_local_map_wrapped):
            return False

        return value is _local_map_wrapped and cls._enabled

    @staticmethod
    # pyrefly: ignore[bad-override]
    def build(**options: Any) -> "TorchHigherOrderOperatorVariable":
        return TorchHigherOrderOperatorVariable.make(
            torch._higher_order_ops.local_map_hop,
            **options,
        )

    def python_type(self) -> type:
        return type(self.value)

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        """
        Goal of this function is to rewrite local_map usage as a HOP:
            local_map(func, ...) -> local_map_hop(gm, ...)
        """

        (
            user_func,
            out_placements,
            in_placements,
            in_grad_placements,
            device_mesh,
            redistribute_inputs,
            *user_args,
        ) = args

        # None placements are used to pass non-Tensors into the local_map function.
        # Containers passed this way can not hold tensors. Thus, Dynamo would have inlined
        # into them, and we handle None placements by assuming they will be desugared away.
        # This will need to be adjusted for dynamic shapes support.
        def check_none_last(placements: Sequence[Any | None]) -> int:
            seen_none = 0
            for p in placements:
                if p is None:
                    seen_none += 1
                else:
                    assert seen_none == 0, (
                        "Tracing local_map is only currently supported with None placements last."
                    )
            return seen_none

        inputs_none_placements = check_none_last(in_placements.value)  # type: ignore[attr-defined]
        output_none_placements = check_none_last(out_placements.value)  # type: ignore[attr-defined]

        local_map_kwargs = {
            "out_placements": out_placements.value,  # type: ignore[attr-defined]
            "in_placements": in_placements.value,  # type: ignore[attr-defined]
            "redistribute_inputs": redistribute_inputs.value,  # type: ignore[attr-defined]
            "in_grad_placements": in_grad_placements.value,  # type: ignore[attr-defined]
            "device_mesh": device_mesh.value,  # type: ignore[attr-defined]
        }
        assert local_map_kwargs["device_mesh"] is not None, (
            "Not yet implemented, please manually provide a device_mesh to local_map."
        )
        mesh = local_map_kwargs["device_mesh"]

        # For Autoparallel, the initial trace is done with global shapes, then we decide model weights sharding,
        # and reuse the graph. Since the sharding decision is after the initial trace, we can't trace with local shapes.
        # For local_map however, since we specify all placements, we can trace with local shapes.

        # Step 1: Validate the annotated function matches the input_placements (i.e. that it can run in eager)
        template = (
            "Expecting {expected} {inputs_or_outputs} to local_map function based on placements"
            ", but found {actual}. Please ensure the count matches for eager. "
        )
        assert len(in_placements.value) == len(user_args), template.format(  # type: ignore[attr-defined]
            expected=len(in_placements.value),  # type: ignore[attr-defined]
            inputs_or_outputs="inputs",
            actual=len(user_args),
        )

        from torch._higher_order_ops.local_map import (
            redistribute_fw_inputs,
            redistribute_fw_outputs,
        )

        # Step 2: Convert inputs to local shapes
        priors = {}
        for placements, vt in zip(in_placements.value, user_args):  # type: ignore[attr-defined]
            if isinstance(vt, variables.lazy.LazyVariableTracker):
                vt = variables.lazy.LazyVariableTracker.realize_all(vt)

            if not vt.is_tensor():
                assert placements is None
                continue

            global_tensor = vt.as_proxy().node.meta["example_value"]
            # NOTE: We don't support local_map region relying on exact grad_fn information
            # This is okay since accessing grad_fn is a graph break.
            local_tensor = redistribute_fw_inputs(
                (global_tensor,),
                (placements,),
                mesh,
            )
            local_tensor = local_tensor[0]

            priors[vt] = global_tensor
            vt.as_proxy().node.meta["example_value"] = local_tensor
            vt.synchronize_attributes(tx)

        # Step 3: Trace local_map subgraph with local tensors
        (
            p_args,
            p_kwargs,
            example_value,
            body_r,
            body_gmod,
            body_name,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(
            tx, user_func, user_args, kwargs, self.value._name, subgraph_name="subgraph"
        )

        # Step 4: Validate traced graph signature still matches placement information
        expected_num_inputs = len(in_placements.value) - inputs_none_placements  # type: ignore[attr-defined]
        actual_num_inputs = len(body_gmod.graph.find_nodes(op="placeholder"))
        expected_num_outputs = len(out_placements.value) - output_none_placements  # type: ignore[attr-defined]
        assert len(body_gmod.graph.find_nodes(op="output")) == 1
        actual_num_outputs = len(body_gmod.graph.find_nodes(op="output")[0].args[0])

        template = (
            "Expecting {expected} {inputs_or_outputs} to local_map function based on placements"
            ", but found {actual}. If the count matches for eager, "
            "Dynamo may have flattened {inputs_or_outputs} to the function or found additional "
            "tensors used via closures. "
            "Please adjust the input placements to match what the traced graph sees: \n{gm_str}."
        )

        def make_error_msg(*args: Any) -> str:
            expected_num, actual_num, inputs_or_outputs = args
            gm_str = body_gmod.print_readable(print_output=False)
            return template.format(
                expected=expected_num,
                inputs_or_outputs=inputs_or_outputs,
                actual=actual_num,
                gm_str=gm_str,
            )

        if expected_num_inputs != actual_num_inputs:
            raise AssertionError(
                make_error_msg(expected_num_inputs, actual_num_inputs, "inputs")
            )
        if expected_num_outputs != actual_num_outputs:
            raise AssertionError(
                make_error_msg(expected_num_outputs, actual_num_outputs, "outputs")
            )

        if inputs_none_placements > 0:
            expected_input_nodes = [
                arg.as_proxy().node for arg in user_args[:-inputs_none_placements]
            ]
        else:
            expected_input_nodes = [arg.as_proxy().node for arg in user_args]
        actual_input_nodes = [proxy.node for proxy in p_args]
        assert actual_input_nodes[0].op == "get_attr"
        assert "subgraph" in actual_input_nodes[0].target  # type: ignore[attr-defined]
        assert len(expected_input_nodes) == len(actual_input_nodes) - 1
        for expected_order, actual_order in zip(
            expected_input_nodes, actual_input_nodes[1:]
        ):
            assert expected_order == actual_order, (
                "Dynamo changed the order of inputs to the local_map function, please adjust "
                f"the order of inputs and input_placements from {expected_input_nodes}, to: {actual_input_nodes[1:]}"
            )
        assert len(p_kwargs) == 0

        # Step 5: Install local_map subgraph
        p_kwargs = {key: value.as_proxy() for key, value in kwargs.items()}
        out = _call_function_with_auto_output_flattening(
            tx,
            self.value,
            p_args,
            p_kwargs,
            example_value,
            body_r,
            body_graph_output_vts,
        )

        # Step 6: Restore inputs and outputs to global shapes
        for vt, global_tensor in priors.items():
            vt.as_proxy().node.meta["example_value"] = global_tensor
            vt.synchronize_attributes(tx)

        outs = out.items if isinstance(out, TupleVariable) else [out]
        assert len(outs) == len(out_placements.value)  # type: ignore[attr-defined]
        for placements, vt in zip(out_placements.value, outs):  # type: ignore[attr-defined]
            if not vt.is_tensor():  # type: ignore[attr-defined]
                assert placements is None
                continue

            local_tensor = vt.as_proxy().node.meta["example_value"]  # type: ignore[attr-defined]

            # NOTE: We don't support code after the local_map region relying on exact grad_fn information
            # This is okay since accessing grad_fn is a graph break.
            global_tensor = redistribute_fw_outputs(
                (local_tensor,),
                (placements,),
                mesh,
                num_activations=0,  # this is not the joint
            )
            global_tensor = global_tensor[0]

            vt.as_proxy().node.meta["example_value"] = global_tensor  # type: ignore[attr-defined]
            vt.synchronize_attributes(tx)  # type: ignore[attr-defined]

        # TODO: Figure out how to handle output order diverging from eager

        # Treat as const, so we don't have to deal with Placement types in fx IR
        # Guarded with EQUALS_MATCH on local_map call's arguments
        body_gmod.meta["local_map_kwargs"] = {
            "out_placements": out_placements.value[:expected_num_outputs],  # type: ignore[attr-defined]
            "in_placements": in_placements.value[:expected_num_inputs],  # type: ignore[attr-defined]
            "redistribute_inputs": redistribute_inputs.value,  # type: ignore[attr-defined]
            "in_grad_placements": in_grad_placements.value,  # type: ignore[attr-defined]
            "device_mesh": device_mesh.value,  # type: ignore[attr-defined]
        }
        assert out is not None
        return out
