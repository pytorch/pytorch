import builtins
import inspect
from collections import namedtuple
from typing import Any, Callable

import torch
import torch.utils._pytree as pytree
from torch._dynamo.convert_frame import FrameInfo, fullgraph_capture, get_compile_id
from torch._dynamo.eval_frame import argument_names
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._guards import compile_context, CompileContext
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo


class ModuleToTrace(torch.nn.Module):
    def __init__(self, foo: Any, in_spec: Any) -> None:
        super().__init__()
        self._export_root = foo
        self.in_spec = in_spec

    def forward(self, *flat_args: Any) -> "ExportTracerOutput":
        args, kwargs = pytree.tree_unflatten(flat_args, self.in_spec)
        res = self._export_root(*args, **kwargs)
        out_flat, out_spec = pytree.tree_flatten(res)
        return ExportTracerOutput(out_flat, out_spec)


ExportTracerOutput = namedtuple("ExportTracerOutput", ["flat_args", "out_spec"])


def _dynamo_graph_capture_for_export(
    mod: torch.nn.Module,
) -> Callable[..., torch.fx.GraphModule]:
    """
    This is lower level API that is used for export to capture dynamo level
    torch IR.

    Notable TODOs:
    1. Are we actually gonna run the bytecode?
    2. Need to attach guards
    """

    def inner(*args: Any, **kwargs: Any) -> torch.fx.GraphModule:
        flat_inputs, in_spec = pytree.tree_flatten((args, kwargs))
        module_to_trace = ModuleToTrace(mod, in_spec)

        signature = inspect.signature(module_to_trace.forward)

        bound_arguments = signature.bind(*flat_inputs)
        bound_arguments.apply_defaults()

        f_locals = {"self": module_to_trace, **bound_arguments.arguments}

        frame = FrameInfo(
            module_to_trace.forward.__func__.__code__,  # type: ignore[attr-defined]
            module_to_trace.forward.__func__.__globals__,  # type: ignore[attr-defined]
            f_locals,
            builtins,  # type: ignore[arg-type]
            closure=(),  # type: ignore[arg-type]
        )

        dynamo_config_ctx = torch._dynamo.config.patch(
            "log_graph_in_out_metadata", True
        )

        with (
            compile_context(CompileContext(get_compile_id({}))),
            get_metrics_context(),
            dynamo_timed("fullgraph_capture"),
            dynamo_config_ctx,
        ):
            out = fullgraph_capture(frame, _is_export_deprecated_do_not_use=True)

            assert out.dynamo_output.tracer_output.output_graph is not None

            export_metadata = (
                out.dynamo_output.tracer_output.output_graph.export_metadata
            )
            graph_inputs = export_metadata.graph_input_idx_to_local_source
            output_return_type = export_metadata.output_return_type
            # We need to extract out_spec here because we are not actually running the bytecode
            out_spec = export_metadata.out_spec

            graph = out.backend_input.graph_module

            # It is not guaranteed that dynamo puts inputs in right order, so we need to
            # map the actual user order to the dynamo order.
            graph_input_order: dict[int, int] = {}
            for inp in graph_inputs:
                source = graph_inputs[inp]
                assert isinstance(source, torch._dynamo.source.GetItemSource)
                graph_input_order[source.index] = len(graph_input_order)

            placeholders = [n for n in list(graph.graph.nodes) if n.op == "placeholder"]
            output = next(n for n in list(graph.graph.nodes) if n.op == "output")
            # Sometimes there can be empty inputs
            anchor = placeholders[0] if len(placeholders) > 0 else output
            inp_to_node = {}

            with graph.graph.inserting_before(anchor):
                for i in range(len(flat_inputs)):
                    node_new = graph.graph.placeholder(f"arg_{i}")
                    if i in graph_input_order:
                        placeholders[graph_input_order[i]]
                        node_new.meta = placeholders[graph_input_order[i]].meta.copy()
                    inp_to_node[i] = node_new

            new_args = []
            for i in output_return_type:
                type, val = output_return_type[i]
                if type == "graph_out":
                    new_args.append(output.args[0][val])
                if type == "input":
                    input_idx = val.index
                    new_args.append(inp_to_node[input_idx])
                if type == "constant":
                    new_args.append(val)
            output.args = (tuple(new_args),)

            for src_idx, i in graph_input_order.items():
                old = placeholders[src_idx]
                new = inp_to_node[i]
                old.replace_all_uses_with(new)
                graph.graph.erase_node(old)

            # Dynamo uses _lazyGraphModule, so we need to force recompile
            from torch.fx._lazy_graph_module import _LazyGraphModule

            _LazyGraphModule.force_recompile(graph)

        graph.graph._codegen = _PyTreeCodeGen(
            _PyTreeInfo(
                argument_names(signature, args, kwargs),  # type: ignore[arg-type]
                in_spec,
                out_spec,
            )
        )

        graph.recompile()
        return graph

    return inner
