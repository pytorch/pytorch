import os
from enum import Enum

import torch
from torch._functorch._aot_autograd.schemas import OutputType
from torch.fx import GraphModule


class RunMode(Enum):
    CODEGEN = "codegen"
    GRAPH_MODULE = "graph_module"
    FX_INTERPRETER = "interpreter"


class JointGraphModule(torch.nn.Module):
    def __init__(
        self,
        params,
        buffers,
        fw_metadata,
        fw_gm: GraphModule,
        bw_gm: GraphModule,
        mode: RunMode,
    ):
        super().__init__()

        self._params = params
        self._buffers = buffers

        self.fw_metadata = fw_metadata

        self.num_user_outputs = get_num_user_outputs(self.fw_metadata)
        # mutated inputs that show up as outputs
        self.num_mutate_inputs = get_num_mutate_inputs(self.fw_metadata)
        self.num_inner_fwd_outputs = self.num_mutate_inputs + self.num_user_outputs

        self.num_fw_output = num_outputs(fw_gm)
        self.num_saved_intermediates = self.num_fw_output - self.num_inner_fwd_outputs
        self.num_bw_inputs = num_inputs(bw_gm)

        self.fw_gm = fw_gm
        self.bw_gm = bw_gm

        cache_dir = "/tmp/aotautograd_runner"
        os.makedirs(cache_dir, exist_ok=True)

        fw_src_path = os.path.join(cache_dir, "fw_gm.py")
        bw_src_path = os.path.join(cache_dir, "bw_gm.py")

        self.fw_runner = graph_runner(fw_gm, fw_src_path, mode=mode)
        self.bw_runner = graph_runner(bw_gm, bw_src_path, mode=mode)

    def forward(self, *args):
        # TODO: apply pytree flattening here
        # TODO: apply DTensor to local_tesnor converstion here

        # Create argument list: (*params, *buffers, user_input, fw_runner, bw_runner)
        flat_args = [
            *self._params,
            *self._buffers,
            *args,
            self.num_mutate_inputs,
            self.num_user_outputs,
            self.fw_runner,
            self.bw_runner,
        ]

        flat_outs = GraphRunner.apply(*flat_args)

        # TODO: apply pytree unflattening here
        # TODO: How to convert local_tensor back to DTensor, placement missing
        if len(flat_outs) == 1:
            return flat_outs[0]
        else:
            return flat_outs


class GraphRunner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        # Extract components: (*params, *buffers, *user_inputs,  num_mutate_inputs, num_user_outputs, fw_gm, bw_gm,)
        num_mutate_inputs = args[-4]
        num_user_outputs = args[-3]
        num_inner_fwd_outputs = num_mutate_inputs + num_user_outputs

        fw_runner = args[-2]
        bw_runner = args[-1]

        # Save backward graph and metadata
        ctx.bw_runner = bw_runner

        # Construct forward arguments: (*params, *buffers, user_input)
        fw_args = args[:-4]

        # Run the forward graph module
        # fw_graph_inputs = (*parameters, *buffers, *user_inputs)
        # fw_graph_outputs = (*updated_inputs, *user_outputs, *saved_intermediates)
        fw_outputs = fw_runner(*fw_args)

        # Save tensors for backward
        saved_intermediates = fw_outputs[num_inner_fwd_outputs:]
        ctx.save_for_backward(*saved_intermediates)

        user_outputs = fw_outputs[num_mutate_inputs:num_inner_fwd_outputs]
        return user_outputs

    @staticmethod
    def backward(ctx, *tangents):
        # Retrieve saved tensors
        saved_intermediates = ctx.saved_tensors
        # assert(len(saved_intermediates) + len(tangents) == num_bw_inputs)

        # Run backward graph
        #  - input signature: (*saved_intermediates, *tangents)
        #  - output signature: (*param_gradients, *user_inputs_gradients)
        bw_args = [*saved_intermediates, *tangents]
        bw_outputs = ctx.bw_runner(*bw_args)

        # assert(len(bw_outputs) == num_params + num_user_inputs)

        # append None as gradient for the last non-tensor inputs
        result = bw_outputs + (None,) * 4
        return tuple(result)


def num_inputs(gm: GraphModule) -> int:
    return sum(1 for node in gm.graph.nodes if node.op == "placeholder")


def num_outputs(gm: GraphModule) -> int:
    output_node = gm.graph.output_node()

    assert len(output_node.args) == 1, "Expecting a single output node"
    assert isinstance(output_node.args[0], tuple), "Expecting output returned as tuple"

    return len(output_node.args[0])


def get_python_code(gm: GraphModule, path: str):
    python_code = gm.graph.python_code(root_module="self", verbose=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(python_code.src)
    else:
        with open(path) as f:
            # override with cached src from disk
            python_code.src = f.read()

    return python_code


def graph_runner(gm: GraphModule, path: str, mode: RunMode = RunMode.CODEGEN):
    from torch.fx.graph_module import _forward_from_src

    if mode == RunMode.CODEGEN:
        python_code = get_python_code(gm, path)
        co_fields = gm._graph._co_fields if hasattr(gm._graph, "_co_fields") else {}
        fn = _forward_from_src(python_code.src, python_code.globals, co_fields)

        # need a wrapper to pass the dummy None as the first argument `self`
        def wrapped_fn(*args):
            return fn(None, *args)

        return wrapped_fn
    elif mode == RunMode.GRAPH_MODULE:
        return gm.forward
    elif mode == RunMode.FX_INTERPRETER:
        interpreter = torch.fx.Interpreter(gm)
        return interpreter.run
    else:
        raise ValueError(f"Unknown mode {mode}")


def get_num_user_outputs(fw_metadata) -> int:
    # TODO: double check what's num_intermediate_bases
    num_user_outputs = (
        len(
            [
                x
                for x in fw_metadata.output_info
                if x.output_type
                in (OutputType.non_alias, OutputType.alias_of_intermediate)
            ]
        )
        + fw_metadata.num_intermediate_bases
    )
    return num_user_outputs


def get_num_mutate_inputs(fw_metadata) -> int:
    return len(
        [x for x in fw_metadata.input_info if x.mutates_data or x.mutates_metadata]
    )
