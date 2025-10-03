from torch.testing._internal.common_quantization import ModelMultipleOps
import os
from enum import Enum

import torch
from torch._functorch._aot_autograd.schemas import OutputType
from torch.fx import GraphModule
import importlib.util

import sys
import hashlib

autograd_function_template = """
import torch

{fw_code}

{bw_code}
class GraphRunner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        # Extract components: (*params, *buffers, *user_inputs, input_mutation_indices, num_mutate_inputs, num_user_outputs, fw_gm, bw_gm)

        input_mutation_indices = args[-5]
        num_mutate_inputs = args[-4]
        num_user_outputs = args[-3]
        num_inner_fwd_outputs = num_mutate_inputs + num_user_outputs

        fw_gm = args[-2]
        bw_gm = args[-1]

        # Save backward graph and metadata
        ctx.bw_gm = bw_gm

        # Construct forward arguments: (gm, *params, *buffers, *user_input)
        fw_args = args[:-5]

        # Run the forward graph module
        # fw_graph_inputs = (gm, *parameters, *buffers, *user_inputs)
        # fw_graph_outputs = (*updated_inputs, *user_outputs, *saved_intermediates)
        fw_outputs = forward_gm(fw_gm, *fw_args)

        # apply mutations
        mutations = fw_outputs[:num_mutate_inputs]
        assert len(input_mutation_indices) == len(mutations)

        for input_idx, mutation in zip(input_mutation_indices, mutations):
            fw_args[input_idx].copy_(mutation)

        # Save tensors for backward
        saved_intermediates = fw_outputs[num_inner_fwd_outputs:]
        ctx.save_for_backward(*saved_intermediates)

        user_outputs = fw_outputs[num_mutate_inputs:num_inner_fwd_outputs]
        return user_outputs

    @staticmethod
    def backward(ctx, *tangents):
        # Retrieve saved tensors
        bw_gm = ctx.bw_gm
        saved_intermediates = ctx.saved_tensors
        # assert(len(saved_intermediates) + len(tangents) == num_bw_inputs)

        # Run backward graph
        #  - input signature: (gm, *saved_intermediates, *tangents)
        #  - output signature: (*param_gradients, *user_inputs_gradients)
        bw_args = [bw_gm, *saved_intermediates, *tangents]
        bw_outputs = backward_gm(*bw_args)

        # assert(len(bw_outputs) == num_params + num_user_inputs)

        # append None as gradient for the last non-tensor inputs
        result = bw_outputs + (None,) * 5
        return tuple(result)

"""


class RunMode(Enum):
    CODEGEN = "codegen"
    CODEGEN_AUTOGRAD = "codegen_autograd"
    GRAPH_MODULE = "graph_module"
    FX_INTERPRETER = "interpreter"

CACHE_DIR = "/tmp/joint_graph_runner"

class JointGraphModule(torch.nn.Module):
    def __init__(
        self,
        params,
        buffers,
        fw_metadata,
        fw_gm: GraphModule,
        bw_gm: GraphModule,
        run_mode: RunMode,
        model_name: str = "",
    ):
        super().__init__()

        self._params = params
        self._buffers = buffers
        self.run_mode = run_mode

        rng_states = [
            n
            for n in fw_gm.graph.find_nodes(op="placeholder")
            if "fwd_rng_state" in n.name
        ]
        fw_metadata.num_graphsafe_rng_states = len(rng_states)
        if rng_states:
            fw_metadata.graphsafe_rng_state_index = (
                rng_states[0].meta["val"].device.index
            )
        self.fw_metadata = fw_metadata

        self.input_mutation_indices = []
        for i, input_info in enumerate(self.fw_metadata.input_info):
            if input_info.mutates_data or input_info.mutates_metadata:
                self.input_mutation_indices.append(i)

        self.num_user_outputs = get_num_user_outputs(self.fw_metadata)
        # mutated inputs that show up as outputs
        self.num_mutate_inputs = get_num_mutate_inputs(self.fw_metadata)
        self.num_inner_fwd_outputs = self.num_mutate_inputs + self.num_user_outputs

        self.num_fw_output = fw_metadata.num_outputs
        self.num_saved_intermediates = self.num_fw_output - self.num_inner_fwd_outputs
        self.num_bw_inputs = num_inputs(bw_gm)

        self.fw_gm = fw_gm
        self.bw_gm = bw_gm

        os.makedirs(CACHE_DIR, exist_ok=True)

        if run_mode == RunMode.CODEGEN_AUTOGRAD:
            self.autograd_func = codegen_autograd_function(fw_gm, bw_gm, model_name)
            self.fw_runner = fw_gm
            self.bw_runner = bw_gm
        else:
            self.fw_runner = graph_runner(fw_gm, f"{model_name}_fw", mode=run_mode)
            self.bw_runner = graph_runner(bw_gm, f"{model_name}_bw", mode=run_mode)


    def forward(self, args):
        assert isinstance(args, tuple)
        # TODO: apply pytree flattening here
        # TODO: apply DTensor to local_tesnor converstion here

        # Create argument list: (*params, *buffers, *user_input, ...)
        flat_args = [
            *self._params,
            *self._buffers,
            *args,
        ]

        flat_args += [None] * self.fw_metadata.num_graphsafe_rng_states

        flat_args += [
            self.input_mutation_indices,
            self.num_mutate_inputs,
            self.num_user_outputs,
        ]   

        flat_args += [self.fw_runner, self.bw_runner]

        # torch.distributed.breakpoint()

# 56, in forward_gm
# [rank0]:[rank0]:     view_6: "f32[128]" = torch.ops.aten.view.default(primals_2, [128]);  primals_2 = None
# [rank0]:[rank0]:                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# [rank0]:[rank0]:   File "/data/users/bahuang/pytorch/torch/_ops.py", line 841, in __call__
# [rank0]:[rank0]:     return self._op(*args, **kwargs)
# [rank0]:[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
# [rank0]:[rank0]: RuntimeError: shape '[128]' is invalid for input of size 8192

        if self.run_mode == RunMode.CODEGEN_AUTOGRAD:
            user_outputs = self.autograd_func.apply(*flat_args)
        else:
            user_outputs = GraphRunner.apply(*flat_args)
        

        # TODO: apply pytree unflattening here
        # TODO: How to convert local_tensor back to DTensor, placement missing
        if len(user_outputs) == 1:
            return user_outputs[0]
        else:
            return user_outputs


class GraphRunner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        # Extract components: (*params, *buffers, *user_inputs, num_mutate_inputs, num_user_outputs, fw_gm, bw_gm)

        input_mutation_indices = args[-5]
        num_mutate_inputs = args[-4]
        num_user_outputs = args[-3]
        num_inner_fwd_outputs = num_mutate_inputs + num_user_outputs

        fw_runner = args[-2]
        bw_runner = args[-1]

        # Save backward graph and metadata
        ctx.bw_runner = bw_runner

        # Construct forward arguments: (*params, *buffers, *user_input)
        fw_args = args[:-5]

        # Run the forward graph module
        # fw_graph_inputs = (*parameters, *buffers, *user_inputs)
        # fw_graph_outputs = (*updated_inputs, *user_outputs, *saved_intermediates)
        fw_outputs = fw_runner(*fw_args)

        # apply mutations
        mutations = fw_outputs[:num_mutate_inputs]
        assert len(input_mutation_indices) == len(mutations)

        for input_idx, mutation in zip(input_mutation_indices, mutations):
            fw_args[input_idx].copy_(mutation)

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
        result = bw_outputs + (None,) * 5
        return tuple(result)


def num_inputs(gm: GraphModule) -> int:
    return sum(1 for node in gm.graph.nodes if node.op == "placeholder")


def num_outputs(gm: GraphModule) -> int:
    output_node = gm.graph.output_node()

    assert len(output_node.args) == 1, "Expecting a single output node"
    assert isinstance(output_node.args[0], tuple), "Expecting output returned as tuple"

    return len(output_node.args[0])



def get_python_code(gm: GraphModule, prefix: str):
    python_code = gm.graph.python_code(root_module="self", verbose=True)

    hs = hashlib.sha256(python_code.src.encode('utf-8')).hexdigest()
    filename = os.path.join(CACHE_DIR, f"{prefix}_{hs[:8]}.py") 

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write(python_code.src)
    else:
        with open(filename) as f:
            # override with cached src from disk
            python_code.src = f.read()

    return python_code



def load_autograd_function(path, module_name):
    filename = os.path.join(path, f"{module_name}.py") 
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Now you can use it
    return module.GraphRunner


def codegen_autograd_function(fw_gm: GraphModule, bw_gm: GraphModule, prefix: str):
    fw_python_code = fw_gm.graph.python_code(root_module="self", verbose=True)
    bw_python_code = bw_gm.graph.python_code(root_module="self", verbose=True)

    fw_code = fw_python_code.src.replace("def forward", "def forward_gm")
    bw_code = bw_python_code.src.replace("def forward", "def backward_gm")

    src = autograd_function_template.format(fw_code=fw_code, bw_code=bw_code)

    # TODO: get hash another way. we shouldn't need to codegen the whole thing if the file already exist
    hs = hashlib.sha256(src.encode('utf-8')).hexdigest()

    module_name = f"{prefix + "_" if prefix else ""}autograd_function_{hs[:8]}"
    filename = os.path.join(CACHE_DIR, f"{module_name}.py") 

    if not os.path.exists(filename): 
        with open(filename, "w") as f:
            f.write(src)

    return load_autograd_function(CACHE_DIR, module_name)




def graph_runner(gm: GraphModule, prefix: str, mode: RunMode = RunMode.CODEGEN):
    from torch.fx.graph_module import _forward_from_src

    if mode == RunMode.CODEGEN:
        python_code = get_python_code(gm, prefix)
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
