import contextlib
import copy
from typing import Callable, Tuple, Generator

import torch
import torch._dynamo as torchdynamo
from torch._decomp import core_aten_decompositions
from torch._dispatch.python import enable_python_dispatcher
from torch.nn.utils import stateless
from torch.utils import _pytree as pytree

from torch._functorch.aot_autograd import (
    AOTConfig,
    create_aot_dispatcher_function,
    default_partition,
    run_functionalized_fw_and_collect_metadata,
)

from torch.fx.experimental.proxy_tensor import (
    get_proxy_slot,
    get_torch_dispatch_modes,
    has_proxy_slot,
    make_fx,
    ProxyTorchDispatchMode,
    set_proxy_slot,
)

from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional

from .workflow import ExportedProgram

class DynamoConfig:
    """
    Manage export-specific configurations of Dynamo.
    """

    def __init__(
        self,
        capture_scalar_outputs: bool = True,
        guard_nn_modules: bool = True,
        dynamic_shapes: bool = True,
        specialize_int_float: bool = True,
        allow_rnn: bool = True,
        verbose: bool = True,
    ) -> None:

        self.capture_scalar_outputs = capture_scalar_outputs
        self.guard_nn_modules = guard_nn_modules
        self.dynamic_shapes = dynamic_shapes
        self.specialize_int_float = specialize_int_float
        self.allow_rnn = allow_rnn
        self.verbose = verbose

    def activate(self) -> None:
        torchdynamo.config.capture_scalar_outputs = self.capture_scalar_outputs
        torchdynamo.config.guard_nn_modules = self.guard_nn_modules
        torchdynamo.config.dynamic_shapes = self.dynamic_shapes
        torchdynamo.config.specialize_int_float = self.specialize_int_float
        torchdynamo.config.allow_rnn = self.allow_rnn
        torchdynamo.config.verbose = self.verbose


@contextlib.contextmanager
def using_config(config: DynamoConfig) -> Generator[DynamoConfig, None, None]:
    config.activate()
    try:
        yield config
    finally:
        pass


def aot_capture(mod, flat_args):
    """
    A wrapper around aot_autograd() to mix AOT Autograd + torch.export.
    Some assumptions were made about the AOT Autograd internal:
    1. The functionalization metadata format.
    2. Calling convention of returned forward graph.
    3. make_fx() internal proxy storage.

    In the current context we're just experimenting the idea so it's possible things
    could break. For the next step we should find a way to upstream something reasonable.
    """
    param_list = [
        *mod.named_parameters(remove_duplicate=False),
        *mod.named_buffers(remove_duplicate=False),
    ]
    params = dict(param_list)
    params_flat, params_spec = pytree.tree_flatten(params)
    params_len = len(params_flat)

    full_args = []
    full_args.extend(params_flat)
    full_args.extend(flat_args)

    assert any(arg.requires_grad for arg in full_args), "AOT Autograd needs at least one graph input to have .requires_grad=True, otherwise functionalization will be skipped. Please make sure at least one input has .requires_grad=True"

    def functional_call(*args):
        with stateless._reparametrize_module(
            mod,
            pytree.tree_unflatten(args[:params_len], params_spec),
        ):
            return torch.fx.Interpreter(mod).run(*args[params_len:])

    out_spec = None

    with enable_python_dispatcher():
        fw_metadata, _ = run_functionalized_fw_and_collect_metadata(
            lambda *args: pytree.tree_flatten(functional_call(*args))[0],
            keep_input_mutations=False,
        )(*copy.deepcopy(full_args))

    assert len(fw_metadata.input_info) == len(full_args)
    mutated_input_indices = [
        i
        for i, input_info in enumerate(fw_metadata.input_info)
        if input_info.mutates_data or input_info.mutates_metadata
    ]

    graph_module = None

    def fw_compiler(gm, inputs):
        nonlocal graph_module
        graph_module = gm

    num_fwd_returns = None

    def partition_fn(joint_module, joint_inputs, *, num_fwd_outputs, **kwargs):
        nonlocal num_fwd_returns
        num_fwd_returns = num_fwd_outputs
        return default_partition(
            joint_module, joint_inputs, num_fwd_outputs=num_fwd_outputs, **kwargs
        )

    def set_state_proxies(state_args):
        modes = get_torch_dispatch_modes()
        proxy_tensor_modes = [m for m in modes if isinstance(m, ProxyTorchDispatchMode)]
        if len(proxy_tensor_modes) == 0:
            return
        assert len(state_args) == len(params_flat)
        for i, arg in enumerate(state_args):
            tracer = next(
                m.tracer for m in proxy_tensor_modes if has_proxy_slot(arg, m.tracer)
            )
            set_proxy_slot(arg, tracer, params_flat[i])

    aot_config = AOTConfig(
        fw_compiler=fw_compiler,
        bw_compiler=lambda gm, inputs: None,
        partition_fn=partition_fn,
        decompositions=core_aten_decompositions(),
        num_params_buffers=params_len,
        aot_id=-1,
        keep_inference_input_mutations=False,
    )

    @contextlib.contextmanager
    def setup_dynamic_shape():
        prev, torch._functorch.config.use_dynamic_shapes = (
            torch._functorch.config.use_dynamic_shapes,
            True,
        )
        try:
            yield
        finally:
            torch._functorch.config.use_dynamic_shapes = prev

    def exported_call(*args):
        state_args = args[:params_len]
        unwrapped_state_args = _unwrap_all_tensors_from_functional(
            state_args, reapply_views=False
        )
        set_state_proxies(unwrapped_state_args)
        with torch.fx.traceback.preserve_node_meta():
            outputs = functional_call(*args)
        nonlocal out_spec
        outputs, out_spec = pytree.tree_flatten(outputs)
        return outputs

    with torch.enable_grad(), setup_dynamic_shape():
        create_aot_dispatcher_function(
            exported_call,
            full_args,
            aot_config,
        )

    assert graph_module is not None

    for i, node in enumerate(graph_module.graph.nodes):
        if i == len(params_flat):
            break
        assert node.op == "placeholder" and len(node.users) == 0
        graph_module.graph.erase_node(node)

    output_node = next(iter(reversed(graph_module.graph.nodes)))
    assert output_node.op == "output" and len(output_node.args) == 1
    assert num_fwd_returns is not None
    assert num_fwd_returns >= fw_metadata.num_intermediate_bases + len(fw_metadata.output_info), "Export cannot proceed because it cannot parse output from AOT Autograd."
    # Turncate the output so we only output what we need.
    output_node.args = (
        output_node.args[0][
            : len(mutated_input_indices) + len(fw_metadata.output_info)
        ],
    )

    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()

    def find_mutation_destinations(gm, w):
        assert isinstance(w, torch.Tensor)
        ret = [
            name for name, x in [*gm.named_parameters(), *gm.named_buffers()] if x is w
        ]
        assert len(ret) != 0, "Cannot find mutation destination."
        return ret

    mutation = [
        (
            "copy_",
            output_node.args[0][k].name,
            find_mutation_destinations(graph_module, param_list[i][1]),
        )
        for k, i in enumerate(mutated_input_indices)
    ]
    assert out_spec is not None
    return graph_module, mutation, out_spec

def export(f: Callable, args: Tuple, is_training=False):
    if is_training:
        NotImplementedError("training mode is not supported yet")

    flattened_args, in_spec = pytree.tree_flatten(args)
    original_flat_args = tuple(flattened_args)
    flat_args = tuple(flattened_args)

    with using_config(DynamoConfig()):
        graph_module, guards = torchdynamo.export(f, *args, aten_graph=False)
    graph_module, _, out_spec = aot_capture(graph_module, flat_args)
    return ExportedProgram(fw_module=graph_module, example_inputs=original_flat_args, in_spec=in_spec, out_spec=out_spec)
