import contextlib
from typing import List, Tuple

import torch


@contextlib.contextmanager
def optimized_execution(should_optimize):
    """Context manager that controls whether the JIT's executor will run optimizations before executing a function."""
    stored_flag = torch._C._get_graph_executor_optimize()
    torch._C._set_graph_executor_optimize(should_optimize)
    try:
        yield
    finally:
        torch._C._set_graph_executor_optimize(stored_flag)


@contextlib.contextmanager
def fuser(name):
    """Context manager that facilitates switching between backend fusers.

    Valid names:
    * ``fuser0`` - enables only legacy fuser
    * ``fuser1`` - enables only NNC
    * ``fuser2`` - enables only nvFuser
    * ``fuser3`` - enables oneDNN Graph
    """
    old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
    old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
    old_texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
    old_nvfuser_state = torch._C._jit_nvfuser_enabled()
    old_llga_state = torch._C._jit_llga_enabled()
    if name == "fuser0":  # legacy fuser
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    elif name == "fuser1":  # NNC
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        old_profiling_mode = torch._C._get_graph_executor_optimize(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    elif name == "fuser2":  # nvFuser
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._jit_set_llga_enabled(False)
    elif name == "fuser3":  # oneDNN Graph
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        old_profiling_mode = torch._C._get_graph_executor_optimize(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(True)
    elif name == "none":  # Turn Pytorch fuser off
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    else:
        raise Exception(f"unrecognized fuser option (name: {name})")
    try:
        yield
    finally:
        if name in ["fuser1", "fuser3"]:  # NNC or oneDNN Graph
            torch._C._jit_set_profiling_executor(old_profiling_executor)
            torch._C._get_graph_executor_optimize(old_profiling_mode)
        # recover the previous values
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(old_gpu_fuse)
        torch._C._jit_set_texpr_fuser_enabled(old_texpr_fuser_state)
        torch._C._jit_set_nvfuser_enabled(old_nvfuser_state)
        torch._C._jit_set_llga_enabled(old_llga_state)


last_executed_optimized_graph = torch._C._last_executed_optimized_graph


def _get_differentiable_graph_node(node, diff_node):
    if node.kind() == "prim::DifferentiableGraph":
        diff_node.append(node)
    else:
        for block in node.blocks():
            for n in block.nodes():
                _get_differentiable_graph_node(n, diff_node)


def _graph_for(self, *args, **kwargs):
    return _script_method_graph_for(self, self, *args, **kwargs)


def _script_method_graph_for(self, parent, *args, **kwargs):
    try:
        dbs = parent.get_debug_state()
        eps = list(dbs.execution_plans.values())
        assert len(eps) == 1
        graph = eps[0].graph.copy()

        # graph_executor_states for differentiable node
        fw_states = eps[0].code.differentiable_op_executor_states()
        diff_nodes: List[torch._C.Node] = []
        for n in graph.nodes():
            _get_differentiable_graph_node(n, diff_nodes)

        assert len(fw_states) == len(diff_nodes)
        # swap each differentiable graph with optimized graph in their execution plan
        for n, state in zip(diff_nodes, fw_states):
            fw_execution_plans = list(state.execution_plans.values())
            # we can only update the subgraph when there's a unique execution
            # plan. Avoid assert here so we would skip the ones that can't be
            # updated while try the best effort to update other nodes.
            if len(fw_execution_plans) == 1:
                n.g_("Subgraph", fw_execution_plans[0].graph)

        return graph
    except Exception:
        # fallback approach, we just ran the graph and return the recorded optimized
        # graph
        self(*args, **kwargs)
        return last_executed_optimized_graph()


def set_fusion_strategy(strategy: List[Tuple[str, int]]):
    """Set the type and number of specializations that can occur during fusion.

    Usage: provide a list of pairs (type, depth) where type is one of "STATIC" or "DYNAMIC"
    and depth is an integer.

    Behavior - static vs dynamic:
        In STATIC fusion, fused ops are compiled to have fixed input shapes. The shape is determined
        based on some initial profiling runs.
        In DYNAMIC fusion, fused ops are compiled to have variable input shapes, so that multiple
        shapes are possible.

    In both cases, we also recompile on new striding behavior, device, or dtype.

    Behavior - fallback functions & depth:
        When an input doesn't match the format required by the specialized compiled op, it will run
        a fallback function. Fallback functions are recursively be compiled and specialized based
        on the observed tensor shapes. Since compilation can be slow, the "depth" parameter is provided to
        limit the number of specializations that can be compiled, before giving up on recompiling and
        falling back to a completely un-fused, un-specialized implementation.

    The list of (type, depth) pairs controls the type of specializations and the number of
    specializations. For example: [("STATIC", 2), ("DYNAMIC", 2)] indicates that the first
    two specializations will use static fusions, the following two specializations will use
    dynamic fusion, and any inputs that satisfy none of the 4 options will run an
    unfused implementation.

    NB: in the future, if more as more fusion backends are added there may be more granular
    apis for specific fusers.
    """
    return torch._C._jit_set_fusion_strategy(strategy)
