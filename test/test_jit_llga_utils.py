import torch
from torch.testing._internal.jit_utils import JitTestCase, warmup_backward, \
    get_execution_plan
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, \
    get_function_arglist, load_tests, repeat_test_for_types, TemporaryFileName

LLGA_FUSION_GROUP = 'prim::LlgaFusionGroup'

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)


def all_backward_graphs(module):
    ge_state = module.get_debug_state()
    fwd_plan = get_execution_plan(ge_state)
    executors = fwd_plan.code.grad_executor_states()
    assert len(executors), 'No backward graph found in the module'
    grad_executor = executors[0]
    bwd_plans = list(grad_executor.execution_plans.values())
    return [p.graph.copy() for p in bwd_plans]


def backward_graph(module):
    graphs = all_backward_graphs(module)
    assert len(graphs), 'Warm up the module before calling backward_graph'
    return graphs[0]


class JitLlgaTestCase(JitTestCase):
    def checkScript(self, m, x):
        requires_grad = any(t.requires_grad for t in x)
        with torch.set_grad_enabled(requires_grad):
            ref = m(*x)
            scripted = torch.jit.script(m)
            y = scripted(*x)
            self.assertEqual(y, ref)
            graph = scripted.graph_for(*x)
        return scripted, graph

    def checkTrace(self, m, x, *args, **kwargs):
        grad = any(t.requires_grad for t in x)
        with torch.set_grad_enabled(grad), \
                torch._jit_internal._disable_emit_hooks():
            traced = super().checkTrace(m, x, inputs_require_grads=grad)
            fwd_graph = traced.graph_for(*x)
        if grad:
            warmup_backward(traced(*x).sum())
            return traced, fwd_graph, backward_graph(traced)
        else:
            return traced, fwd_graph

    def assertFused(self, graph, fused_patterns):
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)
