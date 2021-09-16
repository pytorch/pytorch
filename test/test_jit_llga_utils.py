import torch
from torch.testing._internal.jit_utils import JitTestCase, warmup_backward, \
    get_execution_plan
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, \
    get_function_arglist, load_tests, repeat_test_for_types, TemporaryFileName

LLGA_FUSION_GROUP = 'prim::LlgaFusionGroup'


def warmup_forward(f, *args, profiling_count=2):
    for i in range(profiling_count):
        results = f(*args)

    return results


class JitLlgaTestCase(JitTestCase):
    def checkTrace(self, m, x, *args, **kwargs):
        with torch.no_grad(), \
                torch._jit_internal._disable_emit_hooks():
            traced = torch.jit.trace(m, x)
            warmup_forward(traced, *x)
            fwd_graph = traced.graph_for(*x)

            ref_o = m(*x)
            jit_o = traced(*x)
            self.assertEqual(jit_o, ref_o)
        return traced, fwd_graph

    def assertFused(self, graph, fused_patterns):
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)
