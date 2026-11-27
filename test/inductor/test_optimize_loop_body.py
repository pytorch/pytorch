# Owner(s): ["module: inductor"]

import torch
from torch._inductor import config
from torch._inductor.loop_body import LoopBody
from torch._inductor.optimize_loop_body import eliminate_redundant_lowp_round_trips
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.inductor_utils import requires_triton


class TestOptimizeLoopBody(TestCase):
    def test_eliminate_redundant_lowp_round_trips(self):
        def fn(index):
            value = V.ops.constant(1.25, torch.float32)
            for _ in range(3):
                value = V.ops.to_dtype(value, torch.bfloat16, use_compute_types=False)
                value = V.ops.to_dtype(value, torch.bfloat16)
            return value

        with config.patch(constant_and_index_propagation=False):
            loop_body = LoopBody(fn, ([],), {}, [], [])

        graph = loop_body.root_block.graph
        down1, up1 = graph.find_nodes(op="call_method", target="to_dtype")
        self.assertEqual(down1.args[1].target, "constant")
        self.assertIs(up1.args[1], down1)
        self.assertEqual(list(up1.users), [graph.output_node()])
        self.assertEqual(loop_body.op_counts["to_dtype"], 2)

        self.assertFalse(eliminate_redundant_lowp_round_trips(loop_body))
        self.assertEqual(len(graph.find_nodes(op="call_method", target="to_dtype")), 2)
        self.assertEqual(loop_body.op_counts["to_dtype"], 2)


class TestOptimizeLoopBodyDevice(TestCase):
    @parametrize("lowp_dtype", [torch.float16, torch.bfloat16])
    @config.patch(emulate_precision_casts=True)
    @requires_triton()
    def test_issue_196958(self, device, lowp_dtype):
        def fn(x):
            return (x.to(lowp_dtype) * 0.375).to(torch.float32)

        # Rounding 2055 before multiplication affects both fp16 and bf16, so
        # equality also verifies that the first cast pair is preserved.
        x = torch.tensor([2055], device=device, dtype=torch.int64)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), x
        )

        self.assertEqual(actual, fn(x))
        dtype_name = str(lowp_dtype).removeprefix("torch.")
        self.assertEqual(code.count(f".to(tl.{dtype_name})"), 3)
        self.assertEqual(code.count(".to(tl.float32)"), 4)


instantiate_device_type_tests(
    TestOptimizeLoopBodyDevice, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
