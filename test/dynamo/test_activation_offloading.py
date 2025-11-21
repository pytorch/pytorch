# Owner(s): ["oncall: pt2"]
# flake8: noqa: B950

from functools import partial

import pytest

import torch
import torch._functorch.config
from functorch.compile import (
    aot_function,
    default_decompositions,
    min_cut_rematerialization_partition,
)
from torch._dynamo.graph_bytecode_inputs import reset_user_object_tracking
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


networkx = pytest.importorskip("networkx")


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


def get_fw_bw_graph(
    f, inps, partitioner=min_cut_rematerialization_partition, dynamic=False
):
    fw_graph_cell = [None]
    bw_graph_cell = [None]
    aot_function(
        f,
        fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
        bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
        partition_fn=partitioner,
        decompositions=default_decompositions,
        dynamic=dynamic,
    )(*inps).sum().backward()
    return (fw_graph_cell[0], bw_graph_cell[0])


class ActivationOffloadingTests(TestCase):
    """Tests activation offloading functionality"""

    def setUp(self):
        super().setUp()

        def fn(x):
            return (x[0] + x[1]).sin() + (x[2] + x[3]).sin() + (x[4] + x[5]).sin()

        def mark_one_cos_for_offloading(gm, joint_inputs):
            for node in gm.graph.nodes:
                if node.name == "cos_1":
                    node.meta["should_offload"] = True
            return gm

        dim = 10
        self.x = [
            torch.randn(dim, dim, requires_grad=True, device=GPU_TYPE) for _ in range(6)
        ]
        self.fn = fn
        self.joint_custom_pass = mark_one_cos_for_offloading

    """
    The first set of tests are for the case of adding offload nodes to the fwd and bwd graphs.
    """

    @torch._functorch.config.patch(enable_activation_offloading=True)
    def test_partitioner_offload(self):
        torch._dynamo.reset()
        torch._functorch.config.joint_custom_pass = self.joint_custom_pass
        fw_graph, bw_graph = get_fw_bw_graph(self.fn, [self.x])

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6):
    add = torch.ops.aten.add.Tensor(primals_1, primals_2);  primals_1 = primals_2 = None
    sin = torch.ops.aten.sin.default(add)
    add_1 = torch.ops.aten.add.Tensor(primals_3, primals_4);  primals_3 = primals_4 = None
    sin_1 = torch.ops.aten.sin.default(add_1)
    add_2 = torch.ops.aten.add.Tensor(sin, sin_1);  sin = sin_1 = None
    add_3 = torch.ops.aten.add.Tensor(primals_5, primals_6);  primals_5 = primals_6 = None
    sin_2 = torch.ops.aten.sin.default(add_3)
    add_4 = torch.ops.aten.add.Tensor(add_2, sin_2);  add_2 = sin_2 = None
    cos = torch.ops.aten.cos.default(add_3);  add_3 = None
    cos_1 = torch.ops.aten.cos.default(add_1);  add_1 = None
    cpu_offload_cos_1 = torch.ops.prims.device_put.default(cos_1, device(type='cpu'), non_blocking = True);  cos_1 = None
    cos_2 = torch.ops.aten.cos.default(add);  add = None
    return (add_4, cos, cpu_offload_cos_1, cos_2)""",
        )

        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, cos, cpu_offload_cos_1, cos_2, tangents_1):
    mul = torch.ops.aten.mul.Tensor(tangents_1, cos);  cos = None
    gpu_reload_cos_1 = torch.ops.prims.device_put.default(cpu_offload_cos_1, device(type='cuda', index=0), non_blocking = True);  cpu_offload_cos_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, gpu_reload_cos_1);  gpu_reload_cos_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, cos_2);  tangents_1 = cos_2 = None
    return (mul_2, mul_2, mul_1, mul_1, mul, mul)""",
        )

    def test_inductor_offload(self):
        torch._dynamo.reset()

        def run_compiled():
            torch._functorch.config.enable_activation_offloading = True
            torch._functorch.config.joint_custom_pass = self.joint_custom_pass
            return torch.compile(self.fn)(self.x)

        _, (fw_code, bw_code) = run_fw_bw_and_get_code(run_compiled)

        (
            FileCheck()
            .check("buf3 = empty_strided_cpu_pinned(")
            .check("buf3.copy_(buf2, True)")
            .run(fw_code)
        )

        (
            FileCheck()
            .check("buf1 = empty_strided_cuda(")
            .check("buf1.copy_(cpu_offload_cos_1, True)")
            .check("del cpu_offload_cos_1")
            .run(bw_code)
        )

    @torch._functorch.config.patch(
        enable_activation_offloading=True,
        activation_offload_separate_stream=True,
    )
    def test_partitioner_offload_sep_stream(self):
        reset_user_object_tracking()
        torch._dynamo.reset()
        torch._functorch.config.joint_custom_pass = self.joint_custom_pass
        fw_graph, bw_graph = get_fw_bw_graph(self.fn, [self.x])

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6):
    add = torch.ops.aten.add.Tensor(primals_1, primals_2);  primals_1 = primals_2 = None
    sin = torch.ops.aten.sin.default(add)
    add_1 = torch.ops.aten.add.Tensor(primals_3, primals_4);  primals_3 = primals_4 = None
    sin_1 = torch.ops.aten.sin.default(add_1)
    add_2 = torch.ops.aten.add.Tensor(sin, sin_1);  sin = sin_1 = None
    add_3 = torch.ops.aten.add.Tensor(primals_5, primals_6);  primals_5 = primals_6 = None
    sin_2 = torch.ops.aten.sin.default(add_3)
    add_4 = torch.ops.aten.add.Tensor(add_2, sin_2);  add_2 = sin_2 = None
    cos = torch.ops.aten.cos.default(add_3);  add_3 = None
    cos_1 = torch.ops.aten.cos.default(add_1);  add_1 = None
    record_event_default = torch.ops.streams.record_event.default(2, 0);  record_event_default = None
    stream_in_cpu_offload_cos_1 = torch.ops.streams.fork.default(0, 1);  stream_in_cpu_offload_cos_1 = None
    wait_event_default = torch.ops.streams.wait_event.default(2, 1);  wait_event_default = None
    cpu_offload_cos_1 = torch.ops.prims.device_put.default(cos_1, device(type='cpu'), non_blocking = True);  cos_1 = None
    record_event_default_1 = torch.ops.streams.record_event.default(3, 1);  record_event_default_1 = None
    stream_out_cpu_offload_cos_1 = torch.ops.streams.join.default(1, 0);  stream_out_cpu_offload_cos_1 = None
    wait_event_default_1 = torch.ops.streams.wait_event.default(3, 0);  wait_event_default_1 = None
    cos_2 = torch.ops.aten.cos.default(add);  add = None
    return (add_4, cos, cpu_offload_cos_1, cos_2)""",
        )

        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, cos, cpu_offload_cos_1, cos_2, tangents_1):
    mul = torch.ops.aten.mul.Tensor(tangents_1, cos);  cos = None
    stream_in_gpu_reload_cos_1 = torch.ops.streams.fork.default(4, 5);  stream_in_gpu_reload_cos_1 = None
    wait_stream_default = torch.ops.streams.wait_stream.default(5, 4);  wait_stream_default = None
    gpu_reload_cos_1 = torch.ops.prims.device_put.default(cpu_offload_cos_1, device(type='cuda', index=0), non_blocking = True);  cpu_offload_cos_1 = None
    record_event_default = torch.ops.streams.record_event.default(6, 5);  record_event_default = None
    stream_out_gpu_reload_cos_1 = torch.ops.streams.join.default(5, 4);  stream_out_gpu_reload_cos_1 = None
    wait_event_default = torch.ops.streams.wait_event.default(6, 4);  wait_event_default = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, gpu_reload_cos_1);  gpu_reload_cos_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, cos_2);  tangents_1 = cos_2 = None
    return (mul_2, mul_2, mul_1, mul_1, mul, mul)""",
        )

    @torch._functorch.config.patch(
        enable_activation_offloading=True,
        activation_offload_separate_stream=True,
    )
    def test_partitioner_offload_sep_stream_accuracy(self):
        # Run without compilation to get reference gradients
        x_ref = [x.detach().clone().requires_grad_(True) for x in self.x]
        out_ref = self.fn(x_ref)
        out_ref.sum().backward()
        grads_ref = [inp.grad for inp in x_ref]

        # Run with aot_eager compilation and offloading enabled
        reset_user_object_tracking()
        torch._dynamo.reset()
        torch._functorch.config.joint_custom_pass = self.joint_custom_pass
        x_compile = [x.detach().clone().requires_grad_(True) for x in self.x]
        compiled_fn = torch.compile(self.fn, backend="aot_eager")
        out_compiled = compiled_fn(x_compile)
        out_compiled.sum().backward()
        grads_compiled = [inp.grad for inp in x_compile]

        # Verify gradients match between reference and compiled versions
        for grad_ref, grad_compiled in zip(grads_ref, grads_compiled):
            torch.testing.assert_close(
                grad_compiled,
                grad_ref,
                rtol=1e-5,
                atol=1e-5,
            )


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
