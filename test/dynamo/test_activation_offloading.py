# Owner(s): ["oncall: pt2"]
# flake8: noqa: B950

import unittest
from functools import partial

import pytest

import torch
import torch._functorch.config
from functorch.compile import (
    aot_function,
    default_decompositions,
    default_partition,
    min_cut_rematerialization_partition,
)
from torch._dynamo.graph_bytecode_inputs import reset_user_object_tracking
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, serialTest, TestCase
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


@unittest.skipIf(not HAS_GPU, "requires GPU")
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

    def test_partitioner_offload(self):
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
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
    gpu_reload_cos_1 = torch.ops.prims.device_put.default(cpu_offload_cos_1, device(type='GPU_TYPE', index=0), non_blocking = True);  cpu_offload_cos_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, gpu_reload_cos_1);  gpu_reload_cos_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, cos_2);  tangents_1 = cos_2 = None
    return (mul_2, mul_2, mul_1, mul_1, mul, mul)""".replace("GPU_TYPE", GPU_TYPE),
        )

    def test_inductor_offload(self):
        torch._dynamo.reset()

        def run_compiled():
            return torch.compile(self.fn)(self.x)

        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
            _, (fw_code, bw_code) = run_fw_bw_and_get_code(run_compiled)

        (
            FileCheck()
            .check("buf3 = empty_strided_cpu_pinned(")
            .check("buf3.copy_(buf2, True)")
            .run(fw_code)
        )

        (
            FileCheck()
            .check("buf1 = empty_strided_GPU_TYPE(".replace("GPU_TYPE", GPU_TYPE))
            .check("buf1.copy_(cpu_offload_cos_1, True)")
            .check("del cpu_offload_cos_1")
            .run(bw_code)
        )

    def test_partitioner_offload_sep_stream(self):
        reset_user_object_tracking()
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
            fw_graph, bw_graph = get_fw_bw_graph(self.fn, [self.x])

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6):
    add = torch.ops.aten.add.Tensor(primals_1, primals_2);  primals_1 = primals_2 = None
    sin = torch.ops.aten.sin.default(add)
    add_1 = torch.ops.aten.add.Tensor(primals_3, primals_4);  primals_3 = primals_4 = None
    sin_1 = torch.ops.aten.sin.default(add_1)
    add_2 = torch.ops.aten.add.Tensor(sin, sin_1)
    add_3 = torch.ops.aten.add.Tensor(primals_5, primals_6);  primals_5 = primals_6 = None
    sin_2 = torch.ops.aten.sin.default(add_3)
    add_4 = torch.ops.aten.add.Tensor(add_2, sin_2)
    cos = torch.ops.aten.cos.default(add_3)
    cos_1 = torch.ops.aten.cos.default(add_1)
    subgraph_record_event_default = self.subgraph_record_event_default
    control_deps = torch.ops.higher_order.control_deps((add, sin, add_1, sin_1, add_2, add_3, sin_2, add_4, cos, cos_1), subgraph_record_event_default, add, add_4, cos, cos_1);  add = sin = add_1 = sin_1 = add_2 = add_3 = sin_2 = cos = cos_1 = subgraph_record_event_default = None
    getitem_3 = control_deps[4]
    getitem_2 = control_deps[3]
    getitem_1 = control_deps[2];  getitem_1 = None
    getitem = control_deps[1]
    subgraph_wait_event_default = self.subgraph_wait_event_default
    control_deps_1 = torch.ops.higher_order.control_deps((control_deps,), subgraph_wait_event_default);  control_deps = subgraph_wait_event_default = control_deps_1 = None
    record_stream_cos_1 = torch.ops.streams.record_stream.default(getitem_3, 1);  record_stream_cos_1 = None
    cpu_offload_cos_1 = torch.ops.prims.device_put.default(getitem_3, device(type='cpu'), non_blocking = True)
    subgraph_record_event_default_1 = self.subgraph_record_event_default_1
    control_deps_2 = torch.ops.higher_order.control_deps((cpu_offload_cos_1,), subgraph_record_event_default_1, cpu_offload_cos_1);  subgraph_record_event_default_1 = None
    getitem_4 = control_deps_2[1];  getitem_4 = None
    subgraph_wait_event_default_1 = self.subgraph_wait_event_default_1
    control_deps_3 = torch.ops.higher_order.control_deps((control_deps_2,), subgraph_wait_event_default_1);  control_deps_2 = subgraph_wait_event_default_1 = control_deps_3 = None
    keep_alive_cos_1 = torch.ops.streams.record_stream.default(getitem_3, 0);  getitem_3 = keep_alive_cos_1 = None
    cos_2 = torch.ops.aten.cos.default(getitem);  getitem = None
    return (add_4, getitem_2, cpu_offload_cos_1, cos_2)""",
        )

        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, cos, cpu_offload_cos_1, cos_2, tangents_1):
    mul = torch.ops.aten.mul.Tensor(tangents_1, cos);  cos = None
    gpu_reload_cos_1 = torch.ops.prims.device_put.default(cpu_offload_cos_1, device(type='GPU_TYPE', index=0), non_blocking = True);  cpu_offload_cos_1 = None
    subgraph_record_event_default = self.subgraph_record_event_default
    control_deps = torch.ops.higher_order.control_deps((gpu_reload_cos_1, mul), subgraph_record_event_default, gpu_reload_cos_1, mul);  gpu_reload_cos_1 = mul = subgraph_record_event_default = None
    getitem_1 = control_deps[2]
    getitem = control_deps[1]
    subgraph_wait_event_default = self.subgraph_wait_event_default
    control_deps_1 = torch.ops.higher_order.control_deps((control_deps,), subgraph_wait_event_default);  control_deps = subgraph_wait_event_default = control_deps_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, getitem);  getitem = None
    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, cos_2);  tangents_1 = cos_2 = None
    return (mul_2, mul_2, mul_1, mul_1, getitem_1, getitem_1)""".replace("GPU_TYPE", GPU_TYPE),
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
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
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

    def test_partitioner_offload_sep_stream_reorder(self):
        reset_user_object_tracking()
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            activation_offload_sink_wait=True,
            activation_reload_prefetch=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
            fw_graph, bw_graph = get_fw_bw_graph(self.fn, [self.x])

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6):
    add = torch.ops.aten.add.Tensor(primals_1, primals_2);  primals_1 = primals_2 = None
    sin = torch.ops.aten.sin.default(add)
    add_1 = torch.ops.aten.add.Tensor(primals_3, primals_4);  primals_3 = primals_4 = None
    sin_1 = torch.ops.aten.sin.default(add_1)
    add_2 = torch.ops.aten.add.Tensor(sin, sin_1)
    add_3 = torch.ops.aten.add.Tensor(primals_5, primals_6);  primals_5 = primals_6 = None
    sin_2 = torch.ops.aten.sin.default(add_3)
    add_4 = torch.ops.aten.add.Tensor(add_2, sin_2)
    cos = torch.ops.aten.cos.default(add_3)
    cos_1 = torch.ops.aten.cos.default(add_1)
    subgraph_record_event_default = self.subgraph_record_event_default
    control_deps = torch.ops.higher_order.control_deps((add, sin, add_1, sin_1, add_2, add_3, sin_2, add_4, cos, cos_1), subgraph_record_event_default, add, add_4, cos, cos_1);  add = sin = add_1 = sin_1 = add_2 = add_3 = sin_2 = cos = cos_1 = subgraph_record_event_default = None
    getitem_3 = control_deps[4]
    getitem_2 = control_deps[3]
    getitem_1 = control_deps[2];  getitem_1 = None
    getitem = control_deps[1]
    subgraph_wait_event_default = self.subgraph_wait_event_default
    control_deps_1 = torch.ops.higher_order.control_deps((control_deps,), subgraph_wait_event_default);  control_deps = subgraph_wait_event_default = control_deps_1 = None
    record_stream_cos_1 = torch.ops.streams.record_stream.default(getitem_3, 1);  record_stream_cos_1 = None
    cpu_offload_cos_1 = torch.ops.prims.device_put.default(getitem_3, device(type='cpu'), non_blocking = True)
    subgraph_record_event_default_1 = self.subgraph_record_event_default_1
    control_deps_2 = torch.ops.higher_order.control_deps((cpu_offload_cos_1,), subgraph_record_event_default_1, cpu_offload_cos_1);  subgraph_record_event_default_1 = None
    getitem_4 = control_deps_2[1];  getitem_4 = None
    cos_2 = torch.ops.aten.cos.default(getitem);  getitem = None
    subgraph_wait_event_default_2 = self.subgraph_wait_event_default_2
    control_deps_3 = torch.ops.higher_order.control_deps((control_deps_2, cos_2), subgraph_wait_event_default_2, cos_2);  control_deps_2 = cos_2 = subgraph_wait_event_default_2 = None
    getitem_5 = control_deps_3[1];  control_deps_3 = None
    record_stream_default = torch.ops.streams.record_stream.default(getitem_3, 0);  getitem_3 = record_stream_default = None
    return (add_4, getitem_2, cpu_offload_cos_1, getitem_5)""",
        )

        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, cos, cpu_offload_cos_1, cos_2, tangents_1):
    gpu_reload_cos_1 = torch.ops.prims.device_put.default(cpu_offload_cos_1, device(type='GPU_TYPE', index=0), non_blocking = True);  cpu_offload_cos_1 = None
    subgraph_record_event_default = self.subgraph_record_event_default
    control_deps = torch.ops.higher_order.control_deps((gpu_reload_cos_1,), subgraph_record_event_default, gpu_reload_cos_1);  gpu_reload_cos_1 = subgraph_record_event_default = None
    getitem = control_deps[1]
    mul = torch.ops.aten.mul.Tensor(tangents_1, cos);  cos = None
    subgraph_wait_event_default = self.subgraph_wait_event_default
    control_deps_1 = torch.ops.higher_order.control_deps((control_deps, mul), subgraph_wait_event_default, mul);  control_deps = mul = subgraph_wait_event_default = None
    getitem_1 = control_deps_1[1];  control_deps_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, getitem);  getitem = None
    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, cos_2);  tangents_1 = cos_2 = None
    return (mul_2, mul_2, mul_1, mul_1, getitem_1, getitem_1)""".replace("GPU_TYPE", GPU_TYPE),
        )

    @serialTest()
    def test_partitioner_offload_sep_stream_reorder_accuracy(self):
        # need larger dimension so that memcpy takes longer, and the code is at the risk of
        # premature memory deallocation
        dim = 1024 * 8
        x_larger = [
            torch.randn(dim, dim, requires_grad=True, device=GPU_TYPE) for _ in range(6)
        ]
        # Run without compilation to get reference gradients
        x_ref = [x.detach().clone().requires_grad_(True) for x in x_larger]
        out_ref = self.fn(x_ref)
        out_ref.sum().backward()
        grads_ref = [inp.grad for inp in x_ref]

        # Run with aot_eager compilation and offloading enabled
        reset_user_object_tracking()
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            activation_offload_sink_wait=True,
            activation_reload_prefetch=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
            x_compile = [x.detach().clone().requires_grad_(True) for x in x_larger]
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


@unittest.skipIf(not HAS_GPU, "requires GPU")
class DefaultPartitionerActivationOffloadingTests(TestCase):
    """Tests activation offloading with the default partitioner.

    Note: The default partitioner saves intermediate tensors (like add results)
    and recomputes derived values (like cos) in backward. This differs from
    min-cut partitioner which may save the derived values directly.
    Therefore, we mark 'add_1' for offloading instead of 'cos_1'.
    """

    def setUp(self):
        super().setUp()

        def fn(x):
            return (x[0] + x[1]).sin() + (x[2] + x[3]).sin() + (x[4] + x[5]).sin()

        # Mark add_1 for offloading - this IS saved in forward with default partitioner
        def mark_add_for_offloading(gm, joint_inputs):
            for node in gm.graph.nodes:
                if node.name == "add_1":
                    node.meta["should_offload"] = True
            return gm

        dim = 10
        self.x = [
            torch.randn(dim, dim, requires_grad=True, device=GPU_TYPE) for _ in range(6)
        ]
        self.fn = fn
        self.joint_custom_pass = mark_add_for_offloading

    def test_default_partitioner_offload(self):
        """Test basic offloading with default partitioner"""
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
            fw_graph, bw_graph = get_fw_bw_graph(
                self.fn, [self.x], partitioner=default_partition
            )

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6):
    add = torch.ops.aten.add.Tensor(primals_1, primals_2);  primals_1 = primals_2 = None
    sin = torch.ops.aten.sin.default(add)
    add_1 = torch.ops.aten.add.Tensor(primals_3, primals_4);  primals_3 = primals_4 = None
    sin_1 = torch.ops.aten.sin.default(add_1)
    cpu_offload_add_1 = torch.ops.prims.device_put.default(add_1, device(type='cpu'), non_blocking = True);  add_1 = None
    add_2 = torch.ops.aten.add.Tensor(sin, sin_1);  sin = sin_1 = None
    add_3 = torch.ops.aten.add.Tensor(primals_5, primals_6);  primals_5 = primals_6 = None
    sin_2 = torch.ops.aten.sin.default(add_3)
    add_4 = torch.ops.aten.add.Tensor(add_2, sin_2);  add_2 = sin_2 = None
    return (add_4, add, cpu_offload_add_1, add_3)""",
        )

        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, add, cpu_offload_add_1, add_3, tangents_1):
    cos = torch.ops.aten.cos.default(add_3);  add_3 = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, cos);  cos = None
    gpu_reload_add_1 = torch.ops.prims.device_put.default(cpu_offload_add_1, device(type='GPU_TYPE', index=0), non_blocking = True);  cpu_offload_add_1 = None
    cos_1 = torch.ops.aten.cos.default(gpu_reload_add_1);  gpu_reload_add_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, cos_1);  cos_1 = None
    cos_2 = torch.ops.aten.cos.default(add);  add = None
    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, cos_2);  tangents_1 = cos_2 = None
    return (mul_2, mul_2, mul_1, mul_1, mul, mul)""".replace("GPU_TYPE", GPU_TYPE),
        )

    def test_default_partitioner_offload_sep_stream(self):
        """Test offloading with separate streams using default partitioner"""
        reset_user_object_tracking()
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
            fw_graph, bw_graph = get_fw_bw_graph(
                self.fn, [self.x], partitioner=default_partition
            )

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6):
    add = torch.ops.aten.add.Tensor(primals_1, primals_2);  primals_1 = primals_2 = None
    sin = torch.ops.aten.sin.default(add)
    add_1 = torch.ops.aten.add.Tensor(primals_3, primals_4);  primals_3 = primals_4 = None
    sin_1 = torch.ops.aten.sin.default(add_1)
    subgraph_record_event_default = self.subgraph_record_event_default
    control_deps = torch.ops.higher_order.control_deps((add, sin, add_1, sin_1), subgraph_record_event_default, add, sin, add_1, sin_1);  sin = add_1 = sin_1 = subgraph_record_event_default = None
    getitem_3 = control_deps[4]
    getitem_2 = control_deps[3]
    getitem_1 = control_deps[2]
    getitem = control_deps[1];  getitem = None
    subgraph_wait_event_default = self.subgraph_wait_event_default
    control_deps_1 = torch.ops.higher_order.control_deps((control_deps,), subgraph_wait_event_default);  control_deps = subgraph_wait_event_default = control_deps_1 = None
    record_stream_add_1 = torch.ops.streams.record_stream.default(getitem_2, 1);  record_stream_add_1 = None
    cpu_offload_add_1 = torch.ops.prims.device_put.default(getitem_2, device(type='cpu'), non_blocking = True)
    subgraph_record_event_default_1 = self.subgraph_record_event_default_1
    control_deps_2 = torch.ops.higher_order.control_deps((cpu_offload_add_1,), subgraph_record_event_default_1, cpu_offload_add_1);  subgraph_record_event_default_1 = None
    getitem_4 = control_deps_2[1];  getitem_4 = None
    subgraph_wait_event_default_1 = self.subgraph_wait_event_default_1
    control_deps_3 = torch.ops.higher_order.control_deps((control_deps_2,), subgraph_wait_event_default_1);  control_deps_2 = subgraph_wait_event_default_1 = control_deps_3 = None
    keep_alive_add_1 = torch.ops.streams.record_stream.default(getitem_2, 0);  getitem_2 = keep_alive_add_1 = None
    add_2 = torch.ops.aten.add.Tensor(getitem_1, getitem_3);  getitem_1 = getitem_3 = None
    add_3 = torch.ops.aten.add.Tensor(primals_5, primals_6);  primals_5 = primals_6 = None
    sin_2 = torch.ops.aten.sin.default(add_3)
    add_4 = torch.ops.aten.add.Tensor(add_2, sin_2);  add_2 = sin_2 = None
    return (add_4, add, cpu_offload_add_1, add_3)""",
        )

        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, add, cpu_offload_add_1, add_3, tangents_1):
    cos = torch.ops.aten.cos.default(add_3);  add_3 = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, cos)
    gpu_reload_add_1 = torch.ops.prims.device_put.default(cpu_offload_add_1, device(type='GPU_TYPE', index=0), non_blocking = True);  cpu_offload_add_1 = None
    subgraph_record_event_default = self.subgraph_record_event_default
    control_deps = torch.ops.higher_order.control_deps((gpu_reload_add_1, cos, mul), subgraph_record_event_default, gpu_reload_add_1, mul);  gpu_reload_add_1 = cos = mul = subgraph_record_event_default = None
    getitem_1 = control_deps[2]
    getitem = control_deps[1]
    subgraph_wait_event_default = self.subgraph_wait_event_default
    control_deps_1 = torch.ops.higher_order.control_deps((control_deps,), subgraph_wait_event_default);  control_deps = subgraph_wait_event_default = control_deps_1 = None
    cos_1 = torch.ops.aten.cos.default(getitem);  getitem = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, cos_1);  cos_1 = None
    cos_2 = torch.ops.aten.cos.default(add);  add = None
    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, cos_2);  tangents_1 = cos_2 = None
    return (mul_2, mul_2, mul_1, mul_1, getitem_1, getitem_1)""".replace("GPU_TYPE", GPU_TYPE),
        )

    def test_default_partitioner_offload_accuracy(self):
        """Test that offloading with default partitioner produces correct gradients"""
        # Run without compilation to get reference gradients
        x_ref = [x.detach().clone().requires_grad_(True) for x in self.x]
        out_ref = self.fn(x_ref)
        out_ref.sum().backward()
        grads_ref = [inp.grad for inp in x_ref]

        # Run with aot_eager compilation and offloading enabled using default partitioner
        reset_user_object_tracking()
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
            x_compile = [x.detach().clone().requires_grad_(True) for x in self.x]
            compiled_fn = aot_function(
                self.fn,
                fw_compiler=lambda g, _: g,
                bw_compiler=lambda g, _: g,
                partition_fn=default_partition,
                decompositions=default_decompositions,
            )
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
