# Owner(s): ["oncall: pt2"]

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
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import serialTest
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

    def test_partitioner_offload_ao_ops(self):
        """Test that async offload uses ao.offload + ao.wait_tensor with keepalive."""
        reset_user_object_tracking()
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
            fw_graph, bw_graph = get_fw_bw_graph(self.fn, [self.x])

        fw_code = fw_graph.code
        # Forward: ao.offload produces async CPU tensor, ao.wait_tensor synchronizes
        (
            FileCheck()
            .check("ao.offload.default")
            .check("ao.wait_tensor.default")
            .run(fw_code)
        )
        # No old stream ops should be present
        self.assertNotIn("streams.fork", fw_code)
        self.assertNotIn("streams.join", fw_code)
        self.assertNotIn("streams.record_stream", fw_code)

        # Verify keepalive: ao.wait_tensor for offload should reference the GPU tensor
        # (cos_1) as its 2nd arg to extend its lifetime
        for node in fw_graph.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.ao.wait_tensor.default
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].target == torch.ops.ao.offload.default
            ):
                self.assertEqual(len(node.args), 2)
                offload_node = node.args[0]
                keepalive_node = node.args[1]
                # keepalive should be the same GPU tensor that was offloaded
                self.assertIs(keepalive_node, offload_node.args[0])

        bw_code = bw_graph.code
        # Backward: ao.reload produces async GPU tensor, ao.wait_tensor synchronizes
        (
            FileCheck()
            .check("ao.reload.default")
            .check("ao.wait_tensor.default")
            .run(bw_code)
        )
        self.assertNotIn("streams.fork", bw_code)
        self.assertNotIn("streams.join", bw_code)

        # Verify keepalive: ao.wait_tensor for reload should reference the CPU
        # placeholder as its 2nd arg so its storage is freed after the H2D copy
        for node in bw_graph.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.ao.wait_tensor.default
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].target == torch.ops.ao.reload.default
            ):
                self.assertEqual(len(node.args), 2)
                reload_node = node.args[0]
                keepalive_node = node.args[1]
                # keepalive should be the CPU placeholder that was reloaded
                self.assertIs(keepalive_node, reload_node.args[0])

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

    def test_partitioner_offload_ao_ops_sink_wait(self):
        """Test that sink_wait moves ao.wait_tensor for offload to end of forward graph."""
        reset_user_object_tracking()
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            activation_offload_sink_wait=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
            fw_graph, _ = get_fw_bw_graph(self.fn, [self.x])

        # The ao.wait_tensor for offload should be sunk to just before the output node
        nodes = list(fw_graph.graph.nodes)
        output_node = next(n for n in nodes if n.op == "output")
        output_idx = nodes.index(output_node)

        for node in nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.ao.wait_tensor.default
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].target == torch.ops.ao.offload.default
            ):
                wait_idx = nodes.index(node)
                # ao.wait_tensor should be immediately before output
                self.assertEqual(wait_idx, output_idx - 1)

    def test_partitioner_offload_ao_ops_prefetch(self):
        """Test that prefetch moves ao.reload earlier in backward graph."""
        reset_user_object_tracking()
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            activation_reload_prefetch=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
            _, bw_graph = get_fw_bw_graph(self.fn, [self.x])

        nodes = list(bw_graph.graph.nodes)
        reload_nodes = [
            n
            for n in nodes
            if n.op == "call_function" and n.target == torch.ops.ao.reload.default
        ]
        wait_nodes = [
            n
            for n in nodes
            if n.op == "call_function"
            and n.target == torch.ops.ao.wait_tensor.default
            and isinstance(n.args[0], torch.fx.Node)
            and n.args[0].target == torch.ops.ao.reload.default
        ]
        # Reload should appear before its corresponding wait (prefetched earlier)
        for reload_node in reload_nodes:
            wait_node = next(w for w in wait_nodes if w.args[0] is reload_node)
            self.assertLess(nodes.index(reload_node), nodes.index(wait_node))

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

    def test_wait_tensor_error_no_pending_transfer(self):
        """Test _pop_wait raises RuntimeError for unregistered tensor."""
        from torch._functorch._activation_offloading.offload_ops import (
            _clear_wait_registry,
            _pop_wait,
        )

        _clear_wait_registry()
        x = torch.randn(4, device=GPU_TYPE)
        with self.assertRaisesRegex(RuntimeError, "no pending transfer"):
            _pop_wait(x)

    def test_offload_reload_fake_tensor(self):
        """Test fake tensor implementations for offload, reload, and wait_tensor."""
        import torch._functorch._activation_offloading.offload_ops
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x_gpu = torch.randn(4, 8, device=GPU_TYPE)

            # offload fake: GPU -> CPU with same shape/dtype
            cpu_result = torch.ops.ao.offload.default(x_gpu)
            self.assertEqual(cpu_result.shape, (4, 8))
            self.assertEqual(cpu_result.device.type, "cpu")
            self.assertEqual(cpu_result.dtype, x_gpu.dtype)

            # reload fake: CPU -> GPU with same shape/dtype
            x_cpu = torch.randn(4, 8)
            gpu_result = torch.ops.ao.reload.default(x_cpu, torch.device(GPU_TYPE))
            self.assertEqual(gpu_result.shape, (4, 8))
            self.assertEqual(gpu_result.device.type, GPU_TYPE)
            self.assertEqual(gpu_result.dtype, x_cpu.dtype)

            # wait_tensor fake: returns same tensor (aliasing)
            wait_result = torch.ops.ao.wait_tensor.default(x_gpu)
            self.assertEqual(wait_result.shape, x_gpu.shape)
            self.assertEqual(wait_result.device, x_gpu.device)

    def test_pinned_empty_like_numa_restores_policy(self):
        """Test that _pinned_empty_like_numa restores the thread's prior mempolicy.

        Mocks the NUMA helpers so the test deterministically exercises
        the bind-then-restore path on any host.
        """
        import ctypes
        from unittest.mock import MagicMock, patch

        from torch._functorch._activation_offloading.offload_ops import (
            _MPOL_BIND,
            _pinned_empty_like_numa,
        )

        # Simulate a pre-existing MPOL_BIND to NUMA node 3
        prior_mask = (ctypes.c_ulong * 1)(8)  # bit 3
        saved_policy = (_MPOL_BIND, prior_mask, 64)
        mock_get = MagicMock(return_value=saved_policy)
        mock_set = MagicMock(return_value=True)
        _ops = "torch._functorch._activation_offloading.offload_ops"

        x = torch.randn(64, device=GPU_TYPE)
        with (
            patch(f"{_ops}._init_mempolicy", return_value=True),
            patch(f"{_ops}._gpu_numa_node", return_value=0),
            patch(f"{_ops}._get_mempolicy", mock_get),
            patch(f"{_ops}._set_mempolicy", mock_set),
        ):
            result = _pinned_empty_like_numa(x)

        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(mock_set.call_count, 2)
        bind_call, restore_call = mock_set.call_args_list
        # First call: bind to the GPU's NUMA node
        self.assertEqual(bind_call.args[0], _MPOL_BIND)
        # Second call: restore the original policy (not MPOL_DEFAULT)
        self.assertEqual(restore_call.args[0], _MPOL_BIND)
        self.assertIs(restore_call.args[1], prior_mask)
        self.assertEqual(restore_call.args[2], 64)

    def test_pinned_empty_like_numa_restores_policy_on_failure(self):
        """Test that mempolicy is restored even when allocation raises.

        Mocks _gpu_numa_node and _set_mempolicy/_get_mempolicy to
        deterministically exercise the bind-then-restore path regardless
        of whether the host has NUMA hardware.
        """
        import ctypes
        from unittest.mock import MagicMock, patch

        from torch._functorch._activation_offloading.offload_ops import (
            _MPOL_BIND,
            _pinned_empty_like_numa,
        )

        prior_mask = (ctypes.c_ulong * 1)(8)  # bit 3
        saved_policy = (_MPOL_BIND, prior_mask, 64)
        mock_get = MagicMock(return_value=saved_policy)
        mock_set = MagicMock(return_value=True)
        _ops = "torch._functorch._activation_offloading.offload_ops"

        x = torch.randn(64, device=GPU_TYPE)
        with (
            patch(f"{_ops}._init_mempolicy", return_value=True),
            patch(f"{_ops}._gpu_numa_node", return_value=0),
            patch(f"{_ops}._get_mempolicy", mock_get),
            patch(f"{_ops}._set_mempolicy", mock_set),
            patch("torch.empty_like", side_effect=RuntimeError("OOM")),
        ):
            with self.assertRaises(RuntimeError):
                _pinned_empty_like_numa(x)

        # set_mempolicy must have been called twice: bind then restore
        self.assertEqual(mock_set.call_count, 2)
        bind_call, restore_call = mock_set.call_args_list
        self.assertEqual(bind_call.args[0], _MPOL_BIND)
        self.assertEqual(restore_call.args[0], _MPOL_BIND)
        self.assertIs(restore_call.args[1], prior_mask)
        self.assertEqual(restore_call.args[2], 64)

    def test_pinned_empty_like_numa_fallback(self):
        """Test fallback to default pinned allocation when NUMA lookup fails."""
        from unittest.mock import patch

        from torch._functorch._activation_offloading.offload_ops import (
            _pinned_empty_like_numa,
        )

        x = torch.randn(64, device=GPU_TYPE)
        with patch(
            "torch._functorch._activation_offloading.offload_ops._gpu_numa_node",
            return_value=None,
        ):
            result = _pinned_empty_like_numa(x)
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(result.is_pinned())
        self.assertEqual(result.shape, x.shape)

    def test_gpu_numa_node_uses_device_properties(self):
        """Test that _gpu_numa_node uses torch.cuda.get_device_properties."""
        from torch._functorch._activation_offloading.offload_ops import _gpu_numa_node

        _gpu_numa_node.cache_clear()
        result = _gpu_numa_node(0)
        # On any system with a GPU, result is either a valid node id or None
        if result is not None:
            self.assertGreaterEqual(result, 0)
        _gpu_numa_node.cache_clear()

    def test_wait_tensor_dep_arg_preserved_in_graph(self):
        """Test that the dep arg on wait_tensor is preserved in the FX graph
        and that the op accepts all 3 positional args."""
        import torch._functorch._activation_offloading.offload_ops as _offload_ops  # noqa: F401

        # Verify wait_tensor works with all 3 args at the op level
        x_gpu = torch.randn(4, 4, device=GPU_TYPE)
        cpu_result = torch.ops.ao.offload.default(x_gpu)
        dep = torch.randn(4, device=GPU_TYPE)
        result = torch.ops.ao.wait_tensor.default(cpu_result, x_gpu, dep)
        self.assertEqual(result.shape, cpu_result.shape)
        self.assertEqual(result.device.type, "cpu")

        # Verify the dep arg appears in the FX graph and is not DCE'd
        def fn_with_dep(x, dep_tensor):
            cpu = torch.ops.ao.offload.default(x)
            out = torch.ops.ao.wait_tensor.default(cpu, x, dep_tensor)
            return out

        gm = torch.fx.symbolic_trace(fn_with_dep)
        wait_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.ao.wait_tensor.default
        ]
        self.assertEqual(len(wait_nodes), 1)
        self.assertEqual(len(wait_nodes[0].args), 3)

    def test_wait_tensor_dep_arg_fake_tensor(self):
        """Test wait_tensor fake impl accepts 3 args."""
        import torch._functorch._activation_offloading.offload_ops as _offload_ops  # noqa: F401
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x_gpu = torch.randn(4, 8, device=GPU_TYPE)
            dep = torch.randn(4, device=GPU_TYPE)
            result = torch.ops.ao.wait_tensor.default(x_gpu, None, dep)
            self.assertEqual(result.shape, x_gpu.shape)
            self.assertEqual(result.device, x_gpu.device)

    def test_multiple_tensors_offload_ao_ops(self):
        """Test offloading all saved tensors simultaneously with ao ops."""

        def mark_all_cos_for_offloading(gm, joint_inputs):
            for node in gm.graph.nodes:
                if node.name.startswith("cos"):
                    node.meta["should_offload"] = True
            return gm

        reset_user_object_tracking()
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            joint_custom_pass=mark_all_cos_for_offloading,
        ):
            fw_graph, bw_graph = get_fw_bw_graph(self.fn, [self.x])

        fw_code = fw_graph.code
        self.assertEqual(fw_code.count("ao.offload.default"), 3)
        self.assertEqual(fw_code.count("ao.wait_tensor.default"), 3)

        bw_code = bw_graph.code
        self.assertEqual(bw_code.count("ao.reload.default"), 3)
        self.assertEqual(bw_code.count("ao.wait_tensor.default"), 3)

    def test_multiple_tensors_offload_ao_ops_accuracy(self):
        """Test accuracy when offloading all saved tensors with ao ops."""

        def mark_all_cos_for_offloading(gm, joint_inputs):
            for node in gm.graph.nodes:
                if node.name.startswith("cos"):
                    node.meta["should_offload"] = True
            return gm

        x_ref = [x.detach().clone().requires_grad_(True) for x in self.x]
        out_ref = self.fn(x_ref)
        out_ref.sum().backward()
        grads_ref = [inp.grad for inp in x_ref]

        reset_user_object_tracking()
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            joint_custom_pass=mark_all_cos_for_offloading,
        ):
            fw_graph, bw_graph = get_fw_bw_graph(self.fn, [self.x])

        # Verify ao ops are used (not old stream ops)
        self.assertIn("ao.offload.default", fw_graph.code)
        self.assertIn("ao.reload.default", bw_graph.code)

        reset_user_object_tracking()
        torch._dynamo.reset()
        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            joint_custom_pass=mark_all_cos_for_offloading,
        ):
            x_compile = [x.detach().clone().requires_grad_(True) for x in self.x]
            compiled_fn = torch.compile(self.fn, backend="aot_eager")
            out_compiled = compiled_fn(x_compile)
            out_compiled.sum().backward()
            grads_compiled = [inp.grad for inp in x_compile]

        for grad_ref, grad_compiled in zip(grads_ref, grads_compiled):
            torch.testing.assert_close(grad_compiled, grad_ref, rtol=1e-5, atol=1e-5)

    def test_inductor_offload_ao_ops(self):
        """Test inductor code generation with ao ops."""
        reset_user_object_tracking()
        torch._dynamo.reset()

        def run_compiled():
            return torch.compile(self.fn)(self.x)

        with torch._functorch.config.patch(
            enable_activation_offloading=True,
            activation_offload_separate_stream=True,
            joint_custom_pass=self.joint_custom_pass,
        ):
            _, (fw_code, bw_code) = run_fw_bw_and_get_code(run_compiled)

        (FileCheck().check("ao.offload").check("ao.wait_tensor").run(fw_code))
        (FileCheck().check("ao.reload").check("ao.wait_tensor").run(bw_code))

    def test_estimate_transfer_time_config(self):
        """Test bandwidth config controls transfer time estimation."""
        from torch._functorch._activation_offloading.activation_offloading import (
            _estimate_transfer_time_in_ms,
        )

        size_bytes = 1024**3  # 1 GB
        # 1 GB at 50 GB/s = 20 ms
        with torch._functorch.config.patch(activation_offload_cpu_gpu_bw=50.0):
            self.assertAlmostEqual(_estimate_transfer_time_in_ms(size_bytes), 20.0)
        # 1 GB at 25 GB/s = 40 ms
        with torch._functorch.config.patch(activation_offload_cpu_gpu_bw=25.0):
            self.assertAlmostEqual(_estimate_transfer_time_in_ms(size_bytes), 40.0)


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

    def test_default_partitioner_offload_ao_ops(self):
        """Test ao ops with default partitioner."""
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

        fw_code = fw_graph.code
        (
            FileCheck()
            .check("ao.offload.default")
            .check("ao.wait_tensor.default")
            .run(fw_code)
        )
        self.assertNotIn("streams.fork", fw_code)
        self.assertNotIn("streams.join", fw_code)

        bw_code = bw_graph.code
        (
            FileCheck()
            .check("ao.reload.default")
            .check("ao.wait_tensor.default")
            .run(bw_code)
        )
        self.assertNotIn("streams.fork", bw_code)
        self.assertNotIn("streams.join", bw_code)

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
