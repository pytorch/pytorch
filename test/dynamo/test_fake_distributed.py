# Owner(s): ["module: dynamo"]
from unittest import skipIf

import torch
import torch.distributed as dist
from torch._dynamo.test_case import TestCase as DynamoTestCase
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    normalize_gm,
)
from torch.testing._internal.common_utils import instantiate_parametrized_tests


if dist.is_available():
    from torch.distributed._functional_collectives import (
        all_to_all_single_autograd,
        wait_tensor,
    )
    from torch.distributed.device_mesh import init_device_mesh


def normalize_graph(gm):
    return normalize_gm(gm.print_readable(print_output=False))


@skipIf(not dist.is_available(), "requires distributed")
class TestFakeDistributed(DynamoTestCase):
    def setUp(self):
        # Use FakeProcessGroup to run tests on a single process
        dist.init_process_group(backend="fake", rank=0, world_size=2)
        self.local_rank = 0
        self.world_size = 2

    def tearDown(self):
        dist.destroy_process_group()

    def test_all_to_all_single_autograd(self):
        backend = AotEagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(x):
            return all_to_all_single_autograd(
                x,
                None,  # Will use equal splits
                None,  # Will use equal splits
                group=dist.group.WORLD,
            )

        # Test backed shapes
        x = torch.randn(8, 8, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        wait_tensor(fn(x))
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", primals_2: "Sym(s27)", primals_3: "f32[s77, s27]"):
        floordiv: "Sym((s77//2))" = primals_1 // 2

        all_to_all_single: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.all_to_all_single.default(primals_3, [floordiv, floordiv], [floordiv, floordiv], '0');  primals_3 = None

        wait_tensor: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
        return (wait_tensor, primals_1, primals_2, floordiv)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            normalize_graph(backend.bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", primals_2: "Sym(s27)", floordiv: "Sym((s77//2))", tangents_1: "f32[2*((s77//2)), s27]"):
        all_to_all_single_1: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.all_to_all_single.default(tangents_1, [floordiv, floordiv], [floordiv, floordiv], '0');  tangents_1 = floordiv = None
        wait_tensor_1: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
        return (None, None, wait_tensor_1)
""",  # noqa: B950
        )

        backend.fw_graphs.clear()
        backend.bw_graphs.clear()

        # Test unbacked shapes
        x = torch.randn(8, 8, 8, requires_grad=True)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        torch._dynamo.decorators.mark_unbacked(x, 2)
        wait_tensor(fn(x))
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(u0)", primals_2: "Sym(u1)", primals_3: "Sym(u2)", primals_4: "f32[u0, u1, u2]"):
        ge: "Sym(u0 >= 0)" = primals_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        ge_1: "Sym(u1 >= 0)" = primals_2 >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None
        ge_2: "Sym(u2 >= 0)" = primals_3 >= 0
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u2 >= 0 on node 'ge_2'");  ge_2 = _assert_scalar_2 = None
        floordiv: "Sym((u0//2))" = primals_1 // 2

        all_to_all_single: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.all_to_all_single.default(primals_4, [floordiv, floordiv], [floordiv, floordiv], '0');  primals_4 = None

        wait_tensor: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
        return (wait_tensor, primals_1, primals_2, primals_3, floordiv)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            normalize_graph(backend.bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(u0)", primals_2: "Sym(u1)", primals_3: "Sym(u2)", floordiv: "Sym((u0//2))", tangents_1: "f32[2*((u0//2)), u1, u2]"):
        all_to_all_single_1: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.all_to_all_single.default(tangents_1, [floordiv, floordiv], [floordiv, floordiv], '0');  tangents_1 = floordiv = None
        wait_tensor_1: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
        return (None, None, None, wait_tensor_1)
""",  # noqa: B950
        )

    def test_device_mesh_get_local_rank(self):
        device_mesh = init_device_mesh(
            device_type="cpu",
            mesh_shape=(self.world_size,),
            mesh_dim_names=("dp",),  # data parallel dimension
        )

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            local_rank = device_mesh.get_local_rank()
            global_rank = device_mesh.get_rank()
            if "dp" not in device_mesh.mesh_dim_names:
                x = x * 2
            return x + local_rank + global_rank

        x = torch.ones(10)
        res = fn(x)
        self.assertEqual(res, x)

    def test_device_mesh_flatten(self):
        device_mesh = init_device_mesh(
            device_type="cpu",
            mesh_shape=(
                1,
                self.world_size,
            ),
            mesh_dim_names=("dp", "tp"),
        )
        self.assertEqual(device_mesh.get_coordinate(), [0, 0])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            dm = device_mesh._flatten()
            return x + 1, dm.get_coordinate()

        x = torch.ones(10)
        res = fn(x)
        self.assertEqual(res, (x + 1, [0]))

    def test_device_mesh_init_skip_after_graph_break(self):
        device_mesh = init_device_mesh(
            device_type="cpu",
            mesh_shape=(1, self.world_size),
            mesh_dim_names=("dp", "tp"),
        )

        @torch.compile(backend="eager")
        def fn(x):
            # Graph break so the subsequent DeviceMesh construction happens
            # at runtime, where eval_frame intercepts DeviceMesh.__init__
            # as a top-level frame.
            torch._dynamo.graph_break()
            layout = device_mesh._get_slice_mesh_layout(("tp",))
            sub = device_mesh._create_sub_mesh(layout, ("tp",))
            return x + sub.size()

        x = torch.ones(10)
        res = fn(x)
        self.assertEqual(res, x + self.world_size)


instantiate_parametrized_tests(TestFakeDistributed)


@skipIf(not dist.is_available(), "requires distributed")
class TestFakeDistributedP2P(DynamoTestCase):
    def setUp(self):
        dist.init_process_group(backend="fake", rank=0, world_size=2)

    def tearDown(self):
        dist.destroy_process_group()

    @torch._dynamo.config.patch(enable_p2p_compilation=True)
    def test_compiled_isend_graph(self):
        backend = EagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(tensor):
            req = dist.isend(tensor, 1)
            req.wait()

        tensor = torch.ones(10)
        fn(tensor)
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_tensor_: "f32[10]"):
        l_tensor_ = L_tensor_

        tensor: "f32[0]" = torch.ops._c10d_functional.isend(l_tensor_, 1, 0, '0');  l_tensor_ = None

        wait_tensor: "f32[0]" = torch.distributed._functional_collectives.wait_tensor(tensor);  tensor = wait_tensor = None
        return ()
""",
        )

    @torch._dynamo.config.patch(enable_p2p_compilation=True)
    def test_compiled_fire_and_forget_isend_graph(self):
        backend = EagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(tensor):
            dist.isend(tensor, 1)
            return tensor

        tensor = torch.ones(4)
        fn(tensor)
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_tensor_: "f32[4]"):
        l_tensor_ = L_tensor_

        tensor: "f32[0]" = torch.ops._c10d_functional.isend(l_tensor_, 1, 0, '0');  tensor = None
        return (l_tensor_,)
""",
        )

    @torch._dynamo.config.patch(enable_p2p_compilation=True)
    def test_compiled_irecv_graph(self):
        backend = EagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(tensor):
            req = dist.irecv(tensor, 1)
            req.wait()

        tensor = torch.zeros(10)
        fn(tensor)
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_tensor_: "f32[10]"):
        l_tensor_ = L_tensor_

        tensor: "f32[10]" = torch.ops._c10d_functional.irecv(l_tensor_, 1, 0, '0');  l_tensor_ = None

        req: "f32[10]" = torch.ops._c10d_functional.wait_tensor(tensor);  tensor = None

        wait_tensor_1: "f32[10]" = torch.distributed._functional_collectives.wait_tensor(req);  req = wait_tensor_1 = None
        return ()
""",
        )

    @torch._dynamo.config.patch(enable_p2p_compilation=True)
    def test_compiled_batch_isend_irecv_mixed_graph(self):
        backend = EagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(send_tensor, recv_tensor):
            ops = [
                dist.P2POp(dist.isend, send_tensor, 1),
                dist.P2POp(dist.irecv, recv_tensor, 1),
            ]
            work = dist.batch_isend_irecv(ops)
            for w in work:
                w.wait()

        send_tensor = torch.ones(10)
        recv_tensor = torch.zeros(10)
        fn(send_tensor, recv_tensor)
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            (
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_send_tensor_: "f32[10]", L_recv_tensor_: "f32[10]"):
        l_send_tensor_ = L_send_tensor_
        l_recv_tensor_ = L_recv_tensor_

"""
                "        batch_p2p_ops = torch.ops._c10d_functional.batch_p2p_ops("
                "['isend', 'irecv'], [1, 1], [0, 0], [l_send_tensor_, "
                "l_recv_tensor_], '0');  l_send_tensor_ = l_recv_tensor_ = None\n"
                """\
        t: "f32[0]" = batch_p2p_ops[0]
        t_1: "f32[10]" = batch_p2p_ops[1];  batch_p2p_ops = None

        w: "f32[10]" = torch.ops._c10d_functional.wait_tensor(t_1);  t_1 = None

        wait_tensor_1: "f32[0]" = torch.distributed._functional_collectives.wait_tensor(t);  t = wait_tensor_1 = None
        wait_tensor_2: "f32[10]" = torch.distributed._functional_collectives.wait_tensor(w);  w = wait_tensor_2 = None
        return ()
"""
            ),
        )

    @torch._dynamo.config.patch(enable_p2p_compilation=True)
    def test_compiled_p2p_interleave_graph(self):
        backend = EagerAndRecordGraphs()
        nxt, prv = 1, 1  # rank=0 in world_size=2

        @torch.compile(fullgraph=True, backend=backend)
        def fn(x0, x1, y0, y1):
            work = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, x0, nxt),
                    dist.P2POp(dist.irecv, y0, prv),
                ]
            )
            t0 = x0 * 2 + 1
            for w in work:
                w.wait()
            a = y0 + t0
            work = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, a, nxt),
                    dist.P2POp(dist.irecv, y1, prv),
                ]
            )
            t1 = a * 1.000244140625
            for w in work:
                w.wait()
            return y1 + t1

        M, N = 64, 64
        x0 = torch.ones(M, N)
        x1 = torch.ones(M, N)
        y0 = torch.zeros(M, N)
        y1 = torch.zeros(M, N)
        fn(x0, x1, y0, y1)
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x0_: "f32[64, 64]", L_y0_: "f32[64, 64]", L_y1_: "f32[64, 64]"):
        l_x0_ = L_x0_
        l_y0_ = L_y0_
        l_y1_ = L_y1_

        batch_p2p_ops = torch.ops._c10d_functional.batch_p2p_ops(['isend', 'irecv'], [1, 1], [0, 0], [l_x0_, l_y0_], '0')
        t: "f32[0]" = batch_p2p_ops[0]
        t_1: "f32[64, 64]" = batch_p2p_ops[1];  batch_p2p_ops = None

        w: "f32[64, 64]" = torch.ops._c10d_functional.wait_tensor(t_1);  t_1 = None

        mul: "f32[64, 64]" = l_x0_ * 2;  l_x0_ = None
        t0: "f32[64, 64]" = mul + 1;  mul = None

        wait_tensor_1: "f32[0]" = torch.distributed._functional_collectives.wait_tensor(t);  t = wait_tensor_1 = None
        wait_tensor_2: "f32[64, 64]" = torch.distributed._functional_collectives.wait_tensor(w);  w = wait_tensor_2 = None

        a: "f32[64, 64]" = l_y0_ + t0;  l_y0_ = t0 = None

        batch_p2p_ops_1 = torch.ops._c10d_functional.batch_p2p_ops(['isend', 'irecv'], [1, 1], [0, 0], [a, l_y1_], '0')
        t_2: "f32[0]" = batch_p2p_ops_1[0]
        t_3: "f32[64, 64]" = batch_p2p_ops_1[1];  batch_p2p_ops_1 = None

        w_1: "f32[64, 64]" = torch.ops._c10d_functional.wait_tensor(t_3);  t_3 = None

        t1: "f32[64, 64]" = a * 1.000244140625;  a = None

        wait_tensor_4: "f32[0]" = torch.distributed._functional_collectives.wait_tensor(t_2);  t_2 = wait_tensor_4 = None
        wait_tensor_5: "f32[64, 64]" = torch.distributed._functional_collectives.wait_tensor(w_1);  w_1 = wait_tensor_5 = None

        add_2: "f32[64, 64]" = l_y1_ + t1;  l_y1_ = t1 = None
        return (add_2,)
""",
        )

    @torch._dynamo.config.patch(enable_p2p_compilation=True)
    def test_mutating_p2p_op_graph_breaks(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(tensor):
            op = dist.P2POp(dist.isend, tensor, 1)
            op.tag = 1
            dist.batch_isend_irecv([op])

        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "P2POp"):
            fn(torch.ones(4))


instantiate_parametrized_tests(TestFakeDistributedP2P)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
