# Owner(s): ["oncall: distributed"]

import contextlib

import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import ShardOrderEntry
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._debug_mode import _OpCall, _RedistributeCall, DebugMode
from torch.utils._python_dispatch import TorchDispatchMode


@requires_cuda
class TestDTensorDebugMode(TestCase):
    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        self.world_size = 8
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )
        self.device_type = "cuda"

    def test_debug_mode_mm(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        x = torch.randn(1, 8, requires_grad=False)
        y = torch.randn(1, 32, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)

        with DebugMode(
            record_torchfunction=True, record_ids=True, record_output=True
        ) as debug_mode:
            torch.mm(x_dtensor, y_dtensor).sum()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch.mm(dt$0: f32[8, 8]| S(0), dt$1: f32[8, 32]| S(0))  ->  dt$6: f32[8, 32]| S(0)
    aten::mm(dt$0: f32[8, 8]| S(0), dt$1: f32[8, 32]| S(0))
      redistribute_input(1, S(0) -> R)
        redistribute_input(t$2: f32[1, 32], trace: S(0)->R)
          _c10d_functional::all_gather_into_tensor(t$2: f32[1, 32], 8, 0)  ->  t$3: f32[8, 32]
          _c10d_functional::wait_tensor(t$3: f32[8, 32])  ->  t$3: f32[8, 32]
      aten::mm(t$4: f32[1, 8], t$3: f32[8, 32])  ->  t$5: f32[1, 32]
  <method 'sum' of 'torch._C.TensorBase' objects>(dt$6: f32[8, 32]| S(0))  ->  dt$8: f32[]| P
    aten::sum(dt$6: f32[8, 32]| S(0))
      aten::sum(t$5: f32[1, 32])  ->  t$7: f32[]""",
        )

        self.assertTrue(isinstance(debug_mode.operators[0], _OpCall))
        self.assertTrue(isinstance(debug_mode.operators[2], _RedistributeCall))
        self.assertEqual(next(iter(debug_mode.operators[1])), torch.ops.aten.mm.default)

        # check stringification
        self.assertTrue(hasattr(debug_mode.operators[0], "args_str"))
        self.assertFalse(hasattr(debug_mode.operators[0], "args"))

        # check recording hook
        def mm(x, y):
            return (x @ y).sum()

        eager_out = mm(x_dtensor, y_dtensor)

        # check recording hook for compiled variant
        with (
            DebugMode() as debug_mode,
            DebugMode.record_outputs(),
            DebugMode.log_tensor_hashes(),
        ):
            compiled_out = torch.compile(mm, backend="aot_eager")(x_dtensor, y_dtensor)

        # check numerical equivalence
        self.assertTrue(torch.equal(eager_out, compiled_out))
        sum_op = next(
            iter(
                op
                for op in debug_mode.operators
                if isinstance(op, _OpCall) and str(op.op) == "aten.sum.default"
            )
        )
        self.assertTrue(torch.equal(sum_op.record["output"], eager_out.to_local()))
        self.assertTrue(
            "aten::sum(t: f32[1, 32])  # {'hash': " in debug_mode.debug_string()
        )

    def test_debug_string_inside_context(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        x = torch.randn(1, 8, requires_grad=False)
        y = torch.randn(1, 32, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)

        with DebugMode() as debug_mode:
            torch.mm(x_dtensor, y_dtensor).sum()
            s0 = debug_mode.debug_string()
        s1 = debug_mode.debug_string()
        self.assertEqual(s0, s1)

    def test_debug_mode_backward(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        x = torch.randn(1, 8, requires_grad=True)
        y = torch.randn(8, 1, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(1)], run_check=False)

        with DebugMode(
            record_torchfunction=True, record_stack_trace=True
        ) as debug_mode:
            z = x_dtensor + y_dtensor
            z.sum().backward()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  <method 'add' of 'torch._C.TensorBase' objects>(dt: f32[8, 8]| S(0), dt: f32[8, 8]| S(1))
    aten::add.Tensor(dt: f32[8, 8]| S(0), dt: f32[8, 8]| S(1))
      redistribute_input(1, S(1) -> S(0))
        redistribute_input(t: f32[8, 1], trace: S(1)->S(0))
          _dtensor::shard_dim_alltoall(t: f32[8, 1], 1, 0, 0)
      aten::add.Tensor(t: f32[1, 8], t: f32[1, 8])
  <method 'sum' of 'torch._C.TensorBase' objects>(dt: f32[8, 8]| S(0))
    aten::sum(dt: f32[8, 8]| S(0))
      aten::sum(t: f32[1, 8])
  torch._tensor.backward(dt: f32[]| P, gradient=None, retain_graph=None, create_graph=False, inputs=None)
    aten::ones_like(dt: f32[]| P, pin_memory=False, memory_format=torch.preserve_format)
      aten::ones_like(t: f32[], pin_memory=False, memory_format=torch.preserve_format)
    aten::expand(dt: f32[]| R, [8, 8])
      aten::expand(t: f32[], [8, 8])
      redistribute_input(t: f32[8, 8], trace: R->S(1))
        aten::split.Tensor(t: f32[8, 8], 1, 1)
        aten::clone(t: f32[8, 1])
      aten::_to_copy(t: f32[8, 1], dtype=torch.float32, layout=torch.strided, device=cpu)
      redistribute_input(t: f32[8, 8], trace: R->S(0))
        aten::detach(t: f32[8, 1])
        aten::split.Tensor(t: f32[8, 8], 1)
        aten::clone(t: f32[1, 8])
      aten::_to_copy(t: f32[1, 8], dtype=torch.float32, layout=torch.strided, device=cpu)
      aten::detach(t: f32[1, 8])""",
        )

        # check stack trace
        self.assertTrue("z.sum().backward()" in debug_mode.operators[-1].stack_trace)

    def test_debug_mode_densor_redistribution_trace(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).view(4, 2))

        x = torch.randn(16, 8, requires_grad=True)
        y = torch.randn(8, 16, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0), Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(1), Shard(1)], run_check=False)
        x_dtensor._spec.shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1)),)
        y_dtensor._spec.shard_order = (ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 1)),)
        with DebugMode(record_torchfunction=False) as debug_mode:
            torch.mm(x_dtensor, y_dtensor).sum()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  aten::mm(dt: f32[128, 8]| S(0)[0]S(0)[1], dt: f32[8, 128]| S(1)[0]S(1)[1])
    redistribute_input(0, S(0)[0]S(0)[1] -> S(0)R)
      redistribute_input(t: f32[16, 8], trace: S(0)[0]S(0)[1]->S(0)R)
        _c10d_functional::all_gather_into_tensor(t: f32[16, 8], 2, 3)
        _c10d_functional::wait_tensor(t: f32[32, 8])
    redistribute_input(1, S(1)[0]S(1)[1] -> RS(1))
      redistribute_input(t: f32[8, 16], trace: S(1)[0]S(1)[1]->S(1)R->RR->RS(1))
        _c10d_functional::all_gather_into_tensor(t: f32[8, 16], 2, 3)
        _c10d_functional::wait_tensor(t: f32[16, 16])
        aten::chunk(t: f32[16, 16], 2)
        aten::cat(['t: f32[8, 16]', 't: f32[8, 16]'], 1)
        _c10d_functional::all_gather_into_tensor(t: f32[8, 32], 4, 1)
        _c10d_functional::wait_tensor(t: f32[32, 32])
        aten::chunk(t: f32[32, 32], 4)
        aten::cat(['t: f32[8, 32]', 't: f32[8, 32]', 't: f32[8, 32]', 't: f32[8, 32]'], 1)
        aten::chunk(t: f32[8, 128], 2, 1)
        aten::clone(t: f32[8, 64])
    aten::mm(t: f32[32, 8], t: f32[8, 64])
  aten::sum(dt: f32[128, 128]| S(0)S(1))
    aten::sum(t: f32[32, 64])""",
        )

    def test_debug_mode_einsum(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).view(4, 2))

        # Create test tensors
        a = torch.randn(16, 6, 8)
        b = torch.randn(8, 4, 4)

        a_dt = DTensor.from_local(a, mesh, [Partial(), Replicate()], run_check=False)
        b_dt = DTensor.from_local(b, mesh, [Replicate(), Partial()], run_check=False)

        # Capture the operator decomposition
        with DebugMode(record_torchfunction=True) as debug_mode:
            torch.einsum("bld,dnh->blnh", a_dt, b_dt)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch.functional.einsum(bld,dnh->blnh, dt: f32[16, 6, 8]| PR, dt: f32[8, 4, 4]| RP)
    aten::unsqueeze(dt: f32[16, 6, 8]| PR, 3)
      aten::unsqueeze(t: f32[16, 6, 8], 3)
    aten::unsqueeze(dt: f32[16, 6, 8, 1]| PR, 4)
      aten::unsqueeze(t: f32[16, 6, 8, 1], 4)
    aten::permute(dt: f32[16, 6, 8, 1, 1]| PR, [0, 1, 3, 4, 2])
      aten::permute(t: f32[16, 6, 8, 1, 1], [0, 1, 3, 4, 2])
    aten::unsqueeze(dt: f32[8, 4, 4]| RP, 3)
      aten::unsqueeze(t: f32[8, 4, 4], 3)
    aten::unsqueeze(dt: f32[8, 4, 4, 1]| RP, 4)
      aten::unsqueeze(t: f32[8, 4, 4, 1], 4)
    aten::permute(dt: f32[8, 4, 4, 1, 1]| RP, [3, 4, 1, 2, 0])
      aten::permute(t: f32[8, 4, 4, 1, 1], [3, 4, 1, 2, 0])
    aten::permute(dt: f32[16, 6, 1, 1, 8]| PR, [0, 1, 4, 2, 3])
      aten::permute(t: f32[16, 6, 1, 1, 8], [0, 1, 4, 2, 3])
    aten::view(dt: f32[16, 6, 8, 1, 1]| PR, [1, 96, 8])
      aten::view(t: f32[16, 6, 8, 1, 1], [1, 96, 8])
    aten::permute(dt: f32[1, 1, 4, 4, 8]| RP, [4, 2, 3, 0, 1])
      aten::permute(t: f32[1, 1, 4, 4, 8], [4, 2, 3, 0, 1])
    aten::view(dt: f32[8, 4, 4, 1, 1]| RP, [1, 8, 16])
      aten::view(t: f32[8, 4, 4, 1, 1], [1, 8, 16])
    aten::bmm(dt: f32[1, 96, 8]| PR, dt: f32[1, 8, 16]| RP)
      redistribute_input(0, PR -> S(2)[0]S(2)[1])
        redistribute_input(t: f32[1, 96, 8], trace: PR->S(2)R->S(2)[0]S(2)[1])
          aten::chunk(t: f32[1, 96, 8], 4, 2)
          aten::cat(['t: f32[1, 96, 2]', 't: f32[1, 96, 2]', 't: f32[1, 96, 2]', 't: f32[1, 96, 2]'])
          _c10d_functional::reduce_scatter_tensor(t: f32[4, 96, 2], sum, 4, 1)
          _c10d_functional::wait_tensor(t: f32[1, 96, 2])
          aten::chunk(t: f32[1, 96, 2], 2, 2)
          aten::clone(t: f32[1, 96, 1])
      redistribute_input(1, RP -> S(1)[0]S(1)[1])
        redistribute_input(t: f32[1, 8, 16], trace: RP->S(1)P->S(1)[0]S(1)[1])
          aten::chunk(t: f32[1, 8, 16], 4, 1)
          aten::clone(t: f32[1, 2, 16])
          aten::chunk(t: f32[1, 2, 16], 2, 1)
          aten::cat(['t: f32[1, 1, 16]', 't: f32[1, 1, 16]'])
          _c10d_functional::reduce_scatter_tensor(t: f32[2, 1, 16], sum, 2, 3)
          _c10d_functional::wait_tensor(t: f32[1, 1, 16])
      aten::bmm(t: f32[1, 96, 1], t: f32[1, 1, 16])
    aten::view(dt: f32[1, 96, 16]| PP, [16, 6, 1, 4, 4])
      aten::view(t: f32[1, 96, 16], [16, 6, 1, 4, 4])
    aten::permute(dt: f32[16, 6, 1, 4, 4]| PP, [0, 1, 3, 4, 2])
      aten::permute(t: f32[16, 6, 1, 4, 4], [0, 1, 3, 4, 2])
    aten::view(dt: f32[16, 6, 4, 4, 1]| PP, [16, 6, 4, 4])
      aten::view(t: f32[16, 6, 4, 4, 1], [16, 6, 4, 4])""",
        )

    def test_real_tensor(self):
        x = torch.randn(8, 8, 8)
        linear = torch.nn.Linear(8, 8)

        with DebugMode(record_torchfunction=True) as debug_mode:
            linear(x).sum()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch._C._nn.linear(t: f32[8, 8, 8], t: f32[8, 8], t: f32[8])
      aten::view(t: f32[8, 8, 8], [64, 8])
      aten::t(t: f32[8, 8])
      aten::addmm(t: f32[8], t: f32[64, 8], t: f32[8, 8])
      aten::view(t: f32[64, 8], [8, 8, 8])
  <method 'sum' of 'torch._C.TensorBase' objects>(t: f32[8, 8, 8])
      aten::sum(t: f32[8, 8, 8])""",
        )

    def test_fake_tensor(self):
        with FakeTensorMode():
            x = torch.randn(8, 8)
            y = torch.randn(8, 8, 8)

        with DebugMode(record_torchfunction=True, record_faketensor=True) as debug_mode:
            torch.matmul(y, x)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch.matmul(ft: f32[8, 8, 8], ft: f32[8, 8])
      aten::view(ft: f32[8, 8, 8], [64, 8])
      aten::mm(ft: f32[64, 8], ft: f32[8, 8])
      aten::_unsafe_view(ft: f32[64, 8], [8, 8, 8])""",
        )

    def test_tensor_attributes(self):
        x = torch.randn(8, 8)
        x.a1 = "x1"
        x.a2 = "x2"
        y = torch.randn(8, 8, 8)
        y.a1 = "y"

        with DebugMode(
            record_torchfunction=True,
            record_faketensor=True,
            record_tensor_attributes=["a1", "a2"],
            store_original_args=True,
        ) as debug_mode:
            torch.matmul(y, x)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch.matmul(t: f32[8, 8, 8]{a1=y}, t: f32[8, 8]{a1=x1, a2=x2})
      aten::view(t: f32[8, 8, 8]{a1=y}, [64, 8])
      aten::mm(t: f32[64, 8], t: f32[8, 8]{a1=x1, a2=x2})
      aten::_unsafe_view(t: f32[64, 8], [8, 8, 8])""",
        )

        self.assertTrue(hasattr(debug_mode.operators[0], "args"))
        self.assertEqual(id(debug_mode.operators[0].args[0]), id(y))

    @parametrize("has_inner_mode", [True, False])
    @parametrize("has_outer_mode", [True, False])
    def test_nested_debug_mode(self, has_inner_mode, has_outer_mode):
        class DummyTorchDispatchMode1(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return func(*args, **kwargs)

        class DummyTorchDispatchMode2(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return func(*args, **kwargs)

        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        x = torch.randn(1, 8, requires_grad=True)
        y = torch.randn(1, 32, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)

        inner_mode = (
            DummyTorchDispatchMode1() if has_inner_mode else contextlib.nullcontext()
        )
        outer_mode = (
            DummyTorchDispatchMode2() if has_outer_mode else contextlib.nullcontext()
        )

        with outer_mode:
            with DebugMode() as debug_mode:
                with inner_mode:
                    torch.mm(x_dtensor, y_dtensor)

        self.assertTrue("redistribute_input(1, S(0) -> R)" in debug_mode.debug_string())

    def test_debug_mode_higher_order_cond(self):
        """Test DebugMode with higher order operation."""
        x = torch.randn(1, 8, requires_grad=True)

        with DebugMode(record_torchfunction=True) as debug_mode:
            # rewrite torch.conda as torch.ops.higher_order.cond to avoid compilation
            torch.ops.higher_order.cond(
                torch.tensor(True), lambda x: x + 1, lambda x: x - 1, (x,)
            )

        # Verify that cond operations are captured in debug mode
        self.assertIn("torch.ops.higher_order.cond", debug_mode.debug_string())

    def test_compile(self):
        @torch.compile
        def f(x):
            return x.sin().cos()

        x = torch.randn(8)
        with DebugMode() as debug_mode:
            f(x)
        self.assertEqual(len(debug_mode.debug_string()), 0)

    def test_nn_module(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(4, 4)
                self.l2 = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.l2(self.l1(x))

        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.abc = Foo()
                self.xyz = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.xyz(self.abc(x))

        mod = Bar()
        inp = torch.randn(4, 4)
        with DebugMode(record_nn_module=True) as debug_mode:
            _ = mod(inp)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
    [nn.Mod] Bar
      [nn.Mod] Bar.abc
        [nn.Mod] Bar.abc.l1
          aten::t(t: f32[4, 4])
          aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])
        [nn.Mod] Bar.abc.l2
          aten::t(t: f32[4, 4])
          aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])
      [nn.Mod] Bar.xyz
        aten::t(t: f32[4, 4])
        aten::addmm(t: f32[4], t: f32[4, 4], t: f32[4, 4])""",
        )

        with DebugMode(record_stack_trace=True) as debug_mode:
            out = mod(inp).sum()
            out.backward()

        sum_op = [
            op for op in debug_mode.operators if str(op.op) == "aten.sum.dim_IntList"
        ][-1]
        self.assertTrue("self.l2(self.l1(x))" in sum_op.fwd_stack_trace)

    def test_pretty_print_dtensor_make_fx(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        A = torch.randn(8, 32)
        B = torch.randn(32, 32)
        dA = distribute_tensor(A, mesh, [Shard(0)]).requires_grad_()
        dB = distribute_tensor(B, mesh, [Replicate()]).requires_grad_()

        def f(dA, dB):
            dy = dA @ dB
            loss = dy.sum()
            loss.backward()
            return dA.grad, dB.grad

        # We actually need the tracing_mode='fake' here, or to trace under a FakeTensorMode.
        # make_fx has some logic to ensure we don't accidentally stash real tensors in the graph
        # so we won't stash our DTensors properly if they don't hold Fake inner tensors
        gm = make_fx(f, tracing_mode="fake")(dA, dB)
        # DCE isn't necessary here, there were just a lot of dead detach() nodes that spammed the graph
        gm.graph.eliminate_dead_code()
        gm.recompile()
        # Colored is nice for actual viewing, not using in this test though
        gm_str = gm.print_readable(colored=False, print_output=False)
        self.assertTrue('"DTensor(f32[8, 32], S(0))" = torch.ops.aten.mm' in gm_str)


instantiate_parametrized_tests(TestDTensorDebugMode)


if __name__ == "__main__":
    run_tests()
