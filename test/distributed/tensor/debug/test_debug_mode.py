# Owner(s): ["oncall: distributed"]

import contextlib

import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor import DeviceMesh, DTensor, Partial, Replicate, Shard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._debug_mode import DebugMode
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

        with DebugMode(record_torchfunction=True) as debug_mode:
            torch.mm(x_dtensor, y_dtensor).sum()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch.mm(dt: f32[8, 8][S(0)], dt: f32[8, 32][S(0)])
    aten::mm(dt: f32[8, 8][S(0)], dt: f32[8, 32][S(0)])
      redistribute_input(1, [S(0)] -> [R])
        redistribute_input(t: f32[1, 32], [S(0)] -> [R])
          _c10d_functional::all_gather_into_tensor(t: f32[1, 32], 8, 0)
          _c10d_functional::wait_tensor(t: f32[8, 32])
      aten::mm(t: f32[1, 8], t: f32[8, 32])
  <method 'sum' of 'torch._C.TensorBase' objects>(dt: f32[8, 32][S(0)])
    aten::sum(dt: f32[8, 32][S(0)])
      aten::sum(t: f32[1, 32])""",
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

        with DebugMode(record_torchfunction=True) as debug_mode:
            z = x_dtensor + y_dtensor
            z.sum().backward()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  <method 'add' of 'torch._C.TensorBase' objects>(dt: f32[8, 8][S(0)], dt: f32[8, 8][S(1)])
    aten::add.Tensor(dt: f32[8, 8][S(0)], dt: f32[8, 8][S(1)])
      redistribute_input(1, [S(1)] -> [S(0)])
        redistribute_input(t: f32[8, 1], [S(1)] -> [S(0)])
          _dtensor::shard_dim_alltoall(t: f32[8, 1], 1, 0, 0)
      aten::add.Tensor(t: f32[1, 8], t: f32[1, 8])
  <method 'sum' of 'torch._C.TensorBase' objects>(dt: f32[8, 8][S(0)])
    aten::sum(dt: f32[8, 8][S(0)])
      aten::sum(t: f32[1, 8])
  torch._tensor.backward(dt: f32[][P], gradient=None, retain_graph=None, create_graph=False, inputs=None)
    aten::ones_like(dt: f32[][P], pin_memory=False, memory_format=torch.preserve_format)
      aten::ones_like(t: f32[], pin_memory=False, memory_format=torch.preserve_format)
    aten::expand(dt: f32[][R], [8, 8])
      aten::expand(t: f32[], [8, 8])
      redistribute_input(t: f32[8, 8], [R] -> [S(1)])
        aten::split.Tensor(t: f32[8, 8], 1, 1)
        aten::clone(t: f32[8, 1])
      aten::_to_copy(t: f32[8, 1], dtype=torch.float32, layout=torch.strided, device=cpu)
      redistribute_input(t: f32[8, 8], [R] -> [S(0)])
        aten::detach(t: f32[8, 1])
        aten::split.Tensor(t: f32[8, 8], 1)
        aten::clone(t: f32[1, 8])
      aten::_to_copy(t: f32[1, 8], dtype=torch.float32, layout=torch.strided, device=cpu)
      aten::detach(t: f32[1, 8])""",
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
  torch.functional.einsum(bld,dnh->blnh, dt: f32[16, 6, 8][P, R], dt: f32[8, 4, 4][R, P])
    aten::unsqueeze(dt: f32[16, 6, 8][P, R], 3)
      aten::unsqueeze(t: f32[16, 6, 8], 3)
    aten::unsqueeze(dt: f32[16, 6, 8, 1][P, R], 4)
      aten::unsqueeze(t: f32[16, 6, 8, 1], 4)
    aten::permute(dt: f32[16, 6, 8, 1, 1][P, R], [0, 1, 3, 4, 2])
      aten::permute(t: f32[16, 6, 8, 1, 1], [0, 1, 3, 4, 2])
    aten::unsqueeze(dt: f32[8, 4, 4][R, P], 3)
      aten::unsqueeze(t: f32[8, 4, 4], 3)
    aten::unsqueeze(dt: f32[8, 4, 4, 1][R, P], 4)
      aten::unsqueeze(t: f32[8, 4, 4, 1], 4)
    aten::permute(dt: f32[8, 4, 4, 1, 1][R, P], [3, 4, 1, 2, 0])
      aten::permute(t: f32[8, 4, 4, 1, 1], [3, 4, 1, 2, 0])
    aten::permute(dt: f32[16, 6, 1, 1, 8][P, R], [0, 1, 4, 2, 3])
      aten::permute(t: f32[16, 6, 1, 1, 8], [0, 1, 4, 2, 3])
    aten::view(dt: f32[16, 6, 8, 1, 1][P, R], [1, 96, 8])
      aten::view(t: f32[16, 6, 8, 1, 1], [1, 96, 8])
    aten::permute(dt: f32[1, 1, 4, 4, 8][R, P], [4, 2, 3, 0, 1])
      aten::permute(t: f32[1, 1, 4, 4, 8], [4, 2, 3, 0, 1])
    aten::view(dt: f32[8, 4, 4, 1, 1][R, P], [1, 8, 16])
      aten::view(t: f32[8, 4, 4, 1, 1], [1, 8, 16])
    aten::bmm(dt: f32[1, 96, 8][P, R], dt: f32[1, 8, 16][R, P])
      redistribute_input(0, [P, R] -> [S(2), S(2)])
        redistribute_input(t: f32[1, 96, 8], [P, R] -> [S(2), S(2)])
          aten::chunk(t: f32[1, 96, 8], 4, 2)
          aten::cat(['t: f32[1, 96, 2]', 't: f32[1, 96, 2]', 't: f32[1, 96, 2]', 't: f32[1, 96, 2]'])
          _c10d_functional::reduce_scatter_tensor(t: f32[4, 96, 2], sum, 4, 1)
          _c10d_functional::wait_tensor(t: f32[1, 96, 2])
          aten::chunk(t: f32[1, 96, 2], 2, 2)
          aten::clone(t: f32[1, 96, 1])
      redistribute_input(1, [R, P] -> [S(1), S(1)])
        redistribute_input(t: f32[1, 8, 16], [R, P] -> [S(1), S(1)])
          aten::chunk(t: f32[1, 8, 16], 4, 1)
          aten::clone(t: f32[1, 2, 16])
          aten::chunk(t: f32[1, 2, 16], 2, 1)
          aten::cat(['t: f32[1, 1, 16]', 't: f32[1, 1, 16]'])
          _c10d_functional::reduce_scatter_tensor(t: f32[2, 1, 16], sum, 2, 3)
          _c10d_functional::wait_tensor(t: f32[1, 1, 16])
      aten::bmm(t: f32[1, 96, 1], t: f32[1, 1, 16])
    aten::view(dt: f32[1, 96, 16][P, P], [16, 6, 1, 4, 4])
      aten::view(t: f32[1, 96, 16], [16, 6, 1, 4, 4])
    aten::permute(dt: f32[16, 6, 1, 4, 4][P, P], [0, 1, 3, 4, 2])
      aten::permute(t: f32[16, 6, 1, 4, 4], [0, 1, 3, 4, 2])
    aten::view(dt: f32[16, 6, 4, 4, 1][P, P], [16, 6, 4, 4])
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

        self.assertTrue(
            "redistribute_input(1, [S(0)] -> [R])" in debug_mode.debug_string()
        )

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


instantiate_parametrized_tests(TestDTensorDebugMode)


if __name__ == "__main__":
    run_tests()
