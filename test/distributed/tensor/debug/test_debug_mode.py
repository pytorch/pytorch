# Owner(s): ["oncall: distributed"]

import contextlib
import unittest

import torch
import torch.distributed as dist
from torch._dynamo.testing import CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.placement_utils import ShardOrderEntry
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._debug_mode import (
    _OpCall,
    _RedistributeCall,
    _TritonKernelCall,
    DebugMode,
    hash_tensor_fn,
    norm_hash_fn,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._triton import has_triton_package


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
  <method 'sum' of 'torch._C.TensorBase' objects>(dt$6: f32[8, 32]| S(0))  ->  dt$8: f32[]| P(sum)
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

        # check tuple hash functions
        with (
            DebugMode() as debug_mode,
            DebugMode.log_tensor_hashes(hash_fn=["norm", "hash_tensor"]),
        ):
            mm(x_dtensor, y_dtensor)

        output_hash = debug_mode.operators[-1].log["hash"]
        norm_ = lambda x: norm_hash_fn(x, use_scalar=True)  # noqa: E731
        hash_ = lambda x: hash_tensor_fn(x, use_scalar=True)  # noqa: E731

        self.assertEqual(output_hash[0], norm_(eager_out))
        self.assertEqual(output_hash[1], hash_(eager_out))

        # some edge cases
        self.assertEqual(norm_(torch.tensor(torch.nan)), torch.nan)
        self.assertEqual(norm_(torch.tensor(torch.inf)), torch.inf)
        self.assertEqual(norm_(torch.complex(torch.ones(4), torch.zeros(4))), 4)
        self.assertEqual(hash_(torch.ones(4, dtype=torch.float8_e5m2)), 0)
        self.assertEqual(hash_(torch.ones(4, dtype=torch.int8)), 0)
        self.assertEqual(hash_(torch.ones(5, dtype=torch.int8)), 1)

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
  torch._tensor.backward(dt: f32[]| P(sum), gradient=None, retain_graph=None, create_graph=False, inputs=None)
    aten::ones_like(dt: f32[]| P(sum), pin_memory=False, memory_format=torch.preserve_format)
      aten::ones_like(t: f32[], pin_memory=False, memory_format=torch.preserve_format)
    aten::expand(dt: f32[]| R, [8, 8])
      aten::expand(t: f32[], [8, 8])
      redistribute_input(t: f32[8, 8], trace: R->S(1))
        aten::split.Tensor(t: f32[8, 8], 1, 1)
        aten::clone(t: f32[8, 1])
      aten::_to_copy(t: f32[8, 1], dtype=torch.float32, layout=torch.strided, device=cpu)
      redistribute_input(t: f32[8, 8], trace: R->S(0))
        aten::split.Tensor(t: f32[8, 8], 1)
        aten::clone(t: f32[1, 8])
        aten::detach(t: f32[8, 1])
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
        x_dtensor._spec._maybe_update_placements_given_shard_order(
            (ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1)),)
        )
        y_dtensor._spec._maybe_update_placements_given_shard_order(
            (ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 1)),)
        )
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
  torch.functional.einsum(bld,dnh->blnh, dt: f32[16, 6, 8]| P(sum)R, dt: f32[8, 4, 4]| RP(sum))
    aten::unsqueeze(dt: f32[16, 6, 8]| P(sum)R, 3)
      aten::unsqueeze(t: f32[16, 6, 8], 3)
    aten::unsqueeze(dt: f32[16, 6, 8, 1]| P(sum)R, 4)
      aten::unsqueeze(t: f32[16, 6, 8, 1], 4)
    aten::permute(dt: f32[16, 6, 8, 1, 1]| P(sum)R, [0, 1, 3, 4, 2])
      aten::permute(t: f32[16, 6, 8, 1, 1], [0, 1, 3, 4, 2])
    aten::unsqueeze(dt: f32[8, 4, 4]| RP(sum), 3)
      aten::unsqueeze(t: f32[8, 4, 4], 3)
    aten::unsqueeze(dt: f32[8, 4, 4, 1]| RP(sum), 4)
      aten::unsqueeze(t: f32[8, 4, 4, 1], 4)
    aten::permute(dt: f32[8, 4, 4, 1, 1]| RP(sum), [3, 4, 1, 2, 0])
      aten::permute(t: f32[8, 4, 4, 1, 1], [3, 4, 1, 2, 0])
    aten::permute(dt: f32[16, 6, 1, 1, 8]| P(sum)R, [0, 1, 4, 2, 3])
      aten::permute(t: f32[16, 6, 1, 1, 8], [0, 1, 4, 2, 3])
    aten::view(dt: f32[16, 6, 8, 1, 1]| P(sum)R, [1, 96, 8])
      aten::view(t: f32[16, 6, 8, 1, 1], [1, 96, 8])
    aten::permute(dt: f32[1, 1, 4, 4, 8]| RP(sum), [4, 2, 3, 0, 1])
      aten::permute(t: f32[1, 1, 4, 4, 8], [4, 2, 3, 0, 1])
    aten::view(dt: f32[8, 4, 4, 1, 1]| RP(sum), [1, 8, 16])
      aten::view(t: f32[8, 4, 4, 1, 1], [1, 8, 16])
    aten::bmm(dt: f32[1, 96, 8]| P(sum)R, dt: f32[1, 8, 16]| RP(sum))
      redistribute_input(0, P(sum)R -> S(2)[0]S(2)[1])
        redistribute_input(t: f32[1, 96, 8], trace: P(sum)R->S(2)R->S(2)[0]S(2)[1])
          aten::chunk(t: f32[1, 96, 8], 4, 2)
          aten::cat(['t: f32[1, 96, 2]', 't: f32[1, 96, 2]', 't: f32[1, 96, 2]', 't: f32[1, 96, 2]'])
          _c10d_functional::reduce_scatter_tensor(t: f32[4, 96, 2], sum, 4, 1)
          _c10d_functional::wait_tensor(t: f32[1, 96, 2])
          aten::chunk(t: f32[1, 96, 2], 2, 2)
          aten::clone(t: f32[1, 96, 1])
      redistribute_input(1, RP(sum) -> S(1)[0]S(1)[1])
        redistribute_input(t: f32[1, 8, 16], trace: RP(sum)->S(1)P(sum)->S(1)[0]S(1)[1])
          aten::chunk(t: f32[1, 8, 16], 4, 1)
          aten::clone(t: f32[1, 2, 16])
          aten::chunk(t: f32[1, 2, 16], 2, 1)
          aten::cat(['t: f32[1, 1, 16]', 't: f32[1, 1, 16]'])
          _c10d_functional::reduce_scatter_tensor(t: f32[2, 1, 16], sum, 2, 3)
          _c10d_functional::wait_tensor(t: f32[1, 1, 16])
      aten::bmm(t: f32[1, 96, 1], t: f32[1, 1, 16])
    aten::view(dt: f32[1, 96, 16]| P(sum)P(sum), [16, 6, 1, 4, 4])
      aten::view(t: f32[1, 96, 16], [16, 6, 1, 4, 4])
    aten::permute(dt: f32[16, 6, 1, 4, 4]| P(sum)P(sum), [0, 1, 3, 4, 2])
      aten::permute(t: f32[16, 6, 1, 4, 4], [0, 1, 3, 4, 2])
    aten::view(dt: f32[16, 6, 4, 4, 1]| P(sum)P(sum), [16, 6, 4, 4])
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
        cnt = CompileCounterWithBackend("inductor")

        @torch.compile(backend=cnt)
        def f(x):
            return x.sin().cos()

        x = torch.randn(8)
        f(x)
        with DebugMode() as debug_mode:
            f(x)
            self.assertEqual(len(debug_mode.debug_string()), 0)
            f(x)
        f(x)
        self.assertEqual(
            cnt.frame_count, 1
        )  # check DebugMode doesn't trigger additional recompilations

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
        self.assertTrue(
            "self.l2(self.l1(x))" in debug_mode.debug_string(show_stack_trace=True)
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @unittest.skipIf(not has_triton_package(), "requires triton")
    def test_triton_kernel_logs(self):
        import triton

        from torch.testing._internal.triton_utils import add_kernel_autotuned

        def call_triton(x, y):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            add_kernel_autotuned[grid](x, y, output, n_elements)
            return output

        x = torch.randn(128, device=GPU_TYPE)
        y = torch.randn(128, device=GPU_TYPE)

        with DebugMode() as debug_mode:
            torch.compile(call_triton)(x, y)

        triton_calls = [
            op for op in debug_mode.operators if isinstance(op, _TritonKernelCall)
        ]
        self.assertGreater(len(triton_calls), 0)
        self.assertIn("[triton]", triton_calls[0].render([]))

    def test_check_hash_mismatches(self):
        x = torch.randn(64, 64, device=GPU_TYPE)
        x_different = torch.randn(64, 64, device=GPU_TYPE)

        # Identical runs should have no mismatches
        with DebugMode() as dm1, DebugMode.log_tensor_hashes():
            x.sin().sum()
        with DebugMode() as dm2, DebugMode.log_tensor_hashes():
            x.sin().sum()
        mismatches = DebugMode.check_hash_mismatches(dm1.logs, dm2.logs)
        self.assertEqual(len(mismatches), 0)

        # Different inputs should produce hash mismatches
        with DebugMode() as dm3, DebugMode.log_tensor_hashes():
            x_different.sin().sum()

        # Check that mismatches are detected
        mismatches = DebugMode.check_hash_mismatches(dm1.logs, dm3.logs)
        self.assertEqual(len(mismatches), 2)
        self.assertEqual(
            [call["call"] for call in mismatches], ["aten::sin", "aten::sum"]
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @unittest.skipIf(not has_triton_package(), "requires triton")
    def test_check_triton_hash_mismatches(self):
        import triton

        from torch.testing._internal.triton_utils import add_kernel_autotuned

        def call_triton(x, y):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            add_kernel_autotuned[grid](x, y, output, n_elements)
            return output

        a = torch.randn(128, device=GPU_TYPE)
        b = torch.randn(128, device=GPU_TYPE)
        c = torch.randn(128, device=GPU_TYPE)

        # Run with hash logging to verify triton kernels can be hashed
        with DebugMode() as dm_t1, DebugMode.log_tensor_hashes(hash_inputs=True):
            torch.compile(call_triton)(a, b)

        # Different inputs should have different hashes in triton kernels
        with DebugMode() as dm_t2, DebugMode.log_tensor_hashes(hash_inputs=True):
            torch.compile(call_triton)(a, c)

        # Compare triton kernel hashes
        mismatches = DebugMode.check_hash_mismatches(
            dm_t1.logs, dm_t2.logs, compare_inputs=True
        )
        triton_mismatches = [m for m in mismatches if m["call_type"] == "triton kernel"]
        self.assertGreater(len(triton_mismatches), 0)

        # check both input & output hash mismatches are detected
        self.assertGreater(len([m for m in triton_mismatches if m["is_input_hash"]]), 0)
        self.assertGreater(
            len([m for m in triton_mismatches if not m["is_input_hash"]]), 0
        )

    def test_check_structure_mismatches(self):
        x = torch.randn(32, 32, device=self.device_type)

        with DebugMode() as dm1, DebugMode.log_tensor_hashes():
            x.sin()
        with DebugMode() as dm2, DebugMode.log_tensor_hashes():
            x.cos()
        with DebugMode() as dm3, DebugMode.log_tensor_hashes():
            x.sin().cos()

        with self.assertRaisesRegex(ValueError, "Operators don't match"):
            DebugMode.check_hash_mismatches(dm1.logs, dm2.logs)

        with self.assertRaisesRegex(ValueError, "Log lengths don't match"):
            DebugMode.check_hash_mismatches(dm1.logs, dm3.logs)

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).total_memory < 2**26,
        "Being conservative, test peak memory is 25MB?",
    )
    def test_tensor_hash_waits_on_collective(self):
        # test that hashing collectives gives correct results
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        local_tensor = torch.ones(2**18, device=self.device_type)
        dt = DTensor.from_local(local_tensor, mesh, [Shard(0)], run_check=False)

        with DebugMode() as debug_mode, DebugMode.log_tensor_hashes():
            dt.redistribute(mesh, [Replicate()])

        # Find all_gather hash
        all_gather_logs = [
            op
            for op in debug_mode.logs
            if isinstance(op, _OpCall)
            and op.op == torch.ops._c10d_functional.all_gather_into_tensor.default
        ]
        self.assertEqual(len(all_gather_logs), 1)
        actual_hash = all_gather_logs[0].log["hash"]
        self.assertEqual(actual_hash, float(local_tensor.numel() * self.world_size))

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
