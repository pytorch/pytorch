# Owner(s): ["oncall: distributed"]
import contextlib

import torch
import torch.distributed as dist
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import ShardOrderEntry
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
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

        with DebugMode(record_torchfunction=True) as debug_mode:
            torch.mm(x_dtensor, y_dtensor).sum()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  torch.mm(dt: f32[8, 8]| S(0), dt: f32[8, 32]| S(0))
    aten::mm(dt: f32[8, 8]| S(0), dt: f32[8, 32]| S(0))
      redistribute_input(1, S(0) -> R)
        redistribute_input(t: f32[1, 32], trace: S(0)->R)
          _c10d_functional::all_gather_into_tensor(t: f32[1, 32], 8, 0)
          _c10d_functional::wait_tensor(t: f32[8, 32])
      aten::mm(t: f32[1, 8], t: f32[8, 32])
  <method 'sum' of 'torch._C.TensorBase' objects>(dt: f32[8, 32]| S(0))
    aten::sum(dt: f32[8, 32]| S(0))
      aten::sum(t: f32[1, 32])""",
        )

        self.assertTrue(isinstance(debug_mode.operators[0], _OpCall))
        self.assertTrue(isinstance(debug_mode.operators[2], _RedistributeCall))
        self.assertEqual(next(iter(debug_mode.operators[1])), torch.ops.aten.mm.default)

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

        # test stack trace
        with DebugMode() as debug_mode:
            z = x_dtensor + y_dtensor
            with DebugMode.dispatch_stack_trace(cpp=False):
                z.sum().backward()

        self.assertTrue(debug_mode.operators[0].stack_trace is None)
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

    def test_export(self):
        # inherited from test/distributed/tensor/test_dtensor_export.py
        class SimpleModel(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.mlp_0 = MLPModule(device)
                self.mlp_1 = MLPModule(device)

            def forward(self, input):
                return self.mlp_1(self.mlp_0(input))

        mesh = init_device_mesh(
            self.device_type,
            mesh_shape=(2, 4),
            mesh_dim_names=["dp", "tp"],
        )
        model = SimpleModel(self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, mesh["tp"], parallelize_plan)

        inputs = torch.rand(20, 10, device=self.device_type)
        inputs = distribute_tensor(inputs, mesh["tp"], placements=[Replicate()])

        with torch._dynamo.config.patch(install_free_tensors=True):
            gm = _dynamo_graph_capture_for_export(tp_model)(inputs)

        debug_mode = DebugMode()
        debug_mode.run_graph(gm, inputs)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  [node] l_flat_args_0_: placeholder
  [node] l__self____export_root_mlp_0_net1_weight: get_attr[L__self___mlp_0_net1_weight]()
  [node] l__self____export_root_mlp_0_net1_bias: get_attr[L__self___mlp_0_net1_bias]()
  [node] linear: call_function[torch._C._nn.linear](l_flat_args_0_, l__self____export_root_mlp_0_net1_weight, l__self____export_root_mlp_0_net1_bias)
    aten::t(dt: f32[16, 10]| S(0))
      aten::t(t: f32[4, 10])
    aten::addmm(dt: f32[16]| S(0), dt: f32[20, 10]| R, dt: f32[10, 16]| S(1))
      aten::addmm(t: f32[4], t: f32[20, 10], t: f32[10, 4])
  [node] outputs: call_function[torch._dynamo.variables.tensor.prim_redistribute](linear)
  [node] hook_result: call_function[torch._dynamo.variables.tensor.prim_to_local](outputs)
      aten::view(t: f32[20, 4], [20, 4])
  [node] input_tensor: call_function[torch.nn.functional.relu](hook_result, inplace=False)
      aten::relu(t: f32[20, 4])
  [node] input_tensor_1: call_function[torch._dynamo.variables.torch.prim from_local](input_tensor)
      aten::view(t: f32[20, 4], [20, 4])
  [node] l__self____export_root_mlp_0_net2_weight: get_attr[L__self___mlp_0_net2_weight]()
  [node] l__self____export_root_mlp_0_net2_bias: get_attr[L__self___mlp_0_net2_bias]()
  [node] linear_1: call_function[torch._C._nn.linear](input_tensor_1, l__self____export_root_mlp_0_net2_weight, l__self____export_root_mlp_0_net2_bias)
    aten::t(dt: f32[10, 16]| S(1))
      aten::t(t: f32[10, 4])
    aten::addmm(dt: f32[10]| R, dt: f32[20, 16]| S(1), dt: f32[16, 10]| S(0))
      redistribute_input(0, R -> P)
        redistribute_input(t: f32[10], trace: R->P)
          aten::div.Tensor(t: f32[10], 4)
      aten::addmm(t: f32[10], t: f32[20, 4], t: f32[4, 10])
  [node] outputs_1: call_function[torch._dynamo.variables.tensor.prim_redistribute](linear_1)
      redistribute_input(t: f32[20, 10], trace: P->R)
        _c10d_functional::all_reduce(t: f32[20, 10], sum, 5)
  [node] hook_result_1: call_function[torch._dynamo.variables.tensor.prim_to_local](outputs_1)
  [node] input_tensor_2: call_function[torch._dynamo.variables.torch.prim from_local](hook_result_1)
  [node] l__self____export_root_mlp_1_net1_weight: get_attr[L__self___mlp_1_net1_weight]()
  [node] l__self____export_root_mlp_1_net1_bias: get_attr[L__self___mlp_1_net1_bias]()
  [node] linear_2: call_function[torch._C._nn.linear](input_tensor_2, l__self____export_root_mlp_1_net1_weight, l__self____export_root_mlp_1_net1_bias)
    aten::t(dt: f32[16, 10]| S(0))
      aten::t(t: f32[4, 10])
    aten::addmm(dt: f32[16]| S(0), dt: f32[20, 10]| R, dt: f32[10, 16]| S(1))
  [node] outputs_2: call_function[torch._dynamo.variables.tensor.prim_redistribute](linear_2)
  [node] hook_result_2: call_function[torch._dynamo.variables.tensor.prim_to_local](outputs_2)
      aten::view(t: f32[20, 4], [20, 4])
  [node] input_tensor_3: call_function[torch.nn.functional.relu](hook_result_2, inplace=False)
      aten::relu(t: f32[20, 4])
  [node] input_tensor_4: call_function[torch._dynamo.variables.torch.prim from_local](input_tensor_3)
      aten::view(t: f32[20, 4], [20, 4])
  [node] l__self____export_root_mlp_1_net2_weight: get_attr[L__self___mlp_1_net2_weight]()
  [node] l__self____export_root_mlp_1_net2_bias: get_attr[L__self___mlp_1_net2_bias]()
  [node] linear_3: call_function[torch._C._nn.linear](input_tensor_4, l__self____export_root_mlp_1_net2_weight, l__self____export_root_mlp_1_net2_bias)
    aten::t(dt: f32[10, 16]| S(1))
      aten::t(t: f32[10, 4])
    aten::addmm(dt: f32[10]| R, dt: f32[20, 16]| S(1), dt: f32[16, 10]| S(0))
      redistribute_input(0, R -> P)
        redistribute_input(t: f32[10], trace: R->P)
          aten::div.Tensor(t: f32[10], 4)
      aten::addmm(t: f32[10], t: f32[20, 4], t: f32[4, 10])
  [node] outputs_3: call_function[torch._dynamo.variables.tensor.prim_redistribute](linear_3)
      redistribute_input(t: f32[20, 10], trace: P->R)
        _c10d_functional::all_reduce(t: f32[20, 10], sum, 5)
  [node] hook_result_3: call_function[torch._dynamo.variables.tensor.prim_to_local](outputs_3)
  [node] output: output""",  # NOQA: B950
        )

    def test_fx_annotate_in_graph(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                x = x + 2
                with torch.fx.traceback.annotate({"region": "foo"}):
                    x = x @ y
                    x = x + 2
                x = x * 2
                return x

        mod = Foo()
        x, y = torch.randn(4, 4), torch.randn(4, 8)
        with torch.fx.traceback.preserve_node_meta():
            gm = torch.export.export(mod, (x, y)).module()

        debug_mode = DebugMode()
        debug_mode.run_graph(gm, x, y)
        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  [node] x: placeholder
  [node] y: placeholder
  [node] _guards_fn: call_module[_guards_fn](x, y)
  [node] add: call_function[aten::add.Tensor](x, 2)
      aten::add.Tensor(t: f32[4, 4], 2)
  [node] matmul: call_function[aten::matmul](add, y)  # {"region": foo}
      aten::mm(t: f32[4, 4], t: f32[4, 8])
  [node] add_1: call_function[aten::add.Tensor](matmul, 2)  # {"region": foo}
      aten::add.Tensor(t: f32[4, 8], 2)
  [node] mul: call_function[aten::mul.Tensor](add_1, 2)
      aten::mul.Tensor(t: f32[4, 8], 2)
  [node] output: output""",
        )

    def test_custom_hooks(self):
        def numel_hook(func, types, args, kwargs, result):
            return {"numel": result.numel()}

        # test dispatch hooks
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(4, 5)
                self.l2 = torch.nn.Linear(5, 6)

            def forward(self, x):
                x0 = self.l1(x)
                # local recoding hook & annotation
                with DebugMode.record_outputs():
                    x1 = self.l2(x0)
                return x1, x0

        x = torch.randn(4, 4, device=self.device_type)
        mod = Foo().to(device=self.device_type)

        # global logging hook
        with DebugMode.dispatch_hooks(log_hook=numel_hook), DebugMode() as debug_mode:
            x1, _ = mod(x)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
    aten::t(t: f32[5, 4])  # {'numel': 20}
    aten::addmm(t: f32[5], t: f32[4, 4], t: f32[4, 5])  # {'numel': 20}
    aten::t(t: f32[6, 5])  # {'numel': 30}
    aten::addmm(t: f32[6], t: f32[4, 5], t: f32[5, 6])  # {'numel': 24}""",
        )
        self.assertTrue(debug_mode.operators[0].record is None)
        record = debug_mode.operators[3].record["output"]
        self.assertTrue(torch.allclose(record, x1))

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

    def test_inductor_calls_for_mm(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        x = torch.randn(1, 8, requires_grad=False)
        y = torch.randn(1, 32, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)

        @torch.compile(backend="inductor", fullgraph=True)
        def fn(x, y):
            return torch.mm(x, y)

        with DebugMode(record_torchfunction=False) as debug_mode:
            out = fn(x_dtensor, y_dtensor)
            out.sum().backward()

        self.assertEqual(debug_mode.debug_string().count("inductor_graph_call"), 2)


instantiate_parametrized_tests(TestDTensorDebugMode)


if __name__ == "__main__":
    run_tests()
