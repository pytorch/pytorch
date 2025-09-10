# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.debug import DTensorDebugMode
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


class TestCommMode(TestCase):
    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        self.world_size = 8
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=1, world_size=self.world_size, store=store
        )
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.world_pg = dist.distributed_c10d._get_default_group()

    def test_debug_mode_mm(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        debug_mode = DTensorDebugMode()
        x = torch.randn(1, 8, requires_grad=True)
        y = torch.randn(1, 32, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)

        with debug_mode:
            torch.mm(x_dtensor, y_dtensor)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  aten::mm(dt: f32[8, 8]DM(8)[S(0)], dt: f32[8, 32]DM(8)[S(0)])
    redistribute_input(1, [S(0)], [R])
      _c10d_functional::all_gather_into_tensor(t: f32[1, 32], 8, 0)
      _c10d_functional::wait_tensor(t: f32[8, 32])
    aten::mm(t: f32[1, 8], t: f32[8, 32])""",
        )

    def test_debug_mode_einsum(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).view(4, 2))

        # Create test tensors
        a = torch.randn(16, 6, 8)
        b = torch.randn(8, 4, 4)

        a_dt = DTensor.from_local(a, mesh, [Partial(), Replicate()], run_check=False)
        b_dt = DTensor.from_local(b, mesh, [Replicate(), Partial()], run_check=False)

        # Capture the operator decomposition
        with DTensorDebugMode() as debug_mode:
            torch.einsum("bld,dnh->blnh", a_dt, b_dt)

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
  aten::unsqueeze(dt: f32[16, 6, 8]DM(4, 2)[P, R], 3)
    aten::unsqueeze(t: f32[16, 6, 8], 3)
  aten::unsqueeze(dt: f32[16, 6, 8, 1]DM(4, 2)[P, R], 4)
    aten::unsqueeze(t: f32[16, 6, 8, 1], 4)
  aten::permute(dt: f32[16, 6, 8, 1, 1]DM(4, 2)[P, R], [0, 1, 3, 4, 2])
    aten::permute(t: f32[16, 6, 8, 1, 1], [0, 1, 3, 4, 2])
  aten::unsqueeze(dt: f32[8, 4, 4]DM(4, 2)[R, P], 3)
    aten::unsqueeze(t: f32[8, 4, 4], 3)
  aten::unsqueeze(dt: f32[8, 4, 4, 1]DM(4, 2)[R, P], 4)
    aten::unsqueeze(t: f32[8, 4, 4, 1], 4)
  aten::permute(dt: f32[8, 4, 4, 1, 1]DM(4, 2)[R, P], [3, 4, 1, 2, 0])
    aten::permute(t: f32[8, 4, 4, 1, 1], [3, 4, 1, 2, 0])
  aten::permute(dt: f32[16, 6, 1, 1, 8]DM(4, 2)[P, R], [0, 1, 4, 2, 3])
    aten::permute(t: f32[16, 6, 1, 1, 8], [0, 1, 4, 2, 3])
  aten::view(dt: f32[16, 6, 8, 1, 1]DM(4, 2)[P, R], [1, 96, 8])
    aten::view(t: f32[16, 6, 8, 1, 1], [1, 96, 8])
  aten::permute(dt: f32[1, 1, 4, 4, 8]DM(4, 2)[R, P], [4, 2, 3, 0, 1])
    aten::permute(t: f32[1, 1, 4, 4, 8], [4, 2, 3, 0, 1])
  aten::view(dt: f32[8, 4, 4, 1, 1]DM(4, 2)[R, P], [1, 8, 16])
    aten::view(t: f32[8, 4, 4, 1, 1], [1, 8, 16])
  aten::bmm(dt: f32[1, 96, 8]DM(4, 2)[P, R], dt: f32[1, 8, 16]DM(4, 2)[R, P])
    redistribute_input(0, [P, R], [S(2), S(2)])
      aten::chunk(t: f32[1, 96, 8], 4, 2)
      aten::cat(['t: f32[1, 96, 2]', 't: f32[1, 96, 2]', 't: f32[1, 96, 2]', 't: f32[1, 96, 2]'])
      _c10d_functional::reduce_scatter_tensor(t: f32[4, 96, 2], sum, 4, 2)
      aten::clone(t: f32[1, 96, 1])
    redistribute_input(1, [R, P], [S(1), S(1)])
      aten::chunk(t: f32[1, 8, 16], 4, 1)
      aten::clone(t: f32[1, 2, 16])
      aten::chunk(t: f32[1, 2, 16], 2, 1)
      aten::cat(['t: f32[1, 1, 16]', 't: f32[1, 1, 16]'])
      _c10d_functional::reduce_scatter_tensor(t: f32[2, 1, 16], sum, 2, 3)
      _c10d_functional::wait_tensor(t: f32[1, 1, 16])
    aten::bmm(t: f32[1, 96, 1], t: f32[1, 1, 16])
  aten::view(dt: f32[1, 96, 16]DM(4, 2)[P, P], [16, 6, 1, 4, 4])
    aten::view(t: f32[1, 96, 16], [16, 6, 1, 4, 4])
  aten::permute(dt: f32[16, 6, 1, 4, 4]DM(4, 2)[P, P], [0, 1, 3, 4, 2])
    aten::permute(t: f32[16, 6, 1, 4, 4], [0, 1, 3, 4, 2])
  aten::view(dt: f32[16, 6, 4, 4, 1]DM(4, 2)[P, P], [16, 6, 4, 4])
    aten::view(t: f32[16, 6, 4, 4, 1], [16, 6, 4, 4])""",
        )


if __name__ == "__main__":
    run_tests()
