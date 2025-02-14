import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.export import export
import unittest


class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


def apply_tp(model, mesh):
    parallelize_module(model.net1, mesh, ColwiseParallel(), src_data_rank=None)
    parallelize_module(model.net2, mesh, RowwiseParallel(), src_data_rank=None)


class TestExportDTensor(unittest.TestCase):
    def setUp(self):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29400'
        dist.init_process_group(backend="nccl", init_method="env://")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(self.device)

    def tearDown(self):
        dist.destroy_process_group()

    def test_export_dtensor(self):
        d_hid = 1024
        model = MLPModule(d_hid)
        model = model.to(self.device)
        mesh = DeviceMesh("cuda", list(range(self.world_size)))
        apply_tp(model, mesh)

        bs = 2
        x = torch.rand(bs, d_hid, device=self.device)

        ep = export(model, (x,), strict=True)
        self.assertIsNotNone(ep)

        y = model(x)
        y.wait()
        self.assertEqual(y.shape, (bs, d_hid))


if __name__ == "__main__":
    unittest.main()
