# Owner(s): ["module: dynamo"]

import os
import unittest

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing

from torch._dynamo.testing import normalize_gm


def _get_device_type(world_size):
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        device_type = "cuda"
    else:
        device_type = "cpu"
    return device_type


def _set_env_var(addr="localhost", port="25364", world_size=1, rank=0):
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["RANK"] = f"{rank}"


class DistributedTests(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch(trace_distributed=True)
    @unittest.skipIf(
        not torch.distributed.is_available(), "requires distributed package"
    )
    def test_fsdp_same_storage_size_allowed(self):
        import torch.distributed.fsdp._flat_param as flat_param

        def foo(x, y):
            return flat_param._same_storage_size(x, y)

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        foo_cmp = torch._dynamo.optimize(backend, nopython=True)(foo)
        foo_aot_autograd = torch._dynamo.optimize("aot_eager", nopython=True)(foo)

        x = torch.randn([2, 2])
        y = torch.randn([2, 2])
        self.assertEqual(foo_cmp(x, y), foo(x, y))
        self.assertEqual(foo_aot_autograd(x, y), foo(x, y))
        self.assertEqual(foo_cmp(x, x), foo(x, x))
        self.assertEqual(foo_aot_autograd(x, x), foo(x, x))
        gm = backend.graphs[0]
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        l_x_ = L_x_
        l_y_ = L_y_

        _same_storage_size = torch.distributed.fsdp._flat_param._same_storage_size(l_x_, l_y_);  l_x_ = l_y_ = None
        return (_same_storage_size,)
""",
            )
        else:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s0 : torch.SymInt, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        l_x_ = L_x_
        l_y_ = L_y_

        _same_storage_size = torch.distributed.fsdp._flat_param._same_storage_size(l_x_, l_y_);  l_x_ = l_y_ = None
        return (_same_storage_size,)
""",
            )

    def test_process_group_get_or_create_default_group(self):
        import torch.distributed.device_mesh as device_mesh

        world_size = 1
        rank = 0

        device_type = _get_device_type(world_size)
        _set_env_var(world_size=world_size, rank=rank)
        mesh_tensor = torch.arange(1)

        dmesh = device_mesh.DeviceMesh(device_type, mesh_tensor)
        default = torch.distributed.distributed_c10d._get_default_group()

        def foo(d, default, x):
            group = d._get_or_create_default_group()
            if group == default:
                return x.cos()
            return x.sin()

        x = torch.randn([2, 2])
        eager = foo(dmesh, default, x)
        foo = torch._dynamo.optimize("eager", nopython=True)(foo)
        comp = foo(dmesh, default, x)
        foo = torch._dynamo.optimize("aot_eager", nopython=True)(foo)
        comp_aot = foo(dmesh, default, x)
        self.assertEqual(comp, eager)
        self.assertEqual(comp_aot, comp)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
