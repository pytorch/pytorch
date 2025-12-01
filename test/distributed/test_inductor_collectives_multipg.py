# Owner(s): ["module: dynamo"]
import torch
import torch.distributed as c10d
from torch._inductor.scheduler import BaseSchedulerNode
from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.common_distributed import (
    DynamoDistributedMultiProcTestCase,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import instantiate_parametrized_tests


@requires_accelerator_dist_backend(["nccl"])
@instantiate_parametrized_tests
class TestInductorCollectivesMultiProc(DynamoDistributedMultiProcTestCase):
    device = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_align_runtime_estimations_across_multipg(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        group = c10d.distributed_c10d._get_default_group()
        rank = group.rank()
        import torch.distributed._functional_collectives as funcol

        mesh_tensor = torch.arange(4).reshape(2, 2)
        mesh = DeviceMesh("cuda", mesh_tensor)
        group0 = mesh.get_group(0)
        group0_size = group0.size()
        group0_name = funcol._resolve_group_name(group0)

        group1 = mesh.get_group(1)
        group1_size = group1.size()
        group1_name = funcol._resolve_group_name(group1)

        c10d_func = torch.ops._c10d_functional

        def func(a, b):
            w0 = c10d_func.all_gather_into_tensor(a, group0_size, group0_name)
            ag0 = c10d_func.wait_tensor(w0)
            w1 = c10d_func.all_gather_into_tensor(ag0, group1_size, group1_name)
            ag1 = c10d_func.wait_tensor(w1)
            return torch.ops.aten.mm.default(ag1, b)

        def inps():
            return torch.randn(2, 16, device=self.device), torch.randn(
                16, 16, device=self.device
            )

        ins = inps()
        runtime_estimations = []

        def _post_scheduler_pass(
            snodes: list[BaseSchedulerNode],
        ) -> list[BaseSchedulerNode]:
            nonlocal runtime_estimations
            for snode in snodes:
                runtime_estimations.append(snode.get_estimated_runtime())
            return snodes

        import random

        random.seed(42 + rank)

        def _runtime_estimation(_):
            ret = random.random()
            return ret

        with (
            torch._inductor.config.patch(
                {
                    "estimate_op_runtime": _runtime_estimation,
                    "runtime_estimations_mms_benchmark": True,
                    "reorder_for_compute_comm_overlap": True,
                    "reorder_for_compute_comm_overlap_passes": [
                        _post_scheduler_pass,
                    ],
                }
            ),
            torch._inductor.config_comms.patch(
                {
                    "runtime_estimations_align_across_all_distributed_ranks": True,
                }
            ),
        ):
            compiled = torch.compile(func, fullgraph=True)
            compiled(*ins)
        gathered_runtime_estimations: list[list[float]] = [
            [] for _ in range(group.size())
        ]
        c10d.all_gather_object(gathered_runtime_estimations, runtime_estimations, group)
        row0 = gathered_runtime_estimations[0]
        for row_i in gathered_runtime_estimations[1:]:
            assert row0 == row_i


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
