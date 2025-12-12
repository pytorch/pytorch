import sys

from benchmark_base import BenchmarkBase

import torch
from torch.distributed._tensor import DTensor, Replicate
from torch.testing._internal.distributed.fake_pg import FakeStore


class BenchmarkDTensorDispatch(BenchmarkBase):
    def __init__(self, operator, world_size) -> None:
        super().__init__(
            category=f"dtensor_dispatch_{operator}",
            device="cuda",
        )
        self.world_size = world_size

    def name(self) -> str:
        prefix = f"{self.category()}"
        return prefix

    def description(self) -> str:
        return f"DTensor dispatch time for {self.category()}"

    def _prepare_once(self) -> None:
        self.mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", (self.world_size,), mesh_dim_names=("dp",)
        )
        self.a = DTensor.from_local(
            torch.ones(10, 10, device=self.device()), self.mesh, [Replicate()]
        )
        self.b = DTensor.from_local(
            torch.ones(10, 10, device=self.device()), self.mesh, [Replicate()]
        )

    def _prepare(self) -> None:
        pass


class BenchmarkDetach(BenchmarkDTensorDispatch):
    def __init__(self, world_size) -> None:
        super().__init__(operator="detach", world_size=world_size)

    def _work(self) -> None:
        self.a.detach()


def main():
    world_size = 256
    fake_store = FakeStore()
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )
    result_path = sys.argv[1]
    BenchmarkDetach(world_size).enable_instruction_count().collect_all().append_results(
        result_path
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
