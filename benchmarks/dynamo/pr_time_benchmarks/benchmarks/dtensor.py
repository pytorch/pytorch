import sys

from benchmark_base import BenchmarkBase

import torch
from torch.distributed._tensor import DTensor, Partial, Replicate, Shard
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


class BenchmarkToFromLocal(BenchmarkDTensorDispatch):
    def __init__(self, world_size) -> None:
        super().__init__(operator="to_from_local", world_size=world_size)

    def _work(self) -> None:
        local = self.a.to_local()
        DTensor.from_local(local, self.mesh, [Replicate()])


class BenchmarkCollectives(BenchmarkDTensorDispatch):
    def __init__(self, world_size) -> None:
        super().__init__(operator="collectives", world_size=world_size)

    def _prepare_once(self) -> None:
        super()._prepare_once()
        self.c = DTensor.from_local(
            torch.ones(10, 10, device=self.device()), self.mesh, [Partial()]
        )

    def _work(self) -> None:
        # shard
        a = self.a.redistribute(placements=[Shard(0)])
        # alltoall
        a = a.redistribute(placements=[Shard(1)])
        # allgather
        a = a.redistribute(placements=[Replicate()])
        # allreduce
        self.c.redistribute(placements=[Replicate()])
        # reducescatter
        self.c.redistribute(placements=[Shard(0)])


class BenchmarkAddBackward(BenchmarkDTensorDispatch):
    def __init__(self, world_size) -> None:
        super().__init__(operator="add_backward", world_size=world_size)

    def _prepare_once(self) -> None:
        super()._prepare_once()
        self.a = DTensor.from_local(
            torch.ones(2, 512, device=self.device(), requires_grad=True),
            self.mesh,
            [Shard(0)],
        )
        self.b = DTensor.from_local(
            torch.ones(512, 2, device=self.device(), requires_grad=True),
            self.mesh,
            [Shard(1)],
        )

    def _work(self) -> None:
        out = self.a + self.b
        out.sum().backward()


class BenchmarkInplace(BenchmarkDTensorDispatch):
    def __init__(self, world_size) -> None:
        super().__init__(operator="inplace", world_size=world_size)

    def _work(self) -> None:
        self.a.add_(self.b)


class BenchmarkView(BenchmarkDTensorDispatch):
    def __init__(self, world_size) -> None:
        super().__init__(operator="view", world_size=world_size)

    def _work(self) -> None:
        self.a.view(100)


class BenchmarkRandom(BenchmarkDTensorDispatch):
    def __init__(self, world_size) -> None:
        super().__init__(operator="random", world_size=world_size)

    def _work(self) -> None:
        self.a.uniform_()


class BenchmarkCustomHandler(BenchmarkDTensorDispatch):
    def __init__(self, world_size) -> None:
        super().__init__(operator="custom_handler", world_size=world_size)

    def _work(self) -> None:
        torch.ops.aten.is_same_size(self.a, self.b)


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
    BenchmarkToFromLocal(
        world_size
    ).enable_instruction_count().collect_all().append_results(result_path)
    BenchmarkCollectives(
        world_size
    ).enable_instruction_count().collect_all().append_results(result_path)
    BenchmarkAddBackward(
        world_size
    ).enable_instruction_count().collect_all().append_results(result_path)
    BenchmarkInplace(
        world_size
    ).enable_instruction_count().collect_all().append_results(result_path)
    BenchmarkView(world_size).enable_instruction_count().collect_all().append_results(
        result_path
    )
    BenchmarkRandom(world_size).enable_instruction_count().collect_all().append_results(
        result_path
    )
    BenchmarkCustomHandler(
        world_size
    ).enable_instruction_count().collect_all().append_results(result_path)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
