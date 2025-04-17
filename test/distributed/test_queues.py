import torch
import torch.distributed as dist

from torch.distributed._queue import TensorQueue, TensorStore
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TestCase


class TestQueues(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def create_tcp_store(self):
        store = dist.FileStore(self.file_name, self.world_size)

        KEY = f"{self.id()}/port"

        if self.rank == 0:
            tcp_store = dist.TCPStore(
                "127.0.0.1", 0, self.world_size, is_master=True, wait_for_workers=False
            )
            store.set(KEY, str(tcp_store.port))
            return tcp_store
        else:
            port = int(store.get(KEY))
            return dist.TCPStore(
                "127.0.0.1",
                port,
                self.world_size,
                is_master=False,
                wait_for_workers=False,
            )

    def test_queues(self):
        store = self.create_tcp_store()
        dist.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )

        queue = TensorQueue("foo")

        for i in range(10):
            expected = torch.tensor([1, 2, 3, i])
            out = torch.tensor([0, 0, 0, 0])

            if self.rank == 0:
                queue.push(expected)
            else:
                queue.pop(out)
                self.assertEqual(out, expected)

    def test_tensor_store(self):
        store = self.create_tcp_store()
        dist.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )

        tensor_store = TensorStore("foo")

        tensors = []

        for i in range(10):
            print(f"iter {i=} {self.rank=}")
            key = f"key_{i}"
            if self.rank == 0:
                t = torch.tensor([0])
                tensors.append(t)
                tensor_store.register(key, t)
            else:
                t = torch.tensor([i])
                tensor_store.write(key, t)

                out = torch.tensor([0])
                tensor_store.read(key, out)
                self.assertEqual(out, t)

        # exit barrier
        dist.barrier()

        for i, t in enumerate(tensors):
            self.assertEqual(t, torch.tensor([i]))


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
