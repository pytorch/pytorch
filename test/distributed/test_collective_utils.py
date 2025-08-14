# Owner(s): ["oncall: distributed"]

from unittest import mock

import torch
import torch.distributed as c10d
from torch.distributed.collective_utils import (
    _check_rng_sync,
    _check_rng_sync_internal,
    all_gather,
    broadcast,
)
from torch.testing import FileCheck
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class TestCollectiveUtils(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        super().tearDown()

    def opts(self, threads=2):
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = 50.0
        opts._threads = threads
        return opts

    def test_broadcast_result(self) -> None:
        """
        Basic unit test for broadcast using a process group of default world size.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        pg = c10d.new_group(pg_options=self.opts())

        func = mock.MagicMock()
        func.return_value = pg.rank()

        res = broadcast(data_or_fn=func, rank=0, pg=pg)
        assert res == 0, f"Expect res to be 0 (got {res})"

        if pg.rank() == 0:
            func.assert_called_once()
        else:
            func.assert_not_called()

        func.reset_mock()

        res = broadcast(data_or_fn=func, rank=1, pg=pg)
        assert res == 1, f"Expect res to be 1 (got {res})"

        if pg.rank() == 1:
            func.assert_called_once()
        else:
            func.assert_not_called()

    def test_broadcast_result_no_pg(self) -> None:
        """
        Ensure broadcast has no dependency on torch.distributed when run in single process.
        """
        func = mock.MagicMock()
        broadcast(data_or_fn=func, rank=0)
        func.assert_called_once()

    def test_broadcast_result_raises_exceptions_from_func(
        self,
    ) -> None:
        """
        Ensure broadcast exception is propagated properly.
        """
        # no process group
        func = mock.MagicMock()
        exc = Exception("test exception")
        func.side_effect = exc
        expected_exception = "test exception"
        with self.assertRaisesRegex(Exception, expected_exception):
            broadcast(data_or_fn=func, rank=0)

    def test_all_gather_result(self) -> None:
        """
        Basic unit test for all_gather using a process group of default world size.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        pg = c10d.new_group(pg_options=self.opts())

        func = mock.MagicMock()
        func.return_value = pg.rank()

        res = all_gather(data_or_fn=func, pg=pg)
        func.assert_called_once()
        assert res == list(range(self.world_size)), (
            f"Expect res to be list of 0 through {self.world_size} (got {res})"
        )

    def test_all_gather_result_no_pg(self) -> None:
        """
        Ensure all_gather has no dependency on torch.distributed when run in single process.
        """
        func = mock.MagicMock()
        all_gather(data_or_fn=func)
        func.assert_called_once()

    def test_all_gather_result_raises_exceptions_from_func(
        self,
    ) -> None:
        """
        Ensure all_gather exception is propagated properly.
        """
        # no process group
        func = mock.MagicMock()
        exc = Exception("test exception")
        func.side_effect = exc
        expected_exception = "test exception"
        with self.assertRaisesRegex(Exception, expected_exception):
            all_gather(data_or_fn=func)

    @parametrize("device", ["cpu", "cuda"])
    def test_check_rng_sync(
        self,
        device,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("Cuda is not available")
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        group = torch.distributed.distributed_c10d._get_default_group()
        generator = torch.Generator(device=device)
        generator.manual_seed(123)
        value_ranks, _ = _check_rng_sync_internal(generator, group)
        self.assertEqual(len(value_ranks), 1, value_ranks)
        for actual, expected in zip(value_ranks.values(), [{0, 1, 2, 3}]):
            self.assertEqual(actual, expected, actual)

        if torch.distributed.get_rank() == 1:
            torch.randn((10,), device=device, generator=generator)
        value_ranks, _ = _check_rng_sync_internal(generator, group)
        self.assertEqual(len(value_ranks), 2, value_ranks)
        for actual, expected in zip(value_ranks.values(), [{0, 2, 3}, {1}]):
            self.assertEqual(actual, expected, actual)

        if torch.distributed.get_rank() == 0:
            generator.manual_seed(456)
        value_ranks, _ = _check_rng_sync_internal(generator, group)
        self.assertEqual(len(value_ranks), 3, value_ranks)
        for actual, expected in zip(value_ranks.values(), [{0}, {1}, {2, 3}]):
            self.assertEqual(actual, expected, actual)

        log_str = _check_rng_sync(generator, group)
        FileCheck().check("Generator desync detected").check("Ranks").check("0").check(
            "1"
        ).check("2-3").run(log_str)


instantiate_parametrized_tests(TestCollectiveUtils)

if __name__ == "__main__":
    run_tests()
