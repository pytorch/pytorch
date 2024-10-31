# Owner(s): ["oncall: distributed"]

from unittest import mock

import torch.distributed as c10d
from torch.distributed.collective_utils import all_gather, broadcast
from torch.testing._internal.common_distributed import MultiProcessTestCase


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
        res = broadcast(data_or_fn=func, rank=0)
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
        assert res == list(
            range(self.world_size)
        ), f"Expect res to be list of 0 through {self.world_size} (got {res})"

    def test_all_gather_result_no_pg(self) -> None:
        """
        Ensure all_gather has no dependency on torch.distributed when run in single process.
        """
        func = mock.MagicMock()
        res = all_gather(data_or_fn=func)
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
