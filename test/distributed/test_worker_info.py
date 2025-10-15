# Owner(s): ["oncall: distributed"]

import os
from unittest import mock

import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TestCase, run_tests


class WorkerInfoTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._original_env = {
            key: os.environ.get(key)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        }
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        super().tearDown()

    def test_env_variables_used_when_pg_not_initialized(self) -> None:
        os.environ["RANK"] = "3"
        os.environ["LOCAL_RANK"] = "1"
        os.environ["WORLD_SIZE"] = "8"

        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            info = dist.get_worker_info()

        self.assertEqual(info.rank, 3)
        self.assertEqual(info.local_rank, 1)
        self.assertEqual(info.world_size, 8)
        self.assertTrue(info.is_distributed)

    def test_fallback_values_used_when_env_absent(self) -> None:
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            info = dist.get_worker_info(
                fallback_rank=7, fallback_local_rank=4, fallback_world_size=8
            )

        self.assertEqual(info.rank, 7)
        self.assertEqual(info.local_rank, 4)
        self.assertEqual(info.world_size, 8)
        self.assertTrue(info.is_distributed)

    def test_initialized_pg_takes_precedence(self) -> None:
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=True
        ), mock.patch(
            "torch.distributed.distributed_c10d.get_rank", return_value=5
        ), mock.patch(
            "torch.distributed.distributed_c10d.get_world_size", return_value=16
        ), mock.patch(
            "torch.distributed.distributed_c10d.get_node_local_rank", return_value=2
        ):
            info = dist.get_worker_info()

        self.assertEqual(info.rank, 5)
        self.assertEqual(info.local_rank, 2)
        self.assertEqual(info.world_size, 16)
        self.assertTrue(info.is_distributed)

    def test_env_rank_without_local_rank_defaults_to_rank(self) -> None:
        os.environ["RANK"] = "4"
        os.environ.pop("LOCAL_RANK", None)
        os.environ["WORLD_SIZE"] = "9"

        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            with self.assertWarnsRegex(RuntimeWarning, "LOCAL_RANK.*not set"):
                info = dist.get_worker_info(fallback_local_rank=99)

        self.assertEqual(info.rank, 4)
        self.assertEqual(info.local_rank, 4)
        self.assertEqual(info.world_size, 9)
        self.assertTrue(info.is_distributed)

    def test_env_invalid_values_raise(self) -> None:
        os.environ["RANK"] = "not-a-number"
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            with self.assertRaises(ValueError):
                dist.get_worker_info()

    def test_env_invalid_local_rank_raises(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "not-an-int"
        os.environ["WORLD_SIZE"] = "1"
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            with self.assertRaises(ValueError):
                dist.get_worker_info()

    def test_env_invalid_world_size_raises(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "not-a-number"
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            with self.assertRaises(ValueError):
                dist.get_worker_info()

    def test_world_size_one_reports_non_distributed(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            info = dist.get_worker_info()

        self.assertFalse(info.is_distributed)

    def test_get_node_local_rank_failure_falls_back_to_rank(self) -> None:
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=True
        ), mock.patch(
            "torch.distributed.distributed_c10d.get_rank", return_value=6
        ), mock.patch(
            "torch.distributed.distributed_c10d.get_world_size", return_value=12
        ), mock.patch(
            "torch.distributed.distributed_c10d.get_node_local_rank",
            side_effect=RuntimeError("no local rank"),
        ):
            info = dist.get_worker_info()

        self.assertEqual(info.rank, 6)
        self.assertEqual(info.local_rank, 6)
        self.assertEqual(info.world_size, 12)
        self.assertTrue(info.is_distributed)

    def test_world_size_fallback_when_env_missing(self) -> None:
        os.environ["RANK"] = "2"
        os.environ.pop("WORLD_SIZE", None)
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            with self.assertWarnsRegex(RuntimeWarning, "LOCAL_RANK.*not set"):
                info = dist.get_worker_info(fallback_world_size=3)

        self.assertEqual(info.world_size, 3)
        self.assertTrue(info.is_distributed)

    def test_negative_rank_raises(self) -> None:
        os.environ["RANK"] = "-2"
        os.environ["WORLD_SIZE"] = "1"
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "rank must be non-negative"):
                dist.get_worker_info()

    def test_negative_world_size_raises(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "0"
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "world_size must be at least 1"):
                dist.get_worker_info()

    def test_rank_exceeds_world_size_raises(self) -> None:
        os.environ["RANK"] = "5"
        os.environ["WORLD_SIZE"] = "5"
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            with self.assertWarnsRegex(RuntimeWarning, "LOCAL_RANK.*not set"):
                with self.assertRaisesRegex(
                    ValueError, "rank \\(5\\) must be less than"
                ):
                    dist.get_worker_info()

    def test_multi_node_local_rank_fallback_warns(self) -> None:
        os.environ["RANK"] = "4"
        os.environ.pop("LOCAL_RANK", None)
        os.environ["WORLD_SIZE"] = "8"
        with mock.patch(
            "torch.distributed.distributed_c10d.is_initialized", return_value=False
        ):
            with self.assertWarnsRegex(RuntimeWarning, "LOCAL_RANK.*not set"):
                info = dist.get_worker_info()
        self.assertEqual(info.rank, 4)
        self.assertEqual(info.local_rank, 4)
        self.assertEqual(info.world_size, 8)


class WorkerInfoDistributedTest(MultiProcessTestCase):
    @property
    def world_size(self) -> int:  # type: ignore[override]
        return 2

    def setUp(self) -> None:
        super().setUp()
        if not dist.is_available() or not dist.is_backend_available("gloo"):
            self.skipTest("c10d gloo backend is required for this test")
        self._spawn_processes()

    def test_helper_tracks_process_group_transition(self) -> None:
        if self.rank == self.MAIN_PROCESS_RANK:
            return

        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("LOCAL_RANK", None)

        pre_info = dist.get_worker_info(
            fallback_rank=0, fallback_local_rank=77, fallback_world_size=1
        )
        self.assertEqual(pre_info.rank, 0)
        self.assertEqual(pre_info.local_rank, 77)
        self.assertEqual(pre_info.world_size, 1)
        self.assertFalse(pre_info.is_distributed)

        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )

        post_info = dist.get_worker_info()
        self.assertEqual(post_info.rank, self.rank)
        self.assertEqual(post_info.world_size, self.world_size)
        self.assertEqual(post_info.local_rank, self.rank)
        self.assertTrue(post_info.is_distributed)

        dist.barrier()

    def tearDown(self) -> None:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        finally:
            super().tearDown()


if __name__ == "__main__":
    run_tests()
