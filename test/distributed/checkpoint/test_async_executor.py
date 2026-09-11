# Owner(s): ["oncall: distributed checkpointing"]

import os
import tempfile
import threading
import weakref
from unittest.mock import patch

import torch
from torch.distributed.checkpoint._async_executor import _OwnedStateDictFuture
from torch.distributed.checkpoint._async_thread_executor import (
    _ThreadBasedAsyncCheckpointExecutor,
)
from torch.distributed.checkpoint.state_dict_saver import async_save
from torch.testing._internal.common_utils import run_tests, TestCase


class _Payload:
    pass


class TestOwnedStateDictFuture(TestCase):
    def test_result_releases_state_dict_on_caller(self) -> None:
        save_started = threading.Event()
        allow_save = threading.Event()
        callback_started = threading.Event()
        allow_callback = threading.Event()
        release_threads = []
        cleanup_threads = []

        payload = _Payload()
        payload_ref = weakref.ref(payload)
        weakref.finalize(
            payload,
            lambda: release_threads.append(threading.current_thread().name),
        )
        cache = {"payload": payload}

        def cleanup() -> None:
            cleanup_threads.append(threading.current_thread().name)
            cache.clear()

        def fake_save(state_dict, **kwargs):
            save_started.set()
            self.assertTrue(allow_save.wait(5))
            return "saved"

        future = _OwnedStateDictFuture({"payload": payload}, cleanup)
        del payload
        executor = _ThreadBasedAsyncCheckpointExecutor()

        with patch(
            "torch.distributed.checkpoint.state_dict_saver.save",
            new=fake_save,
        ):
            self.assertIs(executor.execute_save(future), future)
            self.assertTrue(save_started.wait(5))

            def block_worker(completed_future):
                callback_started.set()
                allow_callback.wait(5)

            future.add_done_callback(block_worker)
            allow_save.set()
            self.assertTrue(callback_started.wait(5))
            try:
                self.assertEqual(future.result(timeout=5), "saved")
            finally:
                allow_callback.set()

        caller_thread = threading.current_thread().name
        self.assertEqual(len(cleanup_threads), 1)
        self.assertTrue(cleanup_threads[0].startswith("AsyncCheckpointExecutor"))
        self.assertEqual(release_threads, [caller_thread])
        self.assertIsNone(payload_ref())

    def test_async_save_uses_owned_state_dict_future(self) -> None:
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            future = async_save(
                {"value": torch.ones(2)},
                checkpoint_id=os.path.join(checkpoint_dir, "checkpoint"),
                no_dist=True,
            )
            self.assertIsInstance(future, _OwnedStateDictFuture)
            self.assertIsNotNone(future.result(timeout=30))
            self.assertIsNone(future._state_dict)


if __name__ == "__main__":
    run_tests()
