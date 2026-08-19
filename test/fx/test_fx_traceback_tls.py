# Owner(s): ["module: fx"]

import queue
import threading

import torch
import torch.fx.traceback as fx_traceback
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFXTracebackThreadLocalState(TestCase):
    def test_preserve_node_state_is_thread_local(self):
        graph = torch.fx.Graph()
        replay_node = graph.placeholder("x")
        ready = threading.Event()
        release = threading.Event()
        observed = queue.Queue()
        errors = queue.Queue()

        def owner_thread():
            try:
                with (
                    fx_traceback.preserve_node_meta(),
                    fx_traceback._preserve_node_seq_nr(),
                    fx_traceback.set_current_replay_node(replay_node),
                    fx_traceback.annotate({"owner": "thread-a"}),
                ):
                    fx_traceback.set_stack_trace(["stack-from-thread-a"])
                    ready.set()
                    if not release.wait(timeout=5):
                        raise AssertionError("timed out waiting for observer")
            except Exception as e:
                errors.put(e)
                release.set()

        def observer_thread():
            try:
                if not ready.wait(timeout=5):
                    raise AssertionError("timed out waiting for owner")
                observed.put(
                    {
                        "has_preserved_node_meta": (
                            fx_traceback.has_preserved_node_meta()
                        ),
                        "is_preserving_node_seq_nr": (
                            fx_traceback._is_preserving_node_seq_nr()
                        ),
                        "current_meta": dict(fx_traceback.get_current_meta()),
                        "direct_current_meta": dict(fx_traceback.current_meta),
                        "current_replay_node": fx_traceback.get_current_replay_node(),
                    }
                )
            except Exception as e:
                errors.put(e)
            finally:
                release.set()

        threads = [
            threading.Thread(target=owner_thread),
            threading.Thread(target=observer_thread),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())

        if not errors.empty():
            raise errors.get()

        self.assertEqual(
            observed.get_nowait(),
            {
                "has_preserved_node_meta": False,
                "is_preserving_node_seq_nr": False,
                "current_meta": {},
                "direct_current_meta": {},
                "current_replay_node": None,
            },
        )

    def test_preserve_node_contexts_restore_current_thread(self):
        saved_meta = fx_traceback.current_meta.copy()
        fx_traceback.current_meta.clear()
        try:
            self.assertFalse(fx_traceback.has_preserved_node_meta())
            self.assertFalse(fx_traceback._is_preserving_node_seq_nr())
            fx_traceback.set_stack_trace(["ignored"])
            self.assertNotIn("stack_trace", fx_traceback.current_meta)

            with fx_traceback.preserve_node_meta():
                self.assertTrue(fx_traceback.has_preserved_node_meta())
                fx_traceback.set_stack_trace(["outer"])
                self.assertEqual(fx_traceback.format_stack(), ["outer"])

                with fx_traceback.annotate({"outer": 1}):
                    self.assertEqual(
                        fx_traceback.get_current_meta()["custom"], {"outer": 1}
                    )
                    with fx_traceback.annotate({"inner": 2}):
                        self.assertEqual(
                            fx_traceback.get_current_meta()["custom"],
                            {"outer": 1, "inner": 2},
                        )
                    self.assertEqual(
                        fx_traceback.get_current_meta()["custom"], {"outer": 1}
                    )
                self.assertNotIn("custom", fx_traceback.current_meta)

                with fx_traceback._preserve_node_seq_nr():
                    self.assertTrue(fx_traceback._is_preserving_node_seq_nr())
                    with fx_traceback._preserve_node_seq_nr(False):
                        self.assertFalse(fx_traceback._is_preserving_node_seq_nr())
                    self.assertTrue(fx_traceback._is_preserving_node_seq_nr())
                self.assertFalse(fx_traceback._is_preserving_node_seq_nr())

                with fx_traceback.preserve_node_meta(False):
                    self.assertFalse(fx_traceback.has_preserved_node_meta())
                    fx_traceback.set_stack_trace(["inner"])
                    self.assertEqual(
                        fx_traceback.get_current_meta()["stack_trace"], "outer"
                    )
                self.assertTrue(fx_traceback.has_preserved_node_meta())
                self.assertEqual(fx_traceback.format_stack(), ["outer"])

            self.assertFalse(fx_traceback.has_preserved_node_meta())
            self.assertFalse(fx_traceback._is_preserving_node_seq_nr())
            self.assertNotIn("stack_trace", fx_traceback.current_meta)

            graph = torch.fx.Graph()
            replay_node = graph.placeholder("x")
            other_replay_node = graph.placeholder("y")
            with fx_traceback.set_current_replay_node(replay_node):
                self.assertIs(fx_traceback.get_current_replay_node(), replay_node)
                with fx_traceback.set_current_replay_node(other_replay_node):
                    self.assertIs(
                        fx_traceback.get_current_replay_node(), other_replay_node
                    )
                self.assertIs(fx_traceback.get_current_replay_node(), replay_node)
            self.assertIsNone(fx_traceback.get_current_replay_node())
        finally:
            fx_traceback.current_meta.clear()
            fx_traceback.current_meta.update(saved_meta)


if __name__ == "__main__":
    run_tests()
