import os
import sys

import torch
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_utils import skipIfRocm

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

class TestCUDA(JitTestCase):
    """
    A suite of tests for the CUDA API in TorchScript.
    """

    @skipIfRocm
    def test_simple_stream(self):
        @torch.jit.script
        def fn():
            s = torch.classes.cuda.Stream(0, 0)
            return s is not None
        self.assertEqual(fn(), True, "Could not create Stream!")

    @skipIfRocm
    def test_streams(self):
        @torch.jit.script
        def test_get_stream():
            device_index = torch.cuda.current_device()
            user_stream = torch.classes.cuda.Stream(device_index, 0)
            return device_index == user_stream.device_index()

        @torch.jit.script
        def test_stream_synchronize() -> float:
            device_index = torch.cuda.current_device()
            stream = torch.cuda.current_stream(device_index)
            event = torch.classes.cuda.Event(True, False, False)
            start_event = torch.classes.cuda.Event(True, False, False)
            stream.record_event(start_event)
            stream.record_event(event)
            event.synchronize()
            return start_event.elapsed_time(event)

        self.assertEqual(test_get_stream(), True, "Stream was not created successfully!")
        self.assertEqual(test_default_and_current_stream(), True, "Default stream and current stream are the same")
        self.assertTrue(test_query())
        self.assertGreater(test_stream_synchronize(), 0.0)

    @skipIfRocm
    def test_with_stream(self):
        @torch.jit.script
        def fn():
            device_index = torch.cuda.current_device()
            user_stream = torch.classes.cuda.Stream(device_index, 0, False)
            A = torch.rand(1000, 1000, device = "cuda")

            with torch.cuda.stream(user_stream):
                v = torch.cuda.current_stream_id(device_index).pack()
                k = user_stream.pack()
                d = torch.cuda.default_stream(device_index)
                B = torch.mm(A, A)
            return A, B, v, k, d
        A, B, v, k, d = fn()
        print(v,k, d)
        self.assertEqual(torch.matmul(A, A), B)
        self.assertEqual(v, True)

    @skipIfRocm
    def test_with_multiple_stream(self):
        @torch.jit.script
        def fn():
            s1 = torch.classes.cuda.Stream(0, 0)
            s2 = torch.classes.cuda.Stream(1, 0)

            A = torch.rand(1000, 1000, device = "cuda:0")
            B = torch.rand(1000, 1000, device = "cuda:1")
            with torch.cuda.stream(s1):
                C = torch.mm(A, A)
                with torch.cuda.stream(s2):
                    D = torch.mm(B, B)
                # Wait for D to be computed
                s2.synchronize()
            return A, B, C, D
        A, B, C, D = fn()
        self.assertEqual(torch.matmul(A, A), C)
        self.assertEqual(torch.matmul(B, B), D)

    @skipIfRocm
    def test_simple_event(self):
        @torch.jit.script
        def fn():
            e = torch.classes.cuda.Event(True, False, False)
            return e is not None
        self.assertEqual(fn(), True, "Could not create CUDA Event!")

    @skipIfRocm
    def test_events(self):
        @torch.jit.script
        def test_event_query() -> bool:
            s = torch.classes.cuda.Stream(0, 0)
            e = torch.classes.cuda.Event(True, False, False)
            e.record(s)
            return e.query()

        @torch.jit.script
        def test_event_synchronize() -> float:
            s = torch.classes.cuda.Stream(0, 0)
            e_tik = torch.classes.cuda.Event(True, False, False)
            e_tok = torch.classes.cuda.Event(True, False, False)

            e_tik.record(s)
            s.record_event(e_tok)
            e_tok.synchronize()
            # not necessary to check e_tik and e_tok, as elapsed_time would throw
            # exception if otherwise.
            return e_tik.elapsed_time(e_tok)

        @torch.jit.script
        def test_event_wait() -> float:
            device_index = torch.cuda.current_device()
            s0 = torch.cuda.current_stream(device_index)
            s1 = torch.classes.cuda.Stream(0, 0)
            e_tik = torch.classes.cuda.Event(True, True, False)
            e_tok = torch.classes.cuda.Event(True, True, False)

            e_tik.record(s0)
            e_sync = torch.classes.cuda.Event(True, False, False)
            e_sync.record(torch.cuda.current_stream(device_index))
            e_sync.wait(s1)
            with torch.cuda.stream(s1):
                t = 10
            s1.synchronize()
            e_tok.record(torch.cuda.current_stream(device_index))
            e_tok.synchronize()

            if not s0.query() or not s1.query() or not e_sync.query():
                return -1.0

            # not necessary to check e_tik and e_tok, as elapsed_time would throw
            # exception if otherwise.
            return e_tik.elapsed_time(e_tok)

        self.assertTrue(test_event_query())
        self.assertGreater(test_event_synchronize(), 0.0)
        self.assertGreater(test_event_wait(), 0.0)
