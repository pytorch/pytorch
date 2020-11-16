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
    def test_simple(self):
        @torch.jit.script
        def fn():
            s = torch.classes.cuda.Stream(0, 0)
            return False if s is None else True
        self.assertEqual(fn(), True, "Could not create Stream!")

    @skipIfRocm
    def test_streams(self):
        @torch.jit.script
        def test_get_stream():
            device_index = torch.cuda._cuda_getDevice()
            user_stream = torch.classes.cuda.Stream(device_index, 0)
            return device_index == user_stream.device_index()

        @torch.jit.script
        def test_default_and_current_stream():
            device_index = torch.cuda._cuda_getDevice()
            default_stream = torch.cuda._cuda_getStream(device_index)
            user_stream = torch.classes.cuda.Stream(device_index, 0)
            return default_stream.device_index() == user_stream.device_index()

        @torch.jit.script
        def test_query():
            device_index = torch.cuda._cuda_getDevice()
            user_stream = torch.classes.cuda.Stream(device_index, 0)
            return user_stream.query()

        @torch.jit.script
        def test_stream_synchronize() -> float:
            device_index = torch.cuda._cuda_getDevice()
            stream = torch.cuda._cuda_getStream(device_index)
            event = torch.classes.cuda.Event(True, False, False)
            start_event = torch.classes.cuda.Event(True, False, False)
            stream.record_event(start_event)
            stream.record_event(event)
            event.synchronize()
            return start_event.elapsed_time(event)

        self.assertEqual(test_get_stream(), True, "Stream was not created successfully!")
        self.assertEqual(test_default_and_current_stream(), True, "Default stream and current stream are not the same")
        self.assertTrue(test_query())
        self.assertGreater(test_stream_synchronize(), 0.0)

    @skipIfRocm
    def test_with_multiple_stream(self):
        @torch.jit.script
        def fn():
            device_index = torch.cuda._cuda_getDevice()
            s1 = torch.classes.cuda.Stream(0, 0)
            s2 = torch.classes.cuda.Stream(0, 0)
            c = 10
            with torch.cuda.stream(s1), torch.cuda.stream(s2):
                c = 20
            s2.synchronize()
            return c
        self.assertEqual(fn(), 20)

    @skipIfRocm
    def test_with_stream(self):
        @torch.jit.script
        def fn():
            device_index = torch.cuda._cuda_getDevice()
            user_stream = torch.classes.cuda.Stream(0, 0)
            value = True
            with torch.cuda.stream(user_stream):
                value = torch.cuda._cuda_getStream(device_index).device_index() == user_stream.device_index()
            return value
        self.assertEqual(fn(), True, "Current stream is not set to the user stream")

    @skipIfRocm
    def test_events(self):
        @torch.jit.script
        def test_event_query() -> bool:
            e = torch.classes.cuda.Event(True, False, False)
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
            device_index = torch.cuda._cuda_getDevice()
            s0 = torch.cuda._cuda_getStream(device_index)
            s1 = torch.classes.cuda.Stream(0, 0)
            e_tik = torch.classes.cuda.Event(True, True, False)
            e_tok = torch.classes.cuda.Event(True, True, False)

            e_tik.record(s0)
            e_sync = torch.classes.cuda.Event(True, False, False)
            e_sync.record(torch.cuda._cuda_getStream(device_index))
            e_sync.wait(s1)
            with torch.cuda.stream(s1):
                t = 10
            s1.synchronize()
            e_tok.record(torch.cuda._cuda_getStream(device_index))
            e_tok.synchronize()

            if not s0.query() or not s1.query() or not e_sync.query():
                return -1.0

            # not necessary to check e_tik and e_tok, as elapsed_time would throw
            # exception if otherwise.
            return e_tik.elapsed_time(e_tok)

        self.assertTrue(test_event_query())
        self.assertGreater(test_event_synchronize(), 0.0)
        self.assertGreater(test_event_wait(), 0.0)
