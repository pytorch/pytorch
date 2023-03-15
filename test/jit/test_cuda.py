# Owner(s): ["oncall: jit"]

import os
import sys
import gc
import unittest

import torch
from typing import NamedTuple
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_utils import skipIfRocm, skipCUDANonDefaultStreamIf, NoTest

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# Check if GPU is available
TEST_CUDA = torch.cuda.is_available()
# Check if multiple GPU's are available
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2

# If GPU is not available, then do not run the tests
if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    JitTestCase = NoTest  # noqa: F811

TEST_LARGE_TENSOR = TEST_CUDA

# If GPU is available, then initialize the cuda context and check
# if there is memory available to allocate for LARGE Tensors.
if TEST_CUDA:
    torch.ones(1).cuda()  # initialize cuda context
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 5e9

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
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        super().tearDown()

    @skipIfRocm
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_cuda_synchronize(self):
        # Test device synchronization.

        @torch.jit.script
        def test_device_synchronize():
            prev_current_device_index = torch.cuda.current_device()
            torch.cuda.synchronize()
            torch.cuda.synchronize('cuda')
            torch.cuda.synchronize('cuda:0')
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(torch.device('cuda:1'))
            after_current_device_index = torch.cuda.current_device()

            # Check if the current device index is same as the device index before
            # synchronizing the device.
            return prev_current_device_index == after_current_device_index

        @torch.jit.script
        def test_multi_device_synchronize():
            torch.cuda.synchronize(torch.device('cuda:0'))
            prev_current_device_index = torch.cuda.current_device()
            torch.cuda.synchronize(1)
            after_current_device_index = torch.cuda.current_device()

            # Check if the current device index is same as the device index before
            # synchronizing the device.
            return prev_current_device_index == after_current_device_index

        self.assertTrue(test_device_synchronize)
        FileCheck().check("cuda::synchronize(") \
                   .run(test_device_synchronize.graph)
        self.assertTrue(test_multi_device_synchronize)
        FileCheck().check("cuda::synchronize(") \
                   .run(test_multi_device_synchronize.graph)

    def test_stream_args(self):
        # Test stream creation with default arguments
        @torch.jit.script
        def stream_default_args() -> bool:
            s = torch.cuda.Stream()
            return s.device_index() == torch.cuda.current_device()

        @torch.jit.script
        def stream_default_args_for_device() -> bool:
            s = torch.cuda.Stream(priority=0)
            return s.device_index() == torch.cuda.current_device()

        @torch.jit.script
        def stream_default_args_for_priority() -> bool:
            d = torch.device("cuda:1")
            s = torch.cuda.Stream(d)
            return s.device_index() == 1

        @torch.jit.script
        def stream_args_all() -> bool:
            d = torch.device("cuda:0")
            s = torch.cuda.Stream(d, 0)
            return s.device_index() == 0

        self.assertTrue(stream_default_args)
        self.assertTrue(stream_default_args_for_device)
        self.assertTrue(stream_default_args_for_priority)
        self.assertTrue(stream_args_all)

    def test_event_args(self):
        # Test Event creation with default arguments
        @torch.jit.script
        def event_default_args() -> bool:
            e = torch.cuda.Event()
            return e is not None

        self.assertTrue(event_default_args)

    @skipIfRocm
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_current_stream(self):
        # Test current stream on the device and check if the stream device index
        # matches with the device ID
        @torch.jit.script
        def fn():
            device_index = torch.cuda.current_device()
            device = torch.device("cuda:" + str(device_index))
            s0 = torch.cuda.current_stream(device)
            s1 = torch.cuda.current_stream(torch.device("cuda:1"))
            s2 = torch.cuda.current_stream(torch.device("cuda:0"))

            return s0.device_index(), s1.device_index(), s2.device_index()

        d0, d1, d2 = fn()
        # By default, the current device ID is 0.
        self.assertEqual(0, d0)
        self.assertEqual(1, d1)
        self.assertEqual(0, d2)
        self.assertEqual(d0, d2)

        # Test current_stream API by passing device ID as an argument and
        # and check if the stream device index matches with the device ID
        @torch.jit.script
        def fn_with_device_index_args():
            device_index = torch.cuda.current_device()
            s0 = torch.cuda.current_stream(device_index)
            s1 = torch.cuda.current_stream(1)
            s2 = torch.cuda.current_stream(0)

            return s0.device_index(), s1.device_index(), s2.device_index()

        d0, d1, d2 = fn_with_device_index_args()
        # By default, the current device ID is 0.
        self.assertEqual(0, d0)
        self.assertEqual(1, d1)
        self.assertEqual(0, d2)
        self.assertEqual(d0, d2)

    @skipIfRocm
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @skipCUDANonDefaultStreamIf(True)
    def test_streams_and_events(self):

        # Test default_stream API by passing device ID as an argument and
        # and check if the stream device index matches with the device ID
        @torch.jit.script
        def test_default_streams_with_device_index_args():
            s0 = torch.cuda.default_stream(0)
            s1 = torch.cuda.default_stream(1)
            return s0.device_index(), s1.device_index()

        d0, d1 = test_default_streams_with_device_index_args()

        self.assertEqual(d0, 0)
        self.assertEqual(d1, 1)

        # This test checks for the default stream ID is set to 0 on the device
        @torch.jit.script
        def test_default_streams():
            s0 = torch.cuda.default_stream(torch.device('cuda:0'))
            s1 = torch.cuda.default_stream(torch.device('cuda:1'))

            d = torch.device('cuda:1')

            # Check the current stream id and default id are same
            # on the current device. The current device id by default is 0
            s2 = torch.cuda.current_stream(torch.device('cuda:0'))
            check_s2 = s2.id() == s0.id()
            check_d0 = torch.cuda.current_device() == s2.device_index()

            # Set the current device to d1 and check if the stream
            # has been set to the default stream on d1
            with torch.cuda.device(d):
                s3 = torch.cuda.current_stream(d)
                check_s3 = s3.id() == s1.id()
                check_d1 = torch.cuda.current_device() == s3.device_index()

            # Check if the current device was reset to 0
            is_device_d0 = torch.cuda.current_device() == s2.device_index()

            return s0.device_index(), s1.device_index(), check_s2, check_s3, check_d0, check_d1, is_device_d0

        d0, d1, check_s2, check_s3, check_d0, check_d1, is_device_d0 = test_default_streams()

        self.assertEqual(d0, 0)
        self.assertEqual(d1, 1)
        self.assertTrue(check_s2)
        self.assertTrue(check_s3)
        self.assertTrue(check_d0)
        self.assertTrue(check_d1)
        self.assertTrue(is_device_d0)

        # This test checks if the Stream Context manager is a no op
        # when the stream is none for `with torch.cuda.stream`
        @torch.jit.script
        def test_set_none_stream():
            device_index = torch.cuda.current_device()
            device = torch.device("cuda:" + str(device_index))
            current_stream = torch.cuda.current_stream(device)
            default_stream = torch.cuda.default_stream(device)

            # When stream is none, check if this operation is a no-op
            with torch.cuda.stream(None):
                cur_device_index = torch.cuda.current_device()
                is_device_index_same = cur_device_index == device_index
                is_current_stream_same = torch.cuda.current_stream(device).id() == current_stream.id()
                is_default_stream_same = torch.cuda.default_stream(device).id() == default_stream.id()

            # Check if the device index, current stream and default streams have not changed
            are_streams_same = is_device_index_same and is_current_stream_same and is_default_stream_same
            return are_streams_same
        self.assertTrue(test_set_none_stream())

        # This test checks if the Device Context manager is a no op
        # when the device is none for `with torch.cuda.device`
        @torch.jit.script
        def test_set_device_none():
            device_index = torch.cuda.current_device()
            # When device is none, check if this operation is a no-op
            with torch.cuda.device(None):
                # Check if the current device is the same
                is_device_same = torch.cuda.current_device() == device_index
            return is_device_same
        self.assertTrue(test_set_device_none())

        # Check if a CUDA JIT stream is created
        # on the current_device
        @torch.jit.script
        def test_simple_stream():
            device_index = torch.cuda.current_device()
            s = torch.cuda.Stream()
            return device_index == s.device_index()

        self.assertTrue(test_simple_stream(), "Could not create Stream!")

        # Class used to store results for the test: test_get_stream.
        class Result(NamedTuple):
            t1 : torch.Tensor
            t2 : torch.Tensor
            is_current_and_default_stream_same : bool
            is_default_and_user_stream_not_same : bool
            is_stream_set : bool
            is_stream_reset : bool
            default_stream_query : bool
            default_stream_id : int
            user_stream_id : int

        # The test aims at checking different stream proporties.
        @torch.jit.script
        def test_get_stream():
            device_index = torch.cuda.current_device()
            device = torch.device("cuda:" + str(device_index))
            current_stream = torch.cuda.current_stream(device)
            default_stream = torch.cuda.default_stream(device)
            user_stream = torch.cuda.Stream()

            # Check if the current and default streams are the same on the device
            is_current_and_default_stream_same = current_stream.id() == default_stream.id()
            # Check if user stream and default stream are not the same on the device
            is_default_and_user_stream_not_same = default_stream.id() != user_stream.id()

            with torch.cuda.stream(user_stream):
                is_stream_set = torch.cuda.current_stream(device).id() == user_stream.id()

            # Check if the stream was reset to current_stream
            is_stream_reset = torch.cuda.current_stream(device).id() == current_stream.id()

            tensor1 = torch.rand(10000, 10000, device="cuda")
            tensor2 = torch.mm(tensor1, tensor1).to("cuda")
            default_stream.synchronize()
            default_stream_query = default_stream.query()

            # Capture all the results in the class Result
            res = Result(
                tensor1, tensor2, is_current_and_default_stream_same,
                is_default_and_user_stream_not_same, is_stream_set,
                is_stream_reset, default_stream_query, default_stream.id(), user_stream.id())
            return res

        result = test_get_stream()

        self.assertEqual(torch.matmul(result.t1, result.t1), result.t2)
        self.assertTrue(result.is_current_and_default_stream_same)
        self.assertTrue(result.is_default_and_user_stream_not_same)
        self.assertTrue(result.is_stream_set)
        self.assertTrue(result.is_stream_reset)
        self.assertTrue(result.default_stream_query)
        self.assertEqual(result.default_stream_id, 0)  # Check if the default stream ID is always 0
        self.assertNotEqual(result.user_stream_id, 0)  # Check if the user stream is always non zero

        # Test the stream context manager. This test checks if the stream is switched
        # to the user stream on using the stream context manager.
        @torch.jit.script
        def test_stream_context():
            device_index = torch.cuda.current_device()
            device = torch.device("cuda:" + str(device_index))
            current_stream = torch.cuda.current_stream(device)
            user_stream = torch.cuda.Stream()
            A = torch.rand(1000, 1000, device="cuda")

            with torch.cuda.stream(user_stream):
                check = torch.cuda.current_stream(device).id() == user_stream.id()
                B = torch.mm(A, A).to("cuda")
            # Wait for B to be computed
            user_stream.synchronize()
            # Check if the stream has been reset on the current device
            is_stream_reset = torch.cuda.current_stream(device).id() == current_stream.id()

            return A, B, check, is_stream_reset

        A, B, is_stream_set, is_stream_reset = test_stream_context()
        self.assertEqual(torch.matmul(A, A), B)
        self.assertTrue(is_stream_set, "Error: Current stream was not set to user stream!")
        self.assertTrue(is_stream_reset, "Error: The stream was not restored to previous stream!")

        # Test multiple nested streams. Check if the operations are computed as expected on the streams
        # This test has been adapted from the eager mode tests available at test/test_cuda.py
        @torch.jit.script
        def test_multiple_stream():
            prev_device_index = torch.cuda.current_device()
            device = torch.device("cuda:" + str(prev_device_index))
            prev_current_stream = torch.cuda.current_stream(device)
            d1 = torch.device("cuda:0")
            d2 = torch.device("cuda:1")
            s1 = torch.cuda.Stream(d1, 0)
            s2 = torch.cuda.Stream(d2, 0)

            A = torch.rand(1000, 1000, device="cuda")
            B = torch.rand(1000, 1000, device="cuda")
            with torch.cuda.stream(s1):
                C = torch.mm(A, A).to("cuda")
                # Check if the stream and device have been set to s1
                is_stream_s1 = torch.cuda.current_stream(d1).id() == s1.id()
                is_device_s1 = torch.cuda.current_device() == s1.device_index()
                with torch.cuda.stream(s2):
                    # Check if the stream and device have been set to s2
                    is_stream_s2 = torch.cuda.current_stream(d2).id() == s2.id()
                    is_device_s2 = torch.cuda.current_device() == s2.device_index()
                    D = torch.mm(B, B).to("cuda")
                # Check if the stream and device have been set to s1
                is_stream_s1_after = torch.cuda.current_stream(d1).id() == s1.id()
                is_device_s1_after = torch.cuda.current_device() == s1.device_index()
                # Wait for D to be computed
                s2.synchronize()
            # Wait for C to be computed on S1
            s1.synchronize()

            # Check if the stream and device has been restored to previous stream and device
            is_device_current = torch.cuda.current_device() == prev_device_index
            is_stream_current = torch.cuda.current_stream(device).id() == prev_current_stream.id()

            check_stream = is_stream_s1 and is_stream_s2 and is_stream_s1_after and is_stream_current
            check_device = is_device_s1 and is_device_s2 and is_device_s1_after and is_device_current
            return A, B, C, D, check_stream, check_device
        A, B, C, D, check_stream, check_device = test_multiple_stream()

        self.assertEqual(torch.matmul(A, A), C)
        self.assertEqual(torch.matmul(B, B), D)
        self.assertTrue(check_stream)
        self.assertTrue(check_device)

        # Test multiple streams waiting on each other for the operations to be completed.
        @torch.jit.script
        def test_data_dependency_between_streams():
            device_index = torch.cuda.current_device()
            device = torch.device("cuda:" + str(device_index))
            prev_current_stream = torch.cuda.current_stream(device)
            d = torch.device("cuda:0")
            s1 = torch.cuda.Stream(d, 0)
            s2 = torch.cuda.Stream(d, 0)
            event = torch.cuda.Event(False, False, False)

            A = torch.rand(1000, 1000, device="cuda")
            with torch.cuda.stream(s1):
                is_stream_s1 = torch.cuda.current_stream(device).id() == s1.id()
                B = torch.mm(A, A).to("cuda")
            s1.record_event(event)
            # Check if the current_stream is reset
            is_current_stream_1 = torch.cuda.current_stream(device).id() == prev_current_stream.id()
            # Wait for ops on s1 to be computed
            s2.wait_event(event)
            with torch.cuda.stream(s2):
                is_stream_s2 = torch.cuda.current_stream(device).id() == s2.id()
                C = torch.mm(B, B).to("cuda")
            # Wait for C to be computed
            s2.synchronize()
            # Check if the current_stream is reset
            is_current_stream_2 = torch.cuda.current_stream(device).id() == prev_current_stream.id()

            check_stream = is_current_stream_1 and is_current_stream_2 and is_stream_s1 and is_stream_s2
            return A, B, C, check_stream

        A, B, C, check_stream = test_data_dependency_between_streams()
        self.assertEqual(torch.matmul(A, A), B)
        self.assertEqual(torch.matmul(B, B), C)
        self.assertTrue(check_stream)

        # Test a simple CUDA event. Test if the CUDA event was created successfully
        @torch.jit.script
        def test_simple_event():
            e = torch.cuda.Event(True, False, False)
            return e is not None
        self.assertTrue(test_simple_event(), "Could not create CUDA Event!")

        # Record the CUDA event for operation torch.mm on the current stream
        # and then test if the elapsed time is greater than 0. This test is also
        # an adaption from eager mdoe CUDA tests available at test/test_cuda.py
        @torch.jit.script
        def test_event():
            device_index = torch.cuda.current_device()
            device = torch.device("cuda:" + str(device_index))
            stream = torch.cuda.current_stream(device)
            event = torch.cuda.Event(True, False, False)
            is_true_event_query = event.query()
            start_event = torch.cuda.Event(True, False, False)
            stream.record_event(start_event)
            tensor1 = torch.rand(1000000000, 1000000000, device="cuda")
            tensor2 = torch.mm(tensor1, tensor1).to("cuda")
            stream.record_event(event)
            event.synchronize()
            is_again_true_event_query = event.query()

            if not (is_true_event_query and is_again_true_event_query):
                return -1.0
            return start_event.elapsed_time(event)

        self.assertGreater(test_event(), 0)

        # Check for stream synchronization , when a large tensor multiplication is
        # computed on the stream. The stream.query should be true once the synchroniztion is done
        @torch.jit.script
        def test_stream_synchronize() -> float:
            device_index = torch.cuda.current_device()
            s = torch.cuda.Stream()
            e_tik = torch.cuda.Event(True, False, False)
            e_tok = torch.cuda.Event(True, False, False)

            e_tik.record(s)
            tensor1 = torch.rand(1000000000, 1000000000, device="cuda")
            with torch.cuda.stream(s):
                tensor2 = torch.mm(tensor1, tensor1).to("cuda")
            s.synchronize()
            e_tok.record(s)
            e_tok.synchronize()

            if not s.query():
                return -1.0

            # not necessary to check e_tik and e_tok, as elapsed_time would throw
            # exception if otherwise.
            return e_tik.elapsed_time(e_tok)
        self.assertGreater(test_stream_synchronize(), 0)

        # Test event synchronization for the event that records a stream doing
        # a large tensor multiplication. Check if the elapsed time is greater than 0
        # and the stream.query evaluates to true.
        @torch.jit.script
        def test_event_synchronize() -> float:
            s = torch.cuda.Stream()
            e_tik = torch.cuda.Event(True, False, False)
            e_tok = torch.cuda.Event(True, False, False)

            e_tik.record(s)
            tensor1 = torch.rand(1000000000, 1000000000, device="cuda")
            with torch.cuda.stream(s):
                tensor = torch.mm(tensor1, tensor1).to("cuda")
            s.record_event(e_tok)
            e_tok.synchronize()
            s.synchronize()

            if not s.query():
                return -1.0

            # not necessary to check e_tik and e_tok, as elapsed_time would throw
            # exception if otherwise.
            return e_tik.elapsed_time(e_tok)

        self.assertGreater(test_event_synchronize(), 0)

        # Test for event wait. Check if event waits for the all the operations on
        # the stream to be done. Check for synchronizations and query on the streams
        # and events. This test is adapted from eager mode tests for CUDA. Please refer
        # test/test_cuda.py
        @torch.jit.script
        def test_event_wait() -> float:
            device_index = torch.cuda.current_device()
            device = torch.device("cuda:" + str(device_index))
            s0 = torch.cuda.current_stream(device)
            s1 = torch.cuda.Stream()
            e_tik = torch.cuda.Event(True, True, False)
            e_tok = torch.cuda.Event(True, True, False)

            e_tik.record(s0)
            tensor1 = torch.rand(1000000000, 1000000000, device="cuda")
            with torch.cuda.stream(s0):
                tensor2 = torch.mm(tensor1, tensor1).cuda()
            e_sync = torch.cuda.Event(True, False, False)
            e_sync.record(torch.cuda.current_stream(device))
            e_sync.wait(s1)
            with torch.cuda.stream(s1):
                tensor3 = torch.rand(1000000000, 1000000000, device="cuda")
                tensor4 = torch.mm(tensor3, tensor3).cuda()
            s1.synchronize()
            e_tok.record(torch.cuda.current_stream(device))
            e_tok.synchronize()
            s0.synchronize()

            if not s0.query() or not s1.query() or not e_sync.query():
                return -1.0

            # not necessary to check e_tik and e_tok, as elapsed_time would throw
            # exception if otherwise.
            return e_tik.elapsed_time(e_tok)
        self.assertGreater(test_event_wait(), 0)

        # Test for stream wait_event. Checks if the stream waits on the event
        @torch.jit.script
        def test_wait_event():
            d1 = torch.device('cuda:1')

            with torch.cuda.device(d1):
                s0 = torch.cuda.current_stream(d1)
                tensor1 = torch.rand(1000000000, 1000000000, device="cuda")
                tensor2 = torch.mm(tensor1, tensor1).to("cuda")
                e0 = torch.cuda.Event(False, False, False)
                s0.record_event(e0)

            s1 = torch.cuda.current_stream(torch.device('cuda:0'))
            s1.wait_event(e0)
            s1.synchronize()

            return e0.query() and s0.query() and s1.query()
        self.assertTrue(test_wait_event())

        # Test if a scripted module with cuda streams can be saved, loaded and executed
        def test_save_load(self):
            class Model(torch.nn.Module):
                def forward(self):
                    s = torch.cuda.Stream()
                    a = torch.rand(3, 4, device="cuda")
                    b = torch.rand(3, 4, device="cuda")

                    with torch.cuda.stream(s):
                        is_stream_s = torch.cuda.current_stream(s.device).id() == s.id()
                        c = torch.cat((a, b), 0).cuda()
                    s.synchronize()
                    return is_stream_s, a, b, c

            model = Model()

            # Script the model and save
            script_model = torch.jit.script(model)
            is_stream_s, a, b, c = script_model()
            # Verify if the output is correct
            self.assertTrue(is_stream_s)
            self.assertEqual(torch.cat((a, b), 0), c)

            # Save and load scripted model
            load_model = self.getExportImportCopy(script_model)
            is_stream_s, a_load, b_load, c_load = load_model()
            self.assertTrue(is_stream_s)
            self.assertEqual(torch.cat((a_load, b_load), 0), c_load)

    # Make sure that cuda._exchange_device doesn't get DCE'ed
    @unittest.skipIf(not TEST_CUDA, "Cuda not available")
    def test__exchange_device_op(self):
        def fn(device: int, tensor):
            torch.cuda._exchange_device(device)
            return tensor.cos().relu()

        fn_s = torch.jit.script(fn)
        # Just check the graph, don't run it. Otherwise, we'd  need to
        # run this test on a multi-gpu CI runner, which is overkill.
        g = fn_s.graph
        FileCheck().check("cuda::_exchange_device(").run(g)
        torch._C._jit_pass_inline(g)
        FileCheck().check("cuda::_exchange_device(").run(g)

    # Make sure that cuda._maybe_exchange_device doesn't get DCE'ed
    @unittest.skipIf(not TEST_CUDA, "Cuda not available")
    def test__maybe_exchange_device_op(self):
        def fn(device: int, tensor):
            torch.cuda._maybe_exchange_device(device)
            return tensor.cos().relu()

        fn_s = torch.jit.script(fn)
        # Just check the graph, don't run it. Otherwise, we'd  need to
        # run this test on a multi-gpu CI runner, which is overkill.
        g = fn_s.graph
        FileCheck().check("cuda::_maybe_exchange_device(").run(g)
        torch._C._jit_pass_inline(g)
        FileCheck().check("cuda::_maybe_exchange_device(").run(g)
