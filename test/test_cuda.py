# Owner(s): ["module: cuda"]
import contextlib
import ctypes
import gc
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from copy import deepcopy
from itertools import product
from random import randint

import psutil

import torch
import torch.cuda
from torch import inf, nan
from torch.cuda._memory_viz import (
    _profile_to_snapshot,
    profile_plot,
    segment_plot,
    trace_plot,
)
from torch.testing._internal.autocast_test_lists import AutocastTestLists, TestAutocast
from torch.testing._internal.common_cuda import (
    _create_scaling_case,
    TEST_CUDNN,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCUDA,
    onlyNativeDeviceTypes,
)
from torch.testing._internal.common_optimizers import (
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    optims,
    TensorTracker,
)
from torch.testing._internal.common_utils import (
    EXPANDABLE_SEGMENTS,
    freeze_rng_state,
    gcIfJetson,
    get_cycles_per_ms,
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_FBCODE,
    IS_JETSON,
    IS_LINUX,
    IS_SANDCASTLE,
    IS_WINDOWS,
    load_tests,
    NO_MULTIPROCESSING_SPAWN,
    parametrize,
    run_tests,
    serialTest,
    setBlasBackendsToDefaultFinally,
    skipCUDAMemoryLeakCheckIf,
    skipCUDANonDefaultStreamIf,
    skipIfRocm,
    slowTest,
    subtest,
    TemporaryFileName,
    TEST_CUDA,
    TEST_CUDA_GRAPH,
    TEST_NUMPY,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.viz._cycles import observe_tensor_cycles


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

try:
    import torchvision.models  # noqa: F401
    from torchvision.models import resnet18  # noqa: F401

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

TEST_CUDAMALLOCASYNC = TEST_CUDA and (
    torch.cuda.get_allocator_backend() == "cudaMallocAsync"
)
TEST_LARGE_TENSOR = TEST_CUDA
TEST_MEDIUM_TENSOR = TEST_CUDA
TEST_BF16 = False
TEST_PYNVML = not torch.cuda._HAS_PYNVML
if TEST_CUDA:
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9
    TEST_MEDIUM_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 6e9
    TEST_BF16 = torch.cuda.is_bf16_supported()

_cycles_per_ms = None


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCuda(TestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    FIFTY_MIL_CYCLES = 50000000

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    @property
    def expandable_segments(self):
        return EXPANDABLE_SEGMENTS

    def test_pinned_memory_with_cudaregister(self):
        try:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
            )
            t = torch.ones(20)
            self.assertFalse(t.is_pinned())
            try:
                pinned_t = torch.ones(1 << 21).pin_memory()
                self.assertTrue(pinned_t.is_pinned())
                pinned_t = torch.ones(1 << 24).pin_memory()
                self.assertTrue(pinned_t.is_pinned())
            except RuntimeError as e:
                # Some GPUs don't support same address space on host and device side
                pass
        finally:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:False"
            )

    def test_pinned_memory_with_cudaregister_multithread(self):
        num_threads = 4
        threads = [
            threading.Thread(target=self.test_pinned_memory_with_cudaregister)
            for t in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def test_pinned_memory_empty_cache(self):
        try:
            for alloc_settings in (True, False):
                torch.cuda.memory._set_allocator_settings(
                    f"pinned_use_cuda_host_register:{alloc_settings}"
                )
                try:
                    t = torch.ones(1024 * 1024, pin_memory=True)
                    self.assertTrue(t.is_pinned())
                    del t
                    torch._C._host_emptyCache()
                except RuntimeError as e:
                    # Some GPUs don't support same address space on host and device side
                    pass
        finally:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:False"
            )

    def test_cudart_register(self):
        t = torch.ones(20)
        self.assertFalse(t.is_pinned())
        cudart = torch.cuda.cudart()
        r = cudart.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        self.assertEqual(r, 0)
        self.assertTrue(t.is_pinned())
        r = cudart.cudaHostUnregister(t.data_ptr())
        self.assertEqual(r, 0)
        self.assertFalse(t.is_pinned())

    def test_memory_allocation(self):
        gc.collect()
        torch.cuda.empty_cache()
        mem = None
        size = 1
        prev = 0
        try:
            prev = torch.cuda.memory_allocated()
            mem = torch.cuda.caching_allocator_alloc(size)
            self.assertGreater(torch.cuda.memory_allocated(), prev)
        finally:
            if mem is not None:
                torch.cuda.caching_allocator_delete(mem)
                self.assertEqual(torch.cuda.memory_allocated(), prev)

    def test_check_error(self):
        # Assert this call doesn't raise.
        torch.cuda.check_error(0)

        with self.assertRaisesRegex(
            torch.cuda.CudaError, "out of memory|hipErrorOutOfMemory"
        ):
            torch.cuda.check_error(2)

    def test_cuda_get_device_name(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device)
        device_name_None = torch.cuda.get_device_name(None)
        self.assertEqual(current_device_name, device_name_None)

        # Testing the behaviour for No argument
        device_name_no_argument = torch.cuda.get_device_name()
        self.assertEqual(current_device_name, device_name_no_argument)

    def test_cuda_get_device_capability(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_capability = torch.cuda.get_device_capability(current_device)
        device_capability_None = torch.cuda.get_device_capability(None)
        self.assertEqual(current_device_capability, device_capability_None)

        # Testing the behaviour for No argument
        device_capability_no_argument = torch.cuda.get_device_capability()
        self.assertEqual(current_device_capability, device_capability_no_argument)

    def test_cuda_get_device_properties(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_properties = torch.cuda.get_device_properties(current_device)
        device_properties_None = torch.cuda.get_device_properties(None)
        self.assertEqual(current_device_properties, device_properties_None)

        # Testing the behaviour for No argument
        device_properties_no_argument = torch.cuda.get_device_properties()
        self.assertEqual(current_device_properties, device_properties_no_argument)

    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    def test_out_of_memory(self):
        tensor = torch.zeros(1024, device="cuda")

        oom_regex = (
            "would exceed allowed memory"
            if TEST_CUDAMALLOCASYNC
            else f"Tried to allocate 800000000.00 GiB. GPU {tensor.device.index} has a total capacity of"
        )
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            torch.empty(1024 * 1024 * 1024 * 800000000, dtype=torch.int8, device="cuda")

        with self.assertRaisesRegex(
            RuntimeError, "Tried to allocate more than 1EB memory"
        ):
            torch.empty(
                1024 * 1024 * 1024 * 8000000000, dtype=torch.int8, device="cuda"
            )

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC or IS_JETSON, "Segmentation fault (core dumped)"
    )
    @serialTest()
    def test_out_of_memory_retry(self):
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        oom_regex = (
            "would exceed allowed memory"
            if TEST_CUDAMALLOCASYNC
            else "Tried to allocate"
        )
        size = int(total_memory * 0.5)
        a = torch.empty(size, dtype=torch.int8, device="cuda")
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            b = torch.empty(size, dtype=torch.int8, device="cuda")
        del a
        b = torch.empty(size, dtype=torch.int8, device="cuda")
        del b
        # We used a lot of memory here, clean up so we don't affect other tests too much
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @serialTest()
    def test_set_per_process_memory_fraction(self):
        try:
            # test invalid fraction value.
            with self.assertRaisesRegex(TypeError, "Invalid type"):
                torch.cuda.set_per_process_memory_fraction(1)
            with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
                torch.cuda.set_per_process_memory_fraction(-0.1)
            with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
                torch.cuda.set_per_process_memory_fraction(2.0)

            tensor = torch.zeros(1024, device="cuda")
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(0.5, 0)

            # test 0.499 allocation is ok.
            application = int(total_memory * 0.499) - torch.cuda.max_memory_reserved()
            tmp_tensor = torch.empty(application, dtype=torch.int8, device="cuda")
            del tmp_tensor
            torch.cuda.empty_cache()

            application = int(total_memory * 0.5)
            # it will get OOM when try to allocate more than half memory.
            oom_regex = (
                "would exceed allowed memory"
                if TEST_CUDAMALLOCASYNC
                else "out of memory"
            )
            with self.assertRaisesRegex(RuntimeError, oom_regex):
                torch.empty(application, dtype=torch.int8, device="cuda")

            # ensure out of memory error doesn't disturb subsequent kernel
            tensor.fill_(1)
            self.assertTrue((tensor == 1).all())
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0, 0)

    @serialTest()
    def test_get_per_process_memory_fraction(self):
        # get the initial memory fraction
        init_fraction = torch.cuda.get_per_process_memory_fraction()

        # set and get the limiting cases
        torch.cuda.set_per_process_memory_fraction(1.0)
        self.assertEqual(torch.cuda.get_per_process_memory_fraction(), 1.0)
        torch.cuda.set_per_process_memory_fraction(0.0)
        self.assertEqual(torch.cuda.get_per_process_memory_fraction(), 0.0)

        # test a few random cases
        for val in torch.rand(3):
            torch.cuda.set_per_process_memory_fraction(float(val))
            self.assertEqual(torch.cuda.get_per_process_memory_fraction(), float(val))

        # restore the initial memory fraction
        torch.cuda.set_per_process_memory_fraction(init_fraction)

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "uuid attribute not yet available")
    def test_uuid(self):
        uuid = torch.cuda.get_device_properties(0).uuid
        self.assertEqual(len(str(uuid)), 36)  # xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        self.assertEqual(len(uuid.bytes), 16)

    def test_copy_non_blocking(self):
        def _test_copy_non_blocking(a, b):
            event = torch.cuda.Event()
            a.copy_(b, non_blocking=True)
            event.record()
            event.synchronize()
            self.assertEqual(a, b)

        # 10MB copies
        x = torch.ones(10000000, dtype=torch.uint8).cuda()
        y = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        _test_copy_non_blocking(x, y)

        x = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        y = torch.ones(10000000, dtype=torch.uint8).cuda()
        _test_copy_non_blocking(x, y)

        # Test the case where the pinned data_ptr is not equal to the storage data_ptr.
        x_base = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        x = x_base[1:]
        self.assertTrue(x.is_pinned())
        self.assertTrue(x_base.is_pinned())
        self.assertNotEqual(x_base.data_ptr(), x.data_ptr())
        self.assertEqual(x_base.storage().data_ptr(), x.storage().data_ptr())
        y = torch.ones(10000000 - 1, dtype=torch.uint8).cuda()
        _test_copy_non_blocking(x, y)

    def test_copy_non_blocking_type_conversion(self):
        a = torch.ones(1, device="cuda")
        b = torch.zeros(1, device="cpu", pin_memory=True)
        c = torch.empty(1, device="cuda", dtype=torch.long)
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        b.copy_(a, non_blocking=True)
        c.copy_(b, non_blocking=True)
        self.assertEqual(a, c, exact_dtype=False)

    @serialTest()
    def test_to_non_blocking(self):
        stream = torch.cuda.current_stream()

        def _test_to_non_blocking(a, non_blocking, dst):
            torch.cuda.synchronize()
            # Pushes an 0.1 second spin to stream so if the copy is non blocking,
            # stream will almost surely be active when we query().
            torch.cuda._sleep(int(100 * get_cycles_per_ms()))
            b = a.to(device=dst, non_blocking=non_blocking)
            self.assertEqual(stream.query(), not non_blocking)
            stream.synchronize()
            self.assertEqual(a, b)
            self.assertTrue(b.is_pinned() == (non_blocking and dst == "cpu"))

        for dst, try_non_blocking in product(("cuda", "cpu"), (True, False)):
            # Creates source on the opposite device from destination.
            src = torch.randn(
                1000000,
                device="cuda" if dst == "cpu" else "cpu",
                pin_memory=True if dst == "cuda" else False,
            )
            _test_to_non_blocking(src, try_non_blocking, dst)

    def test_to_cpu_blocking_by_default(self):
        src = torch.randn(1000000, device="cuda")
        torch.cuda.synchronize()
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        dst = src.to(device="cpu")
        self.assertEqual(torch.cuda.current_stream().query(), True)
        self.assertEqual(src, dst)
        self.assertFalse(dst.is_pinned())

    def test_serialization_array_with_storage(self):
        x = torch.randn(5, 5).cuda()
        y = torch.IntTensor(2, 5).fill_(0).cuda()
        q = [x, y, x, y.storage()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(q, f)
            f.seek(0)
            q_copy = torch.load(f)
        self.assertEqual(q_copy, q, atol=0, rtol=0)
        q_copy[0].fill_(5)
        self.assertEqual(q_copy[0], q_copy[2], atol=0, rtol=0)
        self.assertTrue(isinstance(q_copy[0], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[1], torch.cuda.IntTensor))
        self.assertTrue(isinstance(q_copy[2], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[3], torch.storage.TypedStorage))
        self.assertTrue(isinstance(q_copy[3]._untyped_storage, torch.UntypedStorage))
        q_copy[1].fill_(10)
        self.assertEqual(q_copy[3], torch.cuda.IntStorage(10).fill_(10))

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled for async")
    @setBlasBackendsToDefaultFinally
    def test_cublas_workspace_explicit_allocation(self):
        torch.backends.cuda.preferred_blas_library("cublas")
        a = torch.randn(7, 7, device="cuda", requires_grad=False)
        if torch.version.hip:
            default_workspace_size = 1024 * 32 * 1024  # :1024:32  32MiB
            # different size (128 MiB) expected on MI300 GPU
            if torch.cuda.get_device_capability() >= (9, 4):
                default_workspace_size = 1024 * 128 * 1024  # :1024:128
        else:
            default_workspace_size = (
                4096 * 2 * 1024 + 16 * 8 * 1024
            )  # :4096:2:16:8  8MiB
            # different size (32 MiB) expected on Hopper GPU
            if torch.cuda.get_device_capability() == (9, 0):
                default_workspace_size = 4096 * 8 * 1024

        def check_workspace_size(inp):
            torch._C._cuda_clearCublasWorkspaces()
            start = torch.cuda.memory_stats()["active_bytes.all.allocated"]
            with torch.no_grad():
                torch.matmul(inp, inp)
            finish = torch.cuda.memory_stats()["active_bytes.all.allocated"]
            return finish - start

        # check default
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ""
        self.assertTrue(abs(check_workspace_size(a) - default_workspace_size) < 524288)

        # check default with bad user config
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = "-1"
        self.assertTrue(abs(check_workspace_size(a) - default_workspace_size) < 524288)

        # check valid config
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":128:8:64:16:32:32"
        self.assertTrue(abs(check_workspace_size(a) - (3072 * 1024)) < 524288)

        torch._C._cuda_clearCublasWorkspaces()

    def test_cublas_allow_tf32_get_set(self):
        skip_tf32_cublas = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
        )
        if skip_tf32_cublas:
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            return

        orig = torch.backends.cuda.matmul.allow_tf32
        self.assertEqual(torch._C._get_cublas_allow_tf32(), orig)
        torch.backends.cuda.matmul.allow_tf32 = not orig
        self.assertEqual(torch._C._get_cublas_allow_tf32(), not orig)
        torch.backends.cuda.matmul.allow_tf32 = orig

    def test_float32_matmul_precision_get_set(self):
        orig = torch.get_float32_matmul_precision()
        skip_tf32_cublas = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
        )
        # this is really just checking that the environment variable is respected during testing
        # and not overwritten by another function that doesn't revert it to the intitial value
        if not skip_tf32_cublas:
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        else:
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        for p in ("medium", "high"):
            torch.set_float32_matmul_precision(p)
            self.assertEqual(torch.get_float32_matmul_precision(), p)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        torch.set_float32_matmul_precision("highest")
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        torch.set_float32_matmul_precision(orig)

    def test_cublas_allow_fp16_reduced_precision_reduction_get_set(self):
        orig = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(), orig
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not orig
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(), not orig
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig

    def test_cublas_allow_bf16_reduced_precision_reduction_get_set(self):
        orig = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        self.assertEqual(
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(), orig
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not orig
        self.assertEqual(
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(), not orig
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = orig

    def test_cudnn_allow_tf32_get_set(self):
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=False
        ):
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=True
        ):
            self.assertTrue(torch.backends.cudnn.allow_tf32)

    def test_type_conversions(self):
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.cuda().double(), torch.cuda.DoubleTensor)
        self.assertIsInstance(x.cuda().float(), torch.cuda.FloatTensor)
        self.assertIsInstance(x.cuda().float().cpu(), torch.FloatTensor)
        self.assertIsInstance(x.cuda().float().cpu().int(), torch.IntTensor)

        y = x.storage()
        self.assertIsInstance(y.float(), torch.FloatStorage)
        self.assertIsInstance(y.cuda().double(), torch.cuda.DoubleStorage)
        self.assertIsInstance(y.cuda().float(), torch.cuda.FloatStorage)
        self.assertIsInstance(y.cuda().float().cpu(), torch.FloatStorage)
        self.assertIsInstance(y.cuda().float().cpu().int(), torch.IntStorage)

    @unittest.skip("was disabled due to not enough memory, but actually it always fail")
    def test_arithmetic_large_tensor(self):
        x = torch.empty(2**30, device="cuda")

        x.fill_(1)
        self.assertEqual(x.sum(), 2**30)

        x += 1
        self.assertEqual(x.sum(), 2**31)

        x.fill_(1)
        x -= 0.5
        self.assertEqual(x.sum(), 2**29)

        x.fill_(1)
        x *= 2
        self.assertEqual(x.sum(), 2**31)

        x.fill_(1)
        x /= 2
        self.assertEqual(x.sum(), 2**29)

    def test_gather_bool(self):
        t = torch.tensor([[False, True], [True, True]], device="cuda")
        self.assertEqual(
            torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]], device="cuda")),
            torch.tensor([[False, False], [True, True]], device="cuda"),
        )

    def test_torch_manual_seed_seeds_cuda_devices(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            torch.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            self.assertEqual(torch.cuda.initial_seed(), 2)

    def test_manual_seed(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.cuda.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.cuda.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertEqual(x, y)
            self.assertEqual(a, b)
            self.assertEqual(torch.cuda.initial_seed(), 2)

    def test_specify_improper_device_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "tempfile.pt")
            with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
                torch.save(
                    [torch.nn.Parameter(torch.randn(10, 10))],
                    fname,
                    _use_new_zipfile_serialization=True,
                )
                torch.load(fname, "cuda0")

    def test_get_device_index(self):
        from torch.cuda._utils import _get_device_index

        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            _get_device_index("cuda0", optional=True)

        with self.assertRaisesRegex(ValueError, "Expected a cuda device"):
            cpu_device = torch.device("cpu")
            _get_device_index(cpu_device, optional=True)

    def test_serialization_array_with_empty(self):
        x = [torch.randn(4, 4).cuda(), torch.cuda.FloatTensor()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), original.get_device())

    @skipCUDANonDefaultStreamIf(True)
    def test_streams(self):
        default_stream = torch.cuda.current_stream()
        user_stream = torch.cuda.Stream()
        self.assertEqual(torch.cuda.current_stream(), default_stream)
        self.assertNotEqual(default_stream, user_stream)
        self.assertEqual(default_stream.cuda_stream, 0)
        self.assertNotEqual(user_stream.cuda_stream, 0)
        with torch.cuda.stream(user_stream):
            self.assertEqual(torch.cuda.current_stream(), user_stream)
        self.assertTrue(user_stream.query())
        tensor1 = torch.ByteTensor(5).pin_memory()
        tensor2 = tensor1.cuda(non_blocking=True) + 1
        default_stream.synchronize()
        self.assertTrue(default_stream.query())

    def test_stream_event_repr(self):
        s = torch.cuda.current_stream()
        self.assertTrue("torch.cuda.Stream" in s.__repr__())
        e = torch.cuda.Event()
        self.assertTrue("torch.cuda.Event" in e.__repr__())
        s.record_event(e)
        self.assertTrue("torch.cuda.Event" in e.__repr__())

    def test_events(self):
        stream = torch.cuda.current_stream()
        event = torch.cuda.Event(enable_timing=True)
        self.assertTrue(event.query())
        start_event = torch.cuda.Event(enable_timing=True)
        stream.record_event(start_event)
        torch.cuda._sleep(int(50 * get_cycles_per_ms()))
        stream.record_event(event)
        self.assertFalse(event.query())
        event.synchronize()
        self.assertTrue(event.query())
        self.assertGreater(start_event.elapsed_time(event), 0)

    def test_generic_stream_event(self):
        stream = torch.Stream("cuda")
        self.assertEqual(stream.device_index, torch.cuda.current_device())
        cuda_stream = torch.cuda.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )
        self.assertIsInstance(cuda_stream, torch.Stream)
        self.assertTrue(issubclass(type(cuda_stream), torch.Stream))
        self.assertTrue(torch.Stream in type(cuda_stream).mro())
        self.assertEqual(stream.stream_id, cuda_stream.stream_id)
        self.assertNotEqual(stream.stream_id, torch.cuda.current_stream().stream_id)

        event1 = torch.Event("cuda", enable_timing=True)
        event2 = torch.Event("cuda", enable_timing=True)
        self.assertEqual(event1.event_id, 0)
        a = torch.randn(1000)
        b = torch.randn(1000)
        with torch.cuda.stream(cuda_stream):
            a_cuda = a.to("cuda", non_blocking=True)
            b_cuda = b.to("cuda", non_blocking=True)
            self.assertEqual(stream.stream_id, torch.cuda.current_stream().stream_id)
        event1.record(stream)
        event1.synchronize()
        self.assertTrue(event1.query())
        c_cuda = a_cuda + b_cuda
        event2.record()
        event2.synchronize()
        self.assertTrue(event2.query())
        self.assertNotEqual(event1.event_id, event2.event_id)
        self.assertEqual(c_cuda.cpu(), a + b)
        self.assertTrue(event1.elapsed_time(event2) > 0)
        cuda_event = torch.cuda.Event()
        self.assertIsInstance(cuda_event, torch.Event)
        self.assertTrue(issubclass(type(cuda_event), torch.Event))
        self.assertTrue(torch.Event in type(cuda_event).mro())

    def test_record_stream(self):
        cycles_per_ms = get_cycles_per_ms()

        t = torch.FloatTensor([1, 2, 3, 4]).pin_memory()
        result = torch.cuda.FloatTensor(t.size())
        stream = torch.cuda.Stream()
        ptr = [None]

        # Performs the CPU->GPU copy in a background stream
        def perform_copy():
            with torch.cuda.stream(stream):
                tmp = t.cuda(non_blocking=True)
                ptr[0] = tmp.data_ptr()
            torch.cuda.current_stream().wait_stream(stream)
            tmp.record_stream(torch.cuda.current_stream())
            torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
            result.copy_(tmp)

        perform_copy()
        with torch.cuda.stream(stream):
            tmp2 = torch.cuda.FloatTensor(t.size())
            tmp2.zero_()
            self.assertNotEqual(
                tmp2.data_ptr(), ptr[0], msg="allocation re-used to soon"
            )

        self.assertEqual(result.tolist(), [1, 2, 3, 4])

        if not TEST_CUDAMALLOCASYNC:
            # In the native allocator, we expect "tmp"'s side-stream-tagged block will be reused
            # in that side stream after result.copy_(tmp) in the main stream finishes.
            torch.cuda.current_stream().synchronize()
            with torch.cuda.stream(stream):
                tmp3 = torch.cuda.FloatTensor(t.size())
                self.assertEqual(tmp3.data_ptr(), ptr[0], msg="allocation not re-used")

    def test_record_stream_on_shifted_view(self):
        # See issue #27366

        # This test detects unexpected block reallocation. For reliable test,
        # the stream to allocate tensors is isolated. The allocator will not
        # reuse free blocks which were allocated from another stream.
        stream_alloc = torch.cuda.Stream()
        with torch.cuda.stream(stream_alloc):
            base = torch.cuda.FloatTensor([10, 10])

        # Record another stream on a shifted view tensor.
        view = base[5:]
        self.assertTrue(view.storage_offset() > 0)

        stream_record = torch.cuda.Stream()
        with torch.cuda.stream(stream_record):
            torch.cuda._sleep(int(50 * get_cycles_per_ms()))

        view.record_stream(stream_record)

        # Delete those tensors to make the block free soon.
        data_ptr = base.data_ptr()
        del base, view

        # A new tensor should not be allocated to the block above.
        stream_alloc.synchronize()

        with torch.cuda.stream(stream_alloc):
            try_realloc = torch.cuda.FloatTensor([10, 10])

        self.assertNotEqual(try_realloc.data_ptr(), data_ptr)

    def test_noncontiguous_pinned_memory(self):
        # See issue #3266
        x = torch.arange(0, 10).view((2, 5))
        self.assertEqual(x.t(), x.t().pin_memory())

    def test_caching_pinned_memory(self):
        cycles_per_ms = get_cycles_per_ms()

        # check that allocations are re-used after deletion
        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertEqual(t.data_ptr(), ptr, msg="allocation not reused")

        # check that the allocation is not re-used if it's in-use by a copy
        gpu_tensor = torch.cuda.FloatTensor([0])
        torch.cuda._sleep(int(1000 * cycles_per_ms))  # delay the copy by 1s
        gpu_tensor.copy_(t, non_blocking=True)
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, msg="allocation re-used too soon")
        self.assertEqual(list(gpu_tensor), [1])

    def test_caching_allocator_record_stream_oom(self):
        """allocations delayed by a record_stream call should still be freed on
        an out-of-memory in cuda_malloc_retry. see issue #19219"""
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            y = torch.zeros(40 * 1024 * 1024, device="cuda")

        for _ in range(100):
            x = torch.empty(40 * 1024 * 1024, device="cuda")
            with torch.cuda.stream(stream):
                y += x
            # delays re-use of `x` until after all operations in `stream`
            x.record_stream(stream)
            del x

        # we've made a mess by allocating up to the device capacity. free any
        # cached blocks in case it affects future tests.
        torch.cuda.empty_cache()

    # Tests for historic illegal memory access, see #17040.
    def test_reduction_gpu_memory_accessing(self):
        x = torch.ones(512, 8, dtype=torch.float32, device="cuda")
        torch.sum(x, 0)

    def test_sum_fp16(self):
        x = torch.zeros(10, device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum(), 0)

        x = torch.ones(65504, device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum(), 65504)
        self.assertEqual(x.sum(dtype=torch.float32), 65504)

        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum(dtype=torch.float32), 65536)

        a = torch.zeros(1203611).bernoulli_(0.0005)
        x = a.to(device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum().item(), a.sum().item())

        a = torch.zeros(100, 121, 80).bernoulli_(0.0005)
        x = a.to(device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum((0, 2)).float().cpu(), a.sum((0, 2)))

    def test_mean_fp16(self):
        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        self.assertEqual(x.mean(), 1)

        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        self.assertEqual(x.mean(dtype=torch.float32), 1)

    def test_prod_large(self):
        # tests global reduction (should_global_reduce = true) in case of non-zero identity element
        x = torch.ones(240000, device="cuda", dtype=torch.float32)
        self.assertEqual(x.prod(), 1)

        # test for complex types. Note 240k is divisible by 4
        for dtype in [torch.cfloat, torch.cdouble]:
            x = torch.ones(240000, device="cuda", dtype=dtype) * (0 + 1j)
            self.assertEqual(x.prod(), 1)

    def test_multinomial_ext(self):
        # Test two corner cases from older PyTorch (Issue #4858)
        freqs = torch.cuda.FloatTensor(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.03178183361887932,
                0.027680952101945877,
                0.033176131546497345,
                0.046052902936935425,
                0.07742464542388916,
                0.11543981730937958,
                0.14148041605949402,
                0.15784293413162231,
                0.13180233538150787,
                0.08271478116512299,
                0.049702685326337814,
                0.027557924389839172,
                0.018125897273421288,
                0.011851548217236996,
                0.010252203792333603,
                0.007422595750540495,
                0.005372154992073774,
                0.0045109698548913,
                0.0036087757907807827,
                0.0035267581697553396,
                0.0018864056328311563,
                0.0024605290964245796,
                0.0022964938543736935,
                0.0018453967059031129,
                0.0010662291897460818,
                0.0009842115687206388,
                0.00045109697384759784,
                0.0007791675161570311,
                0.00020504408166743815,
                0.00020504408166743815,
                0.00020504408166743815,
                0.00012302644609007984,
                0.0,
                0.00012302644609007984,
                4.100881778867915e-05,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        torch.cuda.manual_seed(11042)
        sample = torch.multinomial(freqs, 1000, True)
        self.assertNotEqual(freqs[sample].min(), 0)

        p = torch.zeros(3421, 2, device="cuda", dtype=torch.float)
        p[:, 1] = 1
        torch.cuda.manual_seed(5214)
        r = torch.multinomial(p, 1)
        self.assertNotEqual(r.min().item(), 0)

        # test corner case from Issue #13867
        torch.cuda.manual_seed(33)
        probs = torch.randn(1000000, device="cuda").clamp(min=0) * 3e-5
        samples = probs.multinomial(1000000, replacement=True)
        self.assertGreater(probs[samples].min().item(), 0)

    def _spawn_test_multinomial_invalid_probs_cuda(self, probs):
        import subprocess

        try:
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"""\
import sys
import torch
from torch import inf, nan
try:
    with torch.random.fork_rng(devices=[0]):
        torch.multinomial(torch.tensor({probs}).to('cuda'), 2, replacement=True)
        torch.cuda.synchronize()
    sys.exit(-1) # Should not be reached
except RuntimeError as e:
    sys.exit(-2)
""",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            out, err = p.communicate(timeout=10)
            p.wait(timeout=10)
        except subprocess.TimeoutExpired as e:
            p.kill()
            out, err = p.communicate()
        expected_messages = [
            "device-side assert triggered",  # CUDA
            "Assertion",  # CUDA
            "HSA_STATUS_ERROR_EXCEPTION",  # ROCm
            "Device-side assertion",  # ROCm
        ]
        self.assertTrue(any(msg in out or msg in err for msg in expected_messages))

    @slowTest
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support device side asserts")
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    def test_multinomial_invalid_probs_cuda(self):
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, -1.0, 1.0])
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, inf, 1.0])
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, -inf, 1.0])
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, 1.0, nan])

    @staticmethod
    def _mute_init():
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stderr.fileno())

    def _spawn_method(self, method, arg):
        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Pool(1, initializer=self._mute_init) as pool:
            errors = pool.map(method, [arg])
            for e in errors:
                if "device-side assert triggered" not in str(e):
                    self.fail(e)

    @staticmethod
    def _test_index_bounds_cuda(idx):
        x = torch.arange(10, device="cuda")
        try:
            y = x[torch.tensor([idx])]
            return f"x[torch.tensor([{idx})]={y}"
        except RuntimeError as err:
            return err

    @slowTest
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    @skipIfRocm
    def test_index_out_of_bounds_exception_cuda(self):
        test_method = TestCuda._test_index_bounds_cuda
        # Test in-bound access works fine
        self.assertEqual(
            test_method(1), "x[torch.tensor([1)]=tensor([1], device='cuda:0')"
        )
        # Test that indexing out of bounds causes assert
        self._spawn_method(test_method, 11)

    @slowTest
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @serialTest()
    def test_huge_index(self):
        src = torch.empty(15000000, 45, device="cuda", dtype=torch.long).random_(
            0, 2**22
        )
        idx = torch.randperm(src.shape[0], device="cuda")
        res = src[idx]
        res_cpu = src.cpu()[idx.cpu()]
        self.assertEqual(res.cpu(), res_cpu)

    def test_randint_randomness_for_large_range(self) -> None:
        # For large ranges, randint generation is slightly different. This lead to a subtle bug where some Philox
        # offsets were not calculated correctly, resulting in reused random states.
        # See https://github.com/pytorch/pytorch/issues/125224
        size = 1_000_000
        high = 6_000_000_000  # Keep this above 2**32

        def run(dev: torch.device) -> int:
            # Measure how many unique numbers are generated in 2 consecutive calls to randint. If random states are
            # reused, this will yield fewer unique numbers.
            gen = torch.Generator(device=dev)
            gen.manual_seed(0)
            t1 = torch.randint(
                0, high, [size], device=dev, generator=gen, dtype=torch.int64
            )
            t2 = torch.randint(
                0, high, [size], device=dev, generator=gen, dtype=torch.int64
            )
            return torch.stack([t1, t2]).unique().shape[0]

        # Use CPU as reference. The results should not deviate too much.
        self.assertTrue(
            abs(run(torch.device("cuda")) - run(torch.device("cpu"))) < 10_000
        )

    @largeTensorTest("20GB", "cuda")
    def test_randint_generation_for_large_numel(self) -> None:
        numel = 2**31 + 1
        s = torch.randint(2, (numel,), device="cuda", dtype=torch.int8).sum()
        self.assertTrue(s > 0, "expected randint in [0, 1] to generate nonzero values")

    @parametrize("dtype", [torch.float32, torch.double])
    def test_random_no_reused_random_states(self, dtype: torch.dtype) -> None:
        # Test if random states do not overlap between consecutive rand/randn calls.
        # See https://github.com/pytorch/pytorch/issues/125224

        def run(func, dev: torch.device, dtype: torch.dtype) -> int:
            # Measure how many unique numbers are generated in 2 consecutive calls. If random states are
            # reused, this will yield fewer unique numbers.
            size = 1000000
            gen = torch.Generator(device=dev)
            gen.manual_seed(0)
            t1 = func((size,), device=dev, generator=gen, dtype=dtype)
            t2 = func((size,), device=dev, generator=gen, dtype=dtype)
            return torch.stack([t1, t2]).unique().shape[0]

        # Use CPU as reference. The results should not deviate too much.
        for func in [torch.rand, torch.randn]:
            deviation = abs(
                run(func, torch.device("cuda"), dtype)
                - run(func, torch.device("cpu"), dtype)
            )
            self.assertTrue(deviation < 50_000, deviation)

    def test_min_max_inits(self):
        # Testing if THC_reduceAll received the correct index initialization.
        # This affects the result of THC_reduceAll operations at extreme values
        x = torch.cuda.ByteTensor([0])
        y = torch.cuda.ByteTensor([255])
        expected = torch.cuda.LongTensor([0])[0]

        _, v = x.max(dim=0)
        self.assertEqual(v, expected)

        _, v = y.min(dim=0)
        self.assertEqual(v, expected)

    def test_nvtx(self):
        # Just making sure we can see the symbols
        torch.cuda.nvtx.range_push("foo")
        torch.cuda.nvtx.mark("bar")
        torch.cuda.nvtx.range_pop()
        range_handle = torch.cuda.nvtx.range_start("range_start")
        torch.cuda.nvtx.range_end(range_handle)

    def test_bincount_ext(self):
        # ensure CUDA code coverage
        input_size = (100000,)
        w = torch.randn(input_size, dtype=torch.double, device="cuda")
        w_cpu = w.cpu()
        # test shared memory impl
        t = torch.randint(50, input_size, dtype=torch.int8, device="cuda")
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))
        # test global memory impl
        #   see `CUDAHistogramMemoryType` in SummaryOps.cu
        #   50000 * sizeof(int64_t) == 390 KiB, which should exceed smem of any known GPU
        t = torch.randint(50000, input_size, dtype=torch.int64, device="cuda")
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        t = torch.zeros([10], dtype=torch.int32, device="cuda")
        # 35488 * 65536 as int32 would cause overflow to negative value
        # giving negative bin offset
        t[0] = 35488
        counted = t.bincount(minlength=65536)
        self.assertEqual(torch.sum(counted), 10)

    def test_tiny_half_norm_(self):
        a = torch.arange(25).cuda().float()
        a /= 100000000
        b = a.half()
        self.assertGreater(b.norm().item(), 0)

    def test_norm_type_conversion(self):
        a = torch.ones(65536).cuda().half()
        self.assertEqual(a.norm(p=0, dtype=torch.float32), 65536)

    def test_cuda_memory_leak_detection_propagates_errors(self):
        with self.assertRaisesRegex(
            RuntimeError, r"The size of tensor a \(3\) must match"
        ):
            with self.assertLeaksNoCudaTensors():
                x = torch.randn(3, 1, device="cuda")
                y = torch.randn(2, 1, device="cuda")
                z = x + y

    @unittest.skipIf(not TEST_MEDIUM_TENSOR, "not enough memory")
    @serialTest()
    def test_cuda_kernel_loop_overflow(self):
        # Issue #24309: In extreme cases, the loop variable could overflow and continue
        # the kernel loop with a negative index, causing a RuntimeError (invalid write):
        x = torch.randn(1, 1, 1, 2**30 + 1, dtype=torch.float16, device="cuda")
        expected = x[0, 0, 0, 2**30]
        y = torch.nn.functional.avg_pool2d(x, kernel_size=1)
        torch.cuda.synchronize()
        self.assertEqual(y[0, 0, 0, 2**30], expected)

    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @gcIfJetson
    @serialTest()
    def test_cuda_kernel_loop_overflow_large(self):
        # Make sure input.numel() > INT_MAX is handled:
        x = torch.randn(1, 1, 1, 2**31, dtype=torch.float16, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "integer out of range"):
            y = torch.nn.functional.avg_pool2d(x, kernel_size=1)

        # Issue #24309: In extreme cases, the loop variable could overflow and continue
        # the kernel loop with a negative index, causing a RuntimeError (invalid write):
        x = torch.randn(1, 1, 1, 2**31 - 1, dtype=torch.float16, device="cuda")
        expected = x[0, 0, 0, 2**31 - 2]
        y = torch.nn.functional.avg_pool2d(x, kernel_size=1)
        torch.cuda.synchronize()
        self.assertEqual(y[0, 0, 0, 2**31 - 2], expected)

    # this might create a reference cycle on self...
    def _make_multiply_in_stream(self):
        class MultiplyInStream(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, val):
                ctx.val = val
                ctx.stream = torch.cuda.current_stream()
                return x * val

            @staticmethod
            def backward(ctx, grad):
                self.assertEqual(torch.cuda.current_stream(), ctx.stream)
                # delays the operation in the background stream
                torch.cuda._sleep(1000 * 5000)
                return grad * ctx.val, None

        return MultiplyInStream

    @skipCUDANonDefaultStreamIf(True)
    def test_streaming_backwards_sync(self):
        default_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()

        MultiplyInStream = self._make_multiply_in_stream()

        # Tests using grads outside the backward() stream context
        # See "Stream semantics of backward passes" on https://pytorch.org/docs/stable/notes/cuda.html
        x = torch.randn(5, 5, device="cuda", requires_grad=True)
        with torch.cuda.stream(stream):
            stream.wait_stream(default_stream)
            output = MultiplyInStream.apply(x, 2)
            output.sum().backward()
        # sync needed
        default_stream.wait_stream(stream)
        self.assertEqual(x.grad, torch.ones_like(x) * 2)
        self.assertEqual(torch.cuda.current_stream(), default_stream)

        # Tests that using grads in the same stream context as backward()
        # is safe regardless what streams bwd ops ran on
        bwd_ambient_stream = torch.cuda.Stream()
        x = torch.randn(5, 5, device="cuda", requires_grad=True)
        with torch.cuda.stream(stream):
            stream.wait_stream(default_stream)
            output = MultiplyInStream.apply(x, 3)
        with torch.cuda.stream(bwd_ambient_stream):
            bwd_ambient_stream.wait_stream(stream)
            output.sum().backward()
            # x was first used on "stream" so its AccumulateGrad leaf should run on "stream".
            # The end of backward() should have synced "bwd_ambient_stream" with "stream"
            # so it should be safe to use x.grad here without any syncs.
            self.assertEqual(x.grad, torch.ones_like(x) * 3)
            self.assertEqual(torch.cuda.current_stream(), bwd_ambient_stream)

    # Skip the test for ROCm as per https://github.com/pytorch/pytorch/issues/53190
    @skipIfRocm(msg="flakey on ROCm https://github.com/pytorch/pytorch/issues/53190")
    def test_streaming_backwards_multiple_streams(self):
        MultiplyInStream = self._make_multiply_in_stream()

        class StreamModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.event = torch.cuda.Event()
                self.stream0 = torch.cuda.Stream()
                self.stream1 = torch.cuda.Stream()

            def forward(self, x, x_first_use_on_ambient):
                if x_first_use_on_ambient:
                    x0 = x.clone()
                self.stream0.wait_stream(torch.cuda.current_stream())
                self.stream1.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream0):
                    if not x_first_use_on_ambient:
                        x0 = x.clone()
                    y0 = MultiplyInStream.apply(x0, 2)
                    self.event.record(stream=torch.cuda.current_stream())

                with torch.cuda.stream(self.stream1):
                    y1 = MultiplyInStream.apply(x, 3)
                    self.stream1.wait_event(self.event)
                    return y0 + y1

        stream = torch.cuda.Stream()

        for x_first_use_on_ambient in (True, False):
            # the out_of_place=False, iters=1 case stresses if proper syncs are inserted
            # when grads are initially None and stolen by backward ops.
            for out_of_place, iters in ((True, 1), (False, 1), (False, 5)):
                with torch.cuda.stream(stream):
                    x = torch.randn(5, 5, device="cuda", requires_grad=True)
                    model = StreamModel().cuda()
                    x.register_hook(
                        lambda grad: self.assertEqual(
                            torch.cuda.current_stream(),
                            stream if x_first_use_on_ambient else model.stream0,
                        )
                    )
                    for p in model.parameters():
                        self.assertTrue(p.grad is None)
                    for i in range(iters):
                        loss = model(x, x_first_use_on_ambient).sum()
                        if out_of_place:
                            x_grad = torch.autograd.grad((loss,), (x,))[0]
                        else:
                            loss.backward()
                # See "Stream semantics of backward passes" on https://pytorch.org/docs/stable/notes/cuda.html
                torch.cuda.current_stream().wait_stream(stream)

                if out_of_place:
                    self.assertEqual(x_grad, torch.ones_like(x) * 5 * iters)
                else:
                    self.assertEqual(x.grad, torch.ones_like(x) * 5 * iters)

    def test_streaming_backwards_sync_graph_root(self):
        # This function tests if bwd ops running on a side stream properly sync with the GraphRoot.
        # The potential bug it targets is a race condition. The test uses multiple trials and
        # torch.cuda._sleep such that if the race condition exists, the test will almost certainly fail,
        # but there's a chance it may spuriously pass. Passing does not guarantee the backend is bug-free,
        # but failure does guarantee there is a bug.
        fwd_bwd_op_stream = torch.cuda.Stream()
        bwd_ambient_stream = torch.cuda.Stream()
        # We need these streams to be different otherwise the test is meaningless.
        self.assertTrue(fwd_bwd_op_stream != bwd_ambient_stream)

        size = int(1e3)

        a = torch.full((size,), 2.0, device="cuda", requires_grad=True)
        b = torch.full((size,), 3.0, device="cuda", requires_grad=True)

        # I don't think we need any manual record_streams below.
        # a and b remain in scope for the entire test.
        # c and grad remain in scope for each iteration, and there's a full sync between iterations.
        for trial in range(5):
            torch.cuda.synchronize()
            a.grad = b.grad = None
            with torch.cuda.stream(fwd_bwd_op_stream):
                c = a * b

            with torch.cuda.stream(bwd_ambient_stream):
                torch.cuda.synchronize()
                # Long-running dummy kernel on bwd_ambient_stream delays filling of grad
                torch.cuda._sleep(int(50 * get_cycles_per_ms()))
                # Fills grad on bwd_ambient_stream
                grad = torch.full((size,), float(trial + 1), device="cuda")

                # Bwd ops still run on fwd_bwd_ops_stream, so the following will likely fail if
                # bwd ops don't sync with bwd_ambient_stream before consuming grad.
                torch.autograd.backward(tensors=c, grad_tensors=grad)

                # See https://github.com/pytorch/pytorch/issues/47028
                # assertEquals below run on bwd_ambient_stream, so this test may also fail
                # if backward() fails to sync with bwd_ambient_stream at the end.
                # Synchronizing here works around the issue until a proper fix can be made.
                torch.cuda.synchronize()
                with torch.no_grad():
                    self.assertEqual(a.grad, grad * b)
                    self.assertEqual(b.grad, grad * a)

    def test_streaming_backwards_callback(self):
        # Tests if autograd callbacks sync properly with respect to leaf streams and
        # the user-facing stream surrounding backward(). If it fails, first suspect is
        # sync logic where  "final_callbacks_" are called in torch/csrc/autograd/engine.cpp
        MultiplyInStream = self._make_multiply_in_stream()

        size = int(1e3)
        a = torch.full((size,), 1, device="cuda", dtype=torch.float, requires_grad=True)
        b = torch.full((size,), 1, device="cuda", dtype=torch.float, requires_grad=True)

        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        stash = []

        # sets up a nontrivial structure of leaf streams
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            c = MultiplyInStream.apply(a, 2)

        s1.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s1):
            d = MultiplyInStream.apply(b, 3)
            s1.wait_stream(s0)
            e = c * d

            def clone_leaf_grads():
                stash.append(a.grad.clone())
                stash.append(b.grad.clone())

            # Use a hook on e to install the callback
            e.register_hook(
                lambda grad: torch.autograd.Variable._execution_engine.queue_callback(
                    clone_leaf_grads
                )
            )

        s2.wait_stream(s1)
        with torch.cuda.stream(s2):
            e.sum().backward()
            # The autograd engine should sync s2 with all leaf streams then run the callback clone_leaf_grads on s2.
            # If those things happened properly, checking the values of the cloned grads on s2 should be safe:
            self.assertEqual(stash[0], torch.full_like(a, 6))
            self.assertEqual(stash[1], torch.full_like(a, 6))

    @unittest.skipIf(
        TEST_WITH_ROCM,
        "In ROCm, kernel asserts are disabled due to performance overhead",
    )
    def test_fixed_cuda_assert_async(self):
        with self.assertRaisesRegex(
            RuntimeError, "Boolean value of Tensor with no values is ambiguous"
        ):
            torch._assert_async(torch.tensor([], device="cuda"))
        with self.assertRaisesRegex(
            RuntimeError,
            "Boolean value of Tensor with more than one value is ambiguous",
        ):
            torch._assert_async(torch.tensor([0, 0], device="cuda"))

        torch._assert_async(torch.tensor(1, device="cuda"))
        torch._assert_async(torch.tensor(0.1, device="cuda"))
        torch._assert_async(torch.tensor(-0.1, device="cuda"))
        torch._assert_async(torch.tensor(True, device="cuda"))
        torch._assert_async(torch.tensor(0 + 0.1j, device="cuda"))

        fail_stmts = [
            "torch._assert_async(torch.tensor(0, device='cuda'))",
            "torch._assert_async(torch.tensor(0.0, device='cuda'))",
            "torch._assert_async(torch.tensor(False, device='cuda'))",
            "torch._assert_async(torch.tensor(0 + 0j, device='cuda'))",
        ]

        import subprocess

        for stmt in fail_stmts:
            with self.subTest(stmt=stmt):
                r = subprocess.call(
                    [
                        sys.executable,
                        "-c",
                        f"""\
import torch

{stmt}
torch.cuda.synchronize()
""",
                    ]
                )
                self.assertTrue(r != 0)

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "FAIL")
    def test_cublas_multiple_threads_same_device(self):
        # Note, these parameters should be very carefully tuned
        # Too small number makes it hard for the racing condition
        # to happen, while too large number sometimes cause hang
        size = 1024
        num_threads = 2
        trials = 3
        test_iters = 100

        weight = torch.ones((size, size), device="cuda")
        results = {}
        barrier = threading.Barrier(num_threads)

        def _worker(t):
            my_stream = torch.cuda.Stream()
            # Hard sync so we don't need to worry about creating and using tensors
            # across streams or the fact that default streams are thread-local.
            # Those issues are not the target of this test.
            torch.cuda.synchronize()
            # Line up threads to increase likelihood of race conditions.
            barrier.wait()
            with torch.cuda.stream(my_stream):
                for i in range(test_iters):
                    # If all threads are sharing the same cublas handle,
                    # the following sequence may occur:
                    # thread 0 calls cublasSetStream()
                    # thread 1 calls cublasSetStream()
                    # thread 0 launches its raw gemm, which it thinks is in
                    #          its own stream, but is actually in thread 1's stream.
                    # thread 0 enqueues its div_, which IS is its own stream,
                    #          but actually now races with its gemm.
                    results[t] = torch.mm(results[t], weight)
                    results[t].div_(float(size))
            torch.cuda.synchronize()

        for _ in range(trials):
            for t in range(num_threads):
                results[t] = torch.ones((size, size), device="cuda")

            threads = [
                threading.Thread(target=_worker, args=(t,)) for t in range(num_threads)
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for t in range(num_threads):
                self.assertEqual(results[t].sum().item(), size * size)

    # Test is flaky on Windows (https://github.com/pytorch/pytorch/issues/57401)
    @unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows (see issue 57401)")
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    @skipIfRocm
    def test_cudnn_multiple_threads_same_device(self):
        # This function is intended to test the lazy creation and reuse of per-thread
        # cudnn handles on each device in aten/src/ATen/cudnn/Handles.cpp.
        # Failure here likely indicates something wrong with that logic.
        weight = torch.ones((1, 1, 2, 2), device="cuda")

        results = {}

        num_threads = 2
        trials = 3
        test_iters = 1000
        barrier = threading.Barrier(num_threads)

        with torch.backends.cudnn.flags(enabled=True):

            def _worker(t):
                my_stream = torch.cuda.Stream()
                # Hard sync so we don't need to worry about creating and using tensors
                # across streams or the fact that default streams are thread-local.
                # Those issues are not the target of this test.
                torch.cuda.synchronize()
                # Line up threads to increase likelihood of race conditions.
                barrier.wait()
                with torch.cuda.stream(my_stream):
                    for _ in range(test_iters):
                        # If all threads are sharing the same cudnn handle,
                        # the following sequence may occur:
                        # thread 0 calls setCuDNNStreamToCurrent()
                        # thread 1 calls setCuDNNStreamToCurrent()
                        # thread 0 launches its raw convolution, which it thinks is in
                        #          its own stream, but is actually in thread 1's stream.
                        # thread 0 enqueues its div_, which IS is its own stream,
                        #          but now races with its convolution.
                        results[t] = torch.nn.functional.conv2d(
                            results[t], weight, padding=0
                        )
                        results[t].div_(4.0)
                torch.cuda.synchronize()

            for _ in range(trials):
                for t in range(num_threads):
                    results[t] = torch.ones((1, 1, 2048, 2048), device="cuda")

                threads = [
                    threading.Thread(target=_worker, args=(t,))
                    for t in range(num_threads)
                ]

                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                for t in range(num_threads):
                    self.assertEqual(
                        results[t].sum().item(),
                        (2048 - test_iters) * (2048 - test_iters),
                    )

    def test_cusparse_multiple_threads_same_device(self):
        size = 1024
        num_threads = 2
        trials = 3
        test_iters = 500

        def ones_sparse(size):
            a = torch.arange(size, device="cuda")
            indices = torch.cartesian_prod(a, a).t()
            values = torch.ones(size * size, device="cuda")
            return torch.sparse_coo_tensor(indices, values)

        weight = ones_sparse(size)
        results = {}
        barrier = threading.Barrier(num_threads)

        def _worker(t):
            my_stream = torch.cuda.Stream()
            # Hard sync so we don't need to worry about creating and using tensors
            # across streams or the fact that default streams are thread-local.
            # Those issues are not the target of this test.
            torch.cuda.synchronize()
            # Line up threads to increase likelihood of race conditions.
            barrier.wait()
            with torch.cuda.stream(my_stream):
                for i in range(test_iters):
                    # If all threads are sharing the same cublas handle,
                    # the following sequence may occur:
                    # thread 0 calls cublasSetStream()
                    # thread 1 calls cublasSetStream()
                    # thread 0 launches its raw gemm, which it thinks is in
                    #          its own stream, but is actually in thread 1's stream.
                    # thread 0 enqueues its div_, which IS is its own stream,
                    #          but actually now races with its gemm.
                    results[t] = weight.mm(results[t])
                    results[t].div_(float(size))
            torch.cuda.synchronize()

        for _ in range(trials):
            for t in range(num_threads):
                results[t] = torch.ones((size, size), device="cuda")

            threads = [
                threading.Thread(target=_worker, args=(t,)) for t in range(num_threads)
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for t in range(num_threads):
                self.assertEqual(results[t].sum().item(), size * size)

    @slowTest
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @serialTest()
    def test_max_large_axis(self):
        x = torch.zeros(2**32, device="cuda", dtype=torch.int8)
        x[-1] = 1
        val, idx = x.max(0)
        self.assertEqual(val, 1)
        self.assertEqual(idx, x.shape[0] - 1)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_to_numpy(self):
        self.assertRaises(TypeError, lambda: torch.empty(1, device="cuda").numpy())

    def test_graph_is_current_stream_capturing(self):
        self.assertFalse(torch.cuda.is_current_stream_capturing())

        if TEST_CUDA and (not TEST_WITH_ROCM):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                g = torch.cuda.CUDAGraph()
                self.assertFalse(torch.cuda.is_current_stream_capturing())
                g.capture_begin()
                self.assertTrue(torch.cuda.is_current_stream_capturing())
                g.capture_end()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_simple(self):
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            a = torch.full((1000,), 1, device="cuda")
            g = torch.cuda.CUDAGraph()
            torch.cuda.empty_cache()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()

        self.assertEqual(b.sum().item(), 11000.0)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graphsafe_set_get_rng_state(self):
        # Define a function to create generator states, with optional graph registration
        def create_states(generator):
            """Initializes generator states and registers them with a CUDA graph if provided."""
            # Ensure the CUDA generator is initialized
            torch.rand(1, device="cuda")
            generator.manual_seed(0)

            # Save the current state of the generator
            old_state = generator.graphsafe_get_state()
            # Create and save a cloned state of the generator
            new_state = generator.clone_state()
            # Return the original generator and its two states
            return generator, old_state, new_state

        def register_states_to_graph(generator_state, graph):
            generator, old_state, new_state = generator_state
            graph.register_generator_state(old_state)
            graph.register_generator_state(new_state)

        # Define a function to perform specific RNG actions using the generator's states
        def perform_random_generation_steps(generator_state):
            generator, old_state, new_state = generator_state
            random_values = []

            # Generate random numbers with the new generator state
            generator.graphsafe_set_state(new_state)
            random_values.append(torch.rand(5, device="cuda", generator=generator))

            # Generate random numbers twice with the old generator state
            generator.graphsafe_set_state(old_state)
            random_values.extend(
                [torch.rand(5, device="cuda", generator=generator) for _ in range(2)]
            )

            return random_values

        # Define a function to retrieve the final offsets of the original and new generator states
        def get_final_offsets_of_states(generator_state):
            generator, old_state, new_state = generator_state
            old_state_offset = old_state.get_offset()
            new_state_offset = new_state.get_offset()
            return old_state_offset, new_state_offset

        # Set up and test a new CUDA generator
        generator = torch.Generator(device="cuda")
        generator_state = create_states(generator)

        # Set up and test the default CUDA generator with a CUDA Graph
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        default_generator = torch.cuda.default_generators[0]
        default_generator_state = create_states(default_generator)
        register_states_to_graph(default_generator_state, g)

        # Perform random number generation within a CUDA graph
        with torch.cuda.stream(s):
            g.capture_begin()
            graphed_random_values = perform_random_generation_steps(
                default_generator_state
            )
            g.capture_end()

        # Synchronize the streams and replay the graph
        torch.cuda.current_stream().wait_stream(s)
        for _ in range(3):
            random_values = perform_random_generation_steps(generator_state)
            g.replay()
            offset = get_final_offsets_of_states(generator_state)
            graph_offset = get_final_offsets_of_states(default_generator_state)

            # Compare the final offsets of states for both generators to ensure consistency
            self.assertEqual(offset, graph_offset)
            # Compare the states generated outside and inside the graph
            self.assertEqual(random_values, graphed_random_values)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_memory_stats_of_multiple_generators_and_graphs(self):
        # Function to clear CUDA cache and collect garbage
        def clear_cuda_cache():
            gc.collect()
            torch.cuda.empty_cache()

        # Executes a simple graph task which includes capturing and executing a random number generation within a CUDA graph.
        def simple_graph_task(graph):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                graph.capture_begin()
                torch.rand(1, device="cuda")
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(s)
            graph.replay()  # Replays the captured operations

        def get_memory_stats():
            stats = torch.cuda.memory_stats()
            num_blocks = stats["active.all.current"]
            total_size = stats["active_bytes.all.current"]
            return num_blocks, total_size

        def test(num_graphs, num_generators):
            baseline = get_memory_stats()
            baseline_num_blocks, baseline_total_size = baseline

            # Allocate CUDA graphs
            graphs = [torch.cuda.CUDAGraph() for _ in range(num_graphs)]

            # Allocate and manage generator states
            default_generator = torch.cuda.default_generators[0]
            generators = [default_generator.graphsafe_get_state()]

            # Starts from 1 as one state is already added
            for _ in range(1, num_generators):
                generators.append(default_generator.clone_state())

            for graph in graphs:
                for generator_state in generators:
                    graph.register_generator_state(generator_state)
                simple_graph_task(graph)

            # Assert conditions after graph tasks
            num_blocks, total_size = get_memory_stats()
            # The allocated blocks should only be proportional to the number of generators
            expected_blocks_diff = 2 * num_generators
            expected_size_diff = 2 * 512 * num_generators  # Each block's size is 512

            self.assertEqual(
                (num_blocks - baseline_num_blocks),
                expected_blocks_diff,
                "Unexpected number of active blocks.",
            )
            self.assertEqual(
                (total_size - baseline_total_size),
                expected_size_diff,
                "Unexpected total memory size.",
            )

            # Cleanup graphs and clear CUDA cache
            while graphs:
                graph = graphs.pop()
                del graph
            clear_cuda_cache()

            # Assert that memory stats return to baseline after cleanup
            self.assertEqual(
                get_memory_stats(),
                baseline,
                "Memory stats do not match baseline after cleanup.",
            )

        # Running the test function with different parameters
        test(1, 1)
        test(3, 2)
        test(10, 20)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_reset_recapture(self):
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            a = torch.full((1000,), 1, device="cuda")
            g = torch.cuda.CUDAGraph()
            torch.cuda.empty_cache()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()

        self.assertEqual(b.sum().item(), 11000.0)

        g.reset()

        with torch.cuda.stream(s):
            g.capture_begin()
            b.fill_(2.0)
            for _ in range(10):
                b = b + 2
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()
        self.assertEqual(b.sum().item(), 22000.0)

        g.reset()
        del g

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_debugdump(self):
        torch.cuda.empty_cache()
        x = torch.randn(10240000, device="cuda")
        y = torch.rand_like(x)
        g = torch.cuda.CUDAGraph()
        g.enable_debug_mode()
        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            g.capture_begin()
            z = x + y
            with torch.cuda.stream(s1):
                s1.wait_stream(s0)
                w = z + y
            s0.wait_stream(s1)
            g.capture_end()
        s0.synchronize()
        torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as tempdir:
            g.debug_dump(os.path.join(tempdir, "out_multi_stream.dot"))

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_error(self):
        # We need to run this test in a separate thread as the error we trigger
        # puts the cuda context in a bad state
        script = """
import torch

g = torch.cuda.CUDAGraph()
try:
    g.capture_begin()
except RuntimeError as e:
    if "CUDA graphs must be captured on a non-default stream." in str(e):
        exit(0)
    else:
        exit(1)
exit(2)
"""
        try:
            a = subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                self.assertTrue(
                    False,
                    "Error raise by starting capture without a stream is not the expected one",
                )
            elif e.returncode == 2:
                self.assertTrue(
                    False,
                    "Error raised by starting capture without a stream was not caught",
                )

    @unittest.skipIf(
        (not TEST_CUDA) or TEST_WITH_ROCM or int(torch.version.cuda.split(".")[0]) < 11,
        "CUDA >= 11.0 required for graphs",
    )
    def test_graph_warn_if_has_zero_nodes(self):
        with warnings.catch_warnings(record=True) as caught:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                g.capture_begin()
                g.capture_end()
        self.assertTrue(
            any("The CUDA Graph is empty" in str(w.message) for w in caught)
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    def test_graph_capture_oom(self):
        oom_regex = (
            "would exceed allowed memory" if TEST_CUDAMALLOCASYNC else "out of memory"
        )
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                torch.zeros(2**40, device="cuda")

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @serialTest()
    @setBlasBackendsToDefaultFinally
    def test_repeat_graph_capture_cublas_workspace_memory(self):
        torch.backends.cuda.preferred_blas_library("cublas")
        (x, y, z) = 1024, 512, 64
        a = torch.rand((x, y), device="cuda")
        b = torch.rand((y, z), device="cuda")

        # warmup
        torch.mm(a, b)

        free_bytes_before, total_bytes = torch.cuda.mem_get_info()
        used_gb_before = (total_bytes - free_bytes_before) / 1e9

        for i in range(100):
            torch_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(torch_graph):
                torch.mm(a, b)
            torch_graph.replay()

        free_bytes_after, _ = torch.cuda.mem_get_info()
        used_gb_after = (total_bytes - free_bytes_after) / 1e9

        self.assertFalse(used_gb_before + 0.1 < used_gb_after)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_rng_functional(self):
        ops_with_kwargs = (
            (torch.nn.functional.dropout, {"p": 0.1}),
            (torch.nn.functional.rrelu, {"training": True}),
        )
        size = 10000

        def run(op, kwargs):
            a = torch.randn((size,), device="cuda", dtype=torch.float)

            # Control
            torch.cuda.manual_seed(5)
            eager_out = a
            for _ in range(6):
                eager_out = op(eager_out, **kwargs)

            graph_in = a.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.manual_seed(5)

                g = torch.cuda.CUDAGraph()
                torch.cuda.empty_cache()
                g.capture_begin()
                graph_out = graph_in
                for _ in range(2):
                    graph_out = op(graph_out, **kwargs)
                g.capture_end()
            torch.cuda.current_stream().wait_stream(stream)

            # Runs a graphed->eager->graphed sequence of RNG ops.
            # replay() plays 2 invocations of the op, so the sequence has 6
            # invocations total, matching Control.
            # replay() reads from graph_in and writes to graph_out.
            g.replay()
            out = op(graph_out, **kwargs)
            out = op(out, **kwargs)
            graph_in.copy_(out)
            g.replay()

            # If replay() updated RNG state correctly, graph_out
            # should now hold data equal to eager_out.
            try:
                self.assertEqual(eager_out, graph_out)
            except Exception as e:
                raise RuntimeError("Failed on ", op) from e

            # Do the same operations varying seeds
            seeds = [6, 128, 9999]

            for seed in seeds:
                torch.cuda.manual_seed(seed)
                graph_in.copy_(a)
                for _ in range(3):
                    g.replay()

                # If the random seed was not updated then the graph would
                # generate the same output as in previous check.
                try:
                    self.assertNotEqual(eager_out, graph_out)
                except Exception as e:
                    raise RuntimeError("Failed on ", op) from e

                # Now repeat the same operations in non-graphed mode.
                torch.cuda.manual_seed(seed)
                for _ in range(3):
                    eager_out.copy_(a)
                    eager_out = op(eager_out, **kwargs)
                    eager_out = op(eager_out, **kwargs)

                # In the end, graph_out and eager_out must be equal
                # as they went under the same set of operations.
                try:
                    self.assertEqual(eager_out, graph_out)
                except Exception as e:
                    raise RuntimeError("Failed on ", op) from e

            # We hold references to all tensors used across streams up til this sync,
            # so no need to call record_stream on those tensors.
            torch.cuda.synchronize()

        for op, kwargs in ops_with_kwargs:
            run(op, kwargs)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_rng_distributions(self):
        size = 10000
        input = torch.rand((size,), device="cuda", dtype=torch.float)
        alloc = torch.empty((size,), device="cuda", dtype=torch.float)

        # Torch ops to test with sample args (tuple) and kwargs (dict)
        torch_with_args = (
            ("bernoulli", (input.clone(),), {}),
            # multinomial uses some uncapturable CUDA calls.
            # TODO: reenable multinomial tests if/when the implementation is capturable.
            # ("multinomial", (input.clone(), size, True), {}),
            # ("multinomial", (input.clone(), size // 2, False), {}),
            # TODO: reenable normal test, where std is a device
            # tensor, when graph test failures are fixed
            # ("normal", (input.clone() + 1, input.clone()), {}),
            ("normal", (input.clone() + 1, 1.0), {}),
            ("poisson", (input.clone(),), {}),
            ("rand", (size,), {"device": "cuda", "dtype": torch.float}),
            ("randint", (0, 3, (size,)), {"device": "cuda", "dtype": torch.float}),
            ("randn", (size,), {"device": "cuda", "dtype": torch.float}),
        )

        # Tensor methods to test with sample args (tuple)
        tensor_with_args = (
            ("bernoulli_", (input.clone(),)),
            ("cauchy_", ()),
            ("exponential_", ()),
            ("geometric_", (0.3,)),
            ("log_normal_", ()),
            ("normal_", ()),
            ("random_", ()),
            ("uniform_", ()),
        )

        def run(module, op, args, kwargs):
            torch.cuda.manual_seed(5)

            # Each path runs a dummy op to increment the state a bit before creating controls.
            if module == "torch":
                dummy = getattr(torch, op)(*args, **kwargs)
                control1 = getattr(torch, op)(*args, **kwargs)
                control2 = getattr(torch, op)(*args, **kwargs)
            else:
                dummy = alloc.clone()
                control1 = alloc.clone()
                control2 = alloc.clone()
                getattr(dummy, op)(*args)
                getattr(control1, op)(*args)
                getattr(control2, op)(*args)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.manual_seed(5)

                g = torch.cuda.CUDAGraph()
                torch.cuda.empty_cache()
                if module == "torch":
                    g.capture_begin()
                    t1 = getattr(torch, op)(*args, **kwargs)
                    t2 = getattr(torch, op)(*args, **kwargs)
                    g.capture_end()
                else:
                    t1 = alloc.clone()
                    t2 = alloc.clone()
                    g.capture_begin()
                    getattr(t1, op)(*args)
                    getattr(t2, op)(*args)
                    g.capture_end()
            torch.cuda.current_stream().wait_stream(stream)

            if not TEST_CUDAMALLOCASYNC:
                # Makes sure values haven't been populated yet
                # (in other words, makes sure capture didn't actually run ops).
                # We can only try this with the native allocator, for which captured
                # addresses are already backed by cudaMalloced memory.
                # If we try it with cudaMallocAsync, CUDA won't event consider
                # the captured addresses allocated until replay(), and if we
                # access them before replay() we get IMAs.
                try:
                    self.assertNotEqual(control1, t1)
                    self.assertNotEqual(control2, t2)
                except Exception as e:
                    raise RuntimeError("Failed on " + module + "." + op) from e

            # Set a new seed to check if graph would use it
            for seed in [6, 314, 271]:
                torch.cuda.manual_seed(seed)
                # Runs a dummy op prelude, as for controls, to make sure replay()
                # picks up the dummy op's state increment.
                if module == "torch":
                    dummy = getattr(torch, op)(*args, **kwargs)
                    control1 = getattr(torch, op)(*args, **kwargs)
                    control2 = getattr(torch, op)(*args, **kwargs)
                else:
                    getattr(dummy, op)(*args)
                    getattr(control1, op)(*args)
                    getattr(control2, op)(*args)

                torch.cuda.manual_seed(seed)
                if module == "torch":
                    dummy = getattr(torch, op)(*args, **kwargs)
                else:
                    getattr(dummy, op)(*args)

                # see above comment on TEST_CUDAMALLOCASYNC
                if not TEST_CUDAMALLOCASYNC:
                    t1.copy_(alloc)
                    t2.copy_(alloc)

                # Runs RNG ops that fill t1 and t2.
                g.replay()

                try:
                    self.assertEqual(control1, t1)
                    self.assertEqual(control2, t2)
                except Exception as e:
                    raise RuntimeError("Failed on " + module + "." + op) from e

            # We hold references to all tensors used across streams up til this sync,
            # so no need to call record_stream on those tensors.
            torch.cuda.synchronize()

        for op_with_args in torch_with_args:
            run("torch", *op_with_args)

        for meth_with_args in tensor_with_args:
            # Adds an empty dict for kwargs, which none of the Tensor methods use
            run("Tensor", *(meth_with_args + ({},)))

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_two_successive(self):
        torch.cuda.empty_cache()

        size = 1000
        kSmallBuffer = 2097152

        def func_with_temps(t, val):
            x = t.clone() + val
            y = t.clone() + val
            return x + y

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()

            a = torch.ones((size,), device="cuda")

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (
                    (torch.cuda.graph_pool_handle(),)
                    if share_mem == "via graph_pool_handle()"
                    else ()
                )
                g0.capture_begin(*g0_args)
                b = a.clone()
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g0.capture_end()

                g1_args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                g1.capture_begin(*g1_args)
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g1.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            # mixes unrelated eager ops with replays
            c = a.clone()
            for _ in range(2):
                c = func_with_temps(c, 3)
            g0.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)
            g1.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)

            self.assertEqual(b.sum().item(), size * 3070)
            self.assertEqual(c.sum().item(), size * 442)

            if not TEST_CUDAMALLOCASYNC:
                # These stat checks are specific to the native allocator.
                if share_mem != "Don't share":
                    self.assertEqual(
                        reserved_no_sharing  # noqa: F821
                        - torch.cuda.memory_stats()["reserved_bytes.all.current"],
                        kSmallBuffer,
                    )
                else:
                    reserved_no_sharing = torch.cuda.memory_stats()[
                        "reserved_bytes.all.current"
                    ]

            del a, b, c, g0, g1
            # Tensors used across streams (a and b) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @unittest.skipIf(
        (not TEST_CUDA_GRAPH)
        or IS_WINDOWS
        or (  # appears to still be broken on Windows as of 11.4+
            torch.version.cuda
            and int(torch.version.cuda.split(".")[0]) == 11
            and int(torch.version.cuda.split(".")[1]) < 4
        ),
        "Graph bindings disallow concurrent replay for CUDA < 11.4, see "
        + "https://github.com/pytorch/pytorch/pull/57556",
    )
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_concurrent_replay(self):
        torch.cuda.empty_cache()

        size = 1000000  # largeish to help expose race conditions

        def func_with_temps(t, val):
            x = t.clone() + val
            y = t.clone() + val
            return x + y

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()

            s0 = torch.cuda.Stream()
            s1 = torch.cuda.Stream()

            a = torch.ones((size,), device="cuda")

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (
                    (torch.cuda.graph_pool_handle(),)
                    if share_mem == "via graph_pool_handle()"
                    else ()
                )
                g0.capture_begin(*g0_args)
                b = a.clone()
                for _ in range(5):
                    b = func_with_temps(b, 1)
                g0.capture_end()

                g1_args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                g1.capture_begin(*g1_args)
                c = a.clone()
                for _ in range(5):
                    c = func_with_temps(c, 2)
                g1.capture_end()

            # To reproduce data corruption, I need g0 and g1's kernels to run concurrently.
            # But replay() (especially cudaGraphLaunch) can incur significant CPU overhead.
            # The following pattern helps align device-side execution of g0 and g1's kernels.
            torch.cuda.synchronize()
            with torch.cuda.stream(s0):
                torch.cuda._sleep(1000000)
                s1.wait_stream(s0)
                g0.replay()
            with torch.cuda.stream(s1):
                g1.replay()
            torch.cuda.current_stream().wait_stream(s0)
            torch.cuda.current_stream().wait_stream(s1)

            if (not TEST_CUDAMALLOCASYNC) and (share_mem != "Don't share"):
                # If we used the native allocator and shared mempools,
                # we expect the concurrent replays corrupted each other.
                self.assertNotEqual(b.sum().item(), size * 94)
                self.assertNotEqual(c.sum().item(), size * 156)
            else:
                # If we EITHER
                #   - used the native allocator without sharing mempools, OR
                #   - used cudaMallocAsync, which ignores graph pool-sharing hints and should always be safe
                # we don't expect memory corruption.
                self.assertEqual(b.sum().item(), size * 94)
                self.assertEqual(c.sum().item(), size * 156)

            del a, b, c, g0, g1
            # Tensors used across streams (a, b, c) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_three_successive(self):
        torch.cuda.empty_cache()

        size = 1000

        s = torch.cuda.Stream()

        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            a = torch.ones((size,), device="cuda")

            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()
            g2 = torch.cuda.CUDAGraph()

            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                g0_args = (
                    (torch.cuda.graph_pool_handle(),)
                    if share_mem == "via graph_pool_handle()"
                    else ()
                )
                g0.capture_begin(*g0_args)
                b = a.clone()
                c = b + 1
                d = b + 2
                g0.capture_end()

                args = (g0.pool(),) if share_mem == "via pool()" else g0_args

                g1.capture_begin(*args)
                e = c + 3
                del c
                g1.capture_end()

                g2.capture_begin(*args)
                f = d + 4
                g2.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            # Tests that replaying in capture order is valid
            g0.replay()
            g1.replay()
            g2.replay()

            self.assertEqual(e.sum().item(), size * 5)
            self.assertEqual(f.sum().item(), size * 7)

            # Tests that replaying as g0, g2, g1 is only valid if they don't share a pool
            g0.replay()
            g2.replay()
            g1.replay()

            expect_corruption = (not TEST_CUDAMALLOCASYNC) and (
                share_mem != "Don't share"
            )
            # If we used the native allocator and shared mempools, g2's capture should have reused c's memory for f.
            # We replayed g2 then g1, so we expect g1's captured "e = c + 3" mistakenly filled e with "f's vals + 3".
            self.assertEqual(
                e.sum().item(), size * (7 + 3) if expect_corruption else size * 5
            )
            self.assertEqual(f.sum().item(), size * 7)

            del a, b, d, e, f, g0, g1, g2
            # Tensors used across streams (a, e, f) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @unittest.skipIf(
        (not TEST_CUDA_GRAPH) or TEST_CUDAMALLOCASYNC,
        "CUDA >= 11.0 or ROCM >= 5.3 required for graphs",
    )
    def test_graph_memory_stats_and_use_result_after_destroy_graph(self):
        kSmallSize = 1048576
        kSmallBuffer = 2097152
        kLargeBuffer = 20971520
        kMinLargeAlloc = 10485760
        kRoundLarge = 2097152

        elem = 4

        # this was annoying to write but stresses the expectations pretty rigorously
        cases = (
            (512 // elem, 1, kSmallBuffer, kSmallBuffer, "small_pool"),
            (kSmallSize // elem, 2, 2 * kSmallBuffer, kSmallBuffer, "small_pool"),
            ((kSmallSize + 512) // elem, 1, kLargeBuffer, kLargeBuffer, "large_pool"),
            (
                (kMinLargeAlloc - 512) // elem,
                2,
                2 * kLargeBuffer,
                kLargeBuffer,
                "large_pool",
            ),
            (
                (kMinLargeAlloc + 512) // elem,
                3,
                3
                * (
                    kRoundLarge
                    * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge)
                ),
                kRoundLarge * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge),
                "large_pool",
            ),
        )

        stats_to_check = ("segment.", "reserved_bytes.", "active.", "active_bytes.")

        gc.collect()
        torch.cuda.empty_cache()

        s = torch.cuda.Stream()

        for (
            numel,
            delta_cudaMallocs,
            delta_cudaMalloc_bytes,
            delta_cudaMalloc_bytes_post_del_g,
            pool_string,
        ) in cases:
            if pool_string == "small_pool":
                delta_active_blocks = 3  # one from "b" plus a sneaky two from CUDAGraph's one-element rng seed and offset holders
                delta_active_bytes = (
                    numel * elem + 1024
                )  # + 1024 for CUDAGraph's rng seed and offset holders each
            else:
                delta_active_blocks = 1  # We only check the large pool, which isn't affected by rng offset holder
                delta_active_bytes = numel * elem

            g = torch.cuda.CUDAGraph()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                # Allocation stat estimates assume input is created on the same stream as capture_begin()
                # (in other words, the same stream silo as the rng offset holder, which is not allocated from the
                # capture's private pool).
                a = torch.ones((numel,), device="cuda")

                precapture_stats = torch.cuda.memory_stats()

                g.capture_begin()
                b = a.clone()
                for _ in range(5):
                    b = b.clone() + 1
                g.capture_end()
            torch.cuda.current_stream().wait_stream(s)

            gc.collect()

            postcapture_stats = torch.cuda.memory_stats()

            expecteds = (
                delta_cudaMallocs,
                delta_cudaMalloc_bytes,
                delta_active_blocks,
                delta_active_bytes,
            )
            # Double checks replay and stats before and after a call to empty_cache
            for i in range(2):
                for stat, expected in zip(stats_to_check, expecteds):
                    stat = stat + pool_string + ".current"
                    current = postcapture_stats[stat] - precapture_stats[stat]

                    # There will only ever be one expandable segment in each of the small and large pools. The way the
                    # bookeeping is done in the allocator means that we never increment the number of segments.
                    if self.expandable_segments and "segment" in stat:
                        expected = 0
                    # These two cases hit an edge case where the PyTorch allocator won't immediately unmap part of an
                    # expandable segment (and as a result reduce the number of reserved bytes) if the block to unmap is
                    # smaller than the page size
                    if (
                        self.expandable_segments
                        and "reserved" in stat
                        and (numel == cases[3][0] or numel == cases[4][0])
                    ):
                        expected = 2 * kLargeBuffer

                    self.assertEqual(
                        current,
                        expected,
                        "Pre to post capture delta of "
                        + stat
                        + f" = {current}, expected = {expected}, numel = {numel}",
                    )

                g.replay()
                self.assertEqual(b.sum().item(), 6 * numel)
                if i == 0:
                    torch.cuda.empty_cache()

            del g
            gc.collect()
            torch.cuda.empty_cache()
            postdel_stats = torch.cuda.memory_stats()

            # Uses graph result b after graph has been deleted
            self.assertEqual(b.sum().item(), 6 * numel)

            # b should be the only live reference remaining from the graph's private pool
            expecteds = (1, delta_cudaMalloc_bytes_post_del_g, 1, numel * elem)
            for stat, expected in zip(stats_to_check, expecteds):
                stat = stat + pool_string + ".current"
                current = postdel_stats[stat] - precapture_stats[stat]

                # There will only ever be one expandable segment in each of the small and large pools. The way the
                # bookeeping is done in the allocator means that we never increment the number of segments.
                if self.expandable_segments and "segment" in stat:
                    expected = 0
                # These two cases hit an edge case where the PyTorch allocator won't immediately unmap part of an
                # expandable segment (and as a result reduce the number of reserved bytes) if the block to unmap is
                # smaller than the page size
                if (
                    self.expandable_segments
                    and "reserved" in stat
                    and numel == cases[3][0]
                ):
                    expected = 2 * kLargeBuffer
                if (
                    self.expandable_segments
                    and "reserved" in stat
                    and numel == cases[4][0]
                ):
                    expected = kLargeBuffer

                self.assertEqual(
                    current,
                    expected,
                    "Pre capture to post graph delete delta of "
                    + stat
                    + f" = {current}, expected = {expected}, numel = {numel}",
                )

            # del a, b before the next case is essential, otherwise overwriting a and b in the next case
            # can throw off its allocation/deallocation counts.
            del a, b
            # Tensors used across streams (a and b) were held until just now, so no need to call record_stream on them.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_record_stream(self):
        # Makes sure graph capture defers attempting to reclaim allocations used across streams. See
        # "Q. Why skip process_events if a capture might be underway?" in c10/cuda/CUDACachingAllocator.cpp
        torch.cuda.empty_cache()

        potential_problem = torch.zeros((3,), device="cuda")
        a = torch.zeros((3,), device="cuda")
        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()

        torch.cuda.synchronize()
        with torch.cuda.stream(s0):
            potential_problem.record_stream(s0)
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            potential_problem.fill_(1.0)
        del potential_problem

        with torch.cuda.stream(s1):
            g.capture_begin()
            # potential_problem's allocation should still be outstanding. if DeviceCachingAllocator::malloc
            # mistakenly calls process_events, it will trigger cudaEventQueries on potential_problem's end-of-life
            # event, which will cause the capture to error.
            b = a.clone()

            # Let's also see what happens if we record_stream on a tensor during capture.
            s2.wait_stream(s1)
            with torch.cuda.stream(s2):
                b.fill_(1.0)
                b.record_stream(s2)  # dummy record_stream
                del b
            s1.wait_stream(s2)
            g.capture_end()
        torch.cuda.synchronize()

        # dummy allocation triggers process_events, Hopefully successfully processes b's end-of-life event.
        c = torch.zeros((3,), device="cuda")

    @skipIfRocm
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # If this test is the first in the process to try cudnn rnns with dropout, it'll initialize
    # DropoutState's long-lived internal buffer. Calling code perceives this (correct) behavior
    # as a memory leak unless we skip the leak check.
    @skipCUDAMemoryLeakCheckIf(True)
    @serialTest()
    def test_graph_cudnn_dropout(self):
        # Tests the interaction of cuda graph capture with DropoutState's syncs in ATen/native/cudnn/RNN.cpp.
        # In particular, if user runs a sequence of captured and noncaptured cudnn rnns, DropoutState should
        # avoid syncing noncapturing streams with captured events or vice versa.
        torch.cuda.empty_cache()

        model = torch.nn.LSTM(512, 512, 2, dropout=0.5).cuda()
        x = torch.ones(100, 192, 512, device="cuda")

        y = model(x)

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g.capture_begin()
            y = model(x)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()

        y = model(x)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @parametrize(
        "with_amp,cache_enabled,allow_unused_input",
        [
            subtest((False, False, True), decorators=[skipIfRocm]),
            subtest((True, False, True), decorators=[skipIfRocm]),
            subtest((True, True, True), decorators=[unittest.expectedFailure]),
            subtest((False, False, False), decorators=[unittest.expectedFailure]),
        ],
        name_fn=lambda x, y, z: "{}{}{}".format(
            {True: "with_amp", False: "without_amp"}[x],
            {True: "_cache_enabled", False: "_cache_disabled"}[y] if x else "",
            {True: "_allow_unused_input", False: "_not_allow_unused_input"}[z],
        ),
    )
    @serialTest()
    def test_graph_make_graphed_callables(
        self, with_amp, cache_enabled, allow_unused_input
    ):
        torch.manual_seed(5)
        torch.cuda.manual_seed(5)

        N, D_in, H, D_out = 640, 4096, 2048, 1024

        class MLP1(torch.nn.Module):
            def __init__(self, D_in: int, H: int, D_out: int):
                super().__init__()
                self.net_1 = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H), torch.nn.Dropout(p=0.1)
                ).cuda()
                self.net_2 = torch.nn.Sequential(
                    torch.nn.Linear(H, D_out), torch.nn.Dropout(p=0.2)
                ).cuda()

            def forward(self, input_dict: dict):
                x = input_dict["x"]
                return self.net_2(self.net_1(x))

        class MLP2(torch.nn.Module):
            def __init__(self, D_in: int, H: int, D_out: int):
                super().__init__()
                self.net_1 = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H), torch.nn.Dropout(p=0.1)
                ).cuda()
                self.net_2 = torch.nn.Sequential(
                    torch.nn.Linear(H, D_out), torch.nn.Dropout(p=0.2)
                ).cuda()

            def forward(self, x):
                return self.net_2(self.net_1(x))

        class ParameterlessModule(torch.nn.Module):
            def forward(self, x):
                idx = (
                    torch.arange(x.size(0), device=x.device)
                    .view(-1, 1)
                    .repeat(1, x.size(1))
                )
                return {"output": torch.gather(x, 0, idx)}

        models = []
        for _ in range(2):
            model_section1 = MLP1(D_in, H, H).cuda()
            model_section2 = MLP2(H, H, D_out).cuda()
            model_section3 = ParameterlessModule().cuda()
            models.append(
                torch.nn.Sequential(model_section1, model_section2, model_section3)
            )

        model_graphed = models[0]
        model_control = models[1]

        model_graphed.load_state_dict(model_control.state_dict())

        opt_graphed = torch.optim.SGD(model_graphed.parameters(), lr=0.1)
        opt_control = torch.optim.SGD(model_control.parameters(), lr=0.1)

        x = torch.randn(N, D_in, device="cuda")
        h = torch.randn(N, H, device="cuda", requires_grad=True)
        h2 = torch.randn(N, D_out, device="cuda", requires_grad=True)
        unused_input = torch.randn(N, H, device="cuda", requires_grad=True)
        y_pred = torch.randn(N, D_out, device="cuda", requires_grad=True)
        y = torch.randn(N, D_out, device="cuda")

        loss_fn_control = torch.nn.functional.mse_loss
        relu_control = torch.nn.functional.relu

        # This is a good stress test. It graphs four callables: two Modules and two python functions.
        with torch.amp.autocast(
            device_type="cuda", enabled=with_amp, cache_enabled=cache_enabled
        ):
            (
                model_graphed[0],
                model_graphed[1],
                model_graphed[2],
                relu_graphed,
                loss_fn_graphed,
            ) = torch.cuda.make_graphed_callables(
                (
                    model_graphed[0],
                    model_graphed[1],
                    model_graphed[2],
                    relu_control,
                    loss_fn_control,
                ),
                (
                    ({"x": x, "unused_input": unused_input},),
                    (h,),
                    (h2,),
                    (y_pred,),
                    (y_pred, y),
                ),
                allow_unused_input=allow_unused_input,
            )

        real_inputs = [torch.rand_like(x) for _ in range(10)]
        real_targets = [torch.rand_like(y) for _ in range(10)]

        for m, opt, relu, loss_fn in zip(
            (model_graphed, model_control),
            (opt_graphed, opt_control),
            (relu_graphed, relu_control),
            (loss_fn_graphed, loss_fn_control),
        ):
            # Resets RNC states before iterations for graphed and ungraphed models,
            # so dropout math should be bitwise identical for both.
            torch.manual_seed(5)
            torch.cuda.manual_seed(5)
            for data, target in zip(real_inputs, real_targets):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda", enabled=with_amp, cache_enabled=cache_enabled
                ):
                    y_pred = m({"x": data, "unused_input": unused_input})["output"]
                    y_pred = relu(y_pred)
                    loss = loss_fn(y_pred, target)
                    loss.backward()
                opt.step()

        for p, pc in zip(model_graphed.parameters(), model_control.parameters()):
            self.assertEqual(p, pc)

        # We graphed the models in training mode. Eval should still run ungraphed.
        model_graphed.eval()
        model_control.eval()
        self.assertEqual(
            model_graphed({"x": real_inputs[0]}), model_control({"x": real_inputs[0]})
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @parametrize(
        "with_amp,cache_enabled,allow_unused_input",
        [
            subtest((False, False, True), decorators=[skipIfRocm]),
            subtest((True, False, True), decorators=[skipIfRocm]),
            subtest((True, True, True), decorators=[unittest.expectedFailure]),
            subtest((False, False, False), decorators=[skipIfRocm]),
        ],
        name_fn=lambda x, y, z: "{}{}{}".format(
            {True: "with_amp", False: "without_amp"}[x],
            {True: "_cache_enabled", False: "_cache_disabled"}[y] if x else "",
            {True: "_allow_unused_input", False: "_not_allow_unused_input"}[z],
        ),
    )
    @serialTest()
    def test_graph_make_graphed_callables_parameterless_nograd_module(
        self, with_amp, cache_enabled, allow_unused_input
    ):
        torch.manual_seed(5)
        torch.cuda.manual_seed(5)

        N, D_in, H, D_out = 640, 4096, 2048, 1024

        class ParameterlessModule(torch.nn.Module):
            def forward(self, input_dict: dict):
                x = input_dict["x"]
                idx = (
                    torch.arange(x.size(0), device=x.device)
                    .view(-1, 1)
                    .repeat(1, x.size(1))
                )
                return {"output": torch.gather(x, 0, idx)}

        models = []
        for _ in range(2):
            model_section1 = ParameterlessModule().cuda()
            models.append(torch.nn.Sequential(model_section1))

        model_graphed = models[0]
        model_control = models[1]

        model_graphed.load_state_dict(model_control.state_dict())

        x = torch.randn(N, D_in, device="cuda", requires_grad=False)
        unused_input = torch.randn(N, H, device="cuda", requires_grad=False)
        y_pred = torch.randn(N, D_in, device="cuda", requires_grad=False)
        y = torch.randn(N, D_in, device="cuda")

        # This is a good stress test. It graphs four callables: two Modules and two python functions.
        with torch.amp.autocast(
            device_type="cuda", enabled=with_amp, cache_enabled=cache_enabled
        ):
            model_graphed[0] = torch.cuda.make_graphed_callables(
                model_graphed[0],
                ({"x": x, "unused_input": unused_input},),
                allow_unused_input=allow_unused_input,
            )

        real_inputs = [torch.rand_like(x, requires_grad=True) for _ in range(10)]
        real_targets = [torch.rand_like(y) for _ in range(10)]

        for m in (model_graphed, model_control):
            # Resets RNC states before iterations for graphed and ungraphed models,
            # so dropout math should be bitwise identical for both.
            torch.manual_seed(5)
            torch.cuda.manual_seed(5)
            for data, _ in zip(real_inputs, real_targets):
                with torch.amp.autocast(
                    device_type="cuda", enabled=with_amp, cache_enabled=cache_enabled
                ):
                    out = m({"x": data, "unused_input": unused_input})["output"]

        # We graphed the models in training mode. Eval should still run ungraphed.
        model_graphed.eval()
        model_control.eval()
        self.assertEqual(
            model_graphed({"x": real_inputs[0]}), model_control({"x": real_inputs[0]})
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_make_graphed_callables_same_pool(self):
        torch.manual_seed(5)
        torch.cuda.manual_seed(5)
        models = []
        num_models = 3
        for _ in range(num_models):
            models.append(
                torch.nn.Sequential(
                    torch.nn.Linear(32, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                ).cuda()
            )
        # we will reuse the same pool for all graph captures
        mempool = torch.cuda.graph_pool_handle()
        graphed_models = []
        for model in models:
            x = torch.randn([64, 32], device="cuda")
            graphed_model = deepcopy(model)
            graphed_model = torch.cuda.make_graphed_callables(
                graphed_model, (x,), pool=mempool
            )
            graphed_models.append(graphed_model)

        for model, graphed_model in zip(models, graphed_models):
            x = torch.randn([64, 32], device="cuda")
            y = model(x)
            yg = graphed_model(x)
            l = y.norm()
            lg = yg.norm()
            l.backward()
            lg.backward()

            self.assertEqual(y, yg)
            self.assertEqual(l, lg)
            for p, pg in zip(model.parameters(), graphed_model.parameters()):
                self.assertEqual(p, pg)
                self.assertEqual(p.grad, pg.grad)
                self.assertNotEqual(p.data_ptr(), pg.data_ptr())
                self.assertNotEqual(p.grad.data_ptr(), pg.grad.data_ptr())

    def _test_graphed_optimizer(
        self, steps_warmup, steps_train, optimizer_ctor, kwargs
    ):
        for actually_do_graphs in (True, False):
            params = [torch.randn((i + 5, i + 5), device="cuda") for i in range(2)] + [
                torch.randn((), device="cuda")
            ]
            params_control = [p.clone().requires_grad_() for p in params]
            params_graphed = [p.clone().requires_grad_() for p in params]

            grads = [
                [torch.randn_like(p) for p in params]
                for _ in range(steps_warmup + steps_train)
            ]

            # Control (capturable=False)

            opt = optimizer_ctor(params_control, capturable=False, **kwargs)

            for i in range(steps_warmup + steps_train):
                for j, p in enumerate(params_control):
                    p.grad = grads[i][j]
                opt.step()

            # capturable=True

            opt = optimizer_ctor(params_graphed, capturable=True, **kwargs)

            for i in range(steps_warmup):
                for j, p in enumerate(params_graphed):
                    p.grad = grads[i][j]
                opt.step()

            if actually_do_graphs:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    opt.step()

            for i in range(steps_train):
                if actually_do_graphs:
                    for j, p in enumerate(params_graphed):
                        p.grad.copy_(grads[i + steps_warmup][j])
                    g.replay()
                else:
                    # Passing capturable=True to the constructor and running without graphs should still be
                    # numerically correct, even if it's not ideal for performance.
                    for j, p in enumerate(params_graphed):
                        p.grad = grads[i + steps_warmup][j]
                    opt.step()

            for p_control, p_graphed in zip(params_control, params_graphed):
                self.assertEqual(p_control, p_graphed)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_optims_with_explicitly_capturable_param_groups(self):
        # mimicking `_test_graphed_optimizer` maladroitly to pass two param_groups to optimizer.__init__
        n_warmup, n_replay = 3, 2
        for optimizer, second_param_group_capturable in product(
            (
                torch.optim.Adam,
                torch.optim.AdamW,
                torch.optim.ASGD,
                torch.optim.Adamax,
                torch.optim.NAdam,
                torch.optim.RAdam,
                torch.optim.Adadelta,
                torch.optim.RMSprop,
                torch.optim.Rprop,
            ),
            (True, False),
        ):
            ref_p1, param1 = (
                torch.nn.Parameter(torch.ones(1, device="cuda")) for _ in range(2)
            )
            ref_p2, param2 = (
                torch.nn.Parameter(torch.ones(1, device="cuda")) for _ in range(2)
            )
            grads1, grads2 = (
                [torch.randn_like(param1) for _ in range(n_warmup + n_replay)]
                for _ in range(2)
            )
            ref_grads1, ref_grads2 = (
                [t.clone() for t in tensors] for tensors in (grads1, grads2)
            )
            params = [
                {"params": [param1], "capturable": True},
                {"params": [param2], "capturable": second_param_group_capturable},
            ]
            opt = optimizer(params)
            opt_ = optimizer(
                [
                    {"params": [ref_p1], "capturable": False},
                    {"params": [ref_p2], "capturable": False},
                ]
            )

            for i in range(n_warmup + n_replay):
                ref_p1.grad = ref_grads1[i]
                ref_p2.grad = ref_grads2[i]
                opt_.step()

            for i in range(n_warmup):
                param1.grad = grads1[i]
                param2.grad = grads2[i]
                opt.step()

            g = torch.cuda.CUDAGraph()
            if not second_param_group_capturable:
                with self.assertRaisesRegex(RuntimeError, "Attempting CUDA graph"):
                    with torch.cuda.graph(g):
                        opt.step()
            else:
                with torch.cuda.graph(g):
                    opt.step()

                for i in range(n_replay):
                    param1.grad.copy_(grads1[n_warmup + i])
                    param2.grad.copy_(grads2[n_warmup + i])
                    g.replay()
                self.assertEqual(ref_p1, param1)
                self.assertEqual(ref_p2, param2)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_cuda_graph_error_options(self):
        def fn():
            x = torch.zeros([2000], device="cuda")
            y = x + x + x
            return y

        mem = None

        def raw_malloc():
            global mem
            mem = None
            stream = torch.cuda.Stream()
            try:
                with torch.cuda.stream(stream):
                    mem = torch.cuda.caching_allocator_alloc(1024)
            except BaseException:
                if mem is None:
                    return
            try:
                torch.cuda.caching_allocator_delete(mem)
                mem = None
                return None
            except BaseException:
                pass

        def throws_on_cuda_event(capture_error_mode):
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fn()
            stream.synchronize()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            try:
                with torch.cuda.graph(
                    graph, stream=stream, capture_error_mode=capture_error_mode
                ):
                    out = fn()
                    thread = threading.Thread(target=raw_malloc)
                    thread.start()
                    thread.join()
            except Exception:
                if mem is not None:
                    torch.cuda.caching_allocator_delete(mem)
                return True

            return False

        self.assertFalse(throws_on_cuda_event("thread_local"))
        self.assertFalse(throws_on_cuda_event("relaxed"))

        # Exception would Corrupt Process and make other tests fail
        # self.assertTrue(throws_on_cuda_event("global"))

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_cuda_graph_allocator_propagates_stream(self):
        segments = torch.cuda.memory_snapshot()
        existing_pools = {s["segment_pool_id"] for s in segments}
        x = torch.randn(10240000, device="cuda")
        y = torch.rand_like(x)
        g = torch.cuda.CUDAGraph()
        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            g.capture_begin()
            z = x + y
        with torch.cuda.stream(s1):
            s1.wait_stream(s0)
            w = z + y
        s0.wait_stream(s1)
        with torch.cuda.stream(s0):
            g.capture_end()
        segments = torch.cuda.memory_snapshot()
        x = [
            s["segment_pool_id"]
            for s in segments
            if s["segment_pool_id"] not in existing_pools
        ]
        self.assertEqual(len(x), 2)
        self.assertEqual(x[0], x[1])

    def test_batch_norm_gather_stats(self):
        input = torch.randn(1, 3, 3, 3, device="cuda")
        mean, invstd = torch.batch_norm_gather_stats(
            input,
            mean=torch.ones(2, 3, device="cuda"),
            invstd=torch.ones(2, 3, device="cuda"),
            running_mean=None,
            running_var=None,
            momentum=0.1,
            eps=1e-5,
            count=2,
        )
        self.assertEqual(mean, torch.ones(3, device="cuda"))
        self.assertEqual(invstd, torch.ones(3, device="cuda"))

    def test_matmul_memory_use(self):
        def get_max_used():
            torch.cuda.synchronize()
            val = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            return val

        a = torch.rand(1, 32, 32, device="cuda")
        b = torch.rand(24, 32, 1, device="cuda")

        get_max_used()

        torch.matmul(a, b)

        matmul_mem = get_max_used()

        a = a.expand(24, 32, 32)
        torch.matmul(a, b)

        matmul_expand_mem = get_max_used()

        torch.bmm(a, b)

        bmm_mem = get_max_used()

        self.assertEqual(matmul_expand_mem, matmul_mem)
        self.assertEqual(bmm_mem, matmul_mem)

    @unittest.skipIf(not TEST_WITH_ROCM, "ROCm-only test")
    def test_rocm_backward_pass_guard(self):
        # The test exercises a ROCm-specific feature.

        class MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor, constant):
                self.assertFalse(torch._C._rocm_is_backward_pass())
                ctx.constant = constant
                return tensor * constant

            @staticmethod
            def backward(ctx, grad_output):
                self.assertTrue(torch._C._rocm_is_backward_pass())
                return grad_output * ctx.constant, None

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.nn.Parameter(torch.randn(()))

            def forward(self, x):
                return MyFunction.apply(x, self.a)

        model = MyModule()
        criterion = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

        x = torch.randn(5, 5)
        result = model(x)
        loss = criterion(result, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_matmul_device_mismatch(self):
        cpu = torch.rand((10, 10))
        cuda = cpu.cuda()
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            cpu @ cuda
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            cuda @ cpu

        for s, m1, m2 in product((cpu, cuda), repeat=3):
            if s.device == m1.device == m2.device:
                torch.addmm(s, m1, m2)
            else:
                with self.assertRaisesRegex(
                    RuntimeError, "Expected all tensors to be on the same device"
                ):
                    torch.addmm(s, m1, m2)

    @unittest.skipIf(TEST_MULTIGPU, "Testing on one GPU is sufficient")
    def test_lazy_init(self):
        """Validate that no CUDA calls are made during `import torch` call"""

        def check_output(script: str) -> str:
            return (
                subprocess.check_output([sys.executable, "-c", script])
                .decode("ascii")
                .strip()
            )

        VISIBLE_DEVICES = (
            "HIP_VISIBLE_DEVICES" if TEST_WITH_ROCM else "CUDA_VISIBLE_DEVICES"
        )
        test_script = f"import os; import torch;os.environ['{VISIBLE_DEVICES}']='32';print(torch.cuda.device_count())"
        rc = check_output(test_script)
        self.assertEqual(rc, "0")
        if not TEST_WITH_ROCM:
            # Check that `cuInit` was not called during the import
            # By using ctypes and calling cuDeviceCountGet() and expect CUDA_ERROR_NOT_INITIALIZED == 3
            # See https://github.com/pytorch/pytorch/issues/116276 for more details
            libcuda_name = "libcuda.so.1" if not IS_WINDOWS else "nvcuda.dll"
            cuda_driver_api_call = (
                f"ctypes.CDLL('{libcuda_name}').cuDeviceGetCount(ctypes.byref(x))"
            )
            rc = check_output(
                f"import torch; import ctypes;x=ctypes.c_int(-1);print({cuda_driver_api_call})"
            )
            self.assertEqual(rc, "3")

    @unittest.skipIf(not TEST_WITH_ROCM, "not relevant for CUDA testing")
    def test_hip_device_count(self):
        """Validate device_count works with both CUDA/HIP visible devices"""
        test_script = """\
import torch
import os
print(f"{torch.cuda.device_count()}")
"""
        custom_envs = [
            {"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": None},
            {"CUDA_VISIBLE_DEVICES": None, "HIP_VISIBLE_DEVICES": "0"},
            {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "HIP_VISIBLE_DEVICES": "0"},
        ]

        for env_config in custom_envs:
            env = os.environ.copy()
            for key, value in env_config.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            r = (
                subprocess.check_output([sys.executable, "-c", test_script], env=env)
                .decode("ascii")
                .strip()
            )
            self.assertEqual("1", r)

    @unittest.skipIf(not TEST_MULTIGPU, "requires multiple devices")
    def test_device_count_not_cached_pre_init(self):
        visible_devices = (
            "HIP_VISIBLE_DEVICES" if torch.version.hip else "CUDA_VISIBLE_DEVICES"
        )
        test_script = f"""\
import torch
import os
r1 = torch.cuda.device_count()
os.environ['{visible_devices}'] = '0'
r2 = torch.cuda.device_count()
torch.empty(10, device='cuda')
print(f"{{r1}}, {{r2}}")
"""

        r = (
            subprocess.check_output([sys.executable, "-c", test_script])
            .decode("ascii")
            .strip()
        )

        x = torch.cuda.device_count()
        self.assertEqual(f"{x}, 1", r)

    @unittest.skip("Disabling as USE_CUFILE=0 by default in builds")
    def test_gds_fails_in_ci(self):
        if IS_WINDOWS or TEST_WITH_ROCM:
            error_msg = "is not supported on this platform"
        else:
            error_msg = "cuFileHandleRegister failed"
        with TemporaryFileName() as f:
            with self.assertRaisesRegex(RuntimeError, error_msg):
                file = torch.cuda.gds._GdsFile(f, os.O_CREAT | os.O_RDWR)

    def _test_copy(self, x, non_blocking):
        # Perform the copy operation, either blocking or non-blocking
        event = torch.cuda.Event()
        x_gpu = x.to(device="cuda", non_blocking=non_blocking)
        event.record()

        if non_blocking:
            event.synchronize()

        self.assertEqual(x, x_gpu.cpu())

    def test_1d_copy(self):
        # Contiguous 1D tensor
        x = torch.ones(10000000, dtype=torch.uint8)
        self._test_copy(x, non_blocking=True)
        self._test_copy(x, non_blocking=False)
        # Discontiguous 1D tensor
        x = torch.ones(1000000, dtype=torch.uint8)[::2]
        self.assertFalse(x.is_contiguous())
        self._test_copy(x, non_blocking=True)
        self._test_copy(x, non_blocking=False)

    def test_2d_copy(self):
        rows, cols = 1000, 1000
        # Contiguous 2D tensor
        x = torch.ones((rows, cols), dtype=torch.float32)
        self._test_copy(x, non_blocking=True)
        self._test_copy(x, non_blocking=False)
        # Discontiguous 2D tensor
        x = torch.randn(rows, cols)[:, :512]
        self.assertFalse(x.is_contiguous())
        self._test_copy(x, non_blocking=True)
        self._test_copy(x, non_blocking=False)


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCudaMallocAsync(TestCase):
    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_memory_snapshot(self):
        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history("state", stacks="python")
            # make x the second block in a segment
            torch.rand(2 * 311, 411, device="cuda")
            unused = torch.rand(310, 410, device="cuda")
            x = torch.rand(311, 411, device="cuda")

            # create a bunch of tensors that all will tile into the
            # same segment to  exercise the history merging code
            # 512B is the minimum block size,
            # so we allocate all the tensors to this size to make sure
            # they tile evenly
            tensors = [torch.rand(128, device="cuda") for _ in range(1000)]
            while tensors:
                del tensors[randint(0, len(tensors) - 1)]

            # exercise the history trimming code
            torch.rand(128 * 5, device="cuda")

            ss = torch.cuda.memory._snapshot()
            found_it = False
            for seg in ss["segments"]:
                self.assertTrue("frames" in seg)
                for b in seg["blocks"]:
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertTrue("test_cuda" in b["frames"][0]["filename"])
                        found_it = True
                        self.assertEqual(x.untyped_storage().data_ptr(), b["address"])
            self.assertTrue(found_it)

            if not IS_WINDOWS:
                with tempfile.NamedTemporaryFile() as f:
                    torch.cuda.memory._save_segment_usage(f.name)
                    with open(f.name) as f2:
                        self.assertTrue("test_cuda.py" in f2.read())
            del unused
            del x
            torch.cuda.empty_cache()
            ss = torch.cuda.memory._snapshot()
            self.assertTrue(
                ss["device_traces"][0][-1]["action"]
                in ("segment_free", "segment_unmap")
            )

        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "x86 linux only cpp unwinding")
    def test_direct_traceback(self):
        from torch._C._profiler import gather_traceback, symbolize_tracebacks  # @manual

        c = gather_traceback(True, True, True)
        (r,) = symbolize_tracebacks([c])
        r = str(r)
        self.assertTrue("test_cuda.py" in r)
        self.assertTrue("unwind" in r)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    def test_memory_snapshot_with_cpp(self):
        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history("state", stacks="all")
            x = torch.rand(311, 411, device="cuda")

            ss = torch.cuda.memory._snapshot()["segments"]
            found_it = False
            for seg in ss:
                for b in seg["blocks"]:
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertTrue("::rand" in str(b["frames"]))
                        found_it = True
            self.assertTrue(found_it)

        finally:
            torch.cuda.memory._record_memory_history(None)

    @skipIfRocm
    def test_memory_profiler_viz(self):
        with torch.profiler.profile(
            with_stack=True, profile_memory=True, record_shapes=True
        ) as prof:
            x = torch.rand(128, 128, device="cuda")
            x * x + x * x
        plot = profile_plot(prof)
        plot = json.dumps(_profile_to_snapshot(prof))
        self.assertTrue("test_cuda.py" in plot)
        self.assertTrue("test_memory_profiler_viz" in plot)
        self.assertTrue("category" in plot)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    def test_cycles(self):
        fired = False

        def observer(html):
            nonlocal fired
            fired = True
            self.assertTrue("torch.Tensor" in html)
            self.assertTrue("test_cuda" in html)
            self.assertTrue("cell_contents" in html)

        disarm = observe_tensor_cycles(observer)

        def noop():
            pass

        try:

            def create():
                x = torch.empty(3, 4, device="cuda")

                def foo(p):
                    if p:
                        return foo(not p)
                    else:
                        return x

                return foo

            create()
            gc.collect()
            # the callback has to run outside of the collect
            # call so it doesn't actual fire until the next
            # method call after a gc.collect
            noop()
            self.assertTrue(fired)
        finally:
            disarm()

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    def test_memory_plots(self):
        for context, stacks in (
            ("all", "all" if IS_LINUX else "python"),
            ("all", "python"),
            (None, "python"),
        ):
            try:
                torch.cuda.memory.empty_cache()
                torch.cuda.memory._record_memory_history(
                    "all", context=context, stacks=stacks
                )

                def run():
                    x = torch.rand(128, 128, device="cuda")
                    x * x + x * x

                run()
                cpp = stacks == "all"
                record_context = context is not None
                ss = torch.cuda.memory._snapshot()

                tplot = trace_plot(ss)
                splot = segment_plot(ss)
                text = json.dumps(ss)

                self.assertTrue(record_context == ("test_memory_plots" in text))
                self.assertTrue(cpp == ("::rand" in text))
                self.assertTrue(str(128 * 128 * 4) in text)

            finally:
                torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    def test_memory_plots_free_stack(self):
        for context in ["alloc", "all", "state"]:
            try:
                torch.cuda.memory.empty_cache()
                torch.cuda.memory._record_memory_history(context=context)
                x = None

                def thealloc():
                    nonlocal x
                    x = torch.rand(3, 4, device="cuda")

                def thefree():
                    nonlocal x
                    del x

                thealloc()
                thefree()
                ss = json.dumps(torch.cuda.memory._snapshot())
                self.assertEqual(("thefree" in ss), (context == "all"))
                self.assertEqual(("thealloc" in ss), (context != "state"))
            finally:
                torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    def test_memory_plots_history_context(self):
        try:
            torch.cuda.memory.empty_cache()
            x = None

            def should_capture1():
                nonlocal x
                x = torch.rand(4, 4, device="cuda")

            def should_not_capture():
                nonlocal x
                x = torch.rand(3, 4, device="cuda")

            def should_capture2():
                nonlocal x
                x = torch.rand(4, 4, device="cuda")

            # Recording with context and python call stacks should capture the call stack.
            torch.cuda.memory._record_memory_history(context="all", stacks="python")
            should_capture1()
            # Recording with context=None should not capture the call stack.
            torch.cuda.memory._record_memory_history(context=None)
            should_not_capture()
            # Recording with context and python call stacks should capture the call stack.
            torch.cuda.memory._record_memory_history(context="all", stacks="python")
            should_capture2()

            ss = json.dumps(torch.cuda.memory._snapshot())
            self.assertTrue("should_capture1" in ss)
            self.assertTrue("should_not_capture" not in ss)
            self.assertTrue("should_capture2" in ss)
        finally:
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    def test_memory_plots_free_segment_stack(self):
        for context in ["alloc", "all", "state"]:
            try:
                torch.cuda.memory.empty_cache()
                torch.cuda.memory._record_memory_history(context=context)
                x = torch.rand(3, 4, device="cuda")
                del x
                torch.cuda.memory.empty_cache()

                ss = json.dumps(torch.cuda.memory._snapshot())
                self.assertEqual(("empty_cache" in ss), (context == "all"))
            finally:
                torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_memory_snapshot_script(self):
        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history("state", stacks="python")

            @torch.jit.script
            def foo():
                return torch.rand(311, 411, device="cuda")

            x = foo()

            ss = torch.cuda.memory._snapshot()["segments"]
            found_it = False
            for seg in ss:
                for b in seg["blocks"]:
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertEqual(b["frames"][0]["name"], "foo")
                        found_it = True
            self.assertTrue(found_it)

        finally:
            torch.cuda.memory._record_memory_history(None)

    def test_max_split_expandable(self):
        try:
            torch.cuda.memory.empty_cache()
            mb = 1024 * 1024
            _, all_memory = torch.cuda.memory.mem_get_info()
            pre_reserved = torch.cuda.memory_reserved()
            total_allowed = 120 * mb + pre_reserved
            fraction_allowed = total_allowed / all_memory
            self.assertEqual(int(fraction_allowed * all_memory), total_allowed)
            torch.cuda.memory.set_per_process_memory_fraction(fraction_allowed)

            def alloc(n):
                return torch.ones(n * mb, dtype=torch.int8, device="cuda")

            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:False,max_split_size_mb:40"
            )
            a = alloc(40)
            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:True,max_split_size_mb:40"
            )
            b = alloc(40)
            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:False,max_split_size_mb:40"
            )
            c = alloc(40)
            with self.assertRaises(torch.OutOfMemoryError):
                alloc(40)
            del a, b, c
            # force release_cached_blocks to run with some expandable segments in the free list
            alloc(120)
        finally:
            torch.cuda.memory.set_per_process_memory_fraction(1.0)

    def test_garbage_collect_expandable(self):
        try:
            torch.cuda.memory.empty_cache()
            mb = 1024 * 1024
            _, all_memory = torch.cuda.memory.mem_get_info()
            pre_reserved = torch.cuda.memory_reserved()
            total_allowed = 120 * mb + pre_reserved
            fraction_allowed = total_allowed / all_memory
            self.assertEqual((fraction_allowed * all_memory), total_allowed)
            torch.cuda.memory.set_per_process_memory_fraction(fraction_allowed)

            def alloc(n):
                return torch.ones(n * mb, dtype=torch.int8, device="cuda")

            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:False,garbage_collection_threshold:0.5"
            )
            a = alloc(40)
            torch.cuda.memory._set_allocator_settings(
                "expandable_segments:True,garbage_collection_threshold:0.5"
            )
            b = alloc(40)
            del a, b
            # causes GC to run. The expandable segment block will be split
            # so GC would not attempt to free it anyway, but this at least makes sure
            # expandable_segment blocks can be in the free list when this is called.
            alloc(80)
        finally:
            torch.cuda.memory.set_per_process_memory_fraction(1.0)

    def test_allocator_settings(self):
        def power2_div(size, div_factor):
            pow2 = 1
            while pow2 < size:
                pow2 = pow2 * 2
            if pow2 == size:
                return pow2
            step = pow2 / 2 / div_factor
            ret = pow2 / 2
            while ret < size:
                ret = ret + step
            return ret

        torch.cuda.memory.empty_cache()
        key_allocated = (
            "active_bytes.all.allocated"
            if not TEST_CUDAMALLOCASYNC
            else "allocated_bytes.all.current"
        )
        key_requested = "requested_bytes.all.allocated"

        nelems = 21 * 1024 * 1024
        nbytes = 4 * nelems  # floats are 4 bytes

        nelems_big = 100 * 1024 * 1024
        nbytes_big = 4 * nelems_big  # floats are 4 bytes

        start_mem = torch.cuda.memory_stats()[key_allocated]
        torch.cuda.memory._set_allocator_settings("")
        x = torch.rand(nelems, device="cuda")

        # test roundup_power2_divisions single value syntax
        reg_mem = torch.cuda.memory_stats()[key_allocated]
        start_requested = torch.cuda.memory_stats()[key_requested]
        torch.cuda.memory._set_allocator_settings("roundup_power2_divisions:4")
        y = torch.rand(nelems, device="cuda")

        pow2_div4_mem = torch.cuda.memory_stats()[key_allocated]
        current_requested = torch.cuda.memory_stats()[key_requested]

        self.assertEqual(reg_mem - start_mem, nbytes)
        if not TEST_CUDAMALLOCASYNC:
            # not supported with the cudaMallocAsync backend
            self.assertEqual(pow2_div4_mem - reg_mem, power2_div(nbytes, 4))
            self.assertEqual(current_requested - start_requested, nbytes)

        torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.5")
        torch.cuda.memory._set_allocator_settings(
            "garbage_collection_threshold:0.5,max_split_size_mb:40"
        )

        # should have reset the power2 divisions now
        torch.cuda.memory.empty_cache()
        start_mem = torch.cuda.memory_stats()[key_allocated]
        z = torch.rand(nelems, device="cuda")
        reg_mem = torch.cuda.memory_stats()[key_allocated]
        self.assertEqual(reg_mem - start_mem, nbytes)

        # roundup_power2_divisions knob array syntax
        torch.cuda.memory.empty_cache()
        torch.cuda.memory._set_allocator_settings(
            "garbage_collection_threshold:0.5,roundup_power2_divisions:[64:8,128:2,256:2,512:2,1024:1,>:1]"
        )
        start_mem = torch.cuda.memory_stats()[key_allocated]
        w = torch.rand(nelems, device="cuda")

        pow2_div8_mem = torch.cuda.memory_stats()[key_allocated]
        if not TEST_CUDAMALLOCASYNC:
            # not supported with the cudaMallocAsync backend
            self.assertEqual(pow2_div8_mem - start_mem, power2_div(nbytes, 8))

        torch.cuda.memory.empty_cache()
        start_mem = torch.cuda.memory_stats()[key_allocated]
        v = torch.rand(nelems_big, device="cuda")

        pow2_div2_mem = torch.cuda.memory_stats()[key_allocated]
        if not TEST_CUDAMALLOCASYNC:
            # not supported with the cudaMallocAsync backend
            self.assertEqual(pow2_div2_mem - start_mem, power2_div(nbytes_big, 2))

        torch.cuda.memory.empty_cache()
        torch.cuda.memory._set_allocator_settings("release_lock_on_cudamalloc:True")
        start_mem = torch.cuda.memory_stats()[key_allocated]
        w = torch.rand(nelems, device="cuda")
        reg_mem = torch.cuda.memory_stats()[key_allocated]
        self.assertEqual(reg_mem - start_mem, nbytes)

        with self.assertRaises(RuntimeError):
            torch.cuda.memory._set_allocator_settings("foo:1,bar:2")

        with self.assertRaises(RuntimeError):
            torch.cuda.memory._set_allocator_settings(
                "garbage_collection_threshold:1.2"
            )

        with self.assertRaises(RuntimeError):
            torch.cuda.memory._set_allocator_settings("max_split_size_mb:2")

        with self.assertRaises(RuntimeError):
            torch.cuda.memory._set_allocator_settings("release_lock_on_cudamalloc:none")

        with self.assertRaises(RuntimeError):
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:none"
            )

        with self.assertRaises(RuntimeError):
            torch.cuda.memory._set_allocator_settings(
                "pinned_num_register_threads:none"
            )

        with self.assertRaises(RuntimeError):
            torch.cuda.memory._set_allocator_settings(
                "pinned_num_register_threads:1024"
            )

    def test_cachingAllocator_raw_alloc(self):
        # Test that raw_alloc respects the setting that
        # activates/deactivates the caching allocator

        # Helper function that calls raw_alloc and returns
        # relevant field in data structure
        def requested_bytes_alloc_stats(raw_alloc_size, stream):
            start = torch.cuda.memory_stats()["requested_bytes.all.allocated"]
            torch._C._cuda_cudaCachingAllocator_raw_alloc(raw_alloc_size, stream)
            finish = torch.cuda.memory_stats()["requested_bytes.all.allocated"]
            return finish - start

        torch.cuda.empty_cache()
        device = torch._C._cuda_getDevice()
        stream = torch._C._cuda_getCurrentRawStream(device)
        torch._C._cuda_resetAccumulatedMemoryStats(device)

        # size of allocation
        raw_alloc_size = 1024 * 1024  # 1 MB

        try:
            # Deactivate the caching allocator
            torch.cuda.caching_allocator_enable(False)

            # For a deactivated caching allocator, result is zero
            cuda_alloc_size = requested_bytes_alloc_stats(raw_alloc_size, stream)
            self.assertEqual(cuda_alloc_size, 0)

        finally:
            # Make sure we get back to the default state that is
            # an activated caching allocator
            torch.cuda.caching_allocator_enable(True)

            # For an active caching allocator, result matches raw_alloc_size
            cuda_alloc_size = requested_bytes_alloc_stats(raw_alloc_size, stream)
            self.assertEqual(cuda_alloc_size, raw_alloc_size)

    @parametrize("max_split_size_mb_setting", [False, True])
    def test_raises_oom(self, max_split_size_mb_setting):
        if max_split_size_mb_setting:
            # CudaCachingAllocator does early return when searching available blocks
            # if max_split_size_mb is not set
            # Setting this triggers more parts of the code
            torch.cuda.memory._set_allocator_settings("max_split_size_mb:1024")
            torch.cuda.memory.empty_cache()
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            torch.empty(1024 * 1024 * 1024 * 1024, device="cuda")

    @unittest.skipIf(
        not (IS_LINUX and os.uname().machine == "x86_64"), "cpp traces only on linux"
    )
    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    def test_cpp_memory_snapshot_pickle(self):
        from torch.utils.cpp_extension import load_inline

        source = """
        #include <torch/csrc/cuda/memory_snapshot.h>
        py::object do_snapshot() {
            std::string data = torch::cuda::_memory_snapshot_pickled();
            return py::bytes(data);
        }
        void record(bool e, bool ctx) {
            torch::cuda::_record_memory_history(e, ctx, 10, ctx, ctx);
        }
        """
        m = load_inline(
            name="snapshot", cpp_sources=[source], functions=["do_snapshot", "record"]
        )
        for ctx in (False, True):
            try:
                m.record(True, ctx)

                @torch.jit.script
                def the_script_fn():
                    return torch.rand(311, 411, device="cuda")

                def run():
                    t = the_script_fn()
                    return pickle.loads(m.do_snapshot())

                mem = run()
                found = False
                for s in mem["segments"]:
                    for b in s["blocks"]:
                        if b["state"] == "active_allocated":
                            if b["requested_size"] == 311 * 411 * 4:
                                if ctx:
                                    frame_text = str(b["frames"])
                                    # C++ frame
                                    self.assertTrue("::rand" in frame_text)
                                    # script frame
                                    self.assertTrue("the_script_fn" in frame_text)
                                    # python frame
                                    self.assertTrue("case.py" in frame_text)
                                found = True
                last_action = mem["device_traces"][0][-1]
                self.assertEqual(last_action["action"], "alloc")
                self.assertEqual(last_action["size"], 311 * 411 * 4)
                self.assertTrue(found)
            finally:
                m.record(False, False)

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled")
    def test_notifies_oom(self):
        x = False

        def cb(device, alloc, device_alloc, device_free):
            nonlocal x
            x = True

        torch._C._cuda_attach_out_of_memory_observer(cb)
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            torch.empty(1024 * 1024 * 1024 * 1024, device="cuda")
        self.assertTrue(x)

    def test_allocator_fuzz(self):
        # fuzz
        state = random.getstate()
        random.seed(123)
        N = 10000
        try:
            mem = []
            total = 0
            c = 0

            def alloc():
                nonlocal total, c
                b = random.randrange(2 * 1024 * 1024 // 4, 20 * 1024 * 1024 // 4)
                mem.append((c, torch.full((b,), c, dtype=torch.int32, device="cuda")))
                c += 1
                total += b

            def free():
                nonlocal total
                idx = random.randrange(0, len(mem))
                v, x = mem.pop(idx)
                self.assertTrue(torch.all(v == x))
                total -= x.numel()

            choices = [alloc, free, torch.cuda.memory.empty_cache]
            for i in range(N):
                while total >= 1024 * 1024 * 1024 / (4 * 10):
                    free()
                (action,) = random.choices(choices, weights=[1, 1 if mem else 0, 0.1])
                action()
        finally:
            random.setstate(state)

    @unittest.skipIf(TEST_PYNVML, "pynvml/amdsmi is not available")
    def test_nvml_get_handler(self):
        if not torch.version.hip:
            self.assertTrue(torch.cuda._get_pynvml_handler() is not None)
        else:
            self.assertTrue(torch.cuda._get_amdsmi_handler() is not None)

    @unittest.skipIf(TEST_PYNVML, "pynvml/amdsmi is not available")
    def test_temperature(self):
        self.assertTrue(0 <= torch.cuda.temperature() <= 150)

    @unittest.skipIf(TEST_WITH_ROCM, "flaky for AMD gpu")
    @unittest.skipIf(TEST_PYNVML, "pynvml/amdsmi is not available")
    def test_device_memory_used(self):
        """
        Verify used device memory in bytes
        """
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        a = torch.cuda.device_memory_used()
        num_bytes = 512 * 1024**2
        _ = torch.empty(num_bytes, dtype=torch.int8, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        b = torch.cuda.device_memory_used()
        mem_bytes = b - a
        # test the order of magnitude
        self.assertTrue(num_bytes // 32 <= mem_bytes <= num_bytes * 32)

    @unittest.skipIf(TEST_PYNVML, "pynvml/amdsmi is not available")
    def test_power_draw(self):
        self.assertTrue(torch.cuda.power_draw() >= 0)

    @unittest.skipIf(TEST_PYNVML, "pynvml/amdsmi is not available")
    def test_clock_speed(self):
        self.assertTrue(torch.cuda.clock_rate() >= 0)

    @unittest.skipIf(TEST_PYNVML, "pynvml/amdsmi is not available")
    @unittest.skipIf(not TEST_WITH_ROCM, "amdsmi specific test")
    def test_raw_amdsmi_device_count(self):
        """
        This unit test will verify if the number of GPUs shown in `amd-smi
        list` is equivalent to the count returned by `_raw_device_count_amdsmi`.
        This should be unaffected by visible device settings.
        """
        raw_device_cnt = int(
            subprocess.check_output(
                "amd-smi list | grep 'GPU' | wc -l", shell=True
            ).strip()
        )
        self.assertEqual(torch.cuda._raw_device_count_amdsmi(), raw_device_cnt)

    @unittest.skipIf(TEST_PYNVML, "pynvml/amdsmi is not available")
    @unittest.skipIf(not TEST_WITH_ROCM, "amdsmi specific test")
    def test_raw_amdsmi_device_uuids(self):
        """
        This unit test will extract a list of UUIDs for each GPU using
        rocminfo information, and check whether each UUID is present in
        the output from `_raw_device_uuid_amdsmi` this allows us to test
        that the pytorch call is returning a correct list of UUIDs.
        """
        cmd = "rocminfo | grep -o 'Uuid:.*GPU-.*' | sed 's/Uuid:.*GPU-//'"
        uuids = (
            subprocess.check_output(cmd, shell=True, universal_newlines=True)
            .strip()
            .split("\n")
        )
        uuids = [s.strip() for s in uuids]
        raw_uuids = torch.cuda._raw_device_uuid_amdsmi()
        for uuid in uuids:
            matching = True
            if not any(uuid in raw_id for raw_id in raw_uuids):
                matching = False
        self.assertEqual(True, matching)

    @unittest.skipIf(TEST_PYNVML, "pynvml/amdsmi is not available")
    @unittest.skipIf(not TEST_WITH_ROCM, "amdsmi specific test")
    def test_uuid_visible_devices(self):
        """
        This unit test will simulate an environment where a UUID is passed
        via CUDA/HIP_VISIBLE_DEVICES and ensure that the correct device count
        is returned. This allows us to test that the visible device functionality
        is operating as expected.
        """
        test_script = """\
import torch
import os
print(f"{torch.cuda.device_count()}")
        """
        cmd = "rocminfo | grep -o 'Uuid:.*GPU-.*' | sed 's/Uuid://'"
        uuids = (
            subprocess.check_output(cmd, shell=True, universal_newlines=True)
            .strip()
            .split("\n")
        )
        uuids = [s.strip() for s in uuids]

        custom_envs = []
        for uuid in uuids:
            custom_envs.append(
                {"CUDA_VISIBLE_DEVICES": f"{uuid}", "HIP_VISIBLE_DEVICES": None}
            )
            custom_envs.append(
                {"HIP_VISIBLE_DEVICES": f"{uuid}", "CUDA_VISIBLE_DEVICES": None}
            )

        for env_config in custom_envs:
            env = os.environ.copy()
            for key, value in env_config.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            r = (
                subprocess.check_output([sys.executable, "-c", test_script], env=env)
                .decode("ascii")
                .strip()
            )
            self.assertEqual("1", r)


MIN_BLOCK_SIZE = 512
SMALL_SIZE = 1048576
SMALL_BUFFER = 2097152
LARGE_BUFFER = 20971520


def get_cudagraph_segments(pool_id):
    segments = torch.cuda.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] == pool_id]


def get_all_cudagraph_segments():
    segments = torch.cuda.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] != (0, 0)]


def cudagraphify(fn, inputs, pool=None):
    if not TEST_CUDA_GRAPH:
        raise unittest.SkipTest("cuda graph test is skipped")

    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=pool):
        static_outputs = fn(*inputs)

    return graph, static_outputs


def int8_cuda(size):
    return torch.ones([size], device="cuda", dtype=torch.uint8)


def live_blocks(pool_id):
    blocks = 0
    seg = get_cudagraph_segments(pool_id)
    for segment in get_cudagraph_segments(pool_id):
        for block in segment["blocks"]:
            blocks += block["state"] == "active_allocated"
    return blocks


def tensor_metadata(x):
    return {
        "nbytes": x.untyped_storage().nbytes(),
        "data_ptr": x.untyped_storage().data_ptr(),
        "size": x.shape,
        "stride": x.stride(),
        "dtype": x.dtype,
        "device": x.device,
        "storage_offset": x.storage_offset(),
    }


def reconstruct_from_tensor_metadata(metadata):
    s = torch._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"]
    )
    t = torch.empty([0], device=metadata["device"], dtype=metadata["dtype"])
    t.set_(
        source=s,
        storage_offset=metadata["storage_offset"],
        size=metadata["size"],
        stride=metadata["stride"],
    )
    return t


@unittest.skipIf(not TEST_CUDA or TEST_CUDAMALLOCASYNC or TEST_WITH_ROCM, "NYI")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestBlockStateAbsorption(TestCase):
    @property
    def expandable_segments(self):
        return EXPANDABLE_SEGMENTS

    def checkCheckpointedBlock(self, before_block, after_block):
        for field in ("size", "state"):
            self.assertEqual(before_block[field], after_block[field])

    def checkCheckpointedState(self, before_segments, after_segments):
        # after may contain additional segments, but all of the segments in before
        # should be exactly equivalent to after
        after_ptr_to_segment = {
            segment["address"]: segment for segment in after_segments
        }

        for before_segment in before_segments:
            self.assertTrue(before_segment["address"] in after_ptr_to_segment)
            after_segment = after_ptr_to_segment[before_segment["address"]]

            for field in (
                "device",
                "total_size",
                "allocated_size",
                "active_size",
                "segment_type",
                "segment_pool_id",
            ):
                self.assertEqual(before_segment[field], after_segment[field])

            self.assertEqual(
                len(before_segment["blocks"]), len(after_segment["blocks"])
            )
            for before_block, after_block in zip(
                before_segment["blocks"], after_segment["blocks"]
            ):
                self.checkCheckpointedBlock(before_block, after_block)

    @staticmethod
    def setCheckpointPoolState(
        device, state, stale_storages_ptr, storages_deleters=None
    ):
        stale_storages_ptr = [t.untyped_storage()._cdata for t in stale_storages_ptr]
        storages_deleters = (
            []
            if not storages_deleters
            else [t.untyped_storage()._cdata for t in storages_deleters]
        )
        torch._C._cuda_setCheckpointPoolState(
            device, state, stale_storages_ptr, storages_deleters
        )

    def checkFunction(self, fn, inputs, pool=None):
        graph, outputs = cudagraphify(fn, inputs, pool=pool)

        pool_id = graph.pool()
        device = outputs[0].device.index

        segments_before_checkpoint = get_cudagraph_segments(pool_id)

        state = torch._C._cuda_getCheckpointState(device, pool_id)
        self.setCheckpointPoolState(device, state, [], [])

        self.checkCheckpointedState(
            segments_before_checkpoint, get_cudagraph_segments(pool_id)
        )

    def setUp(self):
        super().setUp()
        self.segment_length = len(get_all_cudagraph_segments())

    def tearDown(self):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        self.assertEqual(len(get_all_cudagraph_segments()), self.segment_length)

        super().tearDown()

    def test_simple(self):
        def foo():
            x = torch.zeros([SMALL_SIZE * 8], device="cuda", dtype=torch.uint8)
            x = x + x
            x1 = int8_cuda(SMALL_SIZE) + int8_cuda(SMALL_SIZE) + int8_cuda(SMALL_SIZE)
            y = int8_cuda(SMALL_SIZE) + x1
            z = int8_cuda(SMALL_SIZE)
            return x, y, z

        self.checkFunction(foo, [])

    def test_allocated_in_middle_of_segment(self):
        def foo():
            small_buffers = [int8_cuda(MIN_BLOCK_SIZE) for _ in range(11)]
            return small_buffers[5].add_(2)

        self.checkFunction(foo, [])

    def test_multiple_middle_allocations(self):
        def foo():
            small_buffers = [int8_cuda(MIN_BLOCK_SIZE) for _ in range(11)]
            return small_buffers[5], small_buffers[8]

        self.checkFunction(foo, [])

    def test_middle_allocations_contiguous(self):
        def foo():
            small_buffers = [int8_cuda(MIN_BLOCK_SIZE) for _ in range(11)]
            return small_buffers[5], small_buffers[6]

        self.checkFunction(foo, [])

    def test_additional_free_following_checkpoint(self):
        def foo():
            return (int8_cuda(MIN_BLOCK_SIZE),)

        def foo2():
            return (int8_cuda(MIN_BLOCK_SIZE),)

        graph, outputs = cudagraphify(foo, [])
        pool_id = graph.pool()

        segments_before_checkpoint = get_cudagraph_segments(pool_id)

        state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)

        graph2, outputs2 = cudagraphify(foo2, [], pool=graph.pool())

        self.setCheckpointPoolState(outputs[0].device.index, state, outputs2, [])

        del outputs2

        self.checkCheckpointedState(
            segments_before_checkpoint, get_cudagraph_segments(pool_id)
        )

    # TODO: re-enable
    # def test_additional_free_error(self):
    #     def foo():
    #         return int8_cuda(MIN_BLOCK_SIZE),

    #     def foo2():
    #         return int8_cuda(MIN_BLOCK_SIZE),

    #     graph, outputs = cudagraphify(foo, [])
    #     pool_id = graph.pool()

    #     segments_before_checkpoint = get_cudagraph_segments(pool_id)

    #     state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)

    # graph2, outputs2 = cudagraphify(foo2, [], pool=graph.pool())
    # with self.assertRaisesRegex(Exception, "being manually freed must be passed"):
    #     self.setCheckpointPoolState(outputs[0].device.index, state, [], [])

    def test_tensor_dies_after_checkpoint(self):
        def foo():
            return int8_cuda(MIN_BLOCK_SIZE), int8_cuda(MIN_BLOCK_SIZE)

        graph, outputs = cudagraphify(foo, [])
        pool_id = graph.pool()
        device = outputs[0].device.index

        segments_before_checkpoint = get_cudagraph_segments(pool_id)
        state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)

        output_data_ptrs = [output.data_ptr() for output in outputs]

        del outputs

        self.setCheckpointPoolState(device, state, [], [])

        self.assertEqual(live_blocks(pool_id), 2)
        torch._C._cuda_cudaCachingAllocator_raw_delete(output_data_ptrs[0])
        self.assertEqual(live_blocks(pool_id), 1)
        torch._C._cuda_cudaCachingAllocator_raw_delete(output_data_ptrs[1])
        self.assertEqual(live_blocks(pool_id), 0)

    def test_assigning_back_deleter_fns_to_tensor(self):
        def foo(x):
            return (
                int8_cuda(SMALL_BUFFER) + x,
                int8_cuda(SMALL_BUFFER) + x,
                int8_cuda(LARGE_BUFFER) + x,
            )

        inp = torch.tensor([1], device="cuda")
        graph, outputs = cudagraphify(foo, [inp])
        pool_id = graph.pool()
        graph.replay()

        device = outputs[0].device.index

        for i in range(len(outputs)):
            self.assertEqual(outputs[i].mean(dtype=torch.float), 2)

        state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)

        output_ptrs = [output.untyped_storage().data_ptr() for output in outputs]
        ten_metadata = [tensor_metadata(t) for t in outputs]

        self.assertEqual(live_blocks(pool_id), 3)

        del outputs

        self.assertEqual(live_blocks(pool_id), 0)

        reconstructed_tensors = [
            reconstruct_from_tensor_metadata(metadata) for metadata in ten_metadata
        ]

        for i in range(len(reconstructed_tensors)):
            self.assertEqual(reconstructed_tensors[i].mean(dtype=torch.float), 2)

        inp.add_(1)
        graph.replay()

        for i in range(len(reconstructed_tensors)):
            self.assertEqual(reconstructed_tensors[i].mean(dtype=torch.float), 3)

        self.setCheckpointPoolState(
            device, state, [], [reconstructed_tensors[0], reconstructed_tensors[1]]
        )

        self.assertEqual(live_blocks(pool_id), 3)

        reconstructed_tensors[0] = None
        self.assertEqual(live_blocks(pool_id), 2)

        reconstructed_tensors[1] = None
        self.assertEqual(live_blocks(pool_id), 1)

        # should not change, we did not pass it in to swap data ptrs
        reconstructed_tensors[2] = None
        self.assertEqual(live_blocks(pool_id), 1)

        torch._C._cuda_cudaCachingAllocator_raw_delete(output_ptrs[2])

        self.assertEqual(live_blocks(pool_id), 0)

    @skipIfNoTorchVision
    def test_resnet(self):
        import torchvision

        m = torchvision.models.resnet50()
        m.eval()
        m = m.cuda()

        inp = torch.rand([1, 3, 255, 255], device="cuda")
        self.checkFunction(m, [inp])

    def test_check_pool_live_allocations(self):
        def foo():
            return torch.ones([4], device="cuda")

        pool = torch.cuda.graph_pool_handle()
        graph, outputs = cudagraphify(foo, [], pool=pool)

        index = outputs[0].device.index

        def check(live_dps):
            return torch._C._cuda_checkPoolLiveAllocations(index, pool, live_dps)

        self.assertTrue(check({outputs[0].data_ptr()}))

        self.assertFalse(check({outputs[0].data_ptr(), 0}))
        self.assertFalse(check(set()))

        del outputs
        self.assertTrue(check(set()))

    def test_allocate_in_thread_to_pool(self):
        def foo():
            return torch.rand([4], device="cuda")

        pool = torch.cuda.graph_pool_handle()
        graph, outputs = cudagraphify(foo, [], pool=pool)
        device = outputs[0].device.index
        del outputs

        @contextlib.contextmanager
        def _use_cuda_memory_pool_manager(device, mem_pool):
            """
            Context manager to use cuda graph pool for new allocations. If you use this manager
            all cudagraph tensors in use should be reflected in the allocator or they will be overwritten.
            existing_graph should already have been used in a capture, and the mem_pool must already exist.
            """
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            stream_context = torch.cuda.stream(stream)
            stream_context.__enter__()
            torch._C._cuda_beginAllocateCurrentStreamToPool(device, mem_pool)
            try:
                yield
            finally:
                torch._C._cuda_endAllocateCurrentStreamToPool(device, mem_pool)
                torch._C._cuda_releasePool(device, mem_pool)
                stream_context.__exit__(None, None, None)

        segments = get_cudagraph_segments(pool)
        self.assertEqual(len(get_cudagraph_segments(pool)), 1)

        def use_pool():
            def alloc_three():
                a = int8_cuda(LARGE_BUFFER)
                b = int8_cuda(LARGE_BUFFER)
                c = a + b

            with _use_cuda_memory_pool_manager(device, pool):
                # three allocations
                for _ in range(10):
                    alloc_three()

            # three more allocations not in pool
            alloc_three()

        def no_pool():
            # two allocations
            for _ in range(10):
                a = int8_cuda(LARGE_BUFFER)
                b = int8_cuda(LARGE_BUFFER)
                del a, b

        graph_thread = threading.Thread(target=use_pool)
        no_graph_thread = threading.Thread(target=no_pool)
        graph_thread.start()
        no_graph_thread.start()

        graph_thread.join()
        no_graph_thread.join()

        self.assertEqual(
            len(get_cudagraph_segments(pool)), 2 if self.expandable_segments else 4
        )

        del graph

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        self.assertEqual(len(get_cudagraph_segments(pool)), 0)

    def test_no_triton_on_import(self):
        """Test that Trition is not imported on first GPU use"""
        script = "import sys; import torch; torch.rand(2, device='cuda'); print('triton' in sys.modules)"

        rc = (
            subprocess.check_output(
                [sys.executable, "-c", script],
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
            .strip()
            .decode("ascii")
        )
        self.assertEqual(rc, "False", "Triton was imported when importing torch!")


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
class TestMemPool(TestCase):
    def test_mempool_id(self):
        pool1 = torch.cuda.graph_pool_handle()
        pool2 = torch.cuda.MemPool().id

        # first value of id in a user created pool is always zero
        self.assertEqual(pool1[0] == 0, pool2[0] == 0)

        # each call to torch.cuda.graph_pool_handle() or torch.cuda.MemPool()
        # increments the id
        self.assertTrue(abs(pool2[1] - pool1[1]) > 0)

    def test_mempool_with_allocator(self):
        pool = torch.cuda.MemPool()

        # MemPool doesn't have an allocator by default
        self.assertEqual(pool.allocator, None)

        from torch.utils.cpp_extension import load_inline

        dummy_allocator_source = """
        #include <torch/extension.h>
        #include <ATen/cuda/Exceptions.h>
        #include <cuda_runtime_api.h>

        extern "C" {
          C10_EXPORT int called_dummy_alloc = 0;
          C10_EXPORT int called_dummy_free = 0;

          // Note that windows needs __declspec(dllexport): https://stackoverflow.com/a/24575865
          C10_EXPORT void* dummy_alloc(size_t size, int device, void* stream) {
            called_dummy_alloc = 123;
            void* ptr;
            C10_CUDA_CHECK(cudaMallocManaged(&ptr, size));
            return ptr;
          }

          C10_EXPORT void dummy_free(void* ptr, size_t size, int device, void* stream) {
            called_dummy_free = 321;
            C10_CUDA_CHECK(cudaFree(ptr));
          }
        }
        """
        dummy_allocator_libname = "dummy_allocator"
        dummy_allocator = load_inline(
            name=dummy_allocator_libname,
            cpp_sources=dummy_allocator_source,
            is_python_module=False,
            keep_intermediates=False,
            verbose=True,
            with_cuda=True,
        )
        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            dummy_allocator,
            "dummy_alloc",
            "dummy_free",
        )
        pool = torch.cuda.MemPool(allocator.allocator())

        # pool should point to the same allocator as the one passed into it
        self.assertEqual(allocator.allocator(), pool.allocator)

        # pool's use count should be 1 at this point as MemPool object
        # holds a reference
        self.assertEqual(pool.use_count(), 1)

        # no allocations happened yet, so called_dummy_alloc and
        # called_dummy_free should be 0
        alloc_lib = ctypes.CDLL(dummy_allocator)
        called_dummy_alloc = ctypes.c_int.in_dll(alloc_lib, "called_dummy_alloc")
        called_dummy_free = ctypes.c_int.in_dll(alloc_lib, "called_dummy_free")
        self.assertEqual(called_dummy_alloc.value, 0)
        self.assertEqual(called_dummy_free.value, 0)

        nelem_1mb = 1024 * 1024 // 4

        with torch.cuda.use_mem_pool(pool):
            out_0 = torch.randn(nelem_1mb, device="cuda")

            # pool's use count should be 2 at this point as use_mem_pool
            # holds a reference
            self.assertEqual(pool.use_count(), 2)

        # pool's use count should be back to 1 at this point as use_mem_pool
        # released its reference
        self.assertEqual(pool.use_count(), 1)

        # called_dummy_alloc should be 123 if dummy_alloc was used to allocate
        # out tensor
        self.assertEqual(called_dummy_alloc.value, 123)

        with torch.cuda.use_mem_pool(pool):
            # pool should have 1 segment since we made a small allocation (1 MB)
            # above and so the CUDACachingAllocator packed it into a 2 MB buffer
            self.assertEqual(len(pool.snapshot()), 1)

            out_1 = torch.randn(nelem_1mb, device="cuda")

            # pool should still have 1 segment since we made another small allocation
            # (1 MB) that got packed into the existing 2 MB buffer
            self.assertEqual(len(pool.snapshot()), 1)

            out_2 = torch.randn(nelem_1mb, device="cuda")

            # pool now should have 2 segments since the CUDACachingAllocator had
            # to make a new 2 MB buffer to accomodate out_2
            self.assertEqual(len(pool.snapshot()), 2)

        del out_0, out_1, out_2

        # pool's destructor calls emptyCache()
        del pool

        # called_dummy_free should be 321 if dummy_free was used to deallocate
        # out tensor
        self.assertEqual(called_dummy_free.value, 321)

    def test_mempool_context(self):
        active_pool = torch.cuda.MemPoolContext.active_pool()

        # there is no active pool if none was made active
        self.assertEqual(active_pool, None)

        pool = torch.cuda.MemPool()
        ctx = torch.cuda.MemPoolContext(pool)
        active_pool = torch.cuda.MemPoolContext.active_pool()

        # pool was made active
        self.assertEqual(active_pool, pool)

        del ctx
        active_pool = torch.cuda.MemPoolContext.active_pool()

        # ctx was deleted, so active pool is the previous one
        self.assertEqual(active_pool, None)

    def test_mempool_multithread(self):
        pool_ids = []
        active_pool_ids = []

        def create_mempool_and_make_active():
            pool = torch.cuda.MemPool()
            pool_ids.extend([pool.id])

            ctx = torch.cuda.MemPoolContext(pool)
            active_pool = torch.cuda.MemPoolContext.active_pool()
            active_pool_ids.extend([active_pool.id])
            del ctx

        num_threads = 4
        threads = [
            threading.Thread(target=create_mempool_and_make_active)
            for t in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # each thread should create a unique mempool, since
        # mempool id creation is atomic
        self.assertEqual(len(set(pool_ids)), 4)

        # each thread should have different active mempool, since
        # the pointer to the mempool is thread local
        self.assertEqual(len(set(active_pool_ids)), 4)


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCudaOptims(TestCase):
    # These tests will be instantiate with instantiate_device_type_tests
    # to apply the new OptimizerInfo structure.

    @onlyCUDA
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >=5.3 required for graphs"
    )
    @optims(
        [optim for optim in optim_db if optim.has_capturable_arg],
        dtypes=[torch.float32],
    )
    def test_graph_optims(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        steps_warmup = 3
        steps_train = 2

        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs

            # lr and betas as a Tensor is not supported when capturable=False and foreach=True for torch.optim.adam
            # and torch.optim.adamw
            kwargs["lr"] = 0.1
            if optim_cls in (torch.optim.Adam, torch.optim.AdamW):
                kwargs["betas"] = (0.9, 0.99)

            for actually_do_graphs in (True, False):
                params = [
                    torch.randn((i + 5, i + 5), device=device) for i in range(2)
                ] + [torch.randn((), device=device)]
                params_control = [p.clone().requires_grad_() for p in params]
                params_graphed = [p.clone().requires_grad_() for p in params]

                grads = [
                    [torch.randn_like(p) for p in params]
                    for _ in range(steps_warmup + steps_train)
                ]

                # Control (capturable=False)
                kwargs["capturable"] = False

                opt = optim_cls(params_control, **kwargs)
                for i in range(steps_warmup + steps_train):
                    for j, p in enumerate(params_control):
                        p.grad = grads[i][j]
                    opt.step()

                # capturable=True
                kwargs["capturable"] = True
                opt = optim_cls(params_graphed, **kwargs)

                for i in range(steps_warmup):
                    for j, p in enumerate(params_graphed):
                        p.grad = grads[i][j]
                    opt.step()

                if actually_do_graphs:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        opt.step()

                for i in range(steps_train):
                    if actually_do_graphs:
                        for j, p in enumerate(params_graphed):
                            p.grad.copy_(grads[i + steps_warmup][j])
                        g.replay()
                    else:
                        # Passing capturable=True to the constructor and running without graphs should still be
                        # numerically correct, even if it's not ideal for performance.
                        for j, p in enumerate(params_graphed):
                            p.grad = grads[i + steps_warmup][j]
                        opt.step()

                for p_control, p_graphed in zip(params_control, params_graphed):
                    self.assertEqual(p_control, p_graphed)

    @onlyCUDA
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @optims(
        [
            optim
            for optim in optim_db
            if "fused" in optim.supported_impls and "cuda" in optim.supports_fused_on
        ],
        dtypes=[torch.float32],
    )
    def test_graph_scaling_fused_optimizers(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls

        steps_warmup = 3
        steps_train = 2

        optim_inputs = optim_info.optim_inputs_func(device=device)

        for optim_input in optim_inputs:
            kwargs = optim_input.kwargs
            kwargs["fused"] = True

            for actually_do_graphs in (
                (True, False) if optim_info.has_capturable_arg else (True,)
            ):
                params = [torch.randn((i + 5, i + 5), device=device) for i in range(2)]
                params_control = [p.clone().requires_grad_() for p in params]
                params_graphed = [p.clone().requires_grad_() for p in params]

                # `GradScaler` in-place updates gradients thus it's necessary to duplicate gradients.
                grads = [
                    [torch.randn_like(p) for p in params]
                    for _ in range(steps_warmup + steps_train)
                ]
                with torch.no_grad():
                    grads_control = [[g.clone() for g in gs] for gs in grads]
                    grads_graphed = [[g.clone() for g in gs] for gs in grads]

                # Gradient Scaler
                scaler_for_control = torch.cuda.amp.GradScaler(init_scale=128.0)
                with torch.no_grad():
                    scaler_for_control._lazy_init_scale_growth_tracker(device)

                scaler_for_graphed = torch.cuda.amp.GradScaler()
                scaler_for_graphed.load_state_dict(scaler_for_control.state_dict())
                with torch.no_grad():
                    scaler_for_graphed._lazy_init_scale_growth_tracker(device)

                # Control (capturable=False)
                if optim_info.has_capturable_arg:
                    kwargs["capturable"] = False
                opt = optim_cls(params_control, **kwargs)

                for i in range(steps_warmup + steps_train):
                    for j, p in enumerate(params_control):
                        p.grad = grads_control[i][j]
                    scaler_for_control.step(opt)
                    scaler_for_control.update()

                # capturable=True
                if optim_info.has_capturable_arg:
                    kwargs["capturable"] = True
                opt = optim_cls(params_graphed, **kwargs)

                for i in range(steps_warmup):
                    for j, p in enumerate(params_graphed):
                        p.grad = grads_graphed[i][j]
                    scaler_for_graphed.step(opt)
                    scaler_for_graphed.update()

                if actually_do_graphs:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        scaler_for_graphed.step(opt)
                        scaler_for_graphed.update()

                for i in range(steps_train):
                    if actually_do_graphs:
                        for j, p in enumerate(params_graphed):
                            p.grad.copy_(grads_graphed[i + steps_warmup][j])
                        g.replay()
                    else:
                        # Passing capturable=True to the constructor and running without graphs should still be
                        # numerically correct, even if it's not ideal for performance.
                        for j, p in enumerate(params_graphed):
                            p.grad = grads_graphed[i + steps_warmup][j]
                        scaler_for_graphed.step(opt)
                        scaler_for_graphed.update()

                for p_control, p_graphed in zip(params_control, params_graphed):
                    self.assertEqual(p_control, p_graphed)

    @onlyNativeDeviceTypes
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    def test_grad_scaling_autocast_fused_optimizers(self, device, dtype, optim_info):
        device = device.split(":")[0]
        if device not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        optim_inputs = optim_info.optim_inputs_func(device=device)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            for _separate_unscale in (True, False):
                kwargs = optim_input.kwargs
                kwargs["fused"] = True
                torch.manual_seed(20)
                (
                    mod_control,
                    mod_scaling,
                    opt_control,
                    opt_scaling,
                    data,
                    loss_fn,
                    _,
                ) = _create_scaling_case(
                    optimizer_ctor=optim_cls, optimizer_kwargs=kwargs, device=device
                )
                optimizer_kwargs = deepcopy(kwargs)
                optimizer_kwargs["fused"] = False
                if "lr" not in kwargs:
                    # _create_scaling_case will set lr = 1.0 if optimizer_kwargs do not set lr
                    optimizer_kwargs["lr"] = 1.0
                opt_control = optim_cls(mod_control.parameters(), **optimizer_kwargs)
                scaler_scaling = torch.amp.GradScaler(device, init_scale=128.0)
                scaler_control = torch.amp.GradScaler(device, init_scale=128.0)
                tracker = TensorTracker()
                for input, target in data:
                    opt_control.zero_grad()
                    with torch.autocast(device_type=device, dtype=torch.half):
                        output_control = mod_control(input)
                        loss_control = loss_fn(output_control, target)
                    scaler_control.scale(loss_control).backward()
                    scaler_control.step(opt_control)
                    scaler_control.update()

                    opt_scaling.zero_grad()
                    with torch.autocast(device_type=device, dtype=torch.half):
                        output_scaling = mod_scaling(input)
                        loss_scaling = loss_fn(output_scaling, target)
                    scaler_scaling.scale(loss_scaling).backward()
                    if _separate_unscale:
                        scaler_scaling.unscale_(opt_scaling)
                    scaler_scaling.step(opt_scaling)
                    scaler_scaling.update()

                    tracker.add(loss_control)
                    tracker.pop_check_set(loss_scaling, self)
                    for param_control, param_scaling in zip(
                        mod_control.parameters(), mod_scaling.parameters()
                    ):
                        tracker.add(param_control.grad)
                        tracker.pop_check_set(param_scaling.grad, self)
                        tracker.add(param_control)
                        tracker.pop_check_set(param_scaling, self)

                        state_control, state_scaling = (
                            opt_control.state[param_control],
                            opt_scaling.state[param_scaling],
                        )

                        for k in state_control:
                            actual = state_scaling[k]
                            if k == "step":
                                actual = actual.squeeze()
                            tracker.add(state_control[k])
                            tracker.pop_check_set(actual, self)

    @onlyCUDA
    @parametrize("in_place_unscale", [False, True])
    @optims(
        [optim for optim in optim_db if "cuda" in optim.supports_fused_on],
        dtypes=[torch.float32],
    )
    def test_grad_scaler_with_preset_grad_scale(
        self, device, dtype, optim_info, in_place_unscale
    ):
        weight = torch.ones((5, 5), device="cuda", requires_grad=True)
        weight.grad = torch.full_like(weight, fill_value=15)
        opt = optim_info.optim_cls([weight], lr=0.1, fused=True)
        scaler = torch.amp.GradScaler(init_scale=5)

        # simulate scaling a loss
        scaler.scale(torch.ones(5))

        if in_place_unscale:
            scaler.unscale_(opt)
            # the gradient should have been divided in-place
            self.assertEqual(weight.grad, torch.full_like(weight, fill_value=3))

        # the user sets a `grad_scale` value which should be fused with the optimizer step
        opt.grad_scale = torch.Tensor([3]).cuda()
        scaler.step(opt)

        # check that the user's grad_scale was respected (i.e. the gradient was divided by 5 * 3)
        self.assertEqual(weight.grad, torch.full_like(weight, fill_value=1))

    @onlyCUDA
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @parametrize("foreach, fused", [(False, False), (True, False), (False, True)])
    @optims(
        [
            optim
            for optim in optim_db
            if "foreach" in optim.supported_impls and "cuda" in optim.supports_fused_on
        ],
        dtypes=[torch.float32],
    )
    def test_graph_grad_scaling(self, device, dtype, optim_info, foreach, fused):
        torch.cuda.empty_cache()

        scaler = torch.amp.GradScaler(device="cuda", init_scale=4.0)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()

        weight = torch.ones((100,), device="cuda", requires_grad=True)
        opt = optim_info.optim_cls([weight], lr=0.1, foreach=foreach, fused=fused)
        static_input = torch.ones_like(weight)
        static_grad = torch.ones_like(weight)

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            loss = (weight.half() * static_input).sum()
            scaler.scale(loss).backward()
        torch.cuda.current_stream().wait_stream(s)

        opt.zero_grad(set_to_none=True)

        # capture
        with torch.cuda.stream(s):
            g.capture_begin()
            loss = (weight.half() * static_input).sum()
            scaler.scale(loss).backward()
            g.capture_end()

        input_vals = [5, 20000, 5, 40000]
        # If the scale gets updated properly, these are the scale, growth tracker,
        # and grad values we expect.
        expected_scales = [4, 2, 2, 1]
        expected_growth_trackers = [1, 0, 1, 0]
        expected_grad_vals = [5 * 4, float("inf"), 5 * 2, float("inf")]

        for data, scale, growth_tracker, grad_val in zip(
            input_vals, expected_scales, expected_growth_trackers, expected_grad_vals
        ):
            static_input.fill_(data)
            g.replay()
            self.assertEqual(weight.grad, torch.full_like(weight.grad, grad_val))
            scaler.step(opt)
            scaler.update()
            self.assertEqual(scaler._scale, scale)
            self.assertEqual(scaler._growth_tracker, growth_tracker)


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
class TestGDS(TestCase):
    def _get_tmp_dir_fs_type(self):
        my_path = os.path.realpath("/tmp")
        root_type = ""
        for part in psutil.disk_partitions():
            if part.mountpoint == "/":
                root_type = part.fstype
                continue
            if part.mountpoint == my_path:
                return part.fstype
        return root_type

    @unittest.skip("Disabling as USE_CUFILE=0 by default in builds")
    def test_gds_read_write_tensors(self):
        if self._get_tmp_dir_fs_type() not in ("ext4", "xfs"):
            self.skipTest("GPUDirect Storage requires ext4/xfs for local filesystem")
        src1 = torch.randn(1024, device="cuda")
        src2 = torch.randn(2, 1024, device="cuda")
        torch.cuda.gds._gds_register_buffer(src1.untyped_storage())
        torch.cuda.gds._gds_register_buffer(src2.untyped_storage())
        dest1 = torch.empty(1024, device="cuda")
        dest2 = torch.empty(2, 1024, device="cuda")
        with TemporaryFileName() as f:
            file = torch.cuda.gds._GdsFile(f, os.O_CREAT | os.O_RDWR)
            file.save_storage(src1.untyped_storage(), offset=0)
            file.save_storage(src2.untyped_storage(), offset=src1.nbytes)
            file.load_storage(dest1.untyped_storage(), offset=0)
            file.load_storage(dest2.untyped_storage(), offset=src1.nbytes)
        self.assertEqual(src1, dest1)
        self.assertEqual(src2, dest2)
        torch.cuda.gds._gds_deregister_buffer(src1.untyped_storage())
        torch.cuda.gds._gds_deregister_buffer(src2.untyped_storage())


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
class TestCudaAutocast(TestAutocast):
    def setUp(self):
        super().setUp()
        self.autocast_lists = AutocastTestLists(torch.device("cuda:0"))

    def tearDown(self):
        del self.autocast_lists
        super().tearDown()

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_torch_fp16(self):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op_with_args in self.autocast_lists.torch_fp16:
                skip_test = False
                op, args = op_with_args[0], op_with_args[1]
                if len(op_with_args) == 3:
                    skip_test = op_with_args[2]  # TEST_WITH_ROCM
                if not skip_test:
                    self._run_autocast_outofplace(
                        op, args, torch.float16, device="cuda", amp_dtype=torch.float16
                    )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_torch_bf16(self):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op_with_args in self.autocast_lists.torch_fp16:
                skip_test = False
                op, args = op_with_args[0], op_with_args[1]
                if len(op_with_args) == 3:
                    skip_test = op_with_args[2]  # TEST_WITH_ROCM
                should_error_from_cudnn = "cudnn" in op and (
                    "TORCH_CUDNN_V8_API_DISABLED" in os.environ
                    and int(os.environ["TORCH_CUDNN_V8_API_DISABLED"])
                    or torch.cuda.get_device_capability() < (8, 0)
                )
                should_error_from_not_implemented = should_error_from_cudnn
                if not skip_test:
                    if should_error_from_not_implemented:
                        with self.assertRaises(
                            RuntimeError,
                            msg=str(op) + " should not be supported for bfloat16!",
                        ):
                            self._run_autocast_outofplace(
                                op, args, torch.bfloat16, device="cuda"
                            )
                    else:
                        if torch.cuda.is_bf16_supported():
                            self._run_autocast_outofplace(
                                op, args, torch.bfloat16, device="cuda"
                            )
                        else:
                            with self.assertRaisesRegex(
                                RuntimeError, "Device does not support bfloat16"
                            ):
                                self._run_autocast_outofplace(
                                    op, args, torch.bfloat16, device="cuda"
                                )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_torch_fp32(self):
        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cuda",
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_torch_need_autocast_promote(self):
        for op, args in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(
                op, args, torch.float32, device="cuda", amp_dtype=torch.float16
            )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_torch_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cuda",
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_nn_fp16(self):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op, args in self.autocast_lists.nn_fp16:
                self._run_autocast_outofplace(
                    op,
                    args,
                    torch.float16,
                    device="cuda",
                    module=torch._C._nn,
                    amp_dtype=torch.float16,
                )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_nn_bf16(self):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op, args in self.autocast_lists.nn_fp16:
                if torch.cuda.is_bf16_supported():
                    self._run_autocast_outofplace(
                        op, args, torch.bfloat16, device="cuda", module=torch._C._nn
                    )
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "Device does not support bfloat16"
                    ):
                        self._run_autocast_outofplace(
                            op, args, torch.bfloat16, device="cuda", module=torch._C._nn
                        )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_nn_fp32(self):
        for op, args in self.autocast_lists.nn_fp32:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cuda",
                module=torch._C._nn,
                amp_dtype=torch.float16,
            )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_linalg_fp16(self):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op, args in self.autocast_lists.linalg_fp16:
                self._run_autocast_outofplace(
                    op,
                    args,
                    torch.float16,
                    device="cuda",
                    module=torch._C._linalg,
                    amp_dtype=torch.float16,
                )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_methods_fp16(self):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            for op, args in self.autocast_lists.methods_fp16:
                self._run_autocast_outofplace(
                    op,
                    args,
                    torch.float16,
                    device="cuda",
                    module=None,
                    amp_dtype=torch.float16,
                )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_methods_fp32(self):
        for op, args in self.autocast_lists.methods_fp32:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cuda",
                module=None,
                amp_dtype=torch.float16,
            )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_methods_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.methods_expect_builtin_promote:
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cuda",
                module=None,
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    def test_autocast_banned(self):
        with torch.autocast("cuda"):
            for op, args, module in self.autocast_lists.banned:
                with self.assertRaises(RuntimeError):
                    getattr(module, op)(*args)

    def test_autocast_ignored_types(self):
        with torch.autocast("cuda"):
            for ignore_type in (torch.double, torch.int32):
                a_ignore = torch.ones((8, 8), dtype=ignore_type, device="cuda:0")
                b_ignore = torch.ones((8, 8), dtype=ignore_type, device="cuda:0")
                c_16 = torch.ones((8, 8), dtype=torch.float16, device="cuda:0")

                # Tests if CastPolicy::fp16 ops ignore double and int
                # Currently, no ops belonging to this policy support integer inputs.
                if ignore_type is torch.double:
                    with self.assertRaises(RuntimeError):
                        torch.mm(a_ignore, c_16)
                    with torch.autocast("cuda", enabled=False):
                        type_no_autocast = torch.mm(a_ignore, b_ignore).dtype
                    self.assertTrue(
                        torch.mm(a_ignore, b_ignore).dtype is type_no_autocast
                    )

                # Tests if CastPolicy::fp32 ops ignore double and int
                with torch.autocast("cuda", enabled=False):
                    type_no_autocast = torch.pow(a_ignore, 2.0).dtype
                self.assertTrue(torch.pow(a_ignore, 2.0).dtype is type_no_autocast)

                # Tests if CastPolicy::fp32_set_opt_dtype ops ignore double and int
                with torch.autocast("cuda", enabled=False):
                    type_no_autocast = torch.sum(a_ignore).dtype
                self.assertTrue(torch.sum(a_ignore).dtype is type_no_autocast)

                # Tests if CastPolicy::fp32_append_dtype ops ignore double and int
                # Currently, no ops belonging to this policy support integer inputs.
                if ignore_type is torch.double:
                    with torch.autocast("cuda", enabled=False):
                        type_no_autocast = torch.norm(a_ignore).dtype
                    self.assertTrue(torch.norm(a_ignore).dtype is type_no_autocast)

    def test_autocast_custom_enabled(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.amp.custom_fwd(device_type="cuda")
            def forward(ctx, a, b):
                self.assertTrue(a.dtype is torch.float32)
                self.assertTrue(b.dtype is torch.float32)
                self.assertTrue(torch.is_autocast_enabled())
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.amp.custom_bwd(device_type="cuda")
            def backward(ctx, grad):
                self.assertTrue(torch.is_autocast_enabled())
                a, b = ctx.saved_tensors
                a_grad, b_grad = grad.mm(b.t()), a.t().mm(grad)
                self.assertTrue(a_grad.dtype is dtype and b_grad.dtype is dtype)
                return a_grad, b_grad

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="cuda", dtype=torch.float32, requires_grad=True)
        y = torch.randn((8, 8), device="cuda", dtype=torch.float32, requires_grad=True)

        dtypes = (torch.float16, torch.bfloat16) if TEST_BF16 else (torch.float16,)
        for dtype in dtypes:
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = mymm(x, y)
                self.assertTrue(output.dtype is dtype)
                loss = output.sum()
            loss.backward()

    def test_autocast_custom_cast_inputs(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
            def forward(ctx, a, container, expect_type):
                b = container[1][0]
                self.assertTrue(a.dtype is expect_type)
                self.assertTrue(b.dtype is expect_type)
                self.assertFalse(torch.is_autocast_enabled())
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.amp.custom_bwd(device_type="cuda")
            def backward(ctx, grad):
                self.assertFalse(torch.is_autocast_enabled())
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), None, None

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="cuda", dtype=torch.float16, requires_grad=True)
        # Puts one input tensor in a nested container.  y's contained Tensor won't receive a gradient,
        # because torch.autograd.Function can't hand gradients back to non-Tensor forward arguments.
        # Sets requires_grad=False explicitly so we don't lie about expecting a gradient.
        y = (
            0,
            {
                0: torch.randn(
                    (8, 8), device="cuda", dtype=torch.float16, requires_grad=False
                )
            },
        )

        with torch.autocast("cuda"):
            output = mymm(x, y, torch.float32)
            self.assertTrue(output.dtype is torch.float32)
            loss = output.sum()
        loss.backward()

        # Tests if custom_fwd becomes a no-op when mymm runs outside an autocast-enabled region.
        output = mymm(x, y, torch.float16)
        self.assertTrue(output.dtype is torch.float16)
        loss = output.sum()
        loss.backward()

    def test_autocast_custom_deprecated_warning(self):
        with warnings.catch_warnings(record=True) as w:

            class MyMM(torch.autograd.Function):
                @staticmethod
                @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
                def forward(ctx, x, y):
                    ctx.save_for_backward(x, y)
                    self.assertFalse(torch.is_autocast_enabled())
                    return x + y

                @staticmethod
                @torch.cuda.amp.custom_bwd
                def backward(ctx, grad):
                    _, _ = ctx.saved_tensors
                    self.assertFalse(torch.is_autocast_enabled())
                    return grad, grad

        self.assertRegex(
            str(w[0].message), r"`torch.cuda.amp.custom_fwd\(args...\)` is deprecated."
        )
        self.assertRegex(
            str(w[1].message), r"`torch.cuda.amp.custom_bwd\(args...\)` is deprecated."
        )

        mymm = MyMM.apply
        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        with torch.amp.autocast("cuda"):
            output = mymm(x, y)
            loss = output.sum()
        loss.backward()

    def test_autocast_cat_jit(self):
        # Reported at https://github.com/pytorch/pytorch/issues/38958

        class Model(torch.nn.Module):
            def forward(self):
                a = torch.randn(1)
                b = torch.randn(1)
                c = torch.cat((a, b), 0)
                d = torch.stack([c, c], 0)
                return d

        # The JIT here doesn't really matter, we just need to call
        # cat via the boxed API
        model = Model()
        model_jit_script = torch.jit.script(model)

        with torch.autocast("cuda", enabled=True):
            model()
            model_jit_script()

    # cudnn RNNs require special backend handling (weights are cast to FP16 and reflattened)
    # so they get a dedicated test.
    # Despite the large number of RNN cases it tries, the test takes < 15 seconds on a Titan V (similar to V100).
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_autocast_rnn(self):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # seq, batch, features, hidden size
            clses = ("RNN", "GRU", "LSTM")
            T, B, F, H = 3, 4, 5, 6
            dtypes = (torch.float16, torch.float32)
            input_layouts = ("seq_first", "batch_first", "packed")

            for (
                cls,
                num_layers,
                bias,
                input_layout,
                bidirectional,
                try_nonpreflattened_weights,
                input_dtype,
                hidden_dtype,
                weight_dtype,
            ) in product(
                clses,
                (1, 2),
                (True, False),
                input_layouts,
                (True, False),
                (True, False),
                dtypes,
                dtypes,
                dtypes,
            ):
                if input_layout == "seq_first":
                    batch_first = False
                    x = torch.randn((T, B, F), device="cuda", dtype=input_dtype)
                elif input_layout == "batch_first":
                    batch_first = True
                    x = torch.randn((B, T, F), device="cuda", dtype=input_dtype)
                elif input_layout == "packed":
                    batch_first = False
                    x = torch.nn.utils.rnn.pack_padded_sequence(
                        torch.randn((T, B, F), device="cuda", dtype=input_dtype),
                        lengths=(3, 2, 1, 3),
                        enforce_sorted=False,
                    )

                rnn = (
                    getattr(torch.nn, cls)(
                        F,
                        H,
                        num_layers=num_layers,
                        bidirectional=bidirectional,
                        bias=bias,
                        batch_first=batch_first,
                    )
                    .cuda()
                    .to(dtype=weight_dtype)
                )

                if try_nonpreflattened_weights:
                    for p in rnn.parameters():
                        with torch.no_grad():
                            p.set_(p.clone())

                h = torch.randn(
                    (num_layers * (2 if bidirectional else 1), B, H),
                    device="cuda",
                    dtype=hidden_dtype,
                )
                if cls == "LSTM":
                    c = torch.randn(
                        (num_layers * (2 if bidirectional else 1), B, H),
                        device="cuda",
                        dtype=hidden_dtype,
                    )
                    h = (h, c)

                with torch.autocast("cuda"):
                    out, h_out = rnn(x, h)
                out = out.data if input_layout == "packed" else out
                self.assertEqual(out.dtype, torch.float16)
                # Autocast wrapper requires at::_cudnn_rnn is autograd-exposed.  This check can't guarantee
                # at::_cudnn_rnn is autograd-exposed, but if it fires, it indicates some funny business has
                # occurred and we should double check that at::_cudnn_rnn remains autograd-exposed.
                self.assertEqual(
                    out.grad_fn.name(),
                    "MiopenRnnBackward0" if torch.version.hip else "CudnnRnnBackward0",
                )
                out.sum().backward()
                grads = [p.grad.clone() for p in rnn.parameters()]

                rnn.zero_grad()

                if cls == "LSTM":
                    out_control, h_out_control = rnn.to(dtype=torch.float16)(
                        x.half(), (h[0].half(), h[1].half())
                    )
                else:
                    out_control, h_out_control = rnn.to(dtype=torch.float16)(
                        x.half(), h.half()
                    )
                out_control = (
                    out_control.data if input_layout == "packed" else out_control
                )
                out_control.sum().backward()
                grads_control = [p.grad.clone() for p in rnn.parameters()]

                # Compares with default tolerances, even for FP16 execution.  Barring nondeterminism,
                # autocast and control results should be bitwise identical.
                self.assertEqual(out, out_control)

                if cls == "LSTM":
                    self.assertTrue(
                        h_out[0].dtype is torch.float16
                        and h_out[1].dtype is torch.float16
                    )
                    self.assertEqual(h_out[0], h_out_control[0])
                    self.assertEqual(h_out[1], h_out_control[1])
                else:
                    self.assertEqual(h_out.dtype, torch.float16)
                    self.assertEqual(h_out, h_out_control)
                for grad, grad_control in zip(grads, grads_control):
                    self.assertEqual(grad.half(), grad_control)

    def test_autocast_cache_leak(self):
        # Reported at https://github.com/pytorch/pytorch/issues/48049
        # Test is used to check, if autocast recaches the same parameters
        # when executed in a `torch.no_grad()` block.

        linear = torch.nn.Linear(10, 10).to("cuda")
        data = torch.randn(1, 10, device="cuda")

        with torch.autocast("cuda"):
            with torch.no_grad():
                out = linear(data)
                first_iter_mem = torch.cuda.memory_allocated()
                for _ in range(3):
                    out = linear(data)
                self.assertTrue(first_iter_mem == torch.cuda.memory_allocated())

    def test_autocast_checkpointing(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
        ).cuda()
        input = torch.rand(
            (8, 8), device="cuda", dtype=torch.float16, requires_grad=True
        )
        for reentrant in (True, False):
            with torch.autocast("cuda"):
                output = checkpoint_sequential(model, 2, input, use_reentrant=reentrant)
            self.assertTrue(output.requires_grad)
            self.assertTrue(output.dtype is torch.float16)
            output.sum().backward()

    def test_cuda_autocast_deprecated_warning(self):
        with self.assertWarnsRegex(
            FutureWarning,
            r"`torch.cuda.amp.autocast\(args...\)` is deprecated. Please use `torch.amp.autocast\('cuda', args...\)` instead.",
        ):
            with torch.cuda.amp.autocast():
                _ = torch.ones(10)

    def test_cuda_module_loading_env(self):
        torch.cuda.init()
        val = os.environ.get("CUDA_MODULE_LOADING", "")
        self.assertEqual(val, "LAZY")


instantiate_parametrized_tests(TestCuda)
instantiate_parametrized_tests(TestCudaMallocAsync)
instantiate_device_type_tests(TestCudaOptims, globals())

if __name__ == "__main__":
    run_tests()
