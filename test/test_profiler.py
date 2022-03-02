# Owner(s): ["oncall: profiler"]

import collections
import gc
import io
import json
import os
import unittest

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.datapipes as dp
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    TestCase, run_tests, TEST_WITH_ASAN, TEST_WITH_ROCM, IS_WINDOWS,
    TemporaryFileName, TemporaryDirectoryName)
from torch.autograd import (_record_function_with_args_enter, _record_function_with_args_exit)
from torch.autograd.profiler import profile as _profile
from torch.profiler import (
    kineto_available, profile, record_function, supported_activities,
    DeviceType, ProfilerAction, ProfilerActivity
)
from torch.testing._internal.common_device_type import skipCUDAVersionIn

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import pickle


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
@unittest.skipIf(TEST_WITH_ASAN, "Cannot test with ASAN")
@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfilerCUDA(TestCase):

    @skipCUDAVersionIn([(11, 5)])  # https://github.com/pytorch/pytorch/issues/69023
    def test_mem_leak(self):
        """Checks that there's no memory leak when using profiler with CUDA
        """
        t = torch.rand(1, 1).cuda()
        p = psutil.Process()
        last_rss = collections.deque(maxlen=5)
        for outer_idx in range(10):
            with _profile(use_cuda=True):
                for _ in range(1024):
                    t = torch.mm(t, t)

            gc.collect()
            torch.cuda.empty_cache()
            last_rss.append(p.memory_info().rss)

        # with CUDA events leaking the increase in memory was ~7 MB between
        # profiler invocations above
        is_increasing = all(
            [last_rss[idx] > last_rss[idx - 1] for idx in range(1, len(last_rss))])
        max_diff = -1
        for idx in range(1, len(last_rss)):
            max_diff = max(max_diff, last_rss[idx] - last_rss[idx - 1])
        self.assertTrue(not (is_increasing and max_diff > 100 * 1024),
                        msg='memory usage is increasing, {}'.format(str(last_rss)))

class TestRecordFunction(TestCase):
    def _record_function_with_param(self):
        u = torch.randn(3, 4, 5, requires_grad=True)
        with _profile(with_stack=True, use_kineto=kineto_available(), record_shapes=True) as prof:
            with record_function("## TEST 1 ##", "1, 2, 3"):
                rf_handle = _record_function_with_args_enter("## TEST 2 ##", 1, False, 2.5, [u, u], "hello", u)
                _record_function_with_args_exit(rf_handle)
        return prof

    def test_record_function(self):
        prof_result = self._record_function_with_param()
        found_test_1 = False
        found_test_2 = False
        for e in prof_result.function_events:
            if "## TEST 1 ##" == e.name:
                found_test_1 = True
                self.assertTrue(e.input_shapes == [[]])
            elif "## TEST 2 ##" == e.name:
                found_test_2 = True
                self.assertTrue(e.input_shapes == [[], [], [], [], [], [3, 4, 5]])
        self.assertTrue(found_test_1)
        self.assertTrue(found_test_2)

    def test_datapipe_with_record_function(self):
        with _profile(with_stack=True, use_kineto=kineto_available(), record_shapes=True) as prof:
            input_dp1 = dp.iter.IterableWrapper(range(4))
            input_dp2 = dp.iter.IterableWrapper(range(4, 8))
            input_dp3 = dp.iter.IterableWrapper(range(8, 12))
            output_dp = input_dp1.mux(input_dp2, input_dp3)
            output = list(output_dp)

        has_iter = False
        has_mux = False
        for e in prof.function_events:
            if has_iter and has_mux:
                break

            if not has_iter and e.name == "enumerate(DataPipe)#IterableWrapperIterDataPipe":
                has_iter = True
            if not has_mux and e.name == "enumerate(DataPipe)#MultiplexerIterDataPipe":
                has_mux = True
        self.assertTrue(has_iter)
        self.assertTrue(has_mux)

    def test_datapipe_with_record_function_fork(self):
        with _profile(with_stack=True, use_kineto=kineto_available(), record_shapes=True) as prof:
            input_dp = dp.iter.IterableWrapper(range(10))
            dp1, dp2, dp3 = input_dp.fork(num_instances=3)
            output1 = list(dp1)
        has_iter = False
        has_child = False
        for e in prof.function_events:
            if has_iter and has_child:
                break

            if not has_iter and e.name == "enumerate(DataPipe)#IterableWrapperIterDataPipe":
                has_iter = True
            if not has_child and e.name == "enumerate(DataPipe)#_ChildDataPipe":
                has_child = True
        self.assertTrue(has_iter)
        self.assertTrue(has_child)

class TestProfiler(TestCase):
    def test_source(self):
        """Checks that source code attribution works for eager, TS and autograd mode
        """
        # avoid automatic inlining
        prev_opt = torch._C._get_graph_executor_optimize()
        torch._C._set_graph_executor_optimize(False)

        @torch.jit.script
        def ts_method_2(x, y):
            return torch.matmul(x, y)

        @torch.jit.script
        def ts_method_1(x, y, z):
            a = x + z
            w = ts_method_2(x, y) + a
            return w.sum()

        class DummyModule(nn.Module):
            def __init__(self):
                super(DummyModule, self).__init__()
                self.conv = torch.nn.Conv2d(3, 2, kernel_size=1, stride=2, padding=3, bias=False)

            def forward(self, x):
                return self.conv(x)

        mod = DummyModule()

        with _profile(with_stack=True, use_kineto=kineto_available()) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            w = ts_method_1(x, y, z)
            v = 2 * w
            v.backward()
            a = torch.randn(2, 3, 2, 2, requires_grad=True)
            b = mod(a)
            c = b.sum()
            c.backward()

        for e in p.function_events:
            if "aten::add" in e.name or "AddBackward" in e.name:
                self.assertTrue(any(["test_profiler" in entry for entry in e.stack]))
                self.assertTrue(any([(
                    "test_source" in entry or
                    "ts_method_1" in entry or
                    "ts_method_2" in entry) for entry in e.stack]))

        torch._C._set_graph_executor_optimize(prev_opt)

    def payload(self, use_cuda=False):
        x = torch.randn(10, 10)
        if use_cuda:
            x = x.cuda()
        y = torch.randn(10, 10)
        if use_cuda:
            y = y.cuda()
        z = torch.mm(x, y)
        z = z + y
        if use_cuda:
            z = z.cpu()

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_kineto(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with _profile(use_cuda=use_cuda, use_kineto=True):
            self.payload(use_cuda=use_cuda)

        # rerun to avoid initial start overhead
        with _profile(use_cuda=use_cuda, use_kineto=True) as p:
            self.payload(use_cuda=use_cuda)
        output = p.key_averages().table(
            sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total", row_limit=-1)
        # print(output)
        found_gemm = False
        found_memcpy = False
        found_mm = False
        for e in p.function_events:
            if "aten::mm" in e.name:
                found_mm = True
            if "gemm" in e.name:
                found_gemm = True
            if "Memcpy" in e.name or "memcpy" in e.name:
                found_memcpy = True
        if use_cuda:
            self.assertTrue(found_gemm)
            self.assertTrue(found_memcpy)
        else:
            self.assertTrue(found_mm)
        # p.export_chrome_trace("/tmp/test_trace.json")

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(not TEST_MULTIGPU, "Multiple GPUs needed")
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_kineto_multigpu(self):
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA]) as prof:
            for gpu_id in [0, 1]:
                x = torch.randn(10, 10).cuda(gpu_id)
                y = torch.randn(10, 10).cuda(gpu_id)
                z = x.matmul(y)

        found_gemm_0 = False
        found_gemm_1 = False
        found_cuda = False
        for evt in prof.events():
            if "gemm" in evt.name.lower() and evt.device_type == DeviceType.CUDA:
                if evt.device_index == 0:
                    found_gemm_0 = True
                elif evt.device_index == 1:
                    found_gemm_1 = True
            if "cuda" in evt.name.lower() and evt.device_type == DeviceType.CPU:
                found_cuda = True

        self.assertTrue(found_gemm_0)
        self.assertTrue(found_gemm_1)
        self.assertTrue(found_cuda)

    def test_memory_profiler(self):
        def run_profiler(tensor_creation_fn):
            # collecting allocs / deallocs
            with _profile(profile_memory=True, record_shapes=True, use_kineto=kineto_available()) as prof:
                x = None
                with record_function("test_user_scope_alloc"):
                    x = tensor_creation_fn()
                with record_function("test_user_scope_dealloc"):
                    del x
            return prof.key_averages(group_by_input_shape=True)

        def check_metrics(stats, metric, allocs=None, deallocs=None):
            stat_metrics = {}
            for stat in stats:
                stat_metrics[stat.key] = getattr(stat, metric)
            if allocs is not None:
                for alloc_fn in allocs:
                    self.assertTrue(alloc_fn in stat_metrics)
                    self.assertTrue(stat_metrics[alloc_fn] > 0)
            if deallocs is not None:
                for dealloc_fn in deallocs:
                    self.assertTrue(dealloc_fn in stat_metrics)
                    self.assertTrue(stat_metrics[dealloc_fn] < 0)

        def create_cpu_tensor():
            return torch.rand(10, 10)

        def create_cuda_tensor():
            return torch.rand(10, 10).cuda()

        def create_mkldnn_tensor():
            return torch.rand(10, 10, dtype=torch.float32).to_mkldnn()

        stats = run_profiler(create_cpu_tensor)
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=[
                "aten::empty",
                "aten::rand",
                "test_user_scope_alloc",
            ],
            deallocs=[
                "test_user_scope_dealloc",
            ]
        )

        if kineto_available():
            with TemporaryFileName(mode="w+") as fname:
                with profile(profile_memory=True) as prof:
                    x = None
                    with record_function("test_user_scope_alloc"):
                        x = create_cpu_tensor()
                    with record_function("test_user_scope_dealloc"):
                        del x
                prof.export_chrome_trace(fname)
                with io.open(fname, 'r') as f:
                    trace = json.load(f)
                    assert "traceEvents" in trace
                    events = trace["traceEvents"]
                    found_memory_events = False
                    for evt in events:
                        assert "name" in evt
                        if evt["name"] == "[memory]":
                            found_memory_events = True
                            assert "args" in evt
                            assert "Addr" in evt["args"]
                            assert "Device Type" in evt["args"]
                            assert "Device Id" in evt["args"]
                            assert "Bytes" in evt["args"]

                            # Memory should be an instantaneous event.
                            assert "dur" not in evt["args"]
                            assert "cat" not in evt["args"]
                    assert found_memory_events

        if torch.cuda.is_available():
            create_cuda_tensor()
            stats = run_profiler(create_cuda_tensor)
            check_metrics(
                stats,
                "cuda_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::to",
                    "aten::empty_strided",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ]
            )
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "aten::rand",
                    "aten::empty",
                ]
            )

        if torch._C.has_mkldnn:
            create_mkldnn_tensor()
            stats = run_profiler(create_mkldnn_tensor)
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::rand",
                    "aten::empty",
                    "aten::to_mkldnn",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ]
            )

        # check top-level memory events
        with _profile(profile_memory=True, use_kineto=kineto_available()) as prof:
            x = torch.rand(10, 10)
            del x
            if torch.cuda.is_available():
                y = torch.rand(10, 10).cuda()
                del y
            gc.collect()
        stats = prof.key_averages(group_by_input_shape=True)
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=[
                "aten::rand",
                "aten::empty"
            ],
            deallocs=[
                "[memory]"
            ]
        )
        if torch.cuda.is_available():
            check_metrics(
                stats,
                "cuda_memory_usage",
                deallocs=[
                    "[memory]"
                ]
            )

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_module_hierarchy(self):
        class A(nn.Module):
            def __init__(self):
                super(A, self).__init__()

            def my_new_method(self, x):
                return x * 3

            def forward_impl_(self, x, y):
                return self.my_new_method(x) + y

            def forward(self, x, y):
                y = y - 2
                return self.forward_impl_(x, y)

        class B(nn.Module):
            def __init__(self):
                super(B, self).__init__()

            def forward(self, x):
                return x + 2

        class C(nn.Module):
            def __init__(self):
                super(C, self).__init__()
                self.A0 = A()
                self.B0 = B()

            def call_b(self, x):
                return self.B0.forward(x)

            def forward(self, x, y):
                return self.A0.forward(x, y) + self.call_b(x)

        model = C()
        model = torch.jit.script(model)
        input_a = torch.rand(128, 128)
        input_b = torch.rand(128, 128)
        op_to_module_hierarchy = {}
        op_to_module_hierarchy["aten::sub"] = ["TOP(C)::forward.A0(A)::forward."]
        op_to_module_hierarchy["aten::mul"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.SELF(A)::my_new_method."]
        op_to_module_hierarchy["aten::add"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.",
            "TOP(C)::forward.SELF(C)::call_b.B0(B)::forward.", "TOP(C)::forward."]
        with TemporaryFileName(mode="w+") as fname:
            with profile(activities=[torch.profiler.ProfilerActivity.CPU], with_modules=True,) as prof:
                model(input_a, input_b)
            prof.export_chrome_trace(fname)
            with io.open(fname, 'r') as f:
                trace = json.load(f)
                assert "traceEvents" in trace
                events = trace["traceEvents"]
                found_memory_events = False
                for evt in events:
                    assert "name" in evt
                    if "args" in evt:
                        op_name = evt["name"]
                        if "Module Hierarchy" in evt["args"]:
                            hierarchy = evt["args"]["Module Hierarchy"]
                            if op_name in op_to_module_hierarchy:
                                assert hierarchy in op_to_module_hierarchy[op_name]

    def test_high_level_trace(self):
        """Checks that python side high level events are recorded.
        """
        class RepeatedDataset(torch.utils.data.Dataset):
            def __init__(self, N, D_in, D_out):
                self.N = N
                self.x = torch.randn(N, D_in)
                self.y = torch.randn(N, D_out)

            def __len__(self):
                return self.N

            def __getitem__(self, idx):
                return self.x, self.y

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super(TwoLayerNet, self).__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)

            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred

        class CustomSGD(torch.optim.SGD):
            def __init__(self, *args, **kwargs):
                super(CustomSGD, self).__init__(*args, **kwargs)

        def train():
            for _, data in enumerate(dataloader):
                x, y = data[0], data[1]
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        N, D_in, H, D_out = 8, 10, 5, 2
        model = TwoLayerNet(D_in, H, D_out)
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        ds = RepeatedDataset(N, D_in, D_out)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=1)

        try:
            train()
        except Exception:
            self.assertTrue(False, "Expected no exception without profiling.")

        # Create multiple instances, expect each func is hooked only one time.
        # Nested wrappers(repeated patching) will make following test fail.
        optimizer_duplicate = torch.optim.SGD(model.parameters(), lr=1e-4)
        dataloader_duplicate = torch.utils.data.DataLoader(ds, batch_size=1)

        def judge(expected_event_count, prof):
            actual_event_count = {}
            for e in prof.function_events:
                if "#" in e.name:
                    key = e.name
                    if key in expected_event_count.keys():
                        actual_event_count[key] = actual_event_count.setdefault(key, 0) + 1
            for key, count in expected_event_count.items():
                self.assertTrue((key in actual_event_count.keys()) and (count == actual_event_count[key]))

        with _profile(use_kineto=kineto_available()) as prof:
            train()
        expected_event_count = {
            # "+1" because the final iteration will enter __next__ but skip the loop body.
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#SGD.step": N,
            "Optimizer.zero_grad#SGD.zero_grad": N
        }
        judge(expected_event_count, prof)

        # Test on pickle/unpickle. Expect to work in multi-processing.
        optimizer = pickle.loads(pickle.dumps(optimizer))
        with _profile(use_kineto=kineto_available()) as prof:
            train()
        judge(expected_event_count, prof)

        # Test on customized optimizer.
        optimizer = CustomSGD(model.parameters(), lr=1e-4)
        with _profile(use_kineto=kineto_available()) as prof:
            train()
        expected_event_count = {
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#CustomSGD.step": N,
            "Optimizer.zero_grad#CustomSGD.zero_grad": N
        }
        judge(expected_event_count, prof)

    def test_flops(self):
        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        inputs = torch.randn(40, 16, 18, 260)
        with _profile(record_shapes=True, with_flops=True, use_kineto=kineto_available()) as prof:
            model(inputs)
        profiler_output = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10)
        self.assertIn("Total MFLOPs", profiler_output)
        if not (kineto_available() and torch.cuda.is_available()):
            return

        with profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
        ) as kineto_profiler:
            model(inputs)
        profiler_output = kineto_profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1)
        self.assertIn("Total MFLOPs", profiler_output)

    def test_kineto_profiler_api(self):
        called_num = [0]

        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with profile(activities=supported_activities()):
            self.payload(use_cuda=use_cuda)

        def trace_handler(p):
            output = p.key_averages().table(
                sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total", row_limit=-1)
            # print(output)
            # p.export_chrome_trace("/tmp/test_trace_" + str(called_num[0]) + ".json")
            called_num[0] += 1

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=trace_handler
        ) as p:
            for idx in range(8):
                self.payload(use_cuda=use_cuda)
                p.step()

        self.assertEqual(called_num[0], 2)

        # case without schedule
        with profile(
            activities=supported_activities()
        ) as p:
            self.payload(use_cuda=use_cuda)
            self.payload(use_cuda=use_cuda)
        output = p.key_averages().table(
            sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total", row_limit=-1)
        # print(output)

        test_schedule = torch.profiler.schedule(
            skip_first=2,
            wait=1,
            warmup=1,
            active=2,
            repeat=2)
        test_schedule_expected_outputs = [
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])

    def test_export_stacks(self):
        with _profile(with_stack=True, use_kineto=kineto_available()) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)
            z = z + y

        with TemporaryFileName(mode="w+") as fname:
            p.export_stacks(fname)
            with io.open(fname, 'r') as f:
                lines = f.readlines()
            assert len(lines) > 0, "Empty stacks file"
            for line in lines:
                is_int = False
                try:
                    assert int(line.split(" ")[-1]) > 0, "Invalid stacks record"
                    is_int = True
                except ValueError:
                    pass
                assert is_int, "Invalid stacks record"

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
    def test_tensorboard_trace_handler(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with _profile(use_cuda=use_cuda, use_kineto=True):
            self.payload(use_cuda=use_cuda)

        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU
                ] + ([
                    torch.profiler.ProfilerActivity.CUDA
                ] if use_cuda else []),
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2,
                    repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dname)
            ) as p:
                for _ in range(18):
                    self.payload(use_cuda=use_cuda)
                    p.step()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split('.')
                self.assertTrue(len(parts) > 4)
                self.assertTrue(parts[-4].isdigit() and int(parts[-4]) > 0, "Wrong tracing file name pattern")
                self.assertEqual(parts[-3:], ['pt', 'trace', 'json'])
                file_num += 1
            self.assertEqual(file_num, 3)

        # test case for gzip file format
        with TemporaryDirectoryName() as dname:
            p = profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU
                ] + ([
                    torch.profiler.ProfilerActivity.CUDA
                ] if use_cuda else []),
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2,
                    repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dname, use_gzip=True)
            )
            p.start()
            for _ in range(18):
                self.payload(use_cuda=use_cuda)
                p.step()
            p.stop()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split('.')
                self.assertTrue(len(parts) > 4)
                self.assertTrue(parts[-5].isdigit() and int(parts[-5]) > 0, "Wrong tracing file name pattern")
                self.assertEqual(parts[-4:], ['pt', 'trace', 'json', 'gz'])
                file_num += 1
            self.assertEqual(file_num, 3)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_profiler_metadata(self):
        t1, t2 = torch.ones(1), torch.ones(1)
        with profile() as prof:
            torch.add(t1, t2)
            prof.add_metadata("test_key1", "test_value1")
            prof.add_metadata_json("test_key2", "[1,2,3]")

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with io.open(fname, 'r') as f:
                trace = json.load(f)
                assert "test_key1" in trace
                assert trace["test_key1"] == "test_value1"
                assert "test_key2" in trace
                assert trace["test_key2"] == [1, 2, 3]

    def _test_profiler_tracing(self, use_kineto):
        with _profile(use_kineto=use_kineto) as prof:
            t1, t2 = torch.ones(1), torch.ones(1)
            torch.add(t1, t2)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # read the trace and expect valid json
            # if the JSON generated by export_chrome_trace is not valid, this will throw and fail the test.
            with io.open(fname, 'r') as f:
                json.load(f)

        # test empty trace
        with _profile(use_kineto=use_kineto) as prof:
            pass
        # saving an empty trace
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)

        # Same test but for cuda.
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        if not use_cuda:
            return

        device = torch.device("cuda:0")
        with _profile(use_cuda=True, use_kineto=use_kineto) as prof:
            t1, t2 = torch.ones(1, device=device), torch.ones(1, device=device)
            torch.add(t1, t2)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # Now validate the json
            with io.open(fname, 'r') as f:
                json.load(f)

    def test_profiler_tracing(self):
        self._test_profiler_tracing(False)
        if kineto_available():
            self._test_profiler_tracing(True)

    @unittest.skip("Disable forward->backward link to workaround profiler crash")
    def test_profiler_fwd_bwd_link(self):
        with _profile(use_kineto=True) as prof:
            t1, t2 = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with io.open(fname, 'r') as f:
                j = json.load(f)
                events = j["traceEvents"]
                ts_to_name = {}
                flow_s_to_ts = {}
                flow_f_to_ts = {}
                for e in events:
                    if e["ph"] == "X":
                        ts_to_name[e["ts"]] = e["name"]
                    if "cat" in e and "name" in e and e["cat"] == "forward_backward" and e["name"] == "fwd_bwd":
                        if e["ph"] == "s":
                            flow_s_to_ts[e["id"]] = e["ts"]
                        elif e["ph"] == "f":
                            flow_f_to_ts[e["id"]] = e["ts"]
                self.assertTrue(len(flow_s_to_ts) == 2)
                self.assertTrue(len(flow_f_to_ts) == 2)
                self.assertTrue(1 in flow_s_to_ts.keys())
                self.assertTrue(1 in flow_f_to_ts.keys())
                self.assertTrue(2 in flow_s_to_ts.keys())
                self.assertTrue(2 in flow_f_to_ts.keys())
                s_ts_1 = flow_s_to_ts[1]
                f_ts_1 = flow_f_to_ts[1]
                s_ts_2 = flow_s_to_ts[2]
                f_ts_2 = flow_f_to_ts[2]
                self.assertTrue(all([ts in ts_to_name.keys() for ts in [s_ts_1, f_ts_1, s_ts_2, f_ts_2]]))
                self.assertTrue(ts_to_name[s_ts_1] == "aten::binary_cross_entropy_with_logits")
                self.assertTrue(ts_to_name[s_ts_2] == "aten::add")

if __name__ == '__main__':
    run_tests()
