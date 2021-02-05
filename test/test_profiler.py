import collections
import gc
import io
import os
import unittest

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    TestCase, run_tests, TEST_WITH_ASAN, IS_WINDOWS, TemporaryFileName, TemporaryDirectoryName)
from torch.autograd.profiler import profile as _profile
from torch.profiler import (
    kineto_available, profile, record_function, DeviceType, ProfilerActivity
)

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

    def payload(self):
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.mm(x, y)
        z = z + y
        z = z.cpu()

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_kineto(self):
        with _profile(use_cuda=True, use_kineto=True):
            self.payload()

        # rerun to avoid initial start overhead
        with _profile(use_cuda=True, use_kineto=True) as p:
            self.payload()
        output = p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1)
        # print(output)
        found_gemm = False
        found_memcpy = False
        for e in p.function_events:
            if "gemm" in e.name:
                found_gemm = True
            if "Memcpy" in e.name or "memcpy" in e.name:
                found_memcpy = True
        self.assertTrue(found_gemm)
        self.assertTrue(found_memcpy)
        # p.export_chrome_trace("/tmp/test_trace.json")

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(not TEST_MULTIGPU, "Multiple GPUs needed")
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
        def run_profiler(tensor_creation_fn, metric):
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

        stats = run_profiler(create_cpu_tensor, "cpu_memory_usage")
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

        if torch.cuda.is_available():
            create_cuda_tensor()
            stats = run_profiler(create_cuda_tensor, "cuda_memory_usage")
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
            stats = run_profiler(create_mkldnn_tensor, "cpu_memory_usage")
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

        with _profile() as prof:
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
        with _profile() as prof:
            train()
        judge(expected_event_count, prof)

        # Test on customized optimizer.
        optimizer = CustomSGD(model.parameters(), lr=1e-4)
        with _profile() as prof:
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
        self.assertIn("FLOPS", profiler_output)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_kineto_profiler_api(self):
        called_num = [0]

        with _profile(use_cuda=True, use_kineto=True):
            self.payload()

        def trace_handler(p):
            output = p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1)
            # print(output)
            # p.export_chrome_trace("/tmp/test_trace_" + str(called_num[0]) + ".json")
            called_num[0] += 1

        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=trace_handler
        ) as p:
            for idx in range(8):
                self.payload()
                p.step()

        self.assertEqual(called_num[0], 2)

        # case without schedule
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]
        ) as p:
            self.payload()
            self.payload()
        output = p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1)
        # print(output)

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
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_tensorboard_trace_handler(self):
        with _profile(use_cuda=True, use_kineto=True):
            self.payload()

        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dname)
            ) as p:
                for _ in range(8):
                    self.payload()
                    p.step()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split('.')
                self.assertTrue(len(parts) > 4)
                self.assertTrue(parts[-4].isdigit() and int(parts[-4]) > 0, "Wrong tracing file name pattern")
                self.assertEqual(parts[-3:], ['pt', 'trace', 'json'])
                file_num += 1
            self.assertEqual(file_num, 2)


if __name__ == '__main__':
    run_tests()
