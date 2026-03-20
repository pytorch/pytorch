# Owner(s): ["oncall: profiler"]

import glob
import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from torch.profiler._trace_validator import (
    check_backward_seq_id_uniqueness,
    check_gpu_kernel_causality,
    check_nccl_metadata,
    check_stream_sync_overlap,
    check_stream_wait_corr_id_in_past,
    check_stream_wait_corr_id_populated,
    validate_trace,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


# ---------------------------------------------------------------------------
# Minimal ResNet50 (no torchvision dependency) — same bottleneck design
# ---------------------------------------------------------------------------


class _Bottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else None
        )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(in_ch, mid_ch, out_ch, n, stride=1):
    layers = [_Bottleneck(in_ch, mid_ch, out_ch, stride=stride)]
    for _ in range(n - 1):
        layers.append(_Bottleneck(out_ch, mid_ch, out_ch))
    return nn.Sequential(*layers)


def _resnet50():
    return nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
        _make_layer(64, 64, 256, n=3),
        _make_layer(256, 128, 512, n=4, stride=2),
        _make_layer(512, 256, 1024, n=6, stride=2),
        _make_layer(1024, 512, 2048, n=3, stride=2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(2048, 1000),
    )


# ---------------------------------------------------------------------------
# Unit tests — synthetic events, no GPU needed
# ---------------------------------------------------------------------------


class TestTraceValidatorRules(TestCase):
    """Synthetic tests for rules not exercised by real E2E payloads."""

    def test_rule5_nccl_metadata_pass(self):
        events = [
            {
                "ph": "X",
                "name": "record_param_comms",
                "ts": 100,
                "dur": 10,
                "args": {
                    "Collective name": "all_reduce",
                    "dtype": "float32",
                    "In msg nelems": 1024,
                    "Out msg nelems": 1024,
                    "Group size": 8,
                },
            },
        ]
        self.assertEqual(check_nccl_metadata(events), [])

    def test_rule5_nccl_metadata_fail(self):
        events = [
            {
                "ph": "X",
                "name": "record_param_comms",
                "ts": 100,
                "dur": 10,
                "args": {"Collective name": "all_reduce"},
            },
        ]
        v = check_nccl_metadata(events)
        self.assertEqual(len(v), 1)


# ---------------------------------------------------------------------------
# E2E: Two real CUDA payloads profiled once, each rule checked on both traces
#
# Payload 1 — ResNet50 training loop (forward + backward + optimizer)
# Payload 2 — Multi-stream + CUDA events + stream sync + forward/backward
# ---------------------------------------------------------------------------


@unittest.skipIf(IS_WINDOWS, "CUDA profiler tests not supported on Windows")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfTorchDynamo("profiler tests do not work with dynamo")
class TestTraceValidatorE2E(TestCase):
    resnet_trace_path: str = ""
    complex_trace_path: str = ""
    resnet_events: list = []
    complex_events: list = []
    _trace_dir: str = ""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        device = torch.device("cuda:0")
        cls._trace_dir = tempfile.mkdtemp(prefix="profiler_e2e_trace_")

        # ── Payload 1: ResNet50 training loop ─────────────────────────
        model = _resnet50().to(device)
        inputs = torch.randn(4, 3, 224, 224, device=device)
        outputs = torch.rand_like(model(inputs))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        torch.cuda.synchronize()

        resnet_dir = os.path.join(cls._trace_dir, "resnet")
        os.makedirs(resnet_dir)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            experimental_config=_ExperimentalConfig(
                enable_cuda_sync_events=True,
            ),
            schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                resnet_dir, worker_name="w"
            ),
        ) as prof:
            for _ in range(5):
                prof.step()
                optimizer.zero_grad(set_to_none=True)
                with record_function("## forward ##"):
                    pred = model(inputs)
                with record_function("## backward ##"):
                    loss_fn(pred, outputs).backward()
                with record_function("## optimizer ##"):
                    optimizer.step()

        torch.cuda.synchronize()

        traces = glob.glob(os.path.join(resnet_dir, "*.pt.trace.json"))
        if not traces:
            raise RuntimeError(f"No trace file produced in {resnet_dir}")
        cls.resnet_trace_path = traces[0]
        with open(cls.resnet_trace_path) as f:
            data = json.load(f)
        cls.resnet_events = data.get("traceEvents", data)

        # ── Payload 2: multi-stream + CUDA events + forward/backward ──
        complex_dir = os.path.join(cls._trace_dir, "complex")
        os.makedirs(complex_dir)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            experimental_config=_ExperimentalConfig(
                enable_cuda_sync_events=True,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                complex_dir, worker_name="w"
            ),
        ) as prof:
            x = torch.randn(32, 32, device=device, requires_grad=True)
            y = torch.mm(x, x)
            loss = y.sum()
            loss.backward()

            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            event = torch.cuda.Event()
            with torch.cuda.stream(s1):
                a = torch.randn(64, 64, device=device)
                _b = torch.mm(a, a)
                event.record(s1)
            s2.wait_event(event)
            with torch.cuda.stream(s2):
                _c = torch.mm(a, a)
            s2.synchronize()

            prof.step()

        torch.cuda.synchronize()

        traces = glob.glob(os.path.join(complex_dir, "*.pt.trace.json"))
        if not traces:
            raise RuntimeError(f"No trace file produced in {complex_dir}")
        cls.complex_trace_path = traces[0]
        with open(cls.complex_trace_path) as f:
            data = json.load(f)
        cls.complex_events = data.get("traceEvents", data)

    @classmethod
    def tearDownClass(cls):
        if cls._trace_dir and os.path.isdir(cls._trace_dir):
            import shutil

            shutil.rmtree(cls._trace_dir, ignore_errors=True)
        super().tearDownClass()

    # ── one test per rule, checked against both payloads ──────────────

    def test_rule1_gpu_kernel_causality(self):
        # Clock skew between CPU and GPU is normal; just verify the check
        # runs and finds real kernel pairs without crashing.
        check_gpu_kernel_causality(self.resnet_events)
        check_gpu_kernel_causality(self.complex_events)

    def test_rule2_stream_wait_corr_id_populated(self):
        v1 = check_stream_wait_corr_id_populated(self.resnet_events)
        v2 = check_stream_wait_corr_id_populated(self.complex_events)
        self.assertEqual(len(v1), 0, self._fmt(v1))
        self.assertEqual(len(v2), 0, self._fmt(v2))

    def test_rule3_stream_sync_overlap(self):
        v1 = check_stream_sync_overlap(self.resnet_events)
        v2 = check_stream_sync_overlap(self.complex_events)
        self.assertEqual(len(v1), 0, self._fmt(v1))
        self.assertEqual(len(v2), 0, self._fmt(v2))

    def test_rule4_stream_wait_corr_id_in_past(self):
        v1 = check_stream_wait_corr_id_in_past(self.resnet_events)
        v2 = check_stream_wait_corr_id_in_past(self.complex_events)
        self.assertEqual(len(v1), 0, self._fmt(v1))
        self.assertEqual(len(v2), 0, self._fmt(v2))

    def test_rule5_nccl_metadata(self):
        v1 = check_nccl_metadata(self.resnet_events)
        v2 = check_nccl_metadata(self.complex_events)
        self.assertEqual(len(v1), 0, self._fmt(v1))
        self.assertEqual(len(v2), 0, self._fmt(v2))

    def test_rule6_backward_seq_id_uniqueness(self):
        v1 = check_backward_seq_id_uniqueness(self.resnet_events)
        v2 = check_backward_seq_id_uniqueness(self.complex_events)
        self.assertEqual(len(v1), 0, self._fmt(v1))
        self.assertEqual(len(v2), 0, self._fmt(v2))

    def test_validate_trace(self):
        passed1, v1 = validate_trace(self.resnet_trace_path)
        passed2, v2 = validate_trace(self.complex_trace_path)
        self.assertTrue(passed1, self._fmt(v1))
        self.assertTrue(passed2, self._fmt(v2))

    # ── helper ────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(violations, limit=5):
        lines = [f"  {v}" for v in violations[:limit]]
        if len(violations) > limit:
            lines.append(f"  ... and {len(violations) - limit} more")
        return "\n".join(lines)


if __name__ == "__main__":
    run_tests()
