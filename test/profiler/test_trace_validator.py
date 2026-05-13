# Owner(s): ["oncall: profiler"]

import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import profile, ProfilerActivity, record_function
from torch.profiler._trace_validator import (
    _check_backward_seq_id_uniqueness,
    _check_gpu_kernel_causality,
    _check_nccl_metadata,
    _check_stream_sync_overlap,
    _check_stream_wait_corr_id_in_past,
    _check_stream_wait_corr_id_populated,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


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


def _profile_resnet_payload(trace_path):
    """Profile a ResNet50 training loop and return events."""
    device = torch.device("cuda:0")
    model = _resnet50().to(device)
    inputs = torch.randn(4, 3, 224, 224, device=device)
    outputs = torch.rand_like(model(inputs))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        experimental_config=_ExperimentalConfig(enable_cuda_sync_events=True),
    ) as prof:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with record_function("## forward ##"):
                pred = model(inputs)
            with record_function("## backward ##"):
                loss_fn(pred, outputs).backward()
            with record_function("## optimizer ##"):
                optimizer.step()

    torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    return _load_events(trace_path)


def _profile_complex_payload(trace_path):
    """Profile multi-stream + CUDA events + forward/backward and return events."""
    device = torch.device("cuda:0")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        experimental_config=_ExperimentalConfig(enable_cuda_sync_events=True),
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

    torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    return _load_events(trace_path)


def _load_events(trace_path):
    with open(trace_path) as f:
        data = json.load(f)
    return data.get("traceEvents", data)


class TestTraceValidatorRules(TestCase):
    """Synthetic tests for rules not exercised by real E2E payloads."""

    def test_nccl_metadata_pass(self):
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
        self.assertEqual(_check_nccl_metadata(events), [])

    def test_nccl_metadata_fail(self):
        events = [
            {
                "ph": "X",
                "name": "record_param_comms",
                "ts": 100,
                "dur": 10,
                "args": {"Collective name": "all_reduce"},
            },
        ]
        v = _check_nccl_metadata(events)
        self.assertEqual(len(v), 1)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfTorchDynamo("profiler tests do not work with dynamo")
@instantiate_parametrized_tests
class TestTraceValidatorE2E(TestCase):
    _trace_dir: str = ""
    _payloads: dict = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._trace_dir = tempfile.mkdtemp(prefix="profiler_e2e_trace_")
        cls._payloads = {
            "resnet": _profile_resnet_payload(
                os.path.join(cls._trace_dir, "resnet.json")
            ),
            "complex": _profile_complex_payload(
                os.path.join(cls._trace_dir, "complex.json")
            ),
        }

    @classmethod
    def tearDownClass(cls):
        if cls._trace_dir and os.path.isdir(cls._trace_dir):
            shutil.rmtree(cls._trace_dir, ignore_errors=True)
        super().tearDownClass()

    def _events(self, payload):
        return self._payloads[payload]

    @staticmethod
    def _fmt(violations, limit=5):
        lines = [f"  {v}" for v in violations[:limit]]
        if len(violations) > limit:
            lines.append(f"  ... and {len(violations) - limit} more")
        return "\n".join(lines)

    # TODO: unskip once kineto fixes CPU/GPU timestamp synchronization
    @unittest.skip(
        "kineto reports GPU kernel timestamps before cudaLaunchKernel due to clock skew"
    )
    @parametrize("payload", ["resnet", "complex"])
    def test_gpu_kernel_causality(self, payload):
        v = _check_gpu_kernel_causality(self._events(payload))
        self.assertEqual(len(v), 0, self._fmt(v))

    # TODO: unskip once kineto populates wait_on_cuda_event_record_corr_id for cuStreamWaitEvent
    @unittest.skip(
        "kineto does not populate wait_on_cuda_event_record_corr_id for stream wait events (returns -1)"
    )
    @parametrize("payload", ["resnet", "complex"])
    def test_stream_wait_corr_id_populated(self, payload):
        v = _check_stream_wait_corr_id_populated(self._events(payload))
        self.assertEqual(len(v), 0, self._fmt(v))

    # TODO: unskip once kineto stream sync event emission is verified in integration testing
    @unittest.skip(
        "kineto stream sync overlap detection not yet verified in kineto integration testing"
    )
    @parametrize("payload", ["resnet", "complex"])
    def test_stream_sync_overlap(self, payload):
        v = _check_stream_sync_overlap(self._events(payload))
        self.assertEqual(len(v), 0, self._fmt(v))

    # TODO: unskip once kineto populates wait_on_cuda_event_record_corr_id for cuStreamWaitEvent
    @unittest.skip(
        "kineto wait_on_cuda_event_record_corr_id temporal ordering not yet verified in kineto integration testing"
    )
    @parametrize("payload", ["resnet", "complex"])
    def test_stream_wait_corr_id_in_past(self, payload):
        v = _check_stream_wait_corr_id_in_past(self._events(payload))
        self.assertEqual(len(v), 0, self._fmt(v))

    # TODO: unskip once kineto NCCL collective metadata is verified in integration testing
    @unittest.skip(
        "kineto NCCL collective metadata not yet verified in kineto integration testing"
    )
    @parametrize("payload", ["resnet", "complex"])
    def test_nccl_metadata(self, payload):
        v = _check_nccl_metadata(self._events(payload))
        self.assertEqual(len(v), 0, self._fmt(v))

    # TODO: unskip once kineto backward sequence ID emission is verified in integration testing
    @unittest.skip(
        "kineto backward sequence ID uniqueness not yet verified in kineto integration testing"
    )
    @parametrize("payload", ["resnet", "complex"])
    def test_backward_seq_id_uniqueness(self, payload):
        v = _check_backward_seq_id_uniqueness(self._events(payload))
        self.assertEqual(len(v), 0, self._fmt(v))


if __name__ == "__main__":
    run_tests()
