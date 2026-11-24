# Owner(s): ["oncall: r2p"]

import os
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import torch
from torch.monitor import (
    _WaitCounter,
    Aggregation,
    Event,
    log_event,
    register_event_handler,
    Stat,
    TensorboardEventHandler,
    unregister_event_handler,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


class TestMonitor(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_interval_stat(self) -> None:
        events = []

        def handler(event):
            events.append(event)

        handle = register_event_handler(handler)
        s = Stat(
            "asdf",
            (Aggregation.SUM, Aggregation.COUNT),
            timedelta(milliseconds=1),
        )
        self.assertEqual(s.name, "asdf")

        s.add(2)
        for _ in range(100):
            # NOTE: different platforms sleep may be inaccurate so we loop
            # instead (i.e. win)
            time.sleep(1 / 1000)  # ms
            s.add(3)
            if len(events) >= 1:
                break
        self.assertGreaterEqual(len(events), 1)
        unregister_event_handler(handle)

    def test_fixed_count_stat(self) -> None:
        s = Stat(
            "asdf",
            (Aggregation.SUM, Aggregation.COUNT),
            timedelta(hours=100),
            3,
        )
        s.add(1)
        s.add(2)
        name = s.name
        self.assertEqual(name, "asdf")
        self.assertEqual(s.count, 2)
        s.add(3)
        self.assertEqual(s.count, 0)
        self.assertEqual(s.get(), {Aggregation.SUM: 6.0, Aggregation.COUNT: 3})

    def test_log_event(self) -> None:
        e = Event(
            name="torch.monitor.TestEvent",
            timestamp=datetime.now(),
            data={
                "str": "a string",
                "float": 1234.0,
                "int": 1234,
            },
        )
        self.assertEqual(e.name, "torch.monitor.TestEvent")
        self.assertIsNotNone(e.timestamp)
        self.assertIsNotNone(e.data)
        log_event(e)

    @skipIfTorchDynamo("Really weird error")
    def test_event_handler(self) -> None:
        events = []

        def handler(event: Event) -> None:
            events.append(event)

        handle = register_event_handler(handler)
        e = Event(
            name="torch.monitor.TestEvent",
            timestamp=datetime.now(),
            data={},
        )
        log_event(e)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], e)
        log_event(e)
        self.assertEqual(len(events), 2)

        unregister_event_handler(handle)
        log_event(e)
        self.assertEqual(len(events), 2)

    def test_wait_counter(self) -> None:
        wait_counter = _WaitCounter(
            "test_wait_counter",
        )
        with wait_counter.guard():
            pass


@skipIfTorchDynamo("CUDA availability mocks interfere with Dynamo's CUDA RNG setup")
class TestCUDAIsAvailableWaitCounter(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        self.events = []
        self.counter = MagicMock()
        guard = self.counter.return_value.guard.return_value
        guard.__enter__.side_effect = lambda: self.events.append("enter")
        guard.__exit__.side_effect = lambda *args: self.events.append("exit")
        for p in (
            patch.object(torch._C._monitor, "_WaitCounter", self.counter),
            patch.object(torch.cuda, "_is_compiled", return_value=True),
            patch.object(torch.cuda, "device_count", side_effect=self.probe),
            patch.object(
                torch._C, "_cuda_getDeviceCount", create=True, side_effect=self.probe
            ),
        ):
            p.start()
            self.addCleanup(p.stop)

    def probe(self):
        self.events.append("probe")
        return 1

    def test_not_compiled(self):
        with patch.object(torch.cuda, "_is_compiled", return_value=False):
            self.assertFalse(torch.cuda.is_available())
        self.counter.assert_not_called()

    def test_probe_runs_inside_guard(self):
        key = "pytorch.wait_counter.cuda.is_available"
        for nvml_check in ("0", "1"):
            self.events.clear()
            with (
                self.subTest(nvml_check=nvml_check),
                patch.dict(os.environ, {"PYTORCH_NVML_BASED_CUDA_CHECK": nvml_check}),
            ):
                self.assertTrue(torch.cuda.is_available())
                self.assertEqual(self.events, ["enter", "probe", "exit"])
                self.counter.assert_called_with(key)

    def test_probe_error_propagates(self):
        error = RuntimeError("probe failed")
        torch._C._cuda_getDeviceCount.side_effect = error
        with patch.dict(os.environ, {"PYTORCH_NVML_BASED_CUDA_CHECK": "0"}):
            with self.assertRaises(RuntimeError) as caught:
                torch.cuda.is_available()
        self.assertIs(caught.exception, error)
        self.assertEqual(self.events, ["enter", "exit"])


@skipIfTorchDynamo("Really weird error")
class TestMonitorTensorboard(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        global SummaryWriter, event_multiplexer
        try:
            from tensorboard.backend.event_processing import (
                plugin_event_multiplexer as event_multiplexer,
            )
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return self.skipTest("Skip the test since TensorBoard is not installed")
        self.temp_dirs = []

    def create_summary_writer(self):
        temp_dir = tempfile.TemporaryDirectory()  # noqa: P201
        self.temp_dirs.append(temp_dir)
        return SummaryWriter(temp_dir.name)

    def tearDown(self):
        # Remove directories created by SummaryWriter
        for temp_dir in self.temp_dirs:
            temp_dir.cleanup()

    def test_event_handler(self):
        with self.create_summary_writer() as w:
            handle = register_event_handler(TensorboardEventHandler(w))

            s = Stat(
                "asdf",
                (Aggregation.SUM, Aggregation.COUNT),
                timedelta(hours=1),
                5,
            )
            for i in range(10):
                s.add(i)
            self.assertEqual(s.count, 0)

            unregister_event_handler(handle)

        mul = event_multiplexer.EventMultiplexer()
        mul.AddRunsFromDirectory(self.temp_dirs[-1].name)
        mul.Reload()
        scalar_dict = mul.PluginRunToTagToContent("scalars")
        raw_result = {
            tag: mul.Tensors(run, tag)
            for run, run_dict in scalar_dict.items()
            for tag in run_dict
        }
        scalars = {
            tag: [e.tensor_proto.float_val[0] for e in events]
            for tag, events in raw_result.items()
        }
        self.assertEqual(
            scalars,
            {
                "asdf.sum": [10],
                "asdf.count": [5],
            },
        )


if __name__ == "__main__":
    run_tests()
