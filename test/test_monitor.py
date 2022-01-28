# Owner(s): ["oncall: r2p"]

from torch.testing._internal.common_utils import (
    TestCase, run_tests,
)

from datetime import timedelta, datetime
import time

from torch.monitor import (
    Aggregation,
    FixedCountStat,
    IntervalStat,
    Event,
    log_event,
    register_event_handler,
    unregister_event_handler,
    Stat,
)

class TestMonitor(TestCase):
    def test_interval_stat(self) -> None:
        events = []

        def handler(event):
            events.append(event)

        handle = register_event_handler(handler)
        s = IntervalStat(
            "asdf",
            (Aggregation.SUM, Aggregation.COUNT),
            timedelta(milliseconds=1),
        )
        self.assertIsInstance(s, Stat)
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
        s = FixedCountStat(
            "asdf",
            (Aggregation.SUM, Aggregation.COUNT),
            3,
        )
        self.assertIsInstance(s, Stat)
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


if __name__ == '__main__':
    run_tests()
