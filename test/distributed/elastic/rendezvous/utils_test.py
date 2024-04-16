# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import socket
import threading
import time
from datetime import timedelta
from typing import List
from unittest import TestCase
from unittest.mock import patch

from torch.distributed.elastic.rendezvous.utils import (
    _delay,
    _matches_machine_hostname,
    _parse_rendezvous_config,
    _PeriodicTimer,
    _try_parse_port,
    parse_rendezvous_endpoint,
)


class UtilsTest(TestCase):
    def test_parse_rendezvous_config_returns_dict(self) -> None:
        expected_config = {
            "a": "dummy1",
            "b": "dummy2",
            "c": "dummy3=dummy4",
            "d": "dummy5/dummy6",
        }

        config = _parse_rendezvous_config(
            " b= dummy2  ,c=dummy3=dummy4,  a =dummy1,d=dummy5/dummy6"
        )

        self.assertEqual(config, expected_config)

    def test_parse_rendezvous_returns_empty_dict_if_str_is_empty(self) -> None:
        config_strs = ["", "   "]

        for config_str in config_strs:
            with self.subTest(config_str=config_str):
                config = _parse_rendezvous_config(config_str)

                self.assertEqual(config, {})

    def test_parse_rendezvous_raises_error_if_str_is_invalid(self) -> None:
        config_strs = [
            "a=dummy1,",
            "a=dummy1,,c=dummy2",
            "a=dummy1,   ,c=dummy2",
            "a=dummy1,=  ,c=dummy2",
            "a=dummy1, = ,c=dummy2",
            "a=dummy1,  =,c=dummy2",
            " ,  ",
        ]

        for config_str in config_strs:
            with self.subTest(config_str=config_str):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous configuration string must be in format "
                    r"<key1>=<value1>,...,<keyN>=<valueN>.$",
                ):
                    _parse_rendezvous_config(config_str)

    def test_parse_rendezvous_raises_error_if_value_is_empty(self) -> None:
        config_strs = [
            "b=dummy1,a,c=dummy2",
            "b=dummy1,c=dummy2,a",
            "b=dummy1,a=,c=dummy2",
            "  a ",
        ]

        for config_str in config_strs:
            with self.subTest(config_str=config_str):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous configuration option 'a' must have a value specified.$",
                ):
                    _parse_rendezvous_config(config_str)

    def test_try_parse_port_returns_port(self) -> None:
        port = _try_parse_port("123")

        self.assertEqual(port, 123)

    def test_try_parse_port_returns_none_if_str_is_invalid(self) -> None:
        port_strs = [
            "",
            "   ",
            "  1",
            "1  ",
            " 1 ",
            "abc",
        ]

        for port_str in port_strs:
            with self.subTest(port_str=port_str):
                port = _try_parse_port(port_str)

                self.assertIsNone(port)

    def test_parse_rendezvous_endpoint_returns_tuple(self) -> None:
        endpoints = [
            "dummy.com:0",
            "dummy.com:123",
            "dummy.com:65535",
            "dummy-1.com:0",
            "dummy-1.com:123",
            "dummy-1.com:65535",
            "123.123.123.123:0",
            "123.123.123.123:123",
            "123.123.123.123:65535",
            "[2001:db8::1]:0",
            "[2001:db8::1]:123",
            "[2001:db8::1]:65535",
        ]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                host, port = parse_rendezvous_endpoint(endpoint, default_port=123)

                expected_host, expected_port = endpoint.rsplit(":", 1)

                if expected_host[0] == "[" and expected_host[-1] == "]":
                    expected_host = expected_host[1:-1]

                self.assertEqual(host, expected_host)
                self.assertEqual(port, int(expected_port))

    def test_parse_rendezvous_endpoint_returns_tuple_if_endpoint_has_no_port(
        self,
    ) -> None:
        endpoints = ["dummy.com", "dummy-1.com", "123.123.123.123", "[2001:db8::1]"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                host, port = parse_rendezvous_endpoint(endpoint, default_port=123)

                expected_host = endpoint

                if expected_host[0] == "[" and expected_host[-1] == "]":
                    expected_host = expected_host[1:-1]

                self.assertEqual(host, expected_host)
                self.assertEqual(port, 123)

    def test_parse_rendezvous_endpoint_returns_tuple_if_endpoint_is_empty(self) -> None:
        endpoints = ["", "  "]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                host, port = parse_rendezvous_endpoint("", default_port=123)

                self.assertEqual(host, "localhost")
                self.assertEqual(port, 123)

    def test_parse_rendezvous_endpoint_raises_error_if_hostname_is_invalid(
        self,
    ) -> None:
        endpoints = ["~", "dummy.com :123", "~:123", ":123"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The hostname of the rendezvous endpoint '{endpoint}' must be a "
                    r"dot-separated list of labels, an IPv4 address, or an IPv6 address.$",
                ):
                    parse_rendezvous_endpoint(endpoint, default_port=123)

    def test_parse_rendezvous_endpoint_raises_error_if_port_is_invalid(self) -> None:
        endpoints = ["dummy.com:", "dummy.com:abc", "dummy.com:-123", "dummy.com:-"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                    r"between 0 and 65536.$",
                ):
                    parse_rendezvous_endpoint(endpoint, default_port=123)

    def test_parse_rendezvous_endpoint_raises_error_if_port_is_too_big(self) -> None:
        endpoints = ["dummy.com:65536", "dummy.com:70000"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                    r"between 0 and 65536.$",
                ):
                    parse_rendezvous_endpoint(endpoint, default_port=123)

    def test_matches_machine_hostname_returns_true_if_hostname_is_loopback(
        self,
    ) -> None:
        hosts = [
            "localhost",
            "127.0.0.1",
            "::1",
            "0000:0000:0000:0000:0000:0000:0000:0001",
        ]

        for host in hosts:
            with self.subTest(host=host):
                self.assertTrue(_matches_machine_hostname(host))

    def test_matches_machine_hostname_returns_true_if_hostname_is_machine_hostname(
        self,
    ) -> None:
        host = socket.gethostname()

        self.assertTrue(_matches_machine_hostname(host))

    def test_matches_machine_hostname_returns_true_if_hostname_is_machine_fqdn(
        self,
    ) -> None:
        host = socket.getfqdn()

        self.assertTrue(_matches_machine_hostname(host))

    def test_matches_machine_hostname_returns_true_if_hostname_is_machine_address(
        self,
    ) -> None:
        addr_list = socket.getaddrinfo(
            socket.gethostname(), None, proto=socket.IPPROTO_TCP
        )

        for addr in (addr_info[4][0] for addr_info in addr_list):
            with self.subTest(addr=addr):
                self.assertTrue(_matches_machine_hostname(addr))

    def test_matches_machine_hostname_returns_false_if_hostname_does_not_match(
        self,
    ) -> None:
        hosts = ["dummy", "0.0.0.0", "::2"]

        for host in hosts:
            with self.subTest(host=host):
                self.assertFalse(_matches_machine_hostname(host))

    def test_delay_suspends_thread(self) -> None:
        for seconds in 0.2, (0.2, 0.4):
            with self.subTest(seconds=seconds):
                time1 = time.monotonic()

                _delay(seconds)  # type: ignore[arg-type]

                time2 = time.monotonic()

                self.assertGreaterEqual(time2 - time1, 0.2)

    @patch(
        "socket.getaddrinfo",
        side_effect=[
            [(None, None, 0, "a_host", ("1.2.3.4", 0))],
            [(None, None, 0, "a_different_host", ("1.2.3.4", 0))],
        ],
    )
    def test_matches_machine_hostname_returns_true_if_ip_address_match_between_hosts(
        self,
        _0,
    ) -> None:
        self.assertTrue(_matches_machine_hostname("a_host"))

    @patch(
        "socket.getaddrinfo",
        side_effect=[
            [(None, None, 0, "a_host", ("1.2.3.4", 0))],
            [(None, None, 0, "another_host_with_different_ip", ("1.2.3.5", 0))],
        ],
    )
    def test_matches_machine_hostname_returns_false_if_ip_address_not_match_between_hosts(
        self,
        _0,
    ) -> None:
        self.assertFalse(_matches_machine_hostname("a_host"))


class PeriodicTimerTest(TestCase):
    def test_start_can_be_called_only_once(self) -> None:
        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        timer.start()

        with self.assertRaisesRegex(RuntimeError, r"^The timer has already started.$"):
            timer.start()

        timer.cancel()

    def test_cancel_can_be_called_multiple_times(self) -> None:
        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        timer.start()

        timer.cancel()
        timer.cancel()

    def test_cancel_stops_background_thread(self) -> None:
        name = "PeriodicTimer_CancelStopsBackgroundThreadTest"

        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        timer.set_name(name)

        timer.start()

        self.assertTrue(any(t.name == name for t in threading.enumerate()))

        timer.cancel()

        self.assertTrue(all(t.name != name for t in threading.enumerate()))

    def test_delete_stops_background_thread(self) -> None:
        name = "PeriodicTimer_DeleteStopsBackgroundThreadTest"

        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        timer.set_name(name)

        timer.start()

        self.assertTrue(any(t.name == name for t in threading.enumerate()))

        del timer

        self.assertTrue(all(t.name != name for t in threading.enumerate()))

    def test_set_name_cannot_be_called_after_start(self) -> None:
        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        timer.start()

        with self.assertRaisesRegex(RuntimeError, r"^The timer has already started.$"):
            timer.set_name("dummy_name")

        timer.cancel()

    def test_timer_calls_background_thread_at_regular_intervals(self) -> None:
        timer_begin_time: float

        # Call our function every 200ms.
        call_interval = 0.2

        # Keep the log of intervals between each consecutive call.
        actual_call_intervals: List[float] = []

        # Keep the number of times the function was called.
        call_count = 0

        # In order to prevent a flaky test instead of asserting that the
        # function was called an exact number of times we use a lower bound
        # that is guaranteed to be true for a correct implementation.
        min_required_call_count = 4

        timer_stop_event = threading.Event()

        def log_call(self):
            nonlocal timer_begin_time, call_count

            actual_call_intervals.append(time.monotonic() - timer_begin_time)

            call_count += 1
            if call_count == min_required_call_count:
                timer_stop_event.set()

            timer_begin_time = time.monotonic()

        timer = _PeriodicTimer(timedelta(seconds=call_interval), log_call, self)

        timer_begin_time = time.monotonic()

        timer.start()

        # Although this is theoretically non-deterministic, if our timer, which
        # has a 200ms call interval, does not get called 4 times in 60 seconds,
        # there is very likely something else going on.
        timer_stop_event.wait(60)

        timer.cancel()

        self.longMessage = False

        self.assertGreaterEqual(
            call_count,
            min_required_call_count,
            f"The function has been called {call_count} time(s) but expected to be called at least "
            f"{min_required_call_count} time(s).",
        )

        for actual_call_interval in actual_call_intervals:
            self.assertGreaterEqual(
                actual_call_interval,
                call_interval,
                f"The interval between two function calls was {actual_call_interval} second(s) but "
                f"expected to be at least {call_interval} second(s).",
            )
