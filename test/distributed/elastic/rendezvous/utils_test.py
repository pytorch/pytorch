# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from torch.distributed.elastic.rendezvous.utils import (
    _parse_rendezvous_config,
    _parse_rendezvous_endpoint,
    _try_parse_port,
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
                host, port = _parse_rendezvous_endpoint(endpoint, default_port=123)

                expected_host, expected_port = endpoint.rsplit(":", 1)

                if expected_host[0] == "[" and expected_host[-1] == "]":
                    expected_host = expected_host[1:-1]

                self.assertEqual(host, expected_host)
                self.assertEqual(port, int(expected_port))

    def test_parse_rendezvous_endpoint_returns_tuple_if_endpoint_has_no_port(self) -> None:
        endpoints = ["dummy.com", "dummy-1.com", "123.123.123.123", "[2001:db8::1]"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                host, port = _parse_rendezvous_endpoint(endpoint, default_port=123)

                expected_host = endpoint

                if expected_host[0] == "[" and expected_host[-1] == "]":
                    expected_host = expected_host[1:-1]

                self.assertEqual(host, expected_host)
                self.assertEqual(port, 123)

    def test_parse_rendezvous_endpoint_returns_tuple_if_endpoint_is_empty(self) -> None:
        endpoints = ["", "  "]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                host, port = _parse_rendezvous_endpoint("", default_port=123)

                self.assertEqual(host, "localhost")
                self.assertEqual(port, 123)

    def test_parse_rendezvous_endpoint_raises_error_if_hostname_is_invalid(self) -> None:
        endpoints = ["~", "dummy.com :123", "~:123", ":123"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The hostname of the rendezvous endpoint '{endpoint}' must be a "
                    r"dot-separated list of labels, an IPv4 address, or an IPv6 address.$",
                ):
                    _parse_rendezvous_endpoint(endpoint, default_port=123)

    def test_parse_rendezvous_endpoint_raises_error_if_port_is_invalid(self) -> None:
        endpoints = ["dummy.com:", "dummy.com:abc", "dummy.com:-123", "dummy.com:-"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                    r"between 0 and 65536.$",
                ):
                    _parse_rendezvous_endpoint(endpoint, default_port=123)

    def test_parse_rendezvous_endpoint_raises_error_if_port_is_too_big(self) -> None:
        endpoints = ["dummy.com:65536", "dummy.com:70000"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                    r"between 0 and 65536.$",
                ):
                    _parse_rendezvous_endpoint(endpoint, default_port=123)
