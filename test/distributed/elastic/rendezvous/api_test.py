# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, Dict, SupportsInt
from unittest import TestCase

from torch.distributed.elastic.rendezvous import (
    RendezvousHandler,
    RendezvousHandlerRegistry,
    RendezvousInfo,
    RendezvousParameters,
)


class RendezvousParametersTest(TestCase):
    def setUp(self) -> None:
        self._backend = "dummy_backend"
        self._endpoint = "dummy_endpoint"
        self._run_id = "dummy_run_id"
        self._min_nodes = 3
        self._max_nodes = 6
        self._kwargs: Dict[str, Any] = {}

    def _create_params(self) -> RendezvousParameters:
        return RendezvousParameters(
            backend=self._backend,
            endpoint=self._endpoint,
            run_id=self._run_id,
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            **self._kwargs,
        )

    def test_init_initializes_params(self) -> None:
        self._kwargs["dummy_param"] = "x"

        params = self._create_params()

        self.assertEqual(params.backend, self._backend)
        self.assertEqual(params.endpoint, self._endpoint)
        self.assertEqual(params.run_id, self._run_id)
        self.assertEqual(params.min_nodes, self._min_nodes)
        self.assertEqual(params.max_nodes, self._max_nodes)

        self.assertEqual(params.get("dummy_param"), "x")

    def test_init_initializes_params_if_min_nodes_equals_to_1(self) -> None:
        self._min_nodes = 1

        params = self._create_params()

        self.assertEqual(params.min_nodes, self._min_nodes)
        self.assertEqual(params.max_nodes, self._max_nodes)

    def test_init_initializes_params_if_min_and_max_nodes_are_equal(self) -> None:
        self._max_nodes = 3

        params = self._create_params()

        self.assertEqual(params.min_nodes, self._min_nodes)
        self.assertEqual(params.max_nodes, self._max_nodes)

    def test_init_raises_error_if_backend_is_none_or_empty(self) -> None:
        for backend in [None, ""]:
            with self.subTest(backend=backend):
                self._backend = backend  # type: ignore[assignment]

                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous backend name must be a non-empty string.$",
                ):
                    self._create_params()

    def test_init_raises_error_if_min_nodes_is_less_than_1(self) -> None:
        for min_nodes in [0, -1, -5]:
            with self.subTest(min_nodes=min_nodes):
                self._min_nodes = min_nodes

                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The minimum number of rendezvous nodes \({min_nodes}\) must be greater "
                    rf"than zero.$",
                ):
                    self._create_params()

    def test_init_raises_error_if_max_nodes_is_less_than_min_nodes(self) -> None:
        for max_nodes in [2, 1, -2]:
            with self.subTest(max_nodes=max_nodes):
                self._max_nodes = max_nodes

                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The maximum number of rendezvous nodes \({max_nodes}\) must be greater "
                    "than or equal to the minimum number of rendezvous nodes "
                    rf"\({self._min_nodes}\).$",
                ):
                    self._create_params()

    def test_get_returns_none_if_key_does_not_exist(self) -> None:
        params = self._create_params()

        self.assertIsNone(params.get("dummy_param"))

    def test_get_returns_default_if_key_does_not_exist(self) -> None:
        params = self._create_params()

        self.assertEqual(params.get("dummy_param", default="x"), "x")

    def test_get_as_bool_returns_none_if_key_does_not_exist(self) -> None:
        params = self._create_params()

        self.assertIsNone(params.get_as_bool("dummy_param"))

    def test_get_as_bool_returns_default_if_key_does_not_exist(self) -> None:
        params = self._create_params()

        self.assertTrue(params.get_as_bool("dummy_param", default=True))

    def test_get_as_bool_returns_true_if_value_represents_true(self) -> None:
        for value in ["1", "True", "tRue", "T", "t", "yEs", "Y", 1, True]:
            with self.subTest(value=value):
                self._kwargs["dummy_param"] = value

                params = self._create_params()

                self.assertTrue(params.get_as_bool("dummy_param"))

    def test_get_as_bool_returns_false_if_value_represents_false(self) -> None:
        for value in ["0", "False", "faLse", "F", "f", "nO", "N", 0, False]:
            with self.subTest(value=value):
                self._kwargs["dummy_param"] = value

                params = self._create_params()

                self.assertFalse(params.get_as_bool("dummy_param"))

    def test_get_as_bool_raises_error_if_value_is_invalid(self) -> None:
        for value in ["01", "Flse", "Ture", "g", "4", "_", "truefalse", 2, -1]:
            with self.subTest(value=value):
                self._kwargs["dummy_param"] = value

                params = self._create_params()

                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous configuration option 'dummy_param' does not represent a "
                    r"valid boolean value.$",
                ):
                    params.get_as_bool("dummy_param")

    def test_get_as_int_returns_none_if_key_does_not_exist(self) -> None:
        params = self._create_params()

        self.assertIsNone(params.get_as_int("dummy_param"))

    def test_get_as_int_returns_default_if_key_does_not_exist(self) -> None:
        params = self._create_params()

        self.assertEqual(params.get_as_int("dummy_param", default=5), 5)

    def test_get_as_int_returns_integer_if_value_represents_integer(self) -> None:
        for value in ["0", "-10", "5", "  4", "4  ", " 4 ", 0, -4, 3]:
            with self.subTest(value=value):
                self._kwargs["dummy_param"] = value

                params = self._create_params()

                self.assertEqual(
                    params.get_as_int("dummy_param"), int(cast(SupportsInt, value))
                )

    def test_get_as_int_raises_error_if_value_is_invalid(self) -> None:
        for value in ["a", "0a", "3b", "abc"]:
            with self.subTest(value=value):
                self._kwargs["dummy_param"] = value

                params = self._create_params()

                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous configuration option 'dummy_param' does not represent a "
                    r"valid integer value.$",
                ):
                    params.get_as_int("dummy_param")


class _DummyRendezvousHandler(RendezvousHandler):
    def __init__(self, params: RendezvousParameters) -> None:
        self.params = params

    def get_backend(self) -> str:
        return "dummy_backend"

    def next_rendezvous(self) -> RendezvousInfo:
        raise NotImplementedError

    def is_closed(self) -> bool:
        return False

    def set_closed(self) -> None:
        pass

    def num_nodes_waiting(self) -> int:
        return 0

    def get_run_id(self) -> str:
        return ""

    def shutdown(self) -> bool:
        return False


class RendezvousHandlerRegistryTest(TestCase):
    def setUp(self) -> None:
        self._params = RendezvousParameters(
            backend="dummy_backend",
            endpoint="dummy_endpoint",
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
        )

        self._registry = RendezvousHandlerRegistry()

    @staticmethod
    def _create_handler(params: RendezvousParameters) -> RendezvousHandler:
        return _DummyRendezvousHandler(params)

    def test_register_registers_once_if_called_twice_with_same_creator(self) -> None:
        self._registry.register("dummy_backend", self._create_handler)
        self._registry.register("dummy_backend", self._create_handler)

    def test_register_raises_error_if_called_twice_with_different_creators(
        self,
    ) -> None:
        self._registry.register("dummy_backend", self._create_handler)

        other_create_handler = lambda p: _DummyRendezvousHandler(p)  # noqa: E731

        with self.assertRaisesRegex(
            ValueError,
            r"^The rendezvous backend 'dummy_backend' cannot be registered with "
            rf"'{other_create_handler}' as it is already registered with '{self._create_handler}'.$",
        ):
            self._registry.register("dummy_backend", other_create_handler)

    def test_create_handler_returns_handler(self) -> None:
        self._registry.register("dummy_backend", self._create_handler)

        handler = self._registry.create_handler(self._params)

        self.assertIsInstance(handler, _DummyRendezvousHandler)

        self.assertIs(handler.params, self._params)

    def test_create_handler_raises_error_if_backend_is_not_registered(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"^The rendezvous backend 'dummy_backend' is not registered. Did you forget to call "
            r"`register`\?$",
        ):
            self._registry.create_handler(self._params)

    def test_create_handler_raises_error_if_backend_names_do_not_match(self) -> None:
        self._registry.register("dummy_backend_2", self._create_handler)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^The rendezvous backend 'dummy_backend' does not match the requested backend "
            r"'dummy_backend_2'.$",
        ):
            self._params.backend = "dummy_backend_2"

            self._registry.create_handler(self._params)
