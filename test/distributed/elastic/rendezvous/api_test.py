# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from typing import cast, Any, Dict, SupportsInt, Tuple

from torch.distributed.elastic.rendezvous import (
    RendezvousHandler,
    RendezvousHandlerFactory,
    RendezvousParameters,
)


def create_mock_rdzv_handler(ignored: RendezvousParameters) -> RendezvousHandler:
    return MockRendezvousHandler()


class MockRendezvousHandler(RendezvousHandler):
    def next_rendezvous(
        self,
        # pyre-ignore[11]: Annotation `Store` is not defined as a type.
    ) -> Tuple["torch.distributed.Store", int, int]:  # noqa F821
        raise NotImplementedError()

    def get_backend(self) -> str:
        return "mock"

    def is_closed(self) -> bool:
        return False

    def set_closed(self):
        pass

    def num_nodes_waiting(self) -> int:
        return -1

    def get_run_id(self) -> str:
        return ""


class RendezvousHandlerFactoryTest(TestCase):
    def test_double_registration(self):
        factory = RendezvousHandlerFactory()
        factory.register("mock", create_mock_rdzv_handler)
        with self.assertRaises(ValueError):
            factory.register("mock", create_mock_rdzv_handler)

    def test_no_factory_method_found(self):
        factory = RendezvousHandlerFactory()
        rdzv_params = RendezvousParameters(
            backend_name="mock", endpoint="", run_id="foobar", min_num_nodes=1, max_num_nodes=2
        )

        with self.assertRaises(ValueError):
            factory.create_handler(rdzv_params)

    def test_create_handler(self):
        rdzv_params = RendezvousParameters(
            backend_name="mock", endpoint="", run_id="foobar", min_num_nodes=1, max_num_nodes=2
        )

        factory = RendezvousHandlerFactory()
        factory.register("mock", create_mock_rdzv_handler)
        mock_rdzv_handler = factory.create_handler(rdzv_params)
        self.assertTrue(isinstance(mock_rdzv_handler, MockRendezvousHandler))


class RendezvousParametersTest(TestCase):
    def setUp(self) -> None:
        self._backend_name = "dummy_backend"
        self._endpoint = "dummy_endpoint"
        self._run_id = "dummy_run_id"
        self._min_num_nodes = 3
        self._max_num_nodes = 6
        self._kwargs: Dict[str, Any] = {}

    def _create_params(self) -> RendezvousParameters:
        return RendezvousParameters(
            backend_name=self._backend_name,
            endpoint=self._endpoint,
            run_id=self._run_id,
            min_num_nodes=self._min_num_nodes,
            max_num_nodes=self._max_num_nodes,
            **self._kwargs,
        )

    def test_init_initializes_params(self) -> None:
        self._kwargs["dummy_param"] = "x"

        params = self._create_params()

        self.assertEqual(params.backend_name, self._backend_name)
        self.assertEqual(params.endpoint, self._endpoint)
        self.assertEqual(params.run_id, self._run_id)
        self.assertEqual(params.min_num_nodes, self._min_num_nodes)
        self.assertEqual(params.max_num_nodes, self._max_num_nodes)

        self.assertEqual(params.get("dummy_param"), "x")

    def test_init_initializes_params_if_min_num_nodes_equals_to_max_num_nodes(self) -> None:
        self._max_num_nodes = 3

        params = self._create_params()

        self.assertEqual(params.min_num_nodes, self._min_num_nodes)
        self.assertEqual(params.max_num_nodes, self._max_num_nodes)

    def test_init_raises_error_if_backend_name_is_none_or_empty(self) -> None:
        for backend_name in [None, ""]:
            with self.subTest(backend_name=backend_name):
                self._backend_name = backend_name  # type: ignore[assignment]

                with self.assertRaisesRegex(
                    ValueError, r"^The backend name must be a non-empty string.$"
                ):
                    self._create_params()

    def test_init_raises_error_if_min_num_nodes_is_less_than_1(self) -> None:
        for min_num_nodes in [0, -1, -5]:
            with self.subTest(min_num_nodes=min_num_nodes):
                self._min_num_nodes = min_num_nodes

                with self.assertRaisesRegex(
                    ValueError,
                    r"^The minimum number of nodes must be greater than zero.$",
                ):
                    self._create_params()

    def test_init_raises_error_if_max_num_nodes_is_less_than_min_num_nodes(self) -> None:
        for max_num_nodes in [2, 1, -2]:
            with self.subTest(max_num_nodes=max_num_nodes):
                self._max_num_nodes = max_num_nodes

                with self.assertRaisesRegex(
                    ValueError,
                    r"^The maximum number of nodes must be greater than or equal to the minimum "
                    r"number of nodes.$",
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
                    r"^The 'dummy_param' rendezvous config does not represent a valid boolean "
                    r"value.$",
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

                self.assertEqual(params.get_as_int("dummy_param"), int(cast(SupportsInt, value)))

    def test_get_as_int_raises_error_if_value_is_invalid(self) -> None:
        for value in ["a", "0a", "3b", "abc"]:
            with self.subTest(value=value):
                self._kwargs["dummy_param"] = value

                params = self._create_params()

                with self.assertRaisesRegex(
                    ValueError,
                    r"^The 'dummy_param' rendezvous config does not represent a valid integer "
                    r"value.$",
                ):
                    params.get_as_int("dummy_param")
