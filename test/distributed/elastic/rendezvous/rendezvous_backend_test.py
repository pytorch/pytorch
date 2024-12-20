# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Optional, Tuple

from torch.distributed.elastic.rendezvous import RendezvousStateError
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    RendezvousBackend,
    Token,
)


class RendezvousBackendTestMixin(ABC):
    _backend: RendezvousBackend

    # Type hints
    assertEqual: Callable
    assertNotEqual: Callable
    assertIsNone: Callable
    assertIsNotNone: Callable
    assertRaises: Callable

    @abstractmethod
    def _corrupt_state(self) -> None:
        """Corrupts the state stored in the backend."""

    def _set_state(
        self, state: bytes, token: Optional[Any] = None
    ) -> Tuple[bytes, Token, bool]:
        result = self._backend.set_state(state, token)

        self.assertIsNotNone(result)

        return cast(Tuple[bytes, Token, bool], result)

    def test_get_state_returns_backend_state(self) -> None:
        self._backend.set_state(b"x")

        result = self._backend.get_state()

        self.assertIsNotNone(result)

        state, token = cast(Tuple[bytes, Token], result)

        self.assertEqual(b"x", state)
        self.assertIsNotNone(token)

    def test_get_state_returns_none_if_backend_state_does_not_exist(self) -> None:
        result = self._backend.get_state()

        self.assertIsNone(result)

    def test_get_state_raises_error_if_backend_state_is_corrupt(self) -> None:
        self._corrupt_state()

        with self.assertRaises(RendezvousStateError):
            self._backend.get_state()

    def test_set_state_sets_backend_state_if_it_does_not_exist(self) -> None:
        state, token, has_set = self._set_state(b"x")

        self.assertEqual(b"x", state)
        self.assertIsNotNone(token)
        self.assertTrue(has_set)

    def test_set_state_sets_backend_state_if_token_is_current(self) -> None:
        _, token1, has_set1 = self._set_state(b"x")

        state2, token2, has_set2 = self._set_state(b"y", token1)

        self.assertEqual(b"y", state2)
        self.assertNotEqual(token1, token2)
        self.assertTrue(has_set1)
        self.assertTrue(has_set2)

    def test_set_state_returns_current_backend_state_if_token_is_old(self) -> None:
        _, token1, _ = self._set_state(b"x")

        state2, token2, _ = self._set_state(b"y", token1)

        state3, token3, has_set = self._set_state(b"z", token1)

        self.assertEqual(state2, state3)
        self.assertEqual(token2, token3)
        self.assertFalse(has_set)

    def test_set_state_returns_current_backend_state_if_token_is_none(self) -> None:
        state1, token1, _ = self._set_state(b"x")

        state2, token2, has_set = self._set_state(b"y")

        self.assertEqual(state1, state2)
        self.assertEqual(token1, token2)
        self.assertFalse(has_set)

    def test_set_state_returns_current_backend_state_if_token_is_invalid(self) -> None:
        state1, token1, _ = self._set_state(b"x")

        state2, token2, has_set = self._set_state(b"y", token="invalid")

        self.assertEqual(state1, state2)
        self.assertEqual(token1, token2)
        self.assertFalse(has_set)
