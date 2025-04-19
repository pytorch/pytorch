# Owner(s): ["module: dynamo"]

from unittest.mock import Mock

from torch._dynamo.callback import callback_handler
from torch._dynamo.test_case import run_tests, TestCase


class CallbackTests(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._on_compile_start = Mock()
        self._on_compile_end = Mock()
        callback_handler.register_start_callback(self._on_compile_start)
        callback_handler.register_end_callback(self._on_compile_end)

    def tearDown(self) -> None:
        return super().tearDown()
        callback_handler.clear()

    def test_callbacks_without_duplicate_prevention(self) -> None:
        callback_handler._CompilationCallbackHandler__prevent_duplicate_callbacks = (
            False
        )

        with callback_handler.install_callbacks(), callback_handler.install_callbacks():
            self.assertEqual(self._on_compile_start.call_count, 2)
        self.assertEqual(self._on_compile_end.call_count, 2)

    def test_callbacks_with_duplicate_prevention(self) -> None:
        callback_handler._CompilationCallbackHandler__prevent_duplicate_callbacks = True

        with callback_handler.install_callbacks(), callback_handler.install_callbacks():
            self._on_compile_start.assert_called_once()
        self._on_compile_end.assert_called_once()

    def test_counter(self) -> None:
        callback_handler._CompilationCallbackHandler__prevent_duplicate_callbacks = True

        with callback_handler.install_callbacks():
            self.assertEqual(
                callback_handler._CompilationCallbackHandler__pending_callbacks_counter,
                1,
            )
        self.assertEqual(
            callback_handler._CompilationCallbackHandler__pending_callbacks_counter, 0
        )

    def test_counter_assertion(self) -> None:
        callback_handler._CompilationCallbackHandler__prevent_duplicate_callbacks = True
        callback_handler._CompilationCallbackHandler__pending_callbacks_counter -= 1

        with self.assertRaises(
            AssertionError
        ) as e, callback_handler.install_callbacks():
            pass

        self.assertIn(
            "Pending callbacks counter cannot become negative.",
            str(e.exception),
        )

        callback_handler._CompilationCallbackHandler__pending_callbacks_counter += 1


if __name__ == "__main__":
    run_tests()
