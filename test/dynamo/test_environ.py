# Owner(s): ["module: dynamo"]

import os
from contextlib import contextmanager

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


@contextmanager
def env_var(key, value):
    old = os.environ.get(key)
    try:
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


class EnvironTests(torch._dynamo.test_case.TestCase):
    def _check_recompiles_on_change(self, fn, key):
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.ones(4)
        with env_var(key, "1"):
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(cnt.frame_count, 1)
        with env_var(key, "0"):
            self.assertEqual(opt_fn(x), x - 1)
            self.assertEqual(cnt.frame_count, 2)
        with env_var(key, "1"):
            self.assertEqual(opt_fn(x), x + 1)

    def test_getenv(self):
        def fn(x):
            if int(os.getenv("TEST_DYNAMO_ENV_A", "0")):
                return x + 1
            return x - 1

        self._check_recompiles_on_change(fn, "TEST_DYNAMO_ENV_A")

    def test_environ_get(self):
        def fn(x):
            if int(os.environ.get("TEST_DYNAMO_ENV_B", "0")):
                return x + 1
            return x - 1

        self._check_recompiles_on_change(fn, "TEST_DYNAMO_ENV_B")

    def test_environ_getitem(self):
        def fn(x):
            if int(os.environ["TEST_DYNAMO_ENV_C"]):
                return x + 1
            return x - 1

        self._check_recompiles_on_change(fn, "TEST_DYNAMO_ENV_C")

    def test_environ_contains(self):
        def fn(x):
            if "TEST_DYNAMO_ENV_D" in os.environ:
                return x + 1
            return x - 1

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.ones(4)
        with env_var("TEST_DYNAMO_ENV_D", None):
            self.assertEqual(opt_fn(x), x - 1)
            self.assertEqual(cnt.frame_count, 1)
        with env_var("TEST_DYNAMO_ENV_D", "anything"):
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(cnt.frame_count, 2)

    def test_getenv_missing_then_set(self):
        def fn(x):
            return x + int(os.getenv("TEST_DYNAMO_ENV_E", "0"))

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.ones(4)
        with env_var("TEST_DYNAMO_ENV_E", None):
            self.assertEqual(opt_fn(x), x)
            self.assertEqual(cnt.frame_count, 1)
        with env_var("TEST_DYNAMO_ENV_E", "3"):
            self.assertEqual(opt_fn(x), x + 3)
            self.assertEqual(cnt.frame_count, 2)

    def test_getenv_no_default(self):
        def fn(x):
            if os.getenv("TEST_DYNAMO_ENV_F") is None:
                return x - 1
            return x + 1

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.ones(4)
        with env_var("TEST_DYNAMO_ENV_F", None):
            self.assertEqual(opt_fn(x), x - 1)
        with env_var("TEST_DYNAMO_ENV_F", "1"):
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(cnt.frame_count, 2)

    def test_environ_get_non_str_default(self):
        def fn(x):
            return x + int(os.environ.get("TEST_DYNAMO_ENV_G", 2))

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.ones(4)
        with env_var("TEST_DYNAMO_ENV_G", None):
            self.assertEqual(opt_fn(x), x + 2)
        with env_var("TEST_DYNAMO_ENV_G", "1"):
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(cnt.frame_count, 2)

    def test_env_change_during_compile(self):
        # The backend compiler runs between tracing and guard construction
        # and may itself set env vars (inductor does). ENV_MATCH compares
        # against the trace-time snapshot carried by EnvVarSource, so a
        # mid-compile mutation of a guarded variable fails the same-frame
        # guard sanity check loudly instead of revalidating a stale graph
        # forever (which is what re-reading os.environ at guard-construction
        # time used to do).
        def fn(x):
            if int(os.getenv("TEST_DYNAMO_ENV_I", "0")):
                return x + 1
            return x - 1

        def mutating_backend(gm, example_inputs):
            os.environ["TEST_DYNAMO_ENV_I"] = "1"
            return gm.forward

        opt_fn = torch.compile(fn, backend=mutating_backend, fullgraph=True)
        x = torch.ones(4)
        with env_var("TEST_DYNAMO_ENV_I", None):
            with self.assertRaisesRegex(
                AssertionError, "Guard failed on the same frame"
            ):
                opt_fn(x)

    def test_environ_getitem_missing_raises(self):
        def fn(x):
            return x + int(os.environ["TEST_DYNAMO_ENV_H"])

        # Not fullgraph: an exception escaping the compiled region is a graph
        # break by design, same as for a plain dict.
        opt_fn = torch.compile(fn, backend="eager")
        x = torch.ones(4)
        with env_var("TEST_DYNAMO_ENV_H", None):
            with self.assertRaises(KeyError):
                opt_fn(x)
        with env_var("TEST_DYNAMO_ENV_H", "2"):
            self.assertEqual(opt_fn(x), x + 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
