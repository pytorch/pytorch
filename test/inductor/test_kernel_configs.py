# Owner(s): ["module: inductor"]

import contextlib
import os
import re
import tempfile
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.runtime.triton_heuristics import (
    AutotuneCache,
    CachingAutotuner,
    config_to_dict,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import is_big_gpu, run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def _code_for(fn, *args, dynamic=None, **config_kwargs):
    torch._dynamo.reset()
    config_kwargs.setdefault("triton.autotune_at_compile_time", True)
    with config.patch(**config_kwargs):
        result, codes = run_and_get_code(torch.compile(fn, dynamic=dynamic), *args)
    return result, "\n".join(codes)


def _kernels(code):
    return set(re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE))


def _load_from_file(code):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "module.py")
        with open(path, "w") as f:
            f.write(code)
        ns = {"__file__": path, "__name__": "_kernel_configs"}
        exec(compile(code, path, "exec"), ns)
    return ns


@contextlib.contextmanager
def _no_runtime_tuning():
    def tune(*args, **kwargs):
        raise AssertionError("a kernel autotuned at its first launch")

    with (
        mock.patch.object(CachingAutotuner, "autotune_to_one_config", tune),
        mock.patch.object(CachingAutotuner, "_coordinate_descent_tuning", tune),
        # a cached best config would replace the pinned one
        mock.patch.object(AutotuneCache, "create", tune),
    ):
        yield


def _softmax(x):
    return torch.softmax(x * 2, dim=-1)


def _cond_softmax(x):
    return torch.cond(
        x.sum() > 0, lambda t: torch.softmax(t * 2, dim=-1), lambda t: t.cos(), (x,)
    )


def _bucketize(x):
    # its inductor_meta holds an AutotuneHint
    return torch.bucketize(x, torch.linspace(-1, 1, 8, device=x.device))


class TestKernelConfigs(TestCase):
    """Under triton.autotune_at_compile_time, the module launches each kernel with the
    config the compile tuned it to."""

    @requires_cuda_and_triton
    @parametrize(
        "case",
        [
            ("persistent_reduction", _softmax, (64, 128)),
            ("reduction", lambda x: x.sum(-1), (64, 3000)),
            ("pointwise", lambda x: (x.sin() * 2).relu(), (4096,)),
            ("bucketize", _bucketize, (4096,)),
            ("subgraph", _cond_softmax, (64, 128)),
        ],
        name_fn=lambda case: case[0],
    )
    @config.patch(coordinate_descent_tuning=True, max_autotune_pointwise=True)
    def test_kernels_launch_with_their_compile_time_config(self, case):
        _, fn, shape = case
        x = torch.randn(shape, device="cuda")
        _, code = _code_for(fn, x)
        kernels = _kernels(code)
        self.assertTrue(kernels)
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            # -x takes the other torch.cond branch, so every kernel launches
            for arg in (x, -x):
                self.assertEqual(ns["call"]([arg])[0], fn(arg), atol=1e-4, rtol=1e-4)
        self.assertEqual(set(ns["KERNEL_CONFIGS"]), kernels)
        for name, cfg in ns["KERNEL_CONFIGS"].items():
            (launcher,) = ns[name].launchers
            self.assertEqual(config_to_dict(launcher.config), cfg, name)

    @requires_cuda_and_triton
    @config.patch(coordinate_descent_tuning=True)
    def test_dynamic_shape_kernel_keeps_its_config_at_every_size(self):
        # Tuned once on the size hints, it launches with that config at other sizes.
        def fn(x):
            return x.sum(-1)

        _, code = _code_for(fn, torch.randn(64, 3000, device="cuda"), dynamic=True)
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            for shape in [(64, 3000), (17, 5000), (200, 129)]:
                y = torch.randn(shape, device="cuda")
                out = ns["call"]([y, *shape])[0]
                self.assertEqual(out, fn(y), atol=1e-3, rtol=1e-3)
        (cfg,) = ns["KERNEL_CONFIGS"].values()
        (kernel,) = _kernels(code)
        (launcher,) = ns[kernel].launchers
        self.assertEqual(config_to_dict(launcher.config), cfg)

    @requires_cuda_and_triton
    def test_template_keeps_its_one_config(self):
        if not is_big_gpu():
            self.skipTest("Triton GEMM templates need a big GPU")
        a, b = torch.randn(64, 64, device="cuda"), torch.randn(64, 64, device="cuda")
        cfg = {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
        result, code = _code_for(lambda a, b: (a @ b).relu(), a, b, **cfg)
        (template,) = {k for k in _kernels(code) if k.startswith("triton_tem_")}
        ns = _load_from_file(code)
        self.assertNotIn(template, ns.get("KERNEL_CONFIGS", {}))
        self.assertEqual(ns["call"]([a, b])[0], result)

    @requires_cuda_and_triton
    def test_without_compile_time_autotuning_nothing_is_pinned(self):
        x = torch.randn(64, 128, device="cuda")
        cfg = {"triton.autotune_at_compile_time": None}
        result, code = _code_for(_softmax, x, **cfg)
        self.assertNotIn("KERNEL_CONFIGS", code)
        self.assertNotIn("pinned_config", code)
        self.assertEqual(_load_from_file(code)["call"]([x])[0], result)


instantiate_parametrized_tests(TestKernelConfigs)


if __name__ == "__main__":
    run_tests()
