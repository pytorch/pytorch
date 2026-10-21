# Owner(s): ["module: inductor"]

import contextlib
import os
import re
import tempfile
import unittest
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.triton import FixedTritonConfig
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.runtime.triton_heuristics import (
    AutotuneCache,
    cached_autotune,
    CachingAutotuner,
    config_to_dict,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import is_big_gpu, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from torch.utils._triton import has_triton_package


def _code_for(fn, *args, dynamic=None, **config_kwargs):
    torch._dynamo.reset()
    config_kwargs.setdefault("triton.autotune_at_compile_time", True)
    with config.patch(**config_kwargs):
        result, codes = run_and_get_code(torch.compile(fn, dynamic=dynamic), *args)
    return result, "\n".join(codes)


def _kernels(code):
    return set(re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE))


def _load_from_file(code):
    # A module-level kernel reads its source from the module's file, so the file must
    # outlive the load. The test case's fresh cache dir is removed after the test.
    path = os.path.join(tempfile.mkdtemp(dir=cache_dir()), "module.py")
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
        mock.patch.object(CachingAutotuner, "_combo_sequential_autotune", tune),
        # a cached best config would replace the pinned one
        mock.patch.object(AutotuneCache, "create", tune),
    ):
        yield


@contextlib.contextmanager
def _configs_tuned_at_compile_time():
    """Each kernel's config right after its launch in the compile-time autotune block,
    from the kernel object that launched. The wrapper reads the same objects once the
    block is done, so this checks it pins each kernel's own finished config; it does
    not check that tuning chose well."""
    tuned, in_block = {}, [False]
    block = PythonWrapperCodegen.generate_and_run_autotune_block
    run = CachingAutotuner.run

    def block_spy(self, *args, **kwargs):
        in_block[0] = True
        try:
            return block(self, *args, **kwargs)
        finally:
            in_block[0] = False

    def run_spy(self, *args, **kwargs):
        out = run(self, *args, **kwargs)
        name = self.inductor_meta.get("kernel_name")
        if in_block[0] and name is not None and len(self.launchers) == 1:
            tuned[name] = config_to_dict(self.launchers[0].config)
        return out

    with (
        mock.patch.object(
            PythonWrapperCodegen, "generate_and_run_autotune_block", block_spy
        ),
        mock.patch.object(CachingAutotuner, "run", run_spy),
    ):
        yield tuned


def _softmax(x):
    return torch.softmax(x * 2, dim=-1)


def _cond_softmax(x):
    return torch.cond(
        x.sum() > 0, lambda t: torch.softmax(t * 2, dim=-1), lambda t: t.cos(), (x,)
    )


def _bucketize(x):
    # its inductor_meta holds an AutotuneHint
    return torch.bucketize(x, torch.linspace(-1, 1, 8, device=x.device))


def _foreach(x):
    return tuple(torch._foreach_add([x, x.t()], 1))


def _mix_order(x):
    return x.sum(dim=1), x.sum(dim=0)


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
            ("split_scan", lambda x: x.abs().cumsum(0), (2**20,)),
            ("foreach", _foreach, (64, 128)),
            ("mix_order_reduction", _mix_order, (8192, 8192)),
            (
                "cooperative_reduction",
                lambda x: x.sum(-1),
                (8, 2**16),
                {"triton.force_cooperative_reductions": True},
            ),
        ],
        name_fn=lambda case: case[0],
    )
    @config.patch(coordinate_descent_tuning=True, max_autotune_pointwise=True)
    def test_kernels_launch_with_their_compile_time_config(self, case):
        _, fn, shape, *cfg = case
        x = torch.randn(shape, device="cuda")
        with _configs_tuned_at_compile_time() as tuned:
            _, code = _code_for(fn, x, **(cfg[0] if cfg else {}))
        kernels = _kernels(code)
        self.assertTrue(kernels)
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            # -x takes the other torch.cond branch, so every kernel launches
            for arg in (x, -x):
                out, expected = ns["call"]([arg]), fn(arg)
                if not isinstance(expected, tuple):
                    out = out[0]
                self.assertEqual(out, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(set(ns["KERNEL_CONFIGS"]), kernels)
        for name, cfg in ns["KERNEL_CONFIGS"].items():
            self.assertEqual(cfg, tuned[name], name)
            (launcher,) = ns[name].launchers
            self.assertEqual(config_to_dict(launcher.config), cfg, name)

    @requires_cuda_and_triton
    def test_a_fixed_config_kernel_keeps_its_config(self):
        cfg = {"XBLOCK": 128}

        class FixedChoices(InductorChoices):
            def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
                return {**kernel_kwargs, "fixed_config": FixedTritonConfig(cfg)}

        fn = lambda x: (x.sin() * 2).relu()  # noqa: E731
        x = torch.randn(4096, device="cuda")
        with V.set_choices_handler(FixedChoices()):
            _, code = _code_for(fn, x)
        self.assertIn("@triton_heuristics.fixed_config(", code)
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            self.assertEqual(ns["call"]([x])[0], fn(x))
        ((name, pinned),) = ns["KERNEL_CONFIGS"].items()
        self.assertLessEqual(cfg.items(), pinned.items())
        (launcher,) = ns[name].launchers
        self.assertEqual(config_to_dict(launcher.config), pinned)

    @requires_cuda_and_triton
    @config.patch(coordinate_descent_tuning=True)
    def test_inductor_keeps_the_pool_kernel_in_process(self):
        # compiled_kernels_for_load binds the pool-built kernel ahead of the
        # pinned_config check, so inductor's own load does not pin.
        launched = []
        run = CachingAutotuner.run

        def run_spy(self, *args, **kwargs):
            launched.append(self.heuristic_type)
            return run(self, *args, **kwargs)

        x = torch.randn(64, 3000, device="cuda")
        with mock.patch.object(CachingAutotuner, "run", run_spy):
            _, code = _code_for(lambda x: x.sum(-1), x)
        self.assertIn("KERNEL_CONFIGS = {", code)
        self.assertIn(HeuristicType.REDUCTION, launched)
        self.assertNotIn(HeuristicType.FIXED, launched)

    @requires_cuda_and_triton
    @config.patch(coordinate_descent_tuning=True)
    def test_a_subgraph_kernel_pinned_first_writes_the_table(self):
        # The predicate is an input, so the root launches no kernel: the first one
        # pinned is in a branch nested in another branch.
        def fn(p, x):
            def outer(t):
                return torch.cond(p, _softmax, lambda u: u.cos(), (t,))

            return torch.cond(p, outer, lambda t: t.sin() * 2, (x,))

        p = torch.tensor(True, device="cuda")
        x = torch.randn(64, 128, device="cuda")
        with _configs_tuned_at_compile_time() as tuned:
            _, code = _code_for(fn, p, x)
        kernels = _kernels(code)
        self.assertTrue(kernels)
        self.assertNotIn(".run(", code.split("def call(")[-1])
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            for pred in (p, ~p):
                out = ns["call"]([pred, x])[0]
                self.assertEqual(out, fn(pred, x), atol=1e-4, rtol=1e-4)
        self.assertEqual(set(ns["KERNEL_CONFIGS"]), kernels)
        for name, cfg in ns["KERNEL_CONFIGS"].items():
            self.assertEqual(cfg, tuned[name], name)

    @requires_cuda_and_triton
    @parametrize("per_subkernel", [False, True])
    @config.patch(
        combo_kernels=True,
        benchmark_combo_kernel=False,
        combo_kernel_max_distance=-1,
        coordinate_descent_tuning=True,
    )
    def test_combo_kernel_launches_with_its_compile_time_config(self, per_subkernel):
        def fn(a, b):
            return a.relu(), b.sigmoid()

        a, b = torch.randn(10, 10, device="cuda"), torch.randn(20, 20, device="cuda")
        per_subkernel_cfg = {"combo_kernel_per_subkernel_blocks": per_subkernel}
        with _configs_tuned_at_compile_time() as tuned:
            _, code = _code_for(fn, a, b, **per_subkernel_cfg)
        self.assertIn("combo_grid_meta", code)
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            self.assertEqual(ns["call"]([a, b]), fn(a, b))
        ((name, cfg),) = ns["KERNEL_CONFIGS"].items()
        self.assertEqual(cfg, tuned[name])
        (launcher,) = ns[name].launchers
        self.assertEqual(config_to_dict(launcher.config), cfg)
        if per_subkernel:
            self.assertIn("XBLOCK_1", cfg)

    @requires_cuda_and_triton
    @config.patch(coordinate_descent_tuning=True)
    def test_dynamic_shape_kernel_keeps_its_config_at_every_size(self):
        # Tuned once on the size hints, it launches with that config at other sizes.
        def fn(x):
            return x.sum(-1)

        x = torch.randn(64, 3000, device="cuda")
        with _configs_tuned_at_compile_time() as tuned:
            _, code = _code_for(fn, x, dynamic=True)
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            for shape in [(64, 3000), (17, 5000), (200, 129)]:
                y = torch.randn(shape, device="cuda")
                out = ns["call"]([y, *shape])[0]
                self.assertEqual(out, fn(y), atol=1e-3, rtol=1e-3)
        (cfg,) = ns["KERNEL_CONFIGS"].values()
        (kernel,) = _kernels(code)
        self.assertEqual(cfg, tuned[kernel])
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
    def test_user_defined_kernel_keeps_its_autotuning(self):
        import triton
        import triton.language as tl

        @triton.autotune(
            configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})],
            key=["n"],
        )
        @triton.jit
        def add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

        def fn(x):
            out = torch.empty_like(x)
            n = x.numel()
            add_one[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, out, n)
            return out.sin()

        x = torch.randn(4096, device="cuda")
        result, code = _code_for(fn, x)
        self.assertEqual(result, fn(x))
        (user_kernel,) = re.findall(r"^def (add_one\w*)\(", code, re.MULTILINE)
        ns = _load_from_file(code)
        self.assertEqual(set(ns["KERNEL_CONFIGS"]), _kernels(code))
        self.assertNotIn(user_kernel, ns["KERNEL_CONFIGS"])
        self.assertEqual(ns[user_kernel].heuristic_type, HeuristicType.USER_AUTOTUNE)
        self.assertEqual(ns["call"]([x])[0], result)

    @requires_cuda_and_triton
    def test_kernel_without_a_compile_time_config_tunes_on_first_launch(self):
        # As for a kernel the block leaves with more than one launcher.
        block = PythonWrapperCodegen.generate_and_run_autotune_block

        def leave_no_config(self, *args, **kwargs):
            result = block(self, *args, **kwargs)
            self.kernel_configs.update(dict.fromkeys(self.kernel_configs))
            return result

        def fn(x):
            return x.sum(-1)

        x = torch.randn(64, 3000, device="cuda")
        spy = mock.patch.object(
            PythonWrapperCodegen, "generate_and_run_autotune_block", leave_no_config
        )
        with spy:
            # max_autotune, so that the kernel has configs to choose between
            cfg = {"autotune_local_cache": False, "max_autotune": True}
            result, code = _code_for(fn, x, **cfg)
        (kernel,) = _kernels(code)
        ns = _load_from_file(code)
        self.assertEqual(ns["KERNEL_CONFIGS"], {})
        self.assertEqual(ns[kernel].heuristic_type, HeuristicType.REDUCTION)
        self.assertGreater(len(ns[kernel].launchers), 1)
        tune = CachingAutotuner.autotune_to_one_config
        with mock.patch.object(
            CachingAutotuner, "autotune_to_one_config", autospec=True, side_effect=tune
        ) as tuned:
            self.assertEqual(ns["call"]([x])[0], result, atol=1e-3, rtol=1e-3)
        tuned.assert_called_once()

    # Nothing launches, so no GPU is needed.
    @unittest.skipIf(not has_triton_package(), "requires triton")
    def test_pinned_kernel_keeps_the_backends_autotuner_class(self):
        # An out-of-tree backend's heuristics pass their own CachingAutotuner subclass.
        import triton.language as tl

        def triton_fn(XBLOCK: tl.constexpr):
            pass

        class FakeJitFunction:
            fn = staticmethod(triton_fn)

        class BackendAutotuner:
            def __init__(self, *args, configs, heuristic_type, **kwargs):
                self.configs = configs
                self.heuristic_type = heuristic_type

        meta = {"kernel_name": "triton_poi_0", "pinned_config": {"XBLOCK": 128}}
        kernel = cached_autotune(
            [1024],
            [],
            triton_meta={},
            heuristic_type=HeuristicType.POINTWISE,
            inductor_meta=meta,
            caching_autotuner_cls=BackendAutotuner,
        )(FakeJitFunction())
        self.assertIsInstance(kernel, BackendAutotuner)
        self.assertEqual(kernel.heuristic_type, HeuristicType.FIXED)
        (cfg,) = kernel.configs
        self.assertEqual(cfg.kwargs, {"XBLOCK": 128})

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
