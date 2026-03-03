# Owner(s): ["module: dynamo"]

import contextlib
import importlib.util
import os
import re
import tempfile

import torch._dynamo.config
import torch._dynamo.test_case
import torch._inductor.mock_cache as mock_cache
import torch.compiler.config
import torch.nested
from torch._dynamo.testing import CompileCounter
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.utils import clear_caches, fresh_cache
from torch.testing._internal.common_utils import IS_WINDOWS


class PgoTest(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(torch.compiler.config.patch(job_id=self.id()))
        self._test_stack.enter_context(
            torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
        )
        if os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1":
            self._test_stack.enter_context(fresh_cache())
        mock_cache.PatchCaches.setUp()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        self._test_stack.close()
        mock_cache.PatchCaches.tearDown()

    def reset(self):
        torch._dynamo.reset()
        clear_caches()

    def test_basic(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        f(torch.randn(2, 3))
        f(torch.randn(2, 4))
        self.assertEqual(cnts.frame_count, 2)

        self.reset()
        cnts.clear()

        f(torch.randn(2, 5))
        f(torch.randn(2, 6))
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(
        force_parameter_static_shapes=False,
        force_nn_module_property_static_shapes=False,
    )
    def test_whitelist_suggestion(self):
        from torch._dynamo.pgo import (
            _collect_dynamic_sources,
            _collect_missing_sources,
            get_code_state,
            render_code_state,
        )

        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)
                self.attr = torch.randn(4)

            def forward(self, x, y):
                return self.lin(x) + self.attr + y

        sources = [
            "L['x']",
            "L['self']._modules['lin']._parameters['weight']",
            "L['self']._modules['lin']._parameters['bias']",
            "L['self'].attr",
            "L['y']",
        ]

        def check_whitelist(sources_):
            state = render_code_state(get_code_state())
            whitelist = re.search(r'TORCH_COMPILE_DYNAMIC_SOURCES="(.*)"', state).group(
                1
            )
            for src in sources_:
                self.assertTrue(src in whitelist)

        def check_num_missing_whitelist(expected):
            frame_state = next(iter(get_code_state().values()))
            all_dynamic_sources = _collect_dynamic_sources(frame_state)
            missing_whitelist = _collect_missing_sources(all_dynamic_sources)
            self.assertEqual(len(missing_whitelist), expected)

        # check growing whitelist
        f = Foo()
        f(torch.randn(2, 4), torch.randn(4))
        # only x
        f(torch.randn(4, 4), torch.randn(4))
        check_whitelist(sources[:1])
        # x, lin.weight
        f.lin = torch.nn.Linear(8, 4)
        f(torch.randn(8, 8), torch.randn(4))
        check_whitelist(sources[:2])
        # x, y, lin.weight, lin.bias, attr
        f.lin = torch.nn.Linear(8, 8)
        f.attr = torch.randn(8)
        f(torch.randn(8, 8), torch.randn(8))
        check_whitelist(sources)
        check_num_missing_whitelist(5)

        # now use suggested whitelist
        self.reset()
        cnts.clear()
        code_state = get_code_state()
        state = render_code_state(code_state)
        whitelist = re.search(r'TORCH_COMPILE_DYNAMIC_SOURCES="(.*)"', state).group(1)
        with torch.compiler.config.patch(dynamic_sources=whitelist):
            f = Foo()
            f(torch.randn(2, 4), torch.randn(4))
            f(torch.randn(4, 4), torch.randn(4))
            f.lin = torch.nn.Linear(8, 8)
            f.attr = torch.randn(8)
            f(torch.randn(8, 8), torch.randn(8))
            self.assertEqual(cnts.frame_count, 1)
            check_num_missing_whitelist(0)

    def test_no_empty_graph_allowlist(self):
        @torch._dynamo.disable
        def g(x):
            return x * 2 + x

        @torch.compile(backend="eager")
        def f(x):
            return g(x)

        self.reset()
        f(torch.randn(4))
        f(torch.randn(8))
        self.assertEqual(torch._dynamo.pgo._LOGGED_DYNAMIC_ALLOWLIST, False)

        @torch.compile(backend="eager")
        def f1(x):
            return g(x + 2) + 2

        self.reset()
        f1(torch.randn(4))
        f1(torch.randn(8))
        self.assertEqual(torch._dynamo.pgo._LOGGED_DYNAMIC_ALLOWLIST, True)

    def test_pgo_dynamic_false(self):
        @torch.compile(backend="eager", dynamic=False)
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                x += 2
                y += 2
                torch._dynamo.graph_break()
                x -= 2
                y *= 2
                return x, y

        self.reset()
        f = Foo()
        f(torch.randn(2, 4), torch.randn(2, 4))
        f(torch.randn(4, 4), torch.randn(6, 8))

        # check PGO code state is overwritten with static value, both before/after graph break
        for code_state in torch._dynamo.pgo.get_code_state().values():
            self.assertEqual(code_state.automatic_dynamic["L['x']"].size, (4, 4))
            self.assertEqual(code_state.automatic_dynamic["L['y']"].size, (6, 8))

    def test_whitelist_ints_floats(self):
        @torch.compile(backend="eager", fullgraph=True)
        class Bar(torch.nn.Module):
            def __init__(self, c, d):
                super().__init__()
                self.c = c
                self.d = d

            def forward(self, x, y, z):
                if self.d == 2:
                    x += 1
                if self.c == 1.0:
                    return x + y + torch.tensor([z])

        f = Bar(1.0, 2)
        f(2, 1.0, 2.0)
        f.d = 3
        f(3, 1.2, 2.0)
        state = torch._dynamo.pgo.render_code_state(torch._dynamo.pgo.get_code_state())
        whitelist = re.search(r'TORCH_COMPILE_DYNAMIC_SOURCES="(.*)"', state).group(1)
        self.assertTrue("L['x']" in whitelist)
        self.assertTrue("L['y']" in whitelist)
        self.assertTrue(
            "___as_tensor(L['y'])" not in whitelist
        )  # ephemeral FloatTensor source
        self.assertTrue("L['z']" not in whitelist)  # static float
        self.assertTrue("L['self'].c" not in whitelist)  # static float property
        self.assertTrue("L['self'].d" in whitelist)  # dynamic int property

    def test_pgo_dynamic_params(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = None

            def forward(self, x):
                return self.lin(x)

        f = Foo()

        def run():
            self.reset()
            cnts.clear()
            f.lin = torch.nn.Linear(4, 4)
            f(torch.randn(2, 4))
            f(torch.randn(4, 4))
            f.lin = torch.nn.Linear(8, 8)
            f(torch.randn(8, 8))

        # recompile each run
        run()
        self.assertEqual(cnts.frame_count, 3)

        # parameter static shapes are forced static, so we recompile once
        with torch._dynamo.config.patch(
            force_parameter_static_shapes=False,
            force_nn_module_property_static_shapes=False,
        ):
            run()
            self.assertEqual(cnts.frame_count, 2)

            # because flags were flipped, params were included in PGO
            run()
            self.assertEqual(cnts.frame_count, 1)

    def test_njt(self):
        cnts = CompileCounter()

        # NB: PGO doesn't do anything here, the point is to catch pickle
        # problem with nested int

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        x = torch.nested.nested_tensor_from_jagged(
            torch.randn(10, 3), torch.tensor([0, 3, 7, 10]), torch.tensor([1, 2, 3])
        )
        y = torch.nested.nested_tensor_from_jagged(
            torch.randn(13, 3), torch.tensor([0, 3, 7, 13]), torch.tensor([1, 2, 6])
        )

        f(x)
        f(y)
        self.assertEqual(cnts.frame_count, 1)

        self.reset()
        cnts.clear()

        a = torch.nested.nested_tensor_from_jagged(
            torch.randn(14, 3), torch.tensor([0, 3, 7, 14]), torch.tensor([1, 2, 7])
        )
        b = torch.nested.nested_tensor_from_jagged(
            torch.randn(15, 3), torch.tensor([0, 3, 7, 15]), torch.tensor([1, 2, 8])
        )

        f(a)
        f(b)
        self.assertEqual(cnts.frame_count, 1)

    def test_distinct_compile_id(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        with torch.compiler.config.patch(job_id="foo"):
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
        self.assertEqual(cnts.frame_count, 2)

        self.reset()
        cnts.clear()

        with torch.compiler.config.patch(job_id="bar"):
            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
        self.assertEqual(cnts.frame_count, 2)

        torch._dynamo.reset()
        clear_caches()
        cnts.clear()

        with torch.compiler.config.patch(job_id="foo"):
            f(torch.randn(2, 7))
            f(torch.randn(2, 8))
        self.assertEqual(cnts.frame_count, 1)

    # TODO: to test local need to ensure the local filesystem gets cleared out
    @torch._dynamo.config.patch(
        automatic_dynamic_remote_pgo=True, automatic_dynamic_local_pgo=False
    )
    def test_remote_basic(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        with mock_cache.PatchCaches():
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(
                mock_cache.global_stats.dynamo_pgo, mock_cache.Stats(2, 0, 1)
            )

            self.reset()
            cnts.clear()

            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(
                mock_cache.global_stats.dynamo_pgo, mock_cache.Stats(2, 1, 1)
            )

            self.reset()
            cnts.clear()

            with torch.compiler.config.patch({"cache_key_tag": "test"}):
                f(torch.randn(2, 7))
                f(torch.randn(2, 8))
                self.assertEqual(cnts.frame_count, 2)
                self.assertEqual(
                    mock_cache.global_stats.dynamo_pgo, mock_cache.Stats(4, 1, 2)
                )

    # Test that if the same file appears in two different paths for two different compilations PGO still works.
    def test_different_file_paths_local_pgo(self):
        content = """
import torch
def run(cnt):
    @torch.compile(backend=cnt, fullgraph=True)
    def func(x):
        return x*10
    func(torch.rand(10))
    func(torch.rand(20))
    func(torch.rand(30))
"""
        temp_dir1 = tempfile.TemporaryDirectory()
        temp_dir2 = tempfile.TemporaryDirectory()

        # We need normalize_path_separator for Windows file path.
        path1 = normalize_path_separator(os.path.join(temp_dir1.name, "example.py"))
        path2 = normalize_path_separator(os.path.join(temp_dir2.name, "example.py"))
        cnts = CompileCounter()

        if path1 == path2:
            raise AssertionError("Expected path1 != path2")

        def write_load_and_run(path):
            with open(path, "w") as file:
                file.write(content)
            spec = importlib.util.spec_from_file_location("example", path1)
            if spec is None:
                raise AssertionError("Expected spec to not be None")
            module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise AssertionError("Expected spec.loader to not be None")
            spec.loader.exec_module(module)
            module.run(cnts)

        write_load_and_run(path1)
        self.assertEqual(cnts.frame_count, 2)
        state = torch._dynamo.pgo.render_code_state(torch._dynamo.pgo.get_code_state())

        # Windows can't create unification temp path:
        #   hash(a18a3259)C:/Users/Xuhan/AppData/Local/Temp/tmpx3hfkuqa/example.py
        # Skip hash check
        self.assertTrue("hash" if IS_WINDOWS else "hash(390fe689)" in state)
        self.assertTrue("/example.py:4:func:" in state)
        self.assertTrue(" L['x']: tensor size=[?] stride=[1]" in state)
        # We should compile this only once due to PGO.
        cnts.clear()
        write_load_and_run(path2)
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(
        automatic_dynamic_remote_pgo=True, automatic_dynamic_local_pgo=False
    )
    def test_sticky_pgo_read_write(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x, y):
            return x * 2, y * 3

        def t(x, y):
            return torch.randn(x, y)

        with mock_cache.PatchCaches():
            # we pretend to disable the default remote cache, by keying different job ids per run
            with torch.compiler.config.patch(job_id="a"):
                f(t(2, 2), t(2, 2))
                f(t(2, 4), t(2, 2))
                self.assertEqual(cnts.frame_count, 2)

            # first test we're not reading from local/default remote cache;
            # we should recompile when x wobbles
            self.reset()
            cnts.clear()
            with torch.compiler.config.patch(
                job_id="b", pgo_extra_write_key="sticky_0"
            ):
                f(t(2, 2), t(2, 2))
                f(t(2, 4), t(2, 2))
                self.assertEqual(cnts.frame_count, 2)

            # now with the extra sticky_0 key, we start with dynamic x;
            # no recompiles
            self.reset()
            cnts.clear()
            with torch.compiler.config.patch(job_id="c", pgo_extra_read_key="sticky_0"):
                f(t(2, 2), t(2, 2))
                f(t(2, 4), t(2, 2))
                self.assertEqual(cnts.frame_count, 1)

            # last test: wobble y and write to sticky_1 key
            self.reset()
            cnts.clear()
            with torch.compiler.config.patch(
                job_id="d", pgo_extra_write_key="sticky_1"
            ):
                f(t(2, 2), t(2, 2))
                f(t(2, 2), t(2, 4))
                f(t(2, 2), t(4, 4))
                self.assertEqual(cnts.frame_count, 3)

            # start using default remote PGO, create run that wobbles y
            self.reset()
            cnts.clear()
            f(t(2, 2), t(2, 2))
            f(t(2, 4), t(2, 2))
            f(t(4, 2), t(2, 2))

            # with both default remote present, we ignore extra remote.
            self.reset()
            cnts.clear()
            with torch.compiler.config.patch(pgo_extra_read_key="sticky_1"):
                f(t(2, 2), t(2, 2))
                f(t(6, 8), t(2, 2))
                self.assertEqual(cnts.frame_count, 1)
                f(t(2, 2), t(2, 4))
                self.assertEqual(cnts.frame_count, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
