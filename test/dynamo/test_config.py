# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.utils import disable_cache_limit

# NB: do NOT include this test class in test_dynamic_shapes.py


class ConfigTests(torch._dynamo.test_case.TestCase):
    @disable_cache_limit()
    def test_no_automatic_dynamic(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_static = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
            opt_fn = torch._dynamo.optimize(cnt_static)(fn)
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        self.assertEqual(cnt_static.frame_count, 10)

    @disable_cache_limit()
    def test_automatic_dynamic(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=True
        ):
            opt_fn = torch._dynamo.optimize(cnt_dynamic)(fn)
            # NB: must not do 0, 1 as they specialized
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # two graphs now rather than 10
        self.assertEqual(cnt_dynamic.frame_count, 2)

    @disable_cache_limit()
    def test_no_assume_static_by_default(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=False
        ):
            opt_fn = torch._dynamo.optimize(cnt_dynamic)(fn)
            # NB: must not do 0, 1 as they specialized
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # one graph now, as we didn't wait for recompile
        self.assertEqual(cnt_dynamic.frame_count, 1)

    def test_config_compile_ignored(self):
        # Remove from this list if no longer relevant
        dynamo_guarded_config_ignorelist = {
            "log_file_name",
            "verbose",
            "verify_correctness",  # will not affect model, will raise RuntimeError
            # (no silent change to compilation behaviour)
            "cache_size_limit",
            "accumulated_cache_size_limit",
            "replay_record_enabled",
            "cprofile",  # only wraps _compile, not graph
            "repro_after",
            "repro_level",
            "repro_forward_only",
            "repro_tolerance",
            "same_two_models_use_fp64",
            "error_on_recompile",  # safe because: will throw error
            "report_guard_failures",
            "base_dir",  # used for minifying / logging
            "DEBUG_DIR_VAR_NAME",
            "debug_dir_root",
        }
        for k in dynamo_guarded_config_ignorelist:
            assert k in torch._dynamo.config._compile_ignored_keys

    def test_config_hash(self):
        config = torch._dynamo.config
        starting_hash = config.get_hash()

        with config.patch({"verbose": not config.verbose}):
            new_hash = config.get_hash()
            assert "verbose" in config._compile_ignored_keys
            assert new_hash == starting_hash

        new_hash = config.get_hash()
        assert new_hash == starting_hash

        with config.patch({"dead_code_elimination": not config.dead_code_elimination}):
            changed_hash = config.get_hash()
            assert "dead_code_elimination" not in config._compile_ignored_keys
            assert changed_hash != starting_hash

            # Test nested patch
            with config.patch({"verbose": not config.verbose}):
                inner_changed_hash = config.get_hash()
                assert inner_changed_hash == changed_hash
                assert inner_changed_hash != starting_hash

        newest_hash = config.get_hash()
        assert changed_hash != newest_hash
        assert newest_hash == starting_hash

    @disable_cache_limit()
    def test_no_saved_config(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
            opt_fn_static_shape = torch._dynamo.optimize(
                cnt_dynamic, save_config=False
            )(fn)
            opt_fn_static_shape(torch.randn(2), torch.randn(2))
            opt_fn_static_shape(torch.randn(3), torch.randn(3))

        self.assertEqual(cnt_dynamic.frame_count, 2)

        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=False
        ):
            for i in range(2, 12):
                opt_fn_static_shape(
                    torch.randn(i), torch.randn(i)
                )  # will be recompiled under new config

        self.assertEqual(cnt_dynamic.frame_count, 3)

    @disable_cache_limit()
    def test_no_saved_config_nested(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        cnt_dynamic_1 = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=False
        ):
            opt_fn_static_shape = torch._dynamo.optimize(cnt_dynamic, dynamic=False)(fn)

            # Will trigger recompile as compiled as static
            opt_fn_static_shape(torch.randn(2), torch.randn(2))
            opt_fn_static_shape(torch.randn(3), torch.randn(3))

            self.assertEqual(cnt_dynamic.frame_count, 2)

            opt_fn_try_dynamic = torch._dynamo.optimize(
                cnt_dynamic_1, save_config=False
            )(opt_fn_static_shape)

            for i in range(2, 6):
                opt_fn_try_dynamic(torch.randn(i), torch.randn(i))
            self.assertEqual(cnt_dynamic_1.frame_count, 1)

        # Saved config = False will use whatever config is available
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
            for i in range(6, 12):
                opt_fn_try_dynamic(torch.randn(i), torch.randn(i))
            self.assertEqual(cnt_dynamic_1.frame_count, 7)

    @disable_cache_limit()
    def test_config_changed_from_guarded_config_1(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()

        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
            opt_fn_static_shape = torch._dynamo.optimize(cnt_dynamic)(fn)
            res = opt_fn_static_shape(torch.randn(2), torch.randn(2))
            opt_fn_static_shape(torch.randn(3), torch.randn(3))

        self.assertEqual(cnt_dynamic.frame_count, 2)

        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=False
        ):
            for i in range(2, 12):
                # Only 4-11 will now be recompiled under old config
                # 2-3 have been already been compiled under old config
                # and hence will hit cache
                opt_fn_static_shape(torch.randn(i), torch.randn(i))

        self.assertEqual(cnt_dynamic.frame_count, 10)

    @disable_cache_limit()
    def test_config_changed_from_guarded_config_2(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()

        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=False
        ):
            opt_fn_dynamic_shape = torch._dynamo.optimize(cnt_dynamic)(fn)
            opt_fn_dynamic_shape(torch.randn(2), torch.randn(2))
            opt_fn_dynamic_shape(torch.randn(3), torch.randn(3))

        self.assertEqual(cnt_dynamic.frame_count, 1)

        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
            for i in range(2, 12):
                opt_fn_dynamic_shape(
                    torch.randn(i), torch.randn(i)
                )  # will not be recompiled due to automatic dynamic shapes

        self.assertEqual(cnt_dynamic.frame_count, 1)

    @disable_cache_limit()
    def test_nested_compile_outer_wins(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()

        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        cnt_dynamic_1 = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
            opt_fn_static_shape = torch._dynamo.optimize(cnt_dynamic)(fn)
            opt_fn_static_shape(torch.randn(2), torch.randn(2))
            opt_fn_static_shape(torch.randn(3), torch.randn(3))

        self.assertEqual(cnt_dynamic.frame_count, 2)

        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=False
        ):
            opt_fn_dynamic = torch._dynamo.optimize(cnt_dynamic_1)(
                lambda x, y: opt_fn_static_shape(x, y)
            )
            for i in range(2, 12):
                opt_fn_dynamic(
                    torch.randn(i), torch.randn(i)
                )  # will be recompiled under new config

        self.assertEqual(cnt_dynamic.frame_count, 2)
        self.assertEqual(cnt_dynamic_1.frame_count, 1)

    @disable_cache_limit()
    def test_nested_fn_does_not_inherit_outer_config(self):
        def g1(x):
            return x + 1

        def g2(x):
            return x * 2

        def f(x):
            x = g1(x)
            torch._dynamo.graph_break()
            return g2(x)

        torch._dynamo.reset()

        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        cnt_dynamic_1 = torch._dynamo.testing.CompileCounter()

        opt_fn_static_shape = torch._dynamo.optimize(cnt_dynamic, dynamic=False)(f)
        opt_fn_static_shape(torch.randn(2))
        opt_fn_static_shape(torch.randn(3))
        self.assertEqual(cnt_dynamic.frame_count, 4)  # 2 compiles * 2 graphs

        opt_fn_dynamic = torch._dynamo.optimize(cnt_dynamic_1, dynamic=True)(g2)

        for i in range(2, 12):
            opt_fn_dynamic(
                torch.randn(i),
            )  # will be recompiled under new config

        self.assertEqual(cnt_dynamic_1.frame_count, 1)

    @disable_cache_limit()
    def test_multiple_compile_recompiles(self):
        cnt_dynamic = torch._dynamo.testing.CompileCounter()

        def f(dynamic, compile_count):
            @torch._dynamo.optimize(cnt_dynamic, dynamic=dynamic)
            def g(x):
                return x + 1

            for i in range(2, 12):
                g(torch.randn(i))  # will be recompiled under new config
            self.assertEqual(cnt_dynamic.frame_count, compile_count)
            cnt_dynamic.clear()

        f(dynamic=True, compile_count=1)  # first compile
        f(dynamic=False, compile_count=10)  # recompile
        f(dynamic=True, compile_count=0)  # reuse first compile product


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
