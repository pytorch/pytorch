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
    def test_config_changed_from_save_config_1(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()

        cnt_dynamic = torch._dynamo.testing.CompileCounter()
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
            for i in range(2, 12):
                # Only 4-11 will now be recompiled under old config
                # 2-3 have been recompiled under old config due to shape mismatch
                opt_fn_static_shape(torch.randn(i), torch.randn(i))

        self.assertEqual(cnt_dynamic.frame_count, 10)

    @disable_cache_limit()
    def test_config_changed_from_save_config_2(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()

        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=False
        ):
            opt_fn_static_shape = torch._dynamo.optimize(cnt_dynamic)(fn)
            opt_fn_static_shape(torch.randn(2), torch.randn(2))
            opt_fn_static_shape(torch.randn(3), torch.randn(3))

        self.assertEqual(cnt_dynamic.frame_count, 1)

        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
            for i in range(2, 12):
                opt_fn_static_shape(
                    torch.randn(i), torch.randn(i)
                )  # will be recompiled due to shape mismatch under old config

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

    def test_cache_size_limit(self):
        cnt = torch._dynamo.testing.CompileCounter()
        key = "_ConfigTests___test_cache_size_limit_key"
        try:
            torch._dynamo.config._allowed_keys.add(key)
            torch._dynamo.config._ConfigTests___test_cache_size_limit_key = -1
            with torch._dynamo.config.patch(
                {"cache_size_limit": 1, "accumulated_cache_size_limit": 10}
            ):
                for i in range(24):
                    with torch._dynamo.config.patch({key: i % 12}):

                        @torch._dynamo.optimize(cnt)
                        def g(x):
                            return x + 1

                        g(torch.randn(1))
            self.assertEqual(cnt.frame_count, 10)
        finally:
            if key in torch._dynamo.config._allowed_keys:
                torch._dynamo.config._allowed_keys.remove(key)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
