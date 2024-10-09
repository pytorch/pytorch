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
            assert k in torch._dynamo.config._compile_ignored_keys, k

    def test_config_hash(self):
        config = torch._dynamo.config
        starting_hash = config.get_hash()

        with config.patch({"verbose": not config.verbose}):
            new_hash = config.get_hash()
            assert "verbose" in config._compile_ignored_keys
            assert new_hash == starting_hash

        new_hash = config.get_hash()
        assert new_hash == starting_hash

        with config.patch({"suppress_errors": not config.suppress_errors}):
            changed_hash = config.get_hash()
            assert "suppress_errors" not in config._compile_ignored_keys
            assert changed_hash != starting_hash

            # Test nested patch
            with config.patch({"verbose": not config.verbose}):
                inner_changed_hash = config.get_hash()
                assert inner_changed_hash == changed_hash
                assert inner_changed_hash != starting_hash

        newest_hash = config.get_hash()
        assert changed_hash != newest_hash
        assert newest_hash == starting_hash


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
