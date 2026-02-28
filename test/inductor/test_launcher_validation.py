import unittest
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from triton import Config as TritonConfig

class MockDeviceProperties:
    index = 0
    type = "cuda"
    warp_size = 32
    max_threads_per_block = 1024
    max_shared_memory = 65536
    multi_processor_count = 80
    major = 8
    minor = 0

class TestLauncherValidation(unittest.TestCase):
    def test_argument_mismatch_error(self):
        def mock_launcher(a, b, stream):
            pass
        mock_launcher.def_arg_names = ["a", "b"]

        autotuner = CachingAutotuner(
            fn=lambda: None,
            triton_meta={'device': MockDeviceProperties(), 'signature': {}},
            configs=[TritonConfig({})],
            save_cache_hook=None,
            mutated_arg_names=[],
            optimize_mem=False, 
            heuristic_type=None,
            inductor_meta={'kernel_name': 'test_kernel'}
        )

        # Too many positional args
        with self.assertRaisesRegex(TypeError, r"expected at most 2 positional arguments \(a, b\) but got 3"):
            autotuner._validate_launcher_args(mock_launcher, [1, 2, 3], {})

        # Too few positional args, missing kwargs
        with self.assertRaisesRegex(TypeError, r"expected 2 arguments \(a, b\) but only 1 were provided via positional and keyword arguments\. Missing arguments: b"):
            autotuner._validate_launcher_args(mock_launcher, [1], {})

        # 1 positional, 1 correct kwarg
        autotuner._validate_launcher_args(mock_launcher, [1], {"b": 2})

        # 0 positional, 2 correct kwargs
        autotuner._validate_launcher_args(mock_launcher, [], {"a": 1, "b": 2})

        # Too few args, but some are extra kwargs mapping to unknown fields (fails due to missing 'b')
        with self.assertRaisesRegex(TypeError, r"expected 2 arguments \(a, b\) but only 1 were provided via positional and keyword arguments\. Missing arguments: b"):
            autotuner._validate_launcher_args(mock_launcher, [1], {"c": 2})

if __name__ == "__main__":
    unittest.main()
