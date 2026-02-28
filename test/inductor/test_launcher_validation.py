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

        with self.assertRaisesRegex(TypeError, "got 3"):
            autotuner._validate_launcher_args(mock_launcher, [1, 2, 3], {})

        with self.assertRaisesRegex(TypeError, "got 1"):
            autotuner._validate_launcher_args(mock_launcher, [1], {})

if __name__ == "__main__":
    unittest.main()
