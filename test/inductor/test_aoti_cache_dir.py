# Owner(s): ["module: inductor"]

import os
import tempfile
from unittest import mock

import torch
import torch._inductor
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_caches


class TestAOTICacheDir(TestCase):
    def test_aoti_compile_package_with_relative_cache_dir(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 16)

            def forward(self, x):
                return torch.relu(self.linear(x))

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with mock.patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": "cache"}):
                    clear_caches()
                    try:
                        model = Model()
                        example_inputs = (torch.randn(8, 10),)
                        batch = torch.export.Dim("batch", min=1, max=1024)
                        exported = torch.export.export(
                            model,
                            example_inputs,
                            dynamic_shapes={"x": {0: batch}},
                        )
                        package_path = os.path.join(tmpdir, "model.pt2")

                        output_path = torch._inductor.aoti_compile_and_package(
                            exported,
                            package_path=package_path,
                        )

                        expected_cache_dir = os.path.abspath("cache")
                        self.assertEqual(output_path, package_path)
                        self.assertTrue(os.path.exists(output_path))
                        self.assertEqual(
                            os.environ["TORCHINDUCTOR_CACHE_DIR"], expected_cache_dir
                        )
                    finally:
                        clear_caches()
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    run_tests()
