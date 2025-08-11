import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cli.lib.core.vllm as vllm


class TestVllmBuildParameters(unittest.TestCase):
    @patch("cli.lib.core.vllm.local_image_exists", return_value=True)
    @patch("cli.lib.core.vllm.is_path_exist", return_value=True)
    @patch(
        "cli.lib.common.envs_helper.env_path",
        side_effect=lambda name, default=None, resolve=True: {
            "DOCKERFILE_PATH": Path("/abs/vllm/Dockerfile"),
            "TORCH_WHEELS_PATH": Path("/abs/dist"),
            "OUTPUT_DIR": Path("/abs/shared"),
        }.get(name, Path(default) if default is not None else None),
    )
    @patch.dict(
        os.environ,
        {
            "USE_TORCH_WHEEL": "1",
            "USE_LOCAL_BASE_IMAGE": "1",
            "USE_LOCAL_DOCKERFILE": "1",
            "BASE_IMAGE": "my/image:tag",
            "DOCKERFILE_PATH": "vllm/Dockerfile",
            "TORCH_WHEELS_PATH": "dist",
            "OUTPUT_DIR": "shared",
        },
        clear=True,
    )
    def test_params_success_normalizes_and_validates(
        self, mock_env_path, mock_is_path, mock_local_img
    ):
        params = vllm.VllmBuildParameters()
        self.assertEqual(params.torch_whls_path, Path("/abs/dist"))
        self.assertEqual(params.dockerfile_path, Path("/abs/vllm/Dockerfile"))
        self.assertEqual(params.output_dir, Path("/abs/shared"))
        self.assertEqual(params.base_image, "my/image:tag")

    @patch("cli.lib.core.vllm.is_path_exist", return_value=False)
    @patch.dict(
        os.environ, {"USE_TORCH_WHEEL": "1", "TORCH_WHEELS_PATH": "dist"}, clear=True
    )
    def test_params_missing_torch_whls_raises(self, _is_path):
        with self.assertRaises(FileNotFoundError):
            vllm.VllmBuildParameters()

    @patch("cli.lib.core.vllm.local_image_exists", return_value=False)
    @patch.dict(
        os.environ, {"USE_LOCAL_BASE_IMAGE": "1", "BASE_IMAGE": "img:tag"}, clear=True
    )
    def test_params_missing_local_base_image_raises(self, _local_img):
        with self.assertRaises(FileNotFoundError):
            vllm.VllmBuildParameters()

    @patch("cli.lib.core.vllm.is_path_exist", return_value=False)
    @patch.dict(
        os.environ,
        {"USE_LOCAL_DOCKERFILE": "1", "DOCKERFILE_PATH": "Dockerfile"},
        clear=True,
    )
    def test_params_missing_dockerfile_raises(self, _is_path):
        with self.assertRaises(FileNotFoundError):
            vllm.VllmBuildParameters()


class TestBuildCmdAndRun(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            # for VllmDockerBuildArgs
            "output_dir": "/abs/out",
            "TARGET": "export-wheels",
            "TAG": "vllm-wheels",
            "CUDA_VERSION": "12.8.1",
            "PYTHON_VERSION": "3.12",
            "MAX_JOBS": "32",
            "SCCACHE_BUCKET": "my-bucket",
            "SCCACHE_REGION": "us-west-2",
            "TORCH_CUDA_ARCH_LIST": "8.0;9.0",
            # for VllmBuildParameters (used by run(), but we stub it anyway)
            "USE_TORCH_WHEEL": "0",
            "USE_LOCAL_BASE_IMAGE": "1",
            "USE_LOCAL_DOCKERFILE": "0",
            "BASE_IMAGE": "img:tag",
            "OUTPUT_DIR": "/abs/out",
        },
        clear=True,
    )
    @patch("cli.lib.core.vllm.local_image_exists", return_value=True)
    def test_generate_docker_build_cmd_includes_bits(self, _exists):
        runner = vllm.VllmBuildRunner()
        # Craft inputs that simulate a prepared build
        inputs = MagicMock()
        inputs.output_dir = Path("/abs/out")
        inputs.use_local_base_image = True
        inputs.base_image = "img:tag"
        inputs.torch_whls_path = Path("./vllm/tmp")

        cmd = runner._generate_docker_build_cmd(inputs)
        squashed = " ".join(cmd.split())  # normalize whitespace for matching

        self.assertIn("--output type=local,dest=/abs/out", squashed)
        self.assertIn("-f docker/Dockerfile.nightly_torch", squashed)
        self.assertIn("--pull=false", squashed)
        self.assertIn("--build-arg TORCH_WHEELS_PATH=tmp", squashed)
        self.assertIn("--build-arg BUILD_BASE_IMAGE=img:tag", squashed)
        self.assertIn("--build-arg FINAL_BASE_IMAGE=img:tag", squashed)
        self.assertIn("--build-arg max_jobs=32", squashed)
        self.assertIn("--build-arg CUDA_VERSION=12.8.1", squashed)
        self.assertIn("--build-arg PYTHON_VERSION=3.12", squashed)
        self.assertIn("--build-arg USE_SCCACHE=1", squashed)
        self.assertIn("--build-arg SCCACHE_BUCKET_NAME=my-bucket", squashed)
        self.assertIn("--build-arg SCCACHE_REGION_NAME=us-west-2", squashed)
        self.assertIn("--build-arg torch_cuda_arch_list='8.0;9.0'", squashed)
        self.assertIn("--target export-wheels", squashed)
        self.assertIn("-t vllm-wheels", squashed)

    @patch("cli.lib.core.vllm.run_cmd")
    @patch("cli.lib.core.vllm.ensure_dir_exists")
    @patch("cli.lib.core.vllm.clone_vllm")
    @patch.object(
        vllm.VllmBuildRunner,
        "_generate_docker_build_cmd",
        return_value="docker buildx ...",
    )
    @patch.dict(
        os.environ,
        {
            # Make __post_init__ validations pass cheaply
            "USE_TORCH_WHEEL": "0",
            "USE_LOCAL_BASE_IMAGE": "0",
            "USE_LOCAL_DOCKERFILE": "0",
            "OUTPUT_DIR": "shared",
        },
        clear=True,
    )
    def test_run_calls_clone_prepare_and_build(
        self, mock_gen, mock_clone, mock_ensure, mock_run
    ):
        # Stub parameters instance so we avoid FS/Docker accesses in run()
        params = MagicMock()
        params.output_dir = Path("shared")
        params.use_local_dockerfile = False
        params.use_torch_whl = False

        with patch("cli.lib.core.vllm.VllmBuildParameters", return_value=params):
            runner = vllm.VllmBuildRunner()
            runner.run()

        mock_clone.assert_called_once()
        mock_ensure.assert_called_once_with(Path("shared"))
        mock_gen.assert_called_once_with(params)
        mock_run.assert_called_once()
        # ensure we run in vllm workdir
        _, kwargs = mock_run.call_args
        assert kwargs.get("cwd") == "vllm"


if __name__ == "__main__":
    unittest.main()
