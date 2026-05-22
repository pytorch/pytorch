import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cli.lib.core.vllm.vllm_build as vllm_build


_VLLM_BUILD_MODULE = "cli.lib.core.vllm.vllm_build"


class TestVllmBuildParameters(unittest.TestCase):
    @patch(f"{_VLLM_BUILD_MODULE}.local_image_exists", return_value=True)
    @patch(f"{_VLLM_BUILD_MODULE}.is_path_exist", return_value=True)
    @patch(
        "cli.lib.common.envs_helper.env_path_optional",
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
        params = vllm_build.VllmBuildParameters()
        self.assertEqual(params.torch_whls_path, Path("/abs/dist"))
        self.assertEqual(params.dockerfile_path, Path("/abs/vllm/Dockerfile"))
        self.assertEqual(params.output_dir, Path("/abs/shared"))
        self.assertEqual(params.base_image, "my/image:tag")

    @patch(f"{_VLLM_BUILD_MODULE}.is_path_exist", return_value=False)
    @patch.dict(
        os.environ, {"USE_TORCH_WHEEL": "1", "TORCH_WHEELS_PATH": "dist"}, clear=True
    )
    def test_params_missing_torch_whls_raises(self, _is_path):
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            with self.assertRaises(ValueError) as cm:
                vllm_build.VllmBuildParameters(
                    use_local_base_image=False,
                    use_local_dockerfile=False,
                )
        err = cm.exception
        self.assertIn("TORCH_WHEELS_PATH", str(err))

    @patch(f"{_VLLM_BUILD_MODULE}.local_image_exists", return_value=False)
    @patch.dict(
        os.environ, {"USE_LOCAL_BASE_IMAGE": "1", "BASE_IMAGE": "img:tag"}, clear=True
    )
    def test_params_missing_local_base_image_raises(self, _local_img):
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            with self.assertRaises(ValueError) as cm:
                vllm_build.VllmBuildParameters(
                    use_torch_whl=False,
                    use_local_dockerfile=False,
                )
        err = cm.exception
        self.assertIn("BASE_IMAGE", str(err))

    @patch(f"{_VLLM_BUILD_MODULE}.is_path_exist", return_value=False)
    @patch.dict(
        os.environ,
        {"USE_LOCAL_DOCKERFILE": "1", "DOCKERFILE_PATH": "Dockerfile"},
        clear=True,
    )
    def test_params_missing_dockerfile_raises(self, _is_path):
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            with self.assertRaises(ValueError) as cm:
                vllm_build.VllmBuildParameters(
                    use_torch_whl=False,
                    use_local_base_image=False,
                )
        err = cm.exception
        self.assertIn("DOCKERFILE_PATH", str(err))

    @patch(f"{_VLLM_BUILD_MODULE}.is_path_exist", return_value=False)
    @patch.dict(
        os.environ,
        {"OUTPUT_DIR": ""},
        clear=True,
    )
    def test_params_missing_output_dir(self, _is_path):
        with self.assertRaises(FileNotFoundError):
            vllm_build.VllmBuildParameters()


class TestBuildCmdAndRun(unittest.TestCase):
    @patch(f"{_VLLM_BUILD_MODULE}.local_image_exists", return_value=True)
    def test_generate_docker_build_cmd_includes_bits(self, _exists):
        runner = vllm_build.VllmBuildRunner()
        inputs = MagicMock()
        inputs.output_dir = Path("/abs/out")
        inputs.use_local_base_image = True
        inputs.base_image = "img:tag"
        inputs.torch_whls_path = Path("./vllm/tmp")
        inputs.max_jobs = 64
        inputs.cuda_version = "12.8.1"
        inputs.python_version = "3.12"
        inputs.sccache_bucket = "my-bucket"
        inputs.sccache_region = "us-west-2"
        inputs.torch_cuda_arch_list = "8.0;9.0"
        inputs.target_stage = "export-wheels"
        inputs.tag_name = "vllm-wheels"

        cmd = runner._generate_docker_build_cmd(inputs)
        squashed = " ".join(cmd.split())

        self.assertIn("--output type=local,dest=/abs/out", squashed)
        self.assertIn("-f docker/Dockerfile.nightly_torch", squashed)
        self.assertIn("--pull=false", squashed)
        self.assertIn("--build-arg TORCH_WHEELS_PATH=tmp", squashed)
        self.assertIn("--build-arg BUILD_BASE_IMAGE=img:tag", squashed)
        self.assertIn("--build-arg FINAL_BASE_IMAGE=img:tag", squashed)
        self.assertIn("--build-arg max_jobs=64", squashed)
        self.assertIn("--build-arg CUDA_VERSION=12.8.1", squashed)
        self.assertIn("--build-arg PYTHON_VERSION=3.12", squashed)
        self.assertIn("--build-arg USE_SCCACHE=1", squashed)
        self.assertIn("--build-arg SCCACHE_BUCKET_NAME=my-bucket", squashed)
        self.assertIn("--build-arg SCCACHE_REGION_NAME=us-west-2", squashed)
        self.assertIn("--build-arg torch_cuda_arch_list='8.0;9.0'", squashed)
        self.assertIn("--target export-wheels", squashed)
        self.assertIn("-t vllm-wheels", squashed)

    @patch(f"{_VLLM_BUILD_MODULE}.run_command")
    @patch(f"{_VLLM_BUILD_MODULE}.ensure_dir_exists")
    @patch(f"{_VLLM_BUILD_MODULE}.clone_vllm")
    @patch.object(
        vllm_build.VllmBuildRunner,
        "_generate_docker_build_cmd",
        return_value="docker buildx ...",
    )
    @patch.dict(
        os.environ,
        {
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
        params = MagicMock()
        params.output_dir = Path("shared")
        params.use_local_dockerfile = False
        params.use_torch_whl = False

        with patch(f"{_VLLM_BUILD_MODULE}.VllmBuildParameters", return_value=params):
            runner = vllm_build.VllmBuildRunner()
            runner.run()

        mock_clone.assert_called_once()
        mock_ensure.assert_called_once_with(Path("shared"))
        mock_gen.assert_called_once_with(params)
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        if kwargs.get("cwd") != "vllm":
            raise AssertionError(f"Expected cwd='vllm', got cwd={kwargs.get('cwd')!r}")
