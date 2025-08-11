import argparse
import unittest
from unittest.mock import MagicMock, patch

# Import the module under test
from cli.lib.core.vllm import VllmBuildParameters, VllmBuildRunner


# ---------- VllmBuildParameters tests ----------


class TestVllmBuildParameters(unittest.TestCase):
    @patch("cli.lib.core.vllm.get_abs_path", side_effect=lambda p: f"/abs/{p}")
    @patch("cli.lib.core.vllm.is_path_exist", return_value=True)
    @patch("cli.lib.core.vllm.local_image_exists", return_value=True)
    @patch("cli.lib.core.vllm.get_env")
    def test_params_success_normalizes_and_validates(
        self, mock_get_env, mock_local_image_exists, mock_is_path, mock_abs
    ):
        # Flags enabled; provide valid values
        env_map = {
            "USE_TORCH_WHEEL": "1",
            "USE_LOCAL_BASE_IMAGE": "1",
            "USE_LOCAL_DOCKERFILE": "1",
            "BASE_IMAGE": "my/image:tag",
            "DOCKERFILE_PATH": "vllm/Dockerfile",
            "TORCH_WHEELS_PATH": "dist",
            "output_dir": "shared",
        }
        mock_get_env.side_effect = lambda key, default=None: env_map.get(key, default)

        params = VllmBuildParameters()
        # Paths are normalized through get_abs_path when handler present
        self.assertEqual(params.torch_whls_path, "/abs/dist")
        self.assertEqual(params.dockerfile_path, "/abs/vllm/Dockerfile")
        # Base image left as-is
        self.assertEqual(params.base_image, "my/image:tag")
        # Validators called
        mock_local_image_exists.assert_called_once_with("my/image:tag")
        self.assertTrue(mock_is_path.called)

    @patch(
        "cli.lib.core.vllm.get_env",
        side_effect=lambda k, d=None: "1" if k == "USE_TORCH_WHEEL" else d,
    )
    @patch("cli.lib.core.vllm.is_path_exist", return_value=False)
    def test_params_missing_torch_whls_raises(self, _mock_is_path, _mock_env):
        with self.assertRaises(FileNotFoundError):
            VllmBuildParameters()

    @patch(
        "cli.lib.core.vllm.get_env",
        side_effect=lambda k, d=None: {"USE_LOCAL_BASE_IMAGE": "1"}.get(k, d),
    )
    @patch("cli.lib.core.vllm.local_image_exists", return_value=False)
    def test_params_missing_local_base_image_raises(
        self, _mock_local_exists, _mock_env
    ):
        with self.assertRaises(FileNotFoundError):
            VllmBuildParameters()

    @patch(
        "cli.lib.core.vllm.get_env",
        side_effect=lambda k, d=None: {"USE_LOCAL_DOCKERFILE": "1"}.get(k, d),
    )
    @patch("cli.lib.core.vllm.is_path_exist", return_value=False)
    def test_params_missing_dockerfile_raises(self, _mock_is_path, _mock_env):
        with self.assertRaises(FileNotFoundError):
            VllmBuildParameters()


# ---------- VllmBuildRunner helper tests ----------


class TestVllmBuildRunnerHelpers(unittest.TestCase):
    def setUp(self):
        self.runner = VllmBuildRunner()
        self.ns = argparse.Namespace()

    @patch("cli.lib.core.vllm.copy")
    @patch("cli.lib.core.vllm.force_create_dir")
    @patch("cli.lib.core.vllm.get_abs_path", side_effect=lambda p: f"/abs/{p}")
    @patch("cli.lib.core.vllm.is_path_exist", return_value=True)
    def test_cp_torch_whls_if_exist_happy_path(
        self, _mock_exist, mock_abs, mock_force_dir, mock_copy
    ):
        inputs = MagicMock()
        inputs.use_torch_whl = "1"
        inputs.torch_whls_path = "dist"
        tmp = self.runner.cp_torch_whls_if_exist(inputs)
        self.assertEqual(tmp, "./vllm/tmp")
        mock_force_dir.assert_called_once_with("./vllm/tmp")
        mock_copy.assert_called_once_with("/abs/dist", "./vllm/tmp")

    def test_cp_torch_whls_if_exist_flag_off(self):
        inputs = MagicMock(use_torch_whl="0")
        self.assertEqual(self.runner.cp_torch_whls_if_exist(inputs), "")

    @patch("cli.lib.core.vllm.is_path_exist", return_value=False)
    def test_cp_torch_whls_if_exist_missing_raises(self, _mock_exist):
        inputs = MagicMock(use_torch_whl="1", torch_whls_path="dist")
        with self.assertRaises(FileNotFoundError):
            self.runner.cp_torch_whls_if_exist(inputs)

    @patch("cli.lib.core.vllm.copy")
    @patch("cli.lib.core.vllm.get_abs_path", side_effect=lambda p: f"/abs/{p}")
    @patch("cli.lib.core.vllm.is_path_exist", return_value=True)
    def test_cp_dockerfile_if_exist_when_enabled(
        self, _mock_exist, mock_abs, mock_copy
    ):
        inputs = MagicMock(use_local_dockerfile="1", dockerfile_path="Dockerfile")
        self.runner.cp_dockerfile_if_exist(inputs)
        mock_copy.assert_called_once_with(
            "/abs/Dockerfile", "./vllm/docker/Dockerfile.nightly_torch"
        )

    def test_cp_dockerfile_if_exist_flag_zero(self):
        inputs = MagicMock(use_local_dockerfile="0")
        # Should not raise
        self.runner.cp_dockerfile_if_exist(inputs)

    @patch("cli.lib.core.vllm.is_path_exist", return_value=False)
    def test_cp_dockerfile_if_exist_missing_raises(self, _mock_exist):
        inputs = MagicMock(use_local_dockerfile="1", dockerfile_path="Dockerfile")
        with self.assertRaises(FileNotFoundError):
            self.runner.cp_dockerfile_if_exist(inputs)

    def test_get_torch_wheel_path_arg(self):
        self.assertEqual(self.runner._get_torch_wheel_path_arg(""), "")
        self.assertEqual(
            self.runner._get_torch_wheel_path_arg("./vllm/tmp"),
            "--build-arg TORCH_WHEELS_PATH=tmp",
        )

    @patch("cli.lib.core.vllm.local_image_exists", return_value=True)
    def test_get_base_image_args_local_exists(self, _mock_exists):
        inputs = MagicMock(use_local_base_image="1", base_image="img:tag")
        base, final, pull = self.runner._get_base_image_args(inputs)
        self.assertIn("BUILD_BASE_IMAGE=img:tag", base)
        self.assertIn("FINAL_BASE_IMAGE=img:tag", final)
        self.assertEqual(pull, "--pull=false")

    @patch("cli.lib.core.vllm.local_image_exists", return_value=False)
    def test_get_base_image_args_local_missing(self, _mock_exists):
        inputs = MagicMock(use_local_base_image="1", base_image="img:tag")
        base, final, pull = self.runner._get_base_image_args(inputs)
        self.assertIn("BUILD_BASE_IMAGE=img:tag", base)
        self.assertIn("FINAL_BASE_IMAGE=img:tag", final)
        self.assertEqual(pull, "")

    def test_get_base_image_args_flag_zero(self):
        inputs = MagicMock(use_local_base_image="0")
        self.assertEqual(self.runner._get_base_image_args(inputs), ("", "", ""))

    @patch("cli.lib.core.vllm.get_env")
    def test_generate_docker_build_cmd_includes_bits(self, mock_get_env):
        # Configure VllmDockerBuildArgs via get_env
        env_map = {
            "output_dir": "out",
            "TARGET": "export-wheels",
            "TAG": "vllm-wheels",
            "CUDA_VERSION": "12.8.1",
            "PYTHON_VERSION": "3.12",
            "MAX_JOBS": "32",
            "SCCACHE_BUCKET": "my-bucket",
            "SCCACHE_REGION": "us-west-2",
            "TORCH_CUDA_ARCH_LIST": "8.0;9.0",
        }
        mock_get_env.side_effect = lambda k, d=None: env_map.get(k, d)

        inputs = MagicMock()
        with (
            patch.object(
                self.runner,
                "_get_base_image_args",
                return_value=(
                    "--build-arg BUILD_BASE_IMAGE=img:tag",
                    "--build-arg FINAL_BASE_IMAGE=img:tag",
                    "--pull=false",
                ),
            ),
            patch.object(
                self.runner,
                "_get_torch_wheel_path_arg",
                return_value="--build-arg TORCH_WHEELS_PATH=tmp",
            ),
        ):
            cmd = self.runner._generate_docker_build_cmd(
                inputs, "/abs/out", "./vllm/tmp"
            )

        # Spot-check key flags
        self.assertIn("--output type=local,dest=/abs/out", cmd)
        self.assertIn("-f docker/Dockerfile.nightly_torch", cmd)
        self.assertIn("--pull=false", cmd)
        self.assertIn("--build-arg TORCH_WHEELS_PATH=tmp", cmd)
        self.assertIn("--build-arg BUILD_BASE_IMAGE=img:tag", cmd)
        self.assertIn("--build-arg FINAL_BASE_IMAGE=img:tag", cmd)
        self.assertIn("--build-arg max_jobs=32", cmd)
        self.assertIn("--build-arg CUDA_VERSION=12.8.1", cmd)
        self.assertIn("--build-arg PYTHON_VERSION=3.12", cmd)
        self.assertIn("--build-arg USE_SCCACHE=1", cmd)
        self.assertIn("--build-arg SCCACHE_BUCKET_NAME=my-bucket", cmd)
        self.assertIn("--build-arg SCCACHE_REGION_NAME=us-west-2", cmd)
        self.assertIn("--build-arg torch_cuda_arch_list='8.0;9.0'", cmd)
        self.assertIn("--target export-wheels", cmd)
        self.assertIn("-t vllm-wheels", cmd)


# ---------- Full run orchestration ----------


class TestVllmBuildRunnerRun(unittest.TestCase):
    @patch("cli.lib.core.vllm.run_cmd")
    @patch("cli.lib.core.vllm.get_abs_path", side_effect=lambda p: f"/abs/{p}")
    @patch("cli.lib.core.vllm.ensure_dir_exists")
    @patch("cli.lib.core.vllm.clone_vllm")
    def test_run_calls_clone_prepare_and_build(
        self, mock_clone, mock_ensure, mock_abs, mock_run_cmd
    ):
        runner = VllmBuildRunner()

        # Make VllmBuildParameters deterministic & cheap
        fake_params = MagicMock()
        # values used inside run()
        fake_params.output_dir = "shared"
        # Ensure flags so helper methods won't raise
        fake_params.use_local_dockerfile = "0"
        fake_params.use_torch_whl = "0"

        with (
            patch("cli.lib.core.vllm.VllmBuildParameters", return_value=fake_params),
            patch.object(
                runner, "cp_dockerfile_if_exist", return_value=None
            ) as mock_cp_docker,
            patch.object(
                runner, "cp_torch_whls_if_exist", return_value=""
            ) as mock_cp_whl,
            patch.object(
                runner, "_generate_docker_build_cmd", return_value="docker buildx ..."
            ) as mock_gen,
        ):
            runner.run()

        mock_clone.assert_called_once()
        mock_cp_docker.assert_called_once_with(fake_params)
        mock_cp_whl.assert_called_once_with(fake_params)
        mock_ensure.assert_called_once_with("shared")
        mock_abs.assert_called_once_with("shared")
        mock_gen.assert_called_once_with(fake_params, "/abs/shared", "")
        mock_run_cmd.assert_called_once()
        # Assert run_cmd called with cwd="vllm"
        args, kwargs = mock_run_cmd.call_args
        self.assertEqual(kwargs.get("cwd"), "vllm")


if __name__ == "__main__":
    unittest.main()
