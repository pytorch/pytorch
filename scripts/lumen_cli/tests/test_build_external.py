import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from cli.lib.core.vllm import VllmBuildRunner


class TestVllmBuildRunner(unittest.TestCase):
    def setUp(self):
        self.runner = VllmBuildRunner()

    @patch("cli.lib.core.vllm.clone_external_repo")
    @patch("cli.lib.core.vllm.get_abs_path", side_effect=lambda x: f"/abs/{x}")
    @patch("cli.lib.core.vllm.ensure_dir_exists")
    @patch("cli.lib.core.vllm.VllmBuildRunner.cp_torch_whls_if_exist")
    @patch("cli.lib.core.vllm.VllmBuildRunner.cp_dockerfile_if_exist")
    def test_prepare_success(
        self, mock_cp_dockerfile, mock_cp_whls, mock_ensure_dir, mock_abs, mock_clone
    ):
        # simulate config
        self.runner.get_external_build_config = MagicMock(
            return_value={
                "artifact_dir": "artifacts",
                "torch_whl_dir": "whls",
                "dockerfile_path": "docker/Dockerfile.nightly_torch",
                "base_image": "base",
            }
        )
        self.runner.prepare()
        mock_clone.assert_called_once()
        mock_ensure_dir.assert_called_once_with("/abs/artifacts")
        mock_cp_dockerfile.assert_called_once()
        mock_cp_whls.assert_called_once()

    @patch("cli.lib.core.vllm.run_cmd")
    @patch("cli.lib.core.vllm._generate_docker_build_cmd", return_value="echo build")
    @patch("cli.lib.core.vllm._get_torch_wheel_path_arg", return_value="--torch")
    @patch.object(VllmBuildRunner, "prepare")
    def test_run(self, mock_prepare, mock_torch_arg, mock_generate_cmd, mock_run_cmd):
        self.runner.cfg = self.runner._to_vllm_build_config()
        self.runner.run()
        mock_prepare.assert_called_once()
        mock_generate_cmd.assert_called_once()
        mock_run_cmd.assert_called_once_with("echo build", cwd="vllm", env=mock.ANY)

    @patch("cli.lib.core.vllm.is_path_exist", return_value=True)
    @patch("cli.lib.core.vllm.run_cmd")
    @patch("cli.lib.core.vllm.force_create_dir")
    def test_cp_torch_whls(self, mock_force_create, mock_run, mock_exist):
        self.runner.cfg.torch_whl_dir = "some/path"
        self.runner.cp_torch_whls_if_exist()
        mock_force_create.assert_called_once_with("./vllm/tmp")
        mock_run.assert_called_once_with("cp -a some/path/. ./vllm/tmp", log_cmd=True)
