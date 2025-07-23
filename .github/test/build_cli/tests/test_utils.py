import os
import pytest
from unittest.mock import patch
import shlex


from utils import (
    run,
    get_post_build_pinned_commit,
    get_env,
    create_directory,
)

class TestRun:
    def test_run_basic(self, mock_subprocess_run):
        """Test that run calls subprocess.run with the correct arguments"""
        cmd = "echo hello"
        run(cmd)
        mock_subprocess_run.assert_called_once_with(
            shlex.split(cmd),
            check=True,
            cwd=None,
            env=None
        )

    def test_run_with_cwd(self, mock_subprocess_run):
        """Test that run passes the cwd argument correctly"""
        cmd = "echo hello"
        cwd = "test_dir"
        run(cmd, cwd=cwd)
        mock_subprocess_run.assert_called_once_with(
            shlex.split(cmd),
            check=True,
            cwd=cwd,
            env=None
        )

    def test_run_with_env(self, mock_subprocess_run):
        """Test that run passes the env argument correctly"""
        cmd = "echo hello"
        env = {"TEST_VAR": "test_value"}
        run(cmd, env=env)
        mock_subprocess_run.assert_called_once_with(
            shlex.split(cmd),
            check=True,
            cwd=None,
            env=env
        )

    def test_run_with_logging(self, mock_subprocess_run, capsys):
        """Test that run logs the command when logging is enabled"""
        cmd = "echo hello"
        run(cmd, logging=True)
        captured = capsys.readouterr()
        assert f">>> {cmd}" in captured.out
        mock_subprocess_run.assert_called_once()

class TestGetPostBuildPinnedCommit:
    def test_get_post_build_pinned_commit_success(self, mock_path_exists, mock_path_read_text):
        """Test successful retrieval of pinned commit"""
        mock_path_exists.return_value = True
        mock_path_read_text.return_value = "abc123\n"

        result = get_post_build_pinned_commit("vllm")

        assert result == "abc123"
        mock_path_exists.assert_called_once()
        mock_path_read_text.assert_called_once_with(encoding="utf-8")

    def test_get_post_build_pinned_commit_file_not_found(self, mock_path_exists):
        """Test FileNotFoundError is raised when pin file doesn't exist"""
        mock_path_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            get_post_build_pinned_commit("vllm")

class TestGetEnv:
    def test_get_env_existing_var(self, mock_os_environ):
        """Test getting an existing environment variable"""
        mock_os_environ["TEST_VAR"] = "test_value"

        result = get_env("TEST_VAR")

        assert result == "test_value"

    def test_get_env_non_existing_var(self, mock_os_environ):
        """Test getting a non-existing environment variable returns default"""
        # mock_os_environ is empty due to clear=True in the fixture

        result = get_env("NON_EXISTING_VAR", "default_value")

        assert result == "default_value"

    def test_get_env_empty_default(self, mock_os_environ):
        """Test getting a non-existing environment variable with empty default"""
        # mock_os_environ is empty due to clear=True in the fixture

        result = get_env("NON_EXISTING_VAR")

        assert result == ""

class TestCreateDirectory:
    def test_create_directory(self):
        folder_name = "my_folder"
        mocked_path = f"/mocked/abs/{folder_name}"

        with patch("os.path.abspath", return_value=mocked_path) as mock_abspath, \
            patch("os.path.exists", return_value=True) as mock_exists, \
            patch("shutil.rmtree") as mock_rmtree, \
            patch("os.makedirs") as mock_makedirs:

            create_directory(folder_name)

            # abspath should be called twice
            assert mock_abspath.call_count == 2
            mock_exists.assert_called_once_with(mocked_path)
            mock_rmtree.assert_called_once_with(mocked_path)
            mock_makedirs.assert_called_once_with(mocked_path, exist_ok=True)
