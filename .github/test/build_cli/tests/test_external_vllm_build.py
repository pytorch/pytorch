import os
import pytest
import shlex
from unittest.mock import patch, call, MagicMock

from controllers.external_vllm_build import build_vllm, clone_vllm

class TestCloneVllm:
    def test_clone_vllm(self):
        """Test that clone_vllm calls the correct commands"""
        commit = "test-commit-hash"

        # Mock the run function to avoid actual git operations
        with patch('controllers.external_vllm_build.run') as mock_run:
            # Call the function
            clone_vllm(commit)

            # Check that the correct commands were called
            expected_calls = [
                call("git clone https://github.com/vllm-project/vllm.git"),
                call(f"git checkout {commit}", "vllm"),
                call("git submodule update --init --recursive", "vllm")
            ]

            assert mock_run.call_args_list == expected_calls

    def test_clone_vllm_deletes_existing_directory(self, mock_os_path_exists, mock_shutil_rmtree):
        """Test that clone_vllm deletes the directory if it exists"""
        with patch('controllers.external_vllm_build.delete_directory') as mock_delete, \
             patch('controllers.external_vllm_build.run'):  # Mock run to prevent actual git operations
            commit = "test-commit-hash"

            # Call the function
            clone_vllm(commit)

            # Check that delete_directory was called with the correct argument
            mock_delete.assert_called_once_with("vllm")

class TestBuildVllm:
    def test_build_vllm_default_env(self):
        """Test build_vllm with default environment variables"""
        # Setup
        with patch('controllers.external_vllm_build.get_post_build_pinned_commit') as mock_get_commit, \
             patch('controllers.external_vllm_build.clone_vllm') as mock_clone, \
             patch('controllers.external_vllm_build.create_directory') as mock_create_dir, \
             patch('controllers.external_vllm_build.Timer') as mock_timer_class, \
             patch('builtins.print') as mock_print, \
             patch('controllers.external_vllm_build.run') as mock_run, \
             patch('controllers.external_vllm_build.os.environ.copy', return_value={}):

            # Setup the Timer mock to be used as a context manager
            mock_timer = MagicMock()
            mock_timer_class.return_value = mock_timer

            # Setup the Timer mock to be used as a context manager
            mock_timer = MagicMock()
            mock_timer_class.return_value = mock_timer

            mock_get_commit.return_value = "test-commit"

            # Call the function
            build_vllm()

            # Check that the correct functions were called
            mock_get_commit.assert_called_once_with("vllm")
            mock_clone.assert_called_once_with("test-commit")
            mock_create_dir.assert_called_once_with("shared")
            mock_timer.__enter__.assert_called_once()
            mock_timer.__exit__.assert_called_once()

            # Check that the correct commands were run
            assert len(mock_run.call_args_list) == 2  # Copy Dockerfile and docker build

            # Check the docker build command
            docker_cmd_call = mock_run.call_args_list[1]
            docker_cmd = docker_cmd_call[0][0]

            # Verify the docker command contains expected arguments
            assert "docker buildx build" in docker_cmd
            assert "--output type=local,dest=../shared" in docker_cmd
            assert "-f docker/Dockerfile.nightly_torch" in docker_cmd
            assert "--build-arg max_jobs=32" in docker_cmd
            assert "--build-arg CUDA_VERSION=12.8.0" in docker_cmd
            assert "--build-arg PYTHON_VERSION=3.12" in docker_cmd
            assert "--build-arg USE_SCCACHE=0" in docker_cmd
            assert "--build-arg torch_cuda_arch_list=8.6;8.9" in docker_cmd
            assert "--target export-wheels" in docker_cmd
            assert "-t vllm-wheels" in docker_cmd
            assert docker_cmd_call[1].get('cwd') == "vllm"
            assert docker_cmd_call[1].get('logging') is True

    def test_build_vllm_custom_env(self):
        """Test build_vllm with custom environment variables"""
        # Setup
        with patch('controllers.external_vllm_build.get_post_build_pinned_commit') as mock_get_commit, \
             patch('controllers.external_vllm_build.clone_vllm') as mock_clone, \
             patch('controllers.external_vllm_build.create_directory') as mock_create_dir, \
             patch('controllers.external_vllm_build.Timer') as mock_timer_class, \
             patch('builtins.print') as mock_print, \
             patch('controllers.external_vllm_build.run') as mock_run, \
             patch('controllers.external_vllm_build.os.environ.copy', return_value={}):

            mock_get_commit.return_value = "test-commit"

            # Set custom environment variables
            custom_env = {
                "TAG": "custom-tag",
                "CUDA_VERSION": "11.8.0",
                "PYTHON_VERSION": "3.10",
                "MAX_JOBS": "16",
                "TARGET": "custom-target",
                "SCCACHE_BUCKET_NAME": "test-bucket",
                "SCCACHE_REGION_NAME": "test-region",
                "TORCH_CUDA_ARCH_LIST": "7.5;8.0"
            }

            with patch.dict('os.environ', custom_env):
                # Call the function
                build_vllm()

                # Check that the correct functions were called
                mock_get_commit.assert_called_once_with("vllm")
                mock_clone.assert_called_once_with("test-commit")
                mock_create_dir.assert_called_once_with("shared")

                # Check that the correct commands were run
                assert len(mock_run.call_args_list) == 2  # Copy Dockerfile and docker build

                # Check the docker build command
                docker_cmd_call = mock_run.call_args_list[1]
                docker_cmd = docker_cmd_call[0][0]

                # Verify the docker command contains expected arguments
                assert "docker buildx build" in docker_cmd
                assert "--build-arg max_jobs=16" in docker_cmd
                assert "--build-arg CUDA_VERSION=11.8.0" in docker_cmd
                assert "--build-arg PYTHON_VERSION=3.10" in docker_cmd
                assert "--build-arg USE_SCCACHE=1" in docker_cmd
                assert "--build-arg SCCACHE_BUCKET_NAME=test-bucket" in docker_cmd
                assert "--build-arg SCCACHE_REGION_NAME=test-region" in docker_cmd
                assert "--build-arg torch_cuda_arch_list=7.5;8.0" in docker_cmd
                assert "--target custom-target" in docker_cmd
                assert "-t custom-tag" in docker_cmd
                assert docker_cmd_call[1].get('cwd') == "vllm"
                assert docker_cmd_call[1].get('logging') is True
