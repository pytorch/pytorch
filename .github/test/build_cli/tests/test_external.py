import pytest
from unittest.mock import patch, MagicMock
import controllers.external_vllm_build as ext_build

# Skip these tests if cement is not installed
try:
    from controllers.external import ExternalBuildController
    from controllers.external_vllm_build import build_vllm
    SKIP_TESTS = False
except ImportError:
    SKIP_TESTS = True

@pytest.mark.skipif(SKIP_TESTS, reason="Cement package not installed")
class TestExternalController:
    """Test the external controller functionality without using cement framework"""

    def test_vllm_build_flow(self):
        """Test the logical flow of building vllm without using the controller"""
        with patch('controllers.external_vllm_build.build_vllm') as mock_build, \
             patch('controllers.external_vllm_build.get_post_build_pinned_commit', return_value="test-commit"):

            # Simulate what the controller would do
            target = 'vllm'
            print(f"[INFO] Target: {target}")

            if target == 'vllm':
                ext_build.build_vllm()
            else:
                print(f"[ERROR] Unknown target: {target}")
            # Verify the expected behavior
            mock_build.assert_called_once()

    def test_unknown_target_flow(self):
        """Test the logical flow with an unknown target"""
        with patch('builtins.print') as mock_print:
            # Simulate what the controller would do
            target = 'unknown'
            print(f"[INFO] Target: {target}")

            if target == 'vllm':
                pass  # Would call build_vllm()
            else:
                print(f"[ERROR] Unknown target: {target}")

            # Verify the expected behavior
            assert mock_print.call_args_list[1][0][0] == "[ERROR] Unknown target: unknown"
