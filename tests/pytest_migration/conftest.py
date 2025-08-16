"""
PyTest configuration for the new pytest-based test infrastructure.
This configuration is designed to work alongside PyTorch's existing test infrastructure.
"""
import pytest

def pytest_configure(config):
    """Register markers and configure pytest."""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "slow: mark test as slow running")

@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        # If torch is not importable or CUDA check fails, treat as not available
        return False

@pytest.fixture(autouse=True)
def skip_cuda(request, cuda_available):
    """Skip CUDA tests if CUDA is not available."""
    if request.node.get_closest_marker('cuda') and not cuda_available:
        pytest.skip('CUDA not available')

@pytest.fixture(scope="function")
def cuda_device(cuda_available):
    """Provide CUDA device if available."""
    if not cuda_available:
        pytest.skip('CUDA not available')
    import torch
    return torch.device('cuda')

@pytest.fixture(autouse=True)
def cuda_memory_cleanup():
    """Clean up CUDA memory before and after each test."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            yield
            torch.cuda.empty_cache()
            return
    except Exception:
        pass
    # Default when torch is not available
    yield