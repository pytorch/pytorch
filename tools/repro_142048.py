import pytest
import torch

def test_solve_triangular_device_mismatch_mps():
    
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    sq_shape = (3, 3)
    A = torch.normal(torch.zeros(sq_shape), torch.ones(sq_shape)).to(device=torch.device("mps:0"))
    eye = torch.eye(A.shape[0], device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="same device"):
        torch.linalg.solve_triangular(A, eye, upper=True)

def reproduce():
    sq_shape = (3, 3)
    if not torch.backends.mps.is_available():
        print("MPS not available on this machine; this repro expects an MPS device. Skipping.")
        return
    device = torch.device("mps:0")
    A = torch.normal(torch.zeros(sq_shape), torch.ones(sq_shape)).to(device=device)
    eye = torch.eye(A.shape[0], device=torch.device("cpu"))
    print("A device:", A.device, "eye device:", eye.device)
    try:
        torch.linalg.solve_triangular(A, eye, upper=True)
        print("Completed without raising (unexpected).")
    except Exception as e:
        print("Raised:", type(e).__name__, e)

if __name__ == "__main__":
    reproduce()