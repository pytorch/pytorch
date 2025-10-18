import copy

# -----------------------------
# Minimal Mock Tensor
# -----------------------------
class MockTensor:
    def __init__(self, shape, device="cpu", dtype="float32", requires_grad=False):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.is_contiguous_flag = True

    def is_contiguous(self):
        return self.is_contiguous_flag

    def __repr__(self):
        return f"MockTensor(shape={self.shape}, device={self.device}, dtype={self.dtype})"

# -----------------------------
# Minimal FakeTensorMode
# -----------------------------
class FakeTensorMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    def from_real_tensor(t: MockTensor):
        """
        Simulate conversion of a real tensor (CPU/CUDA) to a meta tensor.
        """
        if t.device != "meta":
            # simulate deepcopy conversion to meta
            t_meta = copy.copy(t)  # shallow copy first
            t_meta.device = "meta"
            return t_meta
        return t

# -----------------------------
# Test deepcopy logic
# -----------------------------
def test_deepcopy_real_tensor():
    print("Running mock deepcopy test...")

    # Original tensor simulating CUDA
    x = MockTensor(shape=(2, 3), device="cuda")

    # Use FakeTensorMode to convert
    with FakeTensorMode():
        y = copy.deepcopy(FakeTensorMode.from_real_tensor(x))

    # Verify
    print("Original:", x)
    print("Copied  :", y)

    assert y.shape == x.shape, "Shape mismatch"
    assert y.device == "meta", "Device should be converted to meta"
    assert y.dtype == x.dtype, "Dtype should be preserved"

    print("✅ Mock deepcopy test passed!")

# -----------------------------
# Run the test
# -----------------------------
if __name__ == "__main__":
    test_deepcopy_real_tensor()
