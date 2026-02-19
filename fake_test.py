import torch


def to_fake(tensor):
    return torch._C._to_fake(tensor)


class FakeTensorMode:
    def __enter__(self):
        self._guard = torch._C._IncludeDispatchKeyGuard(
            torch._C.DispatchKey.Fake
        )
        self._guard.__enter__()
        return self

    def __exit__(self, *args):
        self._guard.__exit__(*args)


def test_to_fake_preserves_properties():
    t = torch.randn(3, 4)
    f = to_fake(t)
    assert f.shape == t.shape
    assert f.stride() == t.stride()
    assert f.dtype == t.dtype
    assert f.device == t.device


def test_no_real_storage():
    f = to_fake(torch.randn(3, 4))
    assert f.untyped_storage().device == torch.device("meta")


def test_device_is_cpu():
    f = to_fake(torch.randn(3, 4))
    assert f.device == torch.device("cpu")


def test_add():
    a = to_fake(torch.randn(3, 4))
    b = to_fake(torch.randn(3, 4))
    with FakeTensorMode():
        c = a + b
    assert c.shape == (3, 4)
    assert c.device == torch.device("cpu")
    assert c.untyped_storage().device == torch.device("meta")


def test_matmul():
    a = to_fake(torch.randn(3, 4))
    b = to_fake(torch.randn(4, 5))
    with FakeTensorMode():
        c = a @ b
    assert c.shape == (3, 5)


def test_unary_ops():
    a = to_fake(torch.randn(2, 3))
    with FakeTensorMode():
        assert a.relu().shape == (2, 3)
        assert a.sin().shape == (2, 3)


def test_non_fake_operand_errors():
    real = torch.randn(3, 4)
    fake = to_fake(torch.randn(3, 4))
    try:
        with FakeTensorMode():
            real + fake
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "all tensor operands must be fake tensors" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS: {name}")
    print("All tests passed!")
