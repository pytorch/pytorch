def test_inductor_complex_scalar_add():
    import torch

    def f():
        x = torch.ops.aten.scalar_tensor(2.5, dtype=torch.complex64)
        return x + x

    out = torch.compile(f, backend="inductor")()
    assert out == torch.tensor(5.0 + 0j)


