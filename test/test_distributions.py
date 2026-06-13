import torch
from torch.distributions import Gamma

def test_gamma_generator_basic():
    g = Gamma(torch.tensor(2.0), torch.tensor(1.0))
    gen = torch.Generator().manual_seed(42)
    result = g.sample((5,), generator=gen)
    assert result.shape == torch.Size([5])

def test_gamma_deterministic_with_same_seed():
    g = Gamma(torch.tensor(2.0), torch.tensor(1.0))
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    result1 = g.sample((5,), generator=gen1)
    result2 = g.sample((5,), generator=gen2)
    assert torch.allclose(result1, result2)

def test_gamma_with_different_seeds():
    g = Gamma(torch.tensor(2.0), torch.tensor(1.0))
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(99)
    result1 = g.sample((5,), generator=gen1)
    result2 = g.sample((5,), generator=gen2)
    assert not torch.allclose(result1, result2)

def test_gamma_batched():
    g = Gamma(torch.tensor(2.0), torch.tensor(1.0))
    gen = torch.Generator().manual_seed(42)
    result = g.sample((5, 3), generator=gen)
    assert result.shape == torch.Size([5, 3])

def test_gamma_without_generator():
    g = Gamma(torch.tensor(2.0), torch.tensor(1.0))
    result = g.sample((5,))
    assert result.shape == torch.Size([5])

if __name__ == "__main__":
    print("=" * 50)
    test_gamma_generator_basic()
    print("Test 1 PASSED: generator argument works")
    test_gamma_deterministic_with_same_seed()
    print(" Test 2 PASSED: same seed = same output")
    test_gamma_with_different_seeds()
    print(" Test 3 PASSED: different seeds differ")
    test_gamma_batched()
    print(" Test 4 PASSED: batched shape works")
    test_gamma_without_generator()
    print(" Test 5 PASSED: works without generator")
    print("=" * 50)
    print(" All 5 tests passed! Ready to PR ")
    print("=" * 50)
