# Owner(s): inductor

import pytest

import torch


def f() -> float:
    a = torch.tensor([-1.0])
    assert torch.all(a > 0), "should throw"
    return (a + 1).sum().item()  # forces actual computation


@pytest.mark.parametrize("backend", [None, "eager"])
def test_assert_works(backend):
    torch._dynamo.reset()
    g = torch.compile(f, backend=backend) if backend else torch.compile(f)
    with pytest.raises(RuntimeError, match="should throw"):
        g()


if __name__ == "__main__":
    # lets the test run from `python testfile.py`
    from torch.testing._internal.common_utils import run_tests

    run_tests()
