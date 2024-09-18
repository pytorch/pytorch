# flake8: noqa
import torch


# seed
reveal_type(torch.seed())  # E: int

# manual_seed
reveal_type(torch.manual_seed(3))  # E: torch._C.Generator

# initial_seed
reveal_type(torch.initial_seed())  # E: int

# get_rng_state
reveal_type(torch.get_rng_state())  # E: {Tensor}

# bernoulli
reveal_type(torch.bernoulli(torch.empty(3, 3).uniform_(0, 1)))  # E: {Tensor}

# multinomial
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
reveal_type(torch.multinomial(weights, 2))  # E: {Tensor}

# normal
reveal_type(torch.normal(2, 3, size=(1, 4)))  # E: {Tensor}

# poisson
reveal_type(torch.poisson(torch.rand(4, 4) * 5))  # E: {Tensor}

# rand
reveal_type(torch.rand(4))  # E: {Tensor}
reveal_type(torch.rand(2, 3))  # E: {Tensor}

# rand_like
a = torch.rand(4)
reveal_type(torch.rand_like(a))  # E: {Tensor}

# randint
reveal_type(torch.randint(3, 5, (3,)))  # E: {Tensor}
reveal_type(torch.randint(10, (2, 2)))  # E: {Tensor}
reveal_type(torch.randint(3, 10, (2, 2)))  # E: {Tensor}

# randint_like
b = torch.randint(3, 50, (3, 4))
reveal_type(torch.randint_like(b, 3, 10))  # E: {Tensor}

# randn
reveal_type(torch.randn(4))  # E: {Tensor}
reveal_type(torch.randn(2, 3))  # E: {Tensor}

# randn_like
c = torch.randn(2, 3)
reveal_type(torch.randn_like(c))  # E: {Tensor}

# randperm
reveal_type(torch.randperm(4))  # E: {Tensor}

# soboleng
d = torch.quasirandom.SobolEngine(dimension=5)
reveal_type(d)  # E: torch.quasirandom.SobolEngine
reveal_type(d.draw())  # E: {Tensor}
