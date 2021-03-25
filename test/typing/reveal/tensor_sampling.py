# flake8: noqa
import torch

# seed
# manual_seed
# initial_seed
# get_rng_state
# set_rng_state

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

# randint
reveal_type(torch.randint(3, 5, (3,)))  # E: {Tensor}
reveal_type(torch.randint(10, (2, 2)))  # E: {Tensor}
reveal_type(torch.randint(3, 10, (2, 2)))  # E: {Tensor}

# randint_like

# randn
reveal_type(torch.randn(4))  # E: {Tensor}
reveal_type(torch.randn(2, 3))  # E: {Tensor}

# randn_like

# randperm
reveal_type(torch.randperm(4))  # E: {Tensor}

# soboleng
a = torch.quasirandom.SobolEngine(dimension=5)
reveal_type(a)  # E: torch.quasirandom.SobolEngine
reveal_type(a.draw())  # E: {Tensor}
