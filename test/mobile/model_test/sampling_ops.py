import torch


# https://pytorch.org/docs/stable/torch.html#random-sampling

class SamplingOpsModule(torch.nn.Module):
    def forward(self):
        a = torch.empty(3, 3).uniform_(0.0, 1.0)
        size = (1, 4)
        weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
        return len(
            # torch.seed(),
            # torch.manual_seed(0),
            torch.bernoulli(a),
            # torch.initial_seed(),
            torch.multinomial(weights, 2),
            torch.normal(2.0, 3.0, size),
            torch.poisson(a),
            torch.rand(2, 3),
            torch.rand_like(a),
            torch.randint(10, size),
            torch.randint_like(a, 4),
            torch.rand(4),
            torch.randn_like(a),
            torch.randperm(4),
            a.bernoulli_(),
            a.cauchy_(),
            a.exponential_(),
            a.geometric_(0.5),
            a.log_normal_(),
            a.normal_(),
            a.random_(),
            a.uniform_(),
        )
