import torch
from ..function import Function


class Categorical(Function):
    @staticmethod
    def forward(ctx, probs, num_samples, with_replacement):
        samples = probs.multinomial(num_samples, with_replacement)
        ctx.mark_non_differentiable(samples)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None


class Bernoulli(Function):
    @staticmethod
    def forward(ctx, probs):
        samples = probs.new().resize_as_(probs).bernoulli_(probs)
        ctx.mark_non_differentiable(samples)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        return None


class Normal(Function):
    @staticmethod
    def forward(ctx, means, stddevs=None):
        samples = torch.normal(means, stddevs)
        ctx.mark_non_differentiable(samples)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        return None, None
