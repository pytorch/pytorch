import torch
import pytest
import torch.distributions as dist


def reinforce_estimator(fn, dist, param, num_samples=100000):
    param.requires_grad_(True)
    samples = dist.sample((num_samples,))
    logp = dist.log_prob(samples)
    # Sum out any batch dims
    if dist.batch_shape != ():
        logp = logp.sum([d for d in range(1, logp.ndim)])
    assert logp.shape == (num_samples,)
    y = fn(samples, param)
    assert y.shape == (num_samples,)
    obj = y + y.detach() * logp
    return torch.autograd.grad(obj.mean(), param)[0]

def pathwise_estimator(fn, dist, param, num_samples=100000):
    param.requires_grad_(True)
    samples = dist.rsample((num_samples,))
    obj = fn(samples, param)
    assert obj.shape == (num_samples,)
    return torch.autograd.grad(obj.mean(), param)[0]


@pytest.mark.parametrize("dist_cons, param", [
    (lambda scale: dist.Normal(0., scale), torch.tensor(2.)),
    (lambda conc: dist.Dirichlet(conc), torch.tensor([0.2, 3, 0.5])),
    (lambda conc0: dist.Beta(1., conc0), torch.tensor(0.3)),
    (lambda conc0: dist.Beta(2., conc0), torch.tensor([10., 20.])),
    (lambda conc0: dist.Beta(torch.tensor([0.5, 0.5, 1.]), conc0), torch.tensor([0.8, 0.8, 0.3])),
    (lambda conc: dist.LKJCholesky(3, conc), torch.tensor(0.3)),
    (lambda conc: dist.LKJCholesky(3, conc), torch.tensor(0.5)),
    (lambda conc: dist.LKJCholesky(5, conc), torch.tensor(5.)),
])
def test_reparam_gradient(dist_cons, param):
    torch.manual_seed(1)
    def entropy_monte_carlo(samples, param):
        d = dist_cons(param)
        logp = -d.log_prob(samples)
        if dist.batch_shape != ():
            logp = logp.sum([d for d in range(1, logp.ndim)])
        return logp

    dist = dist_cons(param.requires_grad_(True))
    reinforce_grad = reinforce_estimator(entropy_monte_carlo, dist, param)
    dist = dist_cons(param.requires_grad_(True))
    pathwise_grad = pathwise_estimator(entropy_monte_carlo, dist, param)
    assert torch.allclose(reinforce_grad, pathwise_grad, rtol=0.1)
