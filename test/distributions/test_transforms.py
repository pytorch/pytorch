import pytest

import torch
from torch.autograd import grad
from torch.distributions import Dirichlet, Normal, TransformedDistribution, constraints
from torch.distributions.transforms import (AbsTransform, AffineTransform, CatTransform,
                                            ComposeTransform, CorrCholeskyTransform,
                                            ExpTransform, LowerCholeskyTransform,
                                            PowerTransform, SigmoidTransform, TanhTransform,
                                            SoftmaxTransform, StickBreakingTransform,
                                            identity_transform, StackTransform, Transform,
                                            _InverseTransform)


def get_transforms(cache_size):
    transforms = [
        AbsTransform(cache_size=cache_size),
        ExpTransform(cache_size=cache_size),
        PowerTransform(exponent=2,
                       cache_size=cache_size),
        PowerTransform(exponent=torch.tensor(5.).normal_(),
                       cache_size=cache_size),
        SigmoidTransform(cache_size=cache_size),
        TanhTransform(cache_size=cache_size),
        AffineTransform(0, 1, cache_size=cache_size),
        AffineTransform(1, -2, cache_size=cache_size),
        AffineTransform(torch.randn(5),
                        torch.randn(5),
                        cache_size=cache_size),
        AffineTransform(torch.randn(4, 5),
                        torch.randn(4, 5),
                        cache_size=cache_size),
        SoftmaxTransform(cache_size=cache_size),
        StickBreakingTransform(cache_size=cache_size),
        LowerCholeskyTransform(cache_size=cache_size),
        CorrCholeskyTransform(cache_size=cache_size),
        ComposeTransform([
            AffineTransform(torch.randn(4, 5),
                            torch.randn(4, 5),
                            cache_size=cache_size),
        ]),
        ComposeTransform([
            AffineTransform(torch.randn(4, 5),
                            torch.randn(4, 5),
                            cache_size=cache_size),
            ExpTransform(cache_size=cache_size),
        ]),
        ComposeTransform([
            AffineTransform(0, 1, cache_size=cache_size),
            AffineTransform(torch.randn(4, 5),
                            torch.randn(4, 5),
                            cache_size=cache_size),
            AffineTransform(1, -2, cache_size=cache_size),
            AffineTransform(torch.randn(4, 5),
                            torch.randn(4, 5),
                            cache_size=cache_size),
        ]),
    ]
    transforms += [t.inv for t in transforms]
    return transforms


# Generate pytest ids
def transform_id(x):
    assert isinstance(x, Transform)
    name = f'Inv({type(x._inv).__name__})' if isinstance(x, _InverseTransform) else f'{type(x).__name__}'
    return f'{name}(cache_size={x._cache_size})'


def generate_data(transform):
    torch.manual_seed(1)
    domain = transform.domain
    codomain = transform.codomain
    x = torch.empty(4, 5)
    if domain is constraints.lower_cholesky or codomain is constraints.lower_cholesky:
        x = torch.empty(6, 6)
        x = x.normal_()
        return x
    elif domain is constraints.real:
        return x.normal_()
    elif domain is constraints.real_vector:
        # For corr_cholesky the last dim in the vector
        # must be of size (dim * dim) // 2
        x = torch.empty(3, 6)
        x = x.normal_()
        return x
    elif domain is constraints.positive:
        return x.normal_().exp()
    elif domain is constraints.unit_interval:
        return x.uniform_()
    elif isinstance(domain, constraints.interval):
        x = x.uniform_()
        x = x.mul_(domain.upper_bound - domain.lower_bound).add_(domain.lower_bound)
        return x
    elif domain is constraints.simplex:
        x = x.normal_().exp()
        x /= x.sum(-1, True)
        return x
    elif domain is constraints.corr_cholesky:
        x = torch.empty(4, 5, 5)
        x = x.normal_().tril()
        x /= x.norm(dim=-1, keepdim=True)
        x.diagonal(dim1=-1).copy_(x.diagonal(dim1=-1).abs())
        return x
    raise ValueError('Unsupported domain: {}'.format(domain))


TRANSFORMS_CACHE_ACTIVE = get_transforms(cache_size=1)
TRANSFORMS_CACHE_INACTIVE = get_transforms(cache_size=0)
ALL_TRANSFORMS = TRANSFORMS_CACHE_ACTIVE + TRANSFORMS_CACHE_INACTIVE + [identity_transform]


@pytest.mark.parametrize('transform', ALL_TRANSFORMS, ids=transform_id)
def test_inv_inv(transform, ids=transform_id):
    assert transform.inv.inv is transform


@pytest.mark.parametrize('x', TRANSFORMS_CACHE_INACTIVE, ids=transform_id)
@pytest.mark.parametrize('y', TRANSFORMS_CACHE_INACTIVE, ids=transform_id)
def test_equality(x, y):
    if x is y:
        assert x == y
    else:
        assert x != y
    assert identity_transform == identity_transform.inv


@pytest.mark.parametrize('transform', ALL_TRANSFORMS, ids=transform_id)
def test_with_cache(transform):
    if transform._cache_size == 0:
        transform = transform.with_cache(1)
    assert transform._cache_size == 1
    x = generate_data(transform).requires_grad_()
    try:
        y = transform(x)
    except NotImplementedError:
        pytest.skip('Not implemented.')
    y2 = transform(x)
    assert y2 is y


@pytest.mark.parametrize('transform', ALL_TRANSFORMS, ids=transform_id)
@pytest.mark.parametrize('test_cached', [True, False])
def test_forward_inverse(transform, test_cached):
    x = generate_data(transform).requires_grad_()
    try:
        y = transform(x)
    except NotImplementedError:
        pytest.skip('Not implemented.')
    if test_cached:
        x2 = transform.inv(y)  # should be implemented at least by caching
    else:
        try:
            x2 = transform.inv(y.clone())  # bypass cache
        except NotImplementedError:
            pytest.skip('Not implemented.')
    y2 = transform(x2)
    if transform.bijective:
        # verify function inverse
        assert torch.allclose(x2, x, atol=1e-4, equal_nan=True), '\n'.join([
            '{} t.inv(t(-)) error'.format(transform),
            'x = {}'.format(x),
            'y = t(x) = {}'.format(y),
            'x2 = t.inv(y) = {}'.format(x2),
        ])
    else:
        # verify weaker function pseudo-inverse
        assert torch.allclose(y2, y, atol=1e-4, equal_nan=True), '\n'.join([
            '{} t(t.inv(t(-))) error'.format(transform),
            'x = {}'.format(x),
            'y = t(x) = {}'.format(y),
            'x2 = t.inv(y) = {}'.format(x2),
            'y2 = t(x2) = {}'.format(y2),
        ])


@pytest.mark.parametrize('transform', ALL_TRANSFORMS, ids=transform_id)
def test_univariate_forward_jacobian(transform):
    if transform.event_dim > 0:
        pytest.skip('Not univariate.')
    x = generate_data(transform).requires_grad_()
    try:
        y = transform(x)
        actual = transform.log_abs_det_jacobian(x, y)
    except NotImplementedError:
        pytest.skip('Not implemented.')
    expected = torch.abs(grad([y.sum()], [x])[0]).log()
    assert torch.allclose(actual, expected, atol=1e-5), '\n'.join([
        'Bad {}.log_abs_det_jacobian() disagrees with ()'.format(transform),
        'Expected: {}'.format(expected),
        'Actual: {}'.format(actual),
    ])


@pytest.mark.parametrize('transform', ALL_TRANSFORMS, ids=transform_id)
def test_univariate_inverse_jacobian(transform):
    if transform.event_dim > 0:
        pytest.skip('Not univariate.')
    y = generate_data(transform.inv).requires_grad_()
    try:
        x = transform.inv(y)
        actual = transform.log_abs_det_jacobian(x, y)
    except NotImplementedError:
        pytest.skip('Not implemented.')
    expected = -torch.abs(grad([x.sum()], [y])[0]).log()
    assert torch.allclose(actual, expected, atol=1e-5), '\n'.join([
        '{}.log_abs_det_jacobian() disagrees with .inv()'.format(transform),
        'Expected: {}'.format(expected),
        'Actual: {}'.format(actual),
    ])


@pytest.mark.parametrize('transform', ALL_TRANSFORMS, ids=transform_id)
def test_jacobian_shape(transform):
    x = generate_data(transform)
    try:
        y = transform(x)
        actual = transform.log_abs_det_jacobian(x, y)
    except NotImplementedError:
        pytest.skip('Not implemented.')
    # This is to handle issue in CorrCholeskyTransform
    # where the forward transform adds an additional dimension,
    # and event_dim is defined w.r.t. the shape of the tensor passed
    # to the forward transform.
    if isinstance(transform, _InverseTransform):
        target_shape = y.shape[:y.dim() - transform.event_dim]
    else:
        target_shape = x.shape[:x.dim() - transform.event_dim]
    assert actual.shape == target_shape


def test_compose_transform_shapes():
    transform0 = ExpTransform()
    transform1 = SoftmaxTransform()
    transform2 = LowerCholeskyTransform()

    assert transform0.event_dim == 0
    assert transform1.event_dim == 1
    assert transform2.event_dim == 2
    assert ComposeTransform([transform0, transform1]).event_dim == 1
    assert ComposeTransform([transform0, transform2]).event_dim == 2
    assert ComposeTransform([transform1, transform2]).event_dim == 2


transform0 = ExpTransform()
transform1 = SoftmaxTransform()
transform2 = LowerCholeskyTransform()
base_dist0 = Normal(torch.zeros(4, 4), torch.ones(4, 4))
base_dist1 = Dirichlet(torch.ones(4, 4))
base_dist2 = Normal(torch.zeros(3, 4, 4), torch.ones(3, 4, 4))


@pytest.mark.parametrize('batch_shape, event_shape, dist', [
    ((4, 4), (), base_dist0),
    ((4,), (4,), base_dist1),
    ((4, 4), (), TransformedDistribution(base_dist0, [transform0])),
    ((4,), (4,), TransformedDistribution(base_dist0, [transform1])),
    ((4,), (4,), TransformedDistribution(base_dist0, [transform0, transform1])),
    ((), (4, 4), TransformedDistribution(base_dist0, [transform0, transform2])),
    ((4,), (4,), TransformedDistribution(base_dist0, [transform1, transform0])),
    ((), (4, 4), TransformedDistribution(base_dist0, [transform1, transform2])),
    ((), (4, 4), TransformedDistribution(base_dist0, [transform2, transform0])),
    ((), (4, 4), TransformedDistribution(base_dist0, [transform2, transform1])),
    ((4,), (4,), TransformedDistribution(base_dist1, [transform0])),
    ((4,), (4,), TransformedDistribution(base_dist1, [transform1])),
    ((), (4, 4), TransformedDistribution(base_dist1, [transform2])),
    ((4,), (4,), TransformedDistribution(base_dist1, [transform0, transform1])),
    ((), (4, 4), TransformedDistribution(base_dist1, [transform0, transform2])),
    ((4,), (4,), TransformedDistribution(base_dist1, [transform1, transform0])),
    ((), (4, 4), TransformedDistribution(base_dist1, [transform1, transform2])),
    ((), (4, 4), TransformedDistribution(base_dist1, [transform2, transform0])),
    ((), (4, 4), TransformedDistribution(base_dist1, [transform2, transform1])),
    ((3, 4, 4), (), base_dist2),
    ((3,), (4, 4), TransformedDistribution(base_dist2, [transform2])),
    ((3,), (4, 4), TransformedDistribution(base_dist2, [transform0, transform2])),
    ((3,), (4, 4), TransformedDistribution(base_dist2, [transform1, transform2])),
    ((3,), (4, 4), TransformedDistribution(base_dist2, [transform2, transform0])),
    ((3,), (4, 4), TransformedDistribution(base_dist2, [transform2, transform1])),
])
def test_transformed_distribution_shapes(batch_shape, event_shape, dist):
    assert dist.batch_shape == batch_shape
    assert dist.event_shape == event_shape
    x = dist.rsample()
    try:
        dist.log_prob(x)  # this should not crash
    except NotImplementedError:
        pytest.skip('Not implemented.')


@pytest.mark.parametrize('transform', TRANSFORMS_CACHE_INACTIVE)
def test_jit_fwd(transform):
    x = generate_data(transform).requires_grad_()

    def f(x):
        return transform(x)

    try:
        traced_f = torch.jit.trace(f, (x,))
    except NotImplementedError:
        pytest.skip('Not implemented.')

    # check on different inputs
    x = generate_data(transform).requires_grad_()
    assert torch.allclose(f(x), traced_f(x), atol=1e-5, equal_nan=True)


@pytest.mark.parametrize('transform', TRANSFORMS_CACHE_INACTIVE)
def test_jit_inv(transform):
    y = generate_data(transform.inv).requires_grad_()

    def f(y):
        return transform.inv(y)

    try:
        traced_f = torch.jit.trace(f, (y,))
    except NotImplementedError:
        pytest.skip('Not implemented.')

    # check on different inputs
    y = generate_data(transform.inv).requires_grad_()
    assert torch.allclose(f(y), traced_f(y), atol=1e-5, equal_nan=True)


@pytest.mark.parametrize('transform', TRANSFORMS_CACHE_INACTIVE)
def test_jit_jacobian(transform):
    x = generate_data(transform).requires_grad_()

    def f(x):
        y = transform(x)
        return transform.log_abs_det_jacobian(x, y)

    try:
        traced_f = torch.jit.trace(f, (x,))
    except NotImplementedError:
        pytest.skip('Not implemented.')

    # check on different inputs
    x = generate_data(transform).requires_grad_()
    assert torch.allclose(f(x), traced_f(x), atol=1e-5, equal_nan=True)


if __name__ == '__main__':
    pytest.main([__file__])
