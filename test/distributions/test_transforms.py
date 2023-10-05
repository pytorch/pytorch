# Owner(s): ["module: distributions"]

import io
from numbers import Number

import pytest

import torch
from torch.autograd.functional import jacobian
from torch.distributions import Dirichlet, Independent, Normal, TransformedDistribution, constraints
from torch.distributions.transforms import (AbsTransform, AffineTransform, ComposeTransform,
                                            CorrCholeskyTransform, CumulativeDistributionTransform,
                                            ExpTransform, IndependentTransform,
                                            LowerCholeskyTransform, PowerTransform,
                                            ReshapeTransform, SigmoidTransform, TanhTransform,
                                            SoftmaxTransform, SoftplusTransform, StickBreakingTransform,
                                            identity_transform, Transform, _InverseTransform,
                                            PositiveDefiniteTransform)
from torch.distributions.utils import tril_matrix_to_vec, vec_to_tril_matrix
from torch.testing._internal.common_utils import run_tests


def get_transforms(cache_size):
    transforms = [
        AbsTransform(cache_size=cache_size),
        ExpTransform(cache_size=cache_size),
        PowerTransform(exponent=2,
                       cache_size=cache_size),
        PowerTransform(exponent=torch.tensor(5.).normal_(),
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
        SoftplusTransform(cache_size=cache_size),
        StickBreakingTransform(cache_size=cache_size),
        LowerCholeskyTransform(cache_size=cache_size),
        CorrCholeskyTransform(cache_size=cache_size),
        PositiveDefiniteTransform(cache_size=cache_size),
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
        ReshapeTransform((4, 5), (2, 5, 2)),
        IndependentTransform(
            AffineTransform(torch.randn(5),
                            torch.randn(5),
                            cache_size=cache_size),
            1),
        CumulativeDistributionTransform(Normal(0, 1)),
    ]
    transforms += [t.inv for t in transforms]
    return transforms


def reshape_transform(transform, shape):
    # Needed to squash batch dims for testing jacobian
    if isinstance(transform, AffineTransform):
        if isinstance(transform.loc, Number):
            return transform
        try:
            return AffineTransform(transform.loc.expand(shape), transform.scale.expand(shape), cache_size=transform._cache_size)
        except RuntimeError:
            return AffineTransform(transform.loc.reshape(shape), transform.scale.reshape(shape), cache_size=transform._cache_size)
    if isinstance(transform, ComposeTransform):
        reshaped_parts = []
        for p in transform.parts:
            reshaped_parts.append(reshape_transform(p, shape))
        return ComposeTransform(reshaped_parts, cache_size=transform._cache_size)
    if isinstance(transform.inv, AffineTransform):
        return reshape_transform(transform.inv, shape).inv
    if isinstance(transform.inv, ComposeTransform):
        return reshape_transform(transform.inv, shape).inv
    return transform


# Generate pytest ids
def transform_id(x):
    assert isinstance(x, Transform)
    name = f'Inv({type(x._inv).__name__})' if isinstance(x, _InverseTransform) else f'{type(x).__name__}'
    return f'{name}(cache_size={x._cache_size})'


def generate_data(transform):
    torch.manual_seed(1)
    while isinstance(transform, IndependentTransform):
        transform = transform.base_transform
    if isinstance(transform, ReshapeTransform):
        return torch.randn(transform.in_shape)
    if isinstance(transform.inv, ReshapeTransform):
        return torch.randn(transform.inv.out_shape)
    domain = transform.domain
    while (isinstance(domain, constraints.independent) and
           domain is not constraints.real_vector):
        domain = domain.base_constraint
    codomain = transform.codomain
    x = torch.empty(4, 5)
    positive_definite_constraints = [constraints.lower_cholesky, constraints.positive_definite]
    if domain in positive_definite_constraints:
        x = torch.randn(6, 6)
        x = x.tril(-1) + x.diag().exp().diag_embed()
        if domain is constraints.positive_definite:
            return x @ x.T
        return x
    elif codomain in positive_definite_constraints:
        return torch.randn(6, 6)
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
    raise ValueError(f'Unsupported domain: {domain}')


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
    assert transform.domain.check(x).all()  # verify that the input data are valid
    try:
        y = transform(x)
    except NotImplementedError:
        pytest.skip('Not implemented.')
    assert y.shape == transform.forward_shape(x.shape)
    if test_cached:
        x2 = transform.inv(y)  # should be implemented at least by caching
    else:
        try:
            x2 = transform.inv(y.clone())  # bypass cache
        except NotImplementedError:
            pytest.skip('Not implemented.')
    assert x2.shape == transform.inverse_shape(y.shape)
    y2 = transform(x2)
    if transform.bijective:
        # verify function inverse
        assert torch.allclose(x2, x, atol=1e-4, equal_nan=True), '\n'.join([
            f'{transform} t.inv(t(-)) error',
            f'x = {x}',
            f'y = t(x) = {y}',
            f'x2 = t.inv(y) = {x2}',
        ])
    else:
        # verify weaker function pseudo-inverse
        assert torch.allclose(y2, y, atol=1e-4, equal_nan=True), '\n'.join([
            f'{transform} t(t.inv(t(-))) error',
            f'x = {x}',
            f'y = t(x) = {y}',
            f'x2 = t.inv(y) = {x2}',
            f'y2 = t(x2) = {y2}',
        ])


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


@pytest.mark.parametrize('transform', TRANSFORMS_CACHE_INACTIVE, ids=transform_id)
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


@pytest.mark.parametrize('transform', TRANSFORMS_CACHE_INACTIVE, ids=transform_id)
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


@pytest.mark.parametrize('transform', TRANSFORMS_CACHE_INACTIVE, ids=transform_id)
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


@pytest.mark.parametrize('transform', ALL_TRANSFORMS, ids=transform_id)
def test_jacobian(transform):
    x = generate_data(transform)
    try:
        y = transform(x)
        actual = transform.log_abs_det_jacobian(x, y)
    except NotImplementedError:
        pytest.skip('Not implemented.')
    # Test shape
    target_shape = x.shape[:x.dim() - transform.domain.event_dim]
    assert actual.shape == target_shape

    # Expand if required
    transform = reshape_transform(transform, x.shape)
    ndims = len(x.shape)
    event_dim = ndims - transform.domain.event_dim
    x_ = x.view((-1,) + x.shape[event_dim:])
    n = x_.shape[0]
    # Reshape to squash batch dims to a single batch dim
    transform = reshape_transform(transform, x_.shape)

    # 1. Transforms with unit jacobian
    if isinstance(transform, ReshapeTransform) or isinstance(transform.inv, ReshapeTransform):
        expected = x.new_zeros(x.shape[x.dim() - transform.domain.event_dim])
        expected = x.new_zeros(x.shape[x.dim() - transform.domain.event_dim])
    # 2. Transforms with 0 off-diagonal elements
    elif transform.domain.event_dim == 0:
        jac = jacobian(transform, x_)
        # assert off-diagonal elements are zero
        assert torch.allclose(jac, jac.diagonal().diag_embed())
        expected = jac.diagonal().abs().log().reshape(x.shape)
    # 3. Transforms with non-0 off-diagonal elements
    else:
        if isinstance(transform, CorrCholeskyTransform):
            jac = jacobian(lambda x: tril_matrix_to_vec(transform(x), diag=-1), x_)
        elif isinstance(transform.inv, CorrCholeskyTransform):
            jac = jacobian(lambda x: transform(vec_to_tril_matrix(x, diag=-1)),
                           tril_matrix_to_vec(x_, diag=-1))
        elif isinstance(transform, StickBreakingTransform):
            jac = jacobian(lambda x: transform(x)[..., :-1], x_)
        else:
            jac = jacobian(transform, x_)

        # Note that jacobian will have shape (batch_dims, y_event_dims, batch_dims, x_event_dims)
        # However, batches are independent so this can be converted into a (batch_dims, event_dims, event_dims)
        # after reshaping the event dims (see above) to give a batched square matrix whose determinant
        # can be computed.
        gather_idx_shape = list(jac.shape)
        gather_idx_shape[-2] = 1
        gather_idxs = torch.arange(n).reshape((n,) + (1,) * (len(jac.shape) - 1)).expand(gather_idx_shape)
        jac = jac.gather(-2, gather_idxs).squeeze(-2)
        out_ndims = jac.shape[-2]
        jac = jac[..., :out_ndims]  # Remove extra zero-valued dims (for inverse stick-breaking).
        expected = torch.slogdet(jac).logabsdet

    assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("event_dims",
                         [(0,), (1,), (2, 3), (0, 1, 2), (1, 2, 0), (2, 0, 1)],
                         ids=str)
def test_compose_affine(event_dims):
    transforms = [AffineTransform(torch.zeros((1,) * e), 1, event_dim=e) for e in event_dims]
    transform = ComposeTransform(transforms)
    assert transform.codomain.event_dim == max(event_dims)
    assert transform.domain.event_dim == max(event_dims)

    base_dist = Normal(0, 1)
    if transform.domain.event_dim:
        base_dist = base_dist.expand((1,) * transform.domain.event_dim)
    dist = TransformedDistribution(base_dist, transform.parts)
    assert dist.support.event_dim == max(event_dims)

    base_dist = Dirichlet(torch.ones(5))
    if transform.domain.event_dim > 1:
        base_dist = base_dist.expand((1,) * (transform.domain.event_dim - 1))
    dist = TransformedDistribution(base_dist, transforms)
    assert dist.support.event_dim == max(1, *event_dims)


@pytest.mark.parametrize("batch_shape", [(), (6,), (5, 4)], ids=str)
def test_compose_reshape(batch_shape):
    transforms = [ReshapeTransform((), ()),
                  ReshapeTransform((2,), (1, 2)),
                  ReshapeTransform((3, 1, 2), (6,)),
                  ReshapeTransform((6,), (2, 3))]
    transform = ComposeTransform(transforms)
    assert transform.codomain.event_dim == 2
    assert transform.domain.event_dim == 2
    data = torch.randn(batch_shape + (3, 2))
    assert transform(data).shape == batch_shape + (2, 3)

    dist = TransformedDistribution(Normal(data, 1), transforms)
    assert dist.batch_shape == batch_shape
    assert dist.event_shape == (2, 3)
    assert dist.support.event_dim == 2


@pytest.mark.parametrize("sample_shape", [(), (7,)], ids=str)
@pytest.mark.parametrize("transform_dim", [0, 1, 2])
@pytest.mark.parametrize("base_batch_dim", [0, 1, 2])
@pytest.mark.parametrize("base_event_dim", [0, 1, 2])
@pytest.mark.parametrize("num_transforms", [0, 1, 2, 3])
def test_transformed_distribution(base_batch_dim, base_event_dim, transform_dim,
                                  num_transforms, sample_shape):
    shape = torch.Size([2, 3, 4, 5])
    base_dist = Normal(0, 1)
    base_dist = base_dist.expand(shape[4 - base_batch_dim - base_event_dim:])
    if base_event_dim:
        base_dist = Independent(base_dist, base_event_dim)
    transforms = [AffineTransform(torch.zeros(shape[4 - transform_dim:]), 1),
                  ReshapeTransform((4, 5), (20,)),
                  ReshapeTransform((3, 20), (6, 10))]
    transforms = transforms[:num_transforms]
    transform = ComposeTransform(transforms)

    # Check validation in .__init__().
    if base_batch_dim + base_event_dim < transform.domain.event_dim:
        with pytest.raises(ValueError):
            TransformedDistribution(base_dist, transforms)
        return
    d = TransformedDistribution(base_dist, transforms)

    # Check sampling is sufficiently expanded.
    x = d.sample(sample_shape)
    assert x.shape == sample_shape + d.batch_shape + d.event_shape
    num_unique = len(set(x.reshape(-1).tolist()))
    assert num_unique >= 0.9 * x.numel()

    # Check log_prob shape on full samples.
    log_prob = d.log_prob(x)
    assert log_prob.shape == sample_shape + d.batch_shape

    # Check log_prob shape on partial samples.
    y = x
    while y.dim() > len(d.event_shape):
        y = y[0]
    log_prob = d.log_prob(y)
    assert log_prob.shape == d.batch_shape


def test_save_load_transform():
    # Evaluating `log_prob` will create a weakref `_inv` which cannot be pickled. Here, we check
    # that `__getstate__` correctly handles the weakref, and that we can evaluate the density after.
    dist = TransformedDistribution(Normal(0, 1), [AffineTransform(2, 3)])
    x = torch.linspace(0, 1, 10)
    log_prob = dist.log_prob(x)
    stream = io.BytesIO()
    torch.save(dist, stream)
    stream.seek(0)
    other = torch.load(stream)
    assert torch.allclose(log_prob, other.log_prob(x))


if __name__ == "__main__":
    run_tests()
