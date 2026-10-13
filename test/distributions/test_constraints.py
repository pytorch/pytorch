# Owner(s): ["module: distributions"]


import torch
from torch.distributions import biject_to, constraints, transform_to
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


EXAMPLES = [
    (constraints.symmetric, False, [[2.0, 0], [2.0, 2]]),
    (constraints.positive_semidefinite, False, [[2.0, 0], [2.0, 2]]),
    (constraints.positive_definite, False, [[2.0, 0], [2.0, 2]]),
    (constraints.symmetric, True, [[3.0, -5], [-5.0, 3]]),
    (constraints.positive_semidefinite, False, [[3.0, -5], [-5.0, 3]]),
    (constraints.positive_definite, False, [[3.0, -5], [-5.0, 3]]),
    (constraints.symmetric, True, [[1.0, 2], [2.0, 4]]),
    (constraints.positive_semidefinite, True, [[1.0, 2], [2.0, 4]]),
    (constraints.positive_definite, False, [[1.0, 2], [2.0, 4]]),
    (constraints.symmetric, True, [[[1.0, -2], [-2.0, 1]], [[2.0, 3], [3.0, 2]]]),
    (
        constraints.positive_semidefinite,
        False,
        [[[1.0, -2], [-2.0, 1]], [[2.0, 3], [3.0, 2]]],
    ),
    (
        constraints.positive_definite,
        False,
        [[[1.0, -2], [-2.0, 1]], [[2.0, 3], [3.0, 2]]],
    ),
    (constraints.symmetric, True, [[[1.0, -2], [-2.0, 4]], [[1.0, -1], [-1.0, 1]]]),
    (
        constraints.positive_semidefinite,
        True,
        [[[1.0, -2], [-2.0, 4]], [[1.0, -1], [-1.0, 1]]],
    ),
    (
        constraints.positive_definite,
        False,
        [[[1.0, -2], [-2.0, 4]], [[1.0, -1], [-1.0, 1]]],
    ),
    (constraints.symmetric, True, [[[4.0, 2], [2.0, 4]], [[3.0, -1], [-1.0, 3]]]),
    (
        constraints.positive_semidefinite,
        True,
        [[[4.0, 2], [2.0, 4]], [[3.0, -1], [-1.0, 3]]],
    ),
    (
        constraints.positive_definite,
        True,
        [[[4.0, 2], [2.0, 4]], [[3.0, -1], [-1.0, 3]]],
    ),
]

CONSTRAINTS = [
    (constraints.real,),
    (constraints.real_vector,),
    (constraints.positive,),
    (constraints.greater_than, [-10.0, -2, 0, 2, 10]),
    (constraints.greater_than, 0),
    (constraints.greater_than, 2),
    (constraints.greater_than, -2),
    (constraints.greater_than_eq, 0),
    (constraints.greater_than_eq, 2),
    (constraints.greater_than_eq, -2),
    (constraints.less_than, [-10.0, -2, 0, 2, 10]),
    (constraints.less_than, 0),
    (constraints.less_than, 2),
    (constraints.less_than, -2),
    (constraints.unit_interval,),
    (constraints.interval, [-4.0, -2, 0, 2, 4], [-3.0, 3, 1, 5, 5]),
    (constraints.interval, -2, -1),
    (constraints.interval, 1, 2),
    (constraints.half_open_interval, [-4.0, -2, 0, 2, 4], [-3.0, 3, 1, 5, 5]),
    (constraints.half_open_interval, -2, -1),
    (constraints.half_open_interval, 1, 2),
    (constraints.simplex,),
    (constraints.corr_cholesky,),
    (constraints.lower_cholesky,),
    (constraints.positive_definite,),
]


def build_constraint(constraint_fn, args, device="cpu"):
    if not args:
        return constraint_fn
    return constraint_fn(
        *(
            torch.tensor(x, dtype=torch.double, device=device)
            if isinstance(x, list)
            else x
            for x in args
        )
    )


class TestConstraints(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("constraint_fn, result, value", EXAMPLES)
    def test_constraint(self, device, constraint_fn, result, value):
        t = torch.tensor(value, dtype=torch.double, device=device)
        if constraint_fn.check(t).all() != result:
            raise AssertionError(
                f"Expected {result}, got {constraint_fn.check(t).all()}"
            )

    @parametrize("constraint_fn, args", [(c[0], c[1:]) for c in CONSTRAINTS])
    def test_biject_to(self, constraint_fn, args, device):
        constraint = build_constraint(constraint_fn, args, device=device)
        try:
            t = biject_to(constraint)
        except NotImplementedError:
            self.skipTest("`biject_to` not implemented.")
        if not t.bijective:
            raise AssertionError(f"biject_to({constraint}) is not bijective")
        if constraint_fn is constraints.corr_cholesky:
            # (D * (D-1)) / 2 (where D = 4) = 6 (size of last dim)
            x = torch.randn(6, 6, dtype=torch.double, device=device)
        else:
            x = torch.randn(5, 5, dtype=torch.double, device=device)
        y = t(x)
        if not constraint.check(y).all():
            raise AssertionError(
                "\n".join(
                    [
                        f"Failed to biject_to({constraint})",
                        f"x = {x}",
                        f"biject_to(...)(x) = {y}",
                    ]
                )
            )
        x2 = t.inv(y)
        if not torch.allclose(x, x2):
            raise AssertionError(f"Error in biject_to({constraint}) inverse")

        j = t.log_abs_det_jacobian(x, y)
        if j.shape != x.shape[: x.dim() - t.domain.event_dim]:
            raise AssertionError(
                f"Expected shape {x.shape[: x.dim() - t.domain.event_dim]}, got {j.shape}"
            )

    @parametrize("constraint_fn, args", [(c[0], c[1:]) for c in CONSTRAINTS])
    def test_transform_to(self, constraint_fn, args, device):
        constraint = build_constraint(constraint_fn, args, device=device)
        t = transform_to(constraint)
        if constraint_fn is constraints.corr_cholesky:
            # (D * (D-1)) / 2 (where D = 4) = 6 (size of last dim)
            x = torch.randn(6, 6, dtype=torch.double, device=device)
        else:
            x = torch.randn(5, 5, dtype=torch.double, device=device)
        y = t(x)
        if not constraint.check(y).all():
            raise AssertionError(f"Failed to transform_to({constraint})")
        x2 = t.inv(y)
        y2 = t(x2)
        if not torch.allclose(y, y2):
            raise AssertionError(f"Error in transform_to({constraint}) pseudoinverse")


instantiate_device_type_tests(TestConstraints, globals())

if __name__ == "__main__":
    run_tests()
