# Owner(s): ["module: autograd"]

import torch
from torch.autograd import forward_ad
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    TEST_SCIPY,
    TestCase,
)


if TEST_SCIPY:
    import scipy.special


# name -> spec for the eight low-level Bessel ops made differentiable.
#   dref:   closed-form derivative built from sibling ops (uses J1(x)/x etc.,
#           valid on the non-zero samples used below).
#   domain: "all" allows negative samples; "pos" restricts to x > 0.
#   family/order: selects the SciPy derivative oracle {jvp,yvp,ivp,kvp}(order, x).
_BESSEL_SPECS = {
    "bessel_j0": dict(
        op=torch.special.bessel_j0,
        dref=lambda x: -torch.special.bessel_j1(x),
        domain="all",
        family="j",
        order=0,
    ),
    "bessel_j1": dict(
        op=torch.special.bessel_j1,
        dref=lambda x: torch.special.bessel_j0(x) - torch.special.bessel_j1(x) / x,
        domain="all",
        family="j",
        order=1,
    ),
    "bessel_y0": dict(
        op=torch.special.bessel_y0,
        dref=lambda x: -torch.special.bessel_y1(x),
        domain="pos",
        family="y",
        order=0,
    ),
    "bessel_y1": dict(
        op=torch.special.bessel_y1,
        dref=lambda x: torch.special.bessel_y0(x) - torch.special.bessel_y1(x) / x,
        domain="pos",
        family="y",
        order=1,
    ),
    "modified_bessel_i0": dict(
        op=torch.special.modified_bessel_i0,
        dref=lambda x: torch.special.modified_bessel_i1(x),
        domain="all",
        family="i",
        order=0,
    ),
    "modified_bessel_i1": dict(
        op=torch.special.modified_bessel_i1,
        dref=lambda x: torch.special.modified_bessel_i0(x)
        - torch.special.modified_bessel_i1(x) / x,
        domain="all",
        family="i",
        order=1,
    ),
    "modified_bessel_k0": dict(
        op=torch.special.modified_bessel_k0,
        dref=lambda x: -torch.special.modified_bessel_k1(x),
        domain="pos",
        family="k",
        order=0,
    ),
    "modified_bessel_k1": dict(
        op=torch.special.modified_bessel_k1,
        dref=lambda x: -(
            torch.special.modified_bessel_k0(x)
            + torch.special.modified_bessel_k1(x) / x
        ),
        domain="pos",
        family="k",
        order=1,
    ),
}

# J1 and I1 are finite at x = 0 with derivative 1/2; the closed form is 0/0 there.
_ZERO_LIMIT_OPS = ["bessel_j1", "modified_bessel_i1"]


# Finite-difference gradcheck / gradgradcheck are covered by the OpInfo suite
# (test_ops_gradients.py, TestBwdGradients), which tunes eps/tolerances per op --
# the J/Y forward kernels are only ~1e-7 accurate in float64, so a naive
# gradcheck(eps=1e-6) here would be dominated by forward error. This file instead
# checks independent analytic and SciPy oracles plus the x=0 limit.
class TestBesselAutograd(TestCase):
    def _samples(self, domain, device):
        pos = torch.linspace(0.5, 5.0, 20, dtype=torch.double, device=device)
        if domain == "pos":
            return pos
        return torch.cat([-pos.flip(0), pos])

    @parametrize("name", list(_BESSEL_SPECS))
    def test_grad_matches_recurrence(self, device, name):
        spec = _BESSEL_SPECS[name]
        x = self._samples(spec["domain"], device).requires_grad_(True)
        (grad,) = torch.autograd.grad(spec["op"](x).sum(), x)
        self.assertEqual(grad, spec["dref"](x.detach()))

    @parametrize("name", _ZERO_LIMIT_OPS)
    def test_zero_limit(self, device, name):
        op = _BESSEL_SPECS[name]["op"]
        x = torch.zeros(4, dtype=torch.double, device=device, requires_grad=True)
        (grad,) = torch.autograd.grad(op(x).sum(), x)
        self.assertFalse(torch.isnan(grad).any())
        self.assertEqual(grad, torch.full_like(grad, 0.5))

    @parametrize("name", list(_BESSEL_SPECS))
    def test_forward_ad_matches_backward(self, device, name):
        spec = _BESSEL_SPECS[name]
        x = self._samples(spec["domain"], device)
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(x, torch.ones_like(x))
            jvp = forward_ad.unpack_dual(spec["op"](dual)).tangent

        xr = x.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(spec["op"](xr).sum(), xr)
        self.assertEqual(jvp, grad)

    @parametrize("name", list(_BESSEL_SPECS))
    def test_scipy_oracle(self, device, name):
        if not TEST_SCIPY:
            self.skipTest("SciPy not available")

        spec = _BESSEL_SPECS[name]
        deriv = {
            "j": scipy.special.jvp,
            "y": scipy.special.yvp,
            "i": scipy.special.ivp,
            "k": scipy.special.kvp,
        }[spec["family"]]

        x = self._samples(spec["domain"], device).requires_grad_(True)
        (grad,) = torch.autograd.grad(spec["op"](x).sum(), x)
        expected = torch.as_tensor(
            deriv(spec["order"], x.detach().cpu().numpy()),
            dtype=torch.double,
            device=device,
        )
        self.assertEqual(grad, expected)


instantiate_device_type_tests(TestBesselAutograd, globals())


if __name__ == "__main__":
    run_tests()
