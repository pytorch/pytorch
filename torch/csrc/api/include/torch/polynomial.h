#pragma once

#include <ATen/ATen.h>

namespace torch {
  namespace polynomial {
    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyadd.
    inline Tensor polynomial_polyadd(const Tensor& self) {
      return torch::polynomial_polyadd(self);
    }

    inline Tensor& polynomial_polyadd_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyadd(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polycompanion.
    inline Tensor polynomial_polycompanion(const Tensor& self) {
      return torch::polynomial_polycompanion(self);
    }

    inline Tensor& polynomial_polycompanion_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polycompanion(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyder.
    inline Tensor polynomial_polyder(const Tensor& self) {
      return torch::polynomial_polyder(self);
    }

    inline Tensor& polynomial_polyder_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyder(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polydiv.
    inline Tensor polynomial_polydiv(const Tensor& self) {
      return torch::polynomial_polydiv(self);
    }

    inline Tensor& polynomial_polydiv_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polydiv(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polydomain.
    inline Tensor polynomial_polydomain(const Tensor& self) {
      return torch::polynomial_polydomain(self);
    }

    inline Tensor& polynomial_polydomain_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polydomain(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyfit.
    inline Tensor polynomial_polyfit(const Tensor& self) {
      return torch::polynomial_polyfit(self);
    }

    inline Tensor& polynomial_polyfit_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyfit(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyfromroots.
    inline Tensor polynomial_polyfromroots(const Tensor& self) {
      return torch::polynomial_polyfromroots(self);
    }

    inline Tensor& polynomial_polyfromroots_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyfromroots(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polygrid2d.
    inline Tensor polynomial_polygrid2d(const Tensor& self) {
      return torch::polynomial_polygrid2d(self);
    }

    inline Tensor& polynomial_polygrid2d_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polygrid2d(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polygrid3d.
    inline Tensor polynomial_polygrid3d(const Tensor& self) {
      return torch::polynomial_polygrid3d(self);
    }

    inline Tensor& polynomial_polygrid3d_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polygrid3d(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyint.
    inline Tensor polynomial_polyint(const Tensor& self) {
      return torch::polynomial_polyint(self);
    }

    inline Tensor& polynomial_polyint_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyint(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyline.
    inline Tensor polynomial_polyline(const Tensor& self) {
      return torch::polynomial_polyline(self);
    }

    inline Tensor& polynomial_polyline_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyline(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polymul.
    inline Tensor polynomial_polymul(const Tensor& self) {
      return torch::polynomial_polymul(self);
    }

    inline Tensor& polynomial_polymul_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polymul(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polymulx.
    inline Tensor polynomial_polymulx(const Tensor& self) {
      return torch::polynomial_polymulx(self);
    }

    inline Tensor& polynomial_polymulx_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polymulx(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyone.
    inline Tensor polynomial_polyone(const Tensor& self) {
      return torch::polynomial_polyone(self);
    }

    inline Tensor& polynomial_polyone_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyone(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polypow.
    inline Tensor polynomial_polypow(const Tensor& self) {
      return torch::polynomial_polypow(self);
    }

    inline Tensor& polynomial_polypow_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polypow(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyroots.
    inline Tensor polynomial_polyroots(const Tensor& self) {
      return torch::polynomial_polyroots(self);
    }

    inline Tensor& polynomial_polyroots_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyroots(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polysub.
    inline Tensor polynomial_polysub(const Tensor& self) {
      return torch::polynomial_polysub(self);
    }

    inline Tensor& polynomial_polysub_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polysub(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polytrim.
    inline Tensor polynomial_polytrim(const Tensor& self) {
      return torch::polynomial_polytrim(self);
    }

    inline Tensor& polynomial_polytrim_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polytrim(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyval.
    inline Tensor polynomial_polyval(const Tensor& self) {
      return torch::polynomial_polyval(self);
    }

    inline Tensor& polynomial_polyval_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyval(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyval2d.
    inline Tensor polynomial_polyval2d(const Tensor& self) {
      return torch::polynomial_polyval2d(self);
    }

    inline Tensor& polynomial_polyval2d_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyval2d(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyval3d.
    inline Tensor polynomial_polyval3d(const Tensor& self) {
      return torch::polynomial_polyval3d(self);
    }

    inline Tensor& polynomial_polyval3d_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyval3d(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyvalfromroots.
    inline Tensor polynomial_polyvalfromroots(const Tensor& self) {
      return torch::polynomial_polyvalfromroots(self);
    }

    inline Tensor& polynomial_polyvalfromroots_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyvalfromroots(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyvander.
    inline Tensor polynomial_polyvander(const Tensor& self) {
      return torch::polynomial_polyvander(self);
    }

    inline Tensor& polynomial_polyvander_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyvander(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyvander2d.
    inline Tensor polynomial_polyvander2d(const Tensor& self) {
      return torch::polynomial_polyvander2d(self);
    }

    inline Tensor& polynomial_polyvander2d_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyvander2d(result, self);
    }

    /// See https://pytorch.org/docs/master/polynomial.html#torch.polynomial.polyvander3d.
    inline Tensor polynomial_polyvander3d(const Tensor& self) {
      return torch::polynomial_polyvander3d(self);
    }

    inline Tensor& polynomial_polyvander3d_out(Tensor& result, const Tensor& self) {
      return torch::polynomial_polyvander3d(result, self);
    }
  }
} // torch::polynomial
