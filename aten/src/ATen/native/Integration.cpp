#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "c10/util/Exception.h"
#include <ATen/Parallel.h>
#include <tuple>


namespace at {
namespace native {
namespace {

// The estimated integral of a function y of x,
// sampled at points (y_1, ..., y_n) that are separated by distance (dx_1, ..., dx_{n-1}),
// is given by the trapezoid rule:
//
// \sum_{i=1}^{n-1}  dx_i * (y_i + y_{i+1}) / 2
//
// TODO: if we extend TensorIterator to accept 3 inputs,
// we can probably make this a bit more performant.
static Tensor do_trapz(const Tensor& y, const Tensor& dx, int64_t dim) {
    Tensor left = y.slice(dim, 0, -1);
    Tensor right = y.slice(dim, 1);

    return ((left + right) * dx).sum(dim) / 2.;
}

// When dx is constant, the above formula simplifies
// to dx * [(\sum_{i=1}^n y_i) - (y_1 + y_n)/2]
static Tensor do_trapz(const Tensor& y, double dx, int64_t dim) {
    return (y.sum(dim) - (y.select(dim, 0) + y.select(dim, -1)) * (0.5)) * dx;
}

}

Tensor trapz(const Tensor& y, const Tensor& x, int64_t dim) {
    Tensor x_left = x.slice(dim, 0, -1);
    Tensor x_right = x.slice(dim, 1);

    Tensor dx = x_right - x_left;
    return do_trapz(y, dx, dim);
}

Tensor trapz(const Tensor& y, double dx, int64_t dim) {
    return do_trapz(y, dx, dim);
}

}} // namespace at::native