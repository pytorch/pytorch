#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "c10/util/Exception.h"
#include <ATen/Parallel.h>
#include <tuple>


namespace at {
namespace native {
namespace {

static Tensor do_trapz(const Tensor& y, const Tensor& dx, int64_t dim) {
    Tensor left = y.slice(dim, 0, -1);
    Tensor right = y.slice(dim, 1);

    return ((left + right) * dx).sum(dim) / 2.;
}

// special case for scalar dx, to move the multiplication out of the
// loop
static Tensor do_trapz(const Tensor& y, double dx, int64_t dim) {
    Tensor left = y.slice(dim, 0, -1);
    Tensor right = y.slice(dim, 1);
    return (left + right).sum(dim) * (dx/2);
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