#include <ATen/Tensor.h>

namespace custom {
namespace native {
Tensor& add_3_out(const Tensor& a, const Tensor& b, const Tensor& c, Tensor& out) {
    out = a + b + c;
    return out;
}
}
}
