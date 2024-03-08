#include <ATen/Tensor.h>

namespace custom {
namespace native {
at::Tensor& add_3_out(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, at::Tensor& out) {
    out = a.add(b).add(c);
    return out;
}
}
}
