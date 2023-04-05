#include <ATen/ops/make_dep_token_native.h>
#include <ATen/ops/empty.h>

namespace at {
namespace native {

Tensor make_dep_token(const Tensor& self) {
    return at::empty({0});
}

} // namespace native
} // namespace at
