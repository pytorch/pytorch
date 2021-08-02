#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/irange.h>

#include <vector>

namespace at { namespace native {

namespace detail {

Tensor _philox(const Tensor& self, int64_t ctr) {
  Tensor result;
  uint64_t u_ctr = static_cast<uint64_t>(ctr);
  auto iter = TensorIterator::unary_op(result, self);
  cpu_kernel(iter, [u_ctr](int64_t x){
      return static_cast<int64_t>(at::_philox(static_cast<uint64_t>(x), u_ctr));
  });
  return iter.output();
}

} // namespace detail

Tensor split_key(const Tensor& self, int64_t ctr) {
  TORCH_CHECK(self.is_rng_key(), "`split_key` only accepts PRNGKey created from `torch.PRNGKey`");
  std::vector<Tensor> keys(ctr);
  auto sizes = self.sizes();
  for (const auto i : c10::irange(ctr)) {
    auto tensor = detail::_philox(self, i);
    keys[i] = std::move(tensor);
  }
  auto res = at::stack(keys, 0);
  res._set_rng_key(true);
  return res;
}

} // namespace native
} // namespace at
