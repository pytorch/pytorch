#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <ATen/CPUGeneratorImpl.h>
#ifdef USE_CUDA
#include <ATen/CUDAGeneratorImpl.h>
#endif
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

Tensor randn(IntArrayRef size, const Tensor& key,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  TORCH_CHECK(key.is_rng_key(), "`randn` only accepts key created from `torch.PRNGKey`");
  TORCH_CHECK(key.dim() == 0 && key.numel() == 1, "key is required to be a Scalar PRNGKey");

  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  auto result = at::empty(size, options);

  const auto& device_ = result.device();
  uint64_t seed = static_cast<uint64_t>(*(key.data_ptr<int64_t>()));
  if (device_.is_cpu()) {
    auto gen = at::detail::createCPUGenerator(seed);
    return result.normal_(0, 1, gen);
#ifdef USE_CUDA
  } else if (device_.is_cuda()) {
    auto gen = at::cuda::detail::createCUDAGenerator(device_.index());
    gen.set_current_seed(seed);
    return result.normal_(0, 1, gen);
#endif
  }
  TORCH_CHECK(false, "Device: " + result.device().str() + " is not supported.");
}

} // namespace native

} // namespace at
