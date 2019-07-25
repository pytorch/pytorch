#include <cmath>
#include <iostream>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/native/FastPathBinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native {

namespace {

#define FAST_PATH_BINARY_LOOP(scalar_type, ...)               \
  scalar_type* result_ptr = result.data<scalar_type>();       \
  const scalar_type* self_ptr = self.data<scalar_type>();     \
  const scalar_type* other_ptr = other.data<scalar_type>();   \
  for (int i = begin; i < end; ++i) {                         \
    __VA_ARGS__                                               \
  }

void fast_path_add_kernel(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha_scalar) {
  int64_t num_elements = self.numel();
  at::parallel_for(0, num_elements, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "fast_path_add_cpu", [&]() {
      auto alpha = alpha_scalar.to<scalar_t>();
      FAST_PATH_BINARY_LOOP(scalar_t, result_ptr[i] = self_ptr[i] + other_ptr[i] * alpha;)
    });
  });
}

void fast_path_sub_kernel(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha_scalar) {
  fast_path_add_kernel(result, self, other, -alpha_scalar);
}

void fast_path_mul_kernel(Tensor& result, const Tensor& self, const Tensor& other) {
  int64_t num_elements = self.numel();
  if (self.scalar_type() == ScalarType::Bool) {
    at::parallel_for(0, num_elements, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    FAST_PATH_BINARY_LOOP(bool, result_ptr[i] = self_ptr[i] && other_ptr[i];)
    });
  } else {
    at::parallel_for(0, num_elements, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    AT_DISPATCH_ALL_TYPES(self.scalar_type(), "fast_path_mul_cpu", [&]() {
      FAST_PATH_BINARY_LOOP(scalar_t, result_ptr[i] = self_ptr[i] * other_ptr[i];)
    });
    });
  }
}

void fast_path_div_kernel(Tensor& result, const Tensor& self, const Tensor& other) {
  int64_t num_elements = self.numel();
  at::parallel_for(0, num_elements, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "fast_path_div_cpu", [&]() {
      FAST_PATH_BINARY_LOOP(scalar_t, result_ptr[i] = self_ptr[i] / other_ptr[i];)
    });
  });
}

} // anonymous namespace


REGISTER_DISPATCH(fast_path_add_stub, &fast_path_add_kernel);
REGISTER_DISPATCH(fast_path_sub_stub, &fast_path_sub_kernel);
REGISTER_DISPATCH(fast_path_mul_stub, &fast_path_mul_kernel);
REGISTER_DISPATCH(fast_path_div_stub, &fast_path_div_kernel);

}} // namespace at::native
