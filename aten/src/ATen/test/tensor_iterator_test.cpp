#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

using namespace at;

// An operation with a CUDA tensor and CPU scalar should keep the scalar
// on the CPU (and lift it to a parameter).
TEST(TensorIteratorTest, CPUScalar) {
  if (!at::hasCUDA()) return;
  Tensor out;
  auto x = at::randn({5, 5}, kCUDA);
  auto y = at::ones(1, kCPU).squeeze();
  auto iter = TensorIterator::binary_op(out, x, y);
  EXPECT_TRUE(iter.device(0).is_cuda()) << "result should be CUDA";
  EXPECT_TRUE(iter.device(1).is_cuda()) << "x should be CUDA";
  EXPECT_TRUE(iter.device(2).is_cpu()) << "y should be CPU";
}

// An operation with a CUDA output and CPU scalar inputs should only
// keep a single input as a CPU scalar. (Because we only generate
// specializations in Loops.cuh for a single CPU scalar).
TEST(TensorIteratorTest, CPUScalarInputs) {
  if (!at::hasCUDA()) return;
  Tensor out = at::empty({5, 5}, kCUDA);
  auto x = at::ones(1, kCPU).squeeze();
  auto y = at::ones(1, kCPU).squeeze();
  auto iter = TensorIterator::binary_op(out, x, y);
  EXPECT_TRUE(iter.device(0).is_cuda()) << "result should be CUDA";
  EXPECT_TRUE(iter.device(1).is_cpu()) << "x should be CPU";
  EXPECT_TRUE(iter.device(2).is_cuda()) << "y should be CUDA";
}

// Mixing CPU and CUDA tensors should raise an exception (if neither is a scalar)
TEST(TensorIteratorTest, MixedDevices) {
  if (!at::hasCUDA()) return;
  Tensor out;
  auto x = at::randn({5, 5}, kCUDA);
  auto y = at::ones({5}, kCPU);
  ASSERT_ANY_THROW(TensorIterator::binary_op(out, x, y));
}

namespace at{ namespace native {  // required to use cpu_apply_dim_kernel

Tensor gather_with_broadcast_cpu(IntArrayRef outsizes, const Tensor &src, int64_t dim, const Tensor &index) {
  Tensor result = at::empty(outsizes, src.options());
  auto iter = TensorIterator::dim_apply_op(result, index, src, dim);
  int64_t size = index.size(dim);
  cpu_apply_dim_kernel(iter,
    [=](float *result_data, int64_t result_stride, int64_t *index_data, int64_t index_stride, float *src_data, int64_t src_stride) {
      for (int64_t i = 0; i < size; i++) {
        int64_t index = *(index_data + i * index_stride);
        *(result_data + i * result_stride) = *(src_data + index * src_stride);
      }
    });
  return result;
}

}}  // namespace at::native

// Test TensorIterator's dim_apply CPU implementation by manually implementing gather
TEST(TensorIteratorTest, DimApply) {
  Tensor src = at::randn({20, 1, 20, 10});
  Tensor index = at::randint(20, {100, 10, 20, 1}, ScalarType::Long);
  Tensor result1 = src.expand({20, 10, 20, 10}).gather(0, index.expand({100, 10, 20, 10}));
  Tensor result2 = at::native::gather_with_broadcast_cpu(result1.sizes(), src, 0, index);
  EXPECT_TRUE(at::allclose(result1, result2));
}
