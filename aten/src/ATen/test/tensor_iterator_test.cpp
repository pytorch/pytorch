#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

using namespace at;

// An operation with a CUDA tensor and CPU scalar should keep the scalar
// on the CPU (and lift it to a parameter).
TEST(TensorIteratorTest, CPUScalar) {
  if (!at::hasCUDA()) return;
  Tensor out;
  auto x = at::randn({5, 5}, kCUDA);
  auto y = at::ones(1, kCPU).squeeze();
  auto iter = TensorIterator::binary_op(out, x, y);
  EXPECT_TRUE(iter->device(0).is_cuda()) << "result should be CUDA";
  EXPECT_TRUE(iter->device(1).is_cuda()) << "x should be CUDA";
  EXPECT_TRUE(iter->device(2).is_cpu()) << "y should be CPU";
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
  EXPECT_TRUE(iter->device(0).is_cuda()) << "result should be CUDA";
  EXPECT_TRUE(iter->device(1).is_cpu()) << "x should be CPU";
  EXPECT_TRUE(iter->device(2).is_cuda()) << "y should be CUDA";
}

// Mixing CPU and CUDA tensors should raise an exception (if neither is a scalar)
TEST(TensorIteratorTest, MixedDevices) {
  if (!at::hasCUDA()) return;
  Tensor out;
  auto x = at::randn({5, 5}, kCUDA);
  auto y = at::ones({5}, kCPU);
  ASSERT_ANY_THROW(TensorIterator::binary_op(out, x, y));
}

