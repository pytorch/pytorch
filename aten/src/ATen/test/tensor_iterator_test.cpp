#include <gtest/gtest.h>
#include <thread>

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

Tensor random_tensor_for_type(at::ScalarType scalar_type) {
  if (at::isFloatingType(scalar_type)) {
    return at::randn({5, 5}, kCPU);
  } else {
    return at::randint(1, 10, {5, 5}, kCPU);
  }
}

#define UNARY_TEST_ITER_FOR_TYPE(ctype,name)                                  \
TEST(TensorIteratorTest, SerialLoopUnary_##name) {                            \
  Tensor out;                                                                 \
  auto in = random_tensor_for_type(k##name);                                  \
  auto expected = in.add(1);                                                  \
  auto iter = TensorIterator::unary_op(out, in);                              \
  at::native::cpu_serial_kernel(iter, [=](ctype a) -> int { return a + 1; }); \
  ASSERT_ANY_THROW(out.equal(expected));                                      \
}

#define BINARY_TEST_ITER_FOR_TYPE(ctype,name)                                          \
TEST(TensorIteratorTest, SerialLoopBinary_##name) {                                    \
  Tensor out;                                                                          \
  auto in1 = random_tensor_for_type(k##name);                                          \
  auto in2 = random_tensor_for_type(k##name);                                          \
  auto expected = in1.add(in2);                                                        \
  auto iter = TensorIterator::binary_op(out, in1, in2);                           \
  at::native::cpu_serial_kernel(iter, [=](ctype a, ctype b) -> int { return a + b; }); \
  ASSERT_ANY_THROW(out.equal(expected));                                               \
}

#define POINTWISE_TEST_ITER_FOR_TYPE(ctype,name)                                                    \
TEST(TensorIteratorTest, SerialLoopPointwise_##name) {                                              \
  Tensor out;                                                                                       \
  auto in1 = random_tensor_for_type(k##name);                                                       \
  auto in2 = random_tensor_for_type(k##name);                                                       \
  auto in3 = random_tensor_for_type(k##name);                                                       \
  auto expected = in1.add(in2).add(in3);                                                            \
  auto iter = at::TensorIterator();                                                                 \
  iter.add_output(out);                                                                             \
  iter.add_input(in1);                                                                              \
  iter.add_input(in2);                                                                              \
  iter.add_input(in3);                                                                              \
  iter.build();                                                                                     \
  at::native::cpu_serial_kernel(iter, [=](ctype a, ctype b, ctype c) -> int { return a + b + c; }); \
  ASSERT_ANY_THROW(out.equal(expected));                                                            \
}

AT_FORALL_SCALAR_TYPES(UNARY_TEST_ITER_FOR_TYPE)
AT_FORALL_SCALAR_TYPES(BINARY_TEST_ITER_FOR_TYPE)
AT_FORALL_SCALAR_TYPES(POINTWISE_TEST_ITER_FOR_TYPE)

TEST(TensorIteratorTest, SerialLoopSingleThread) {
  std::thread::id thread_id = std::this_thread::get_id();
  Tensor out;
  auto x = at::zeros({50000}, kCPU);
  auto iter = TensorIterator::unary_op(out, x);
  at::native::cpu_serial_kernel(iter, [=](int a) -> int {
    std::thread::id lambda_thread_id = std::this_thread::get_id();
    EXPECT_TRUE(lambda_thread_id == thread_id);
    return a + 1;
  });
}

