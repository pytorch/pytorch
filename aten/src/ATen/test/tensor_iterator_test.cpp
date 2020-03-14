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
    return at::randn({5, 5}, at::device(kCPU).dtype(scalar_type));
  } else if (scalar_type == kBool) {
    return at::randint(0, 2, {5, 5}, at::device(kCPU).dtype(scalar_type));
  } else {
    return at::randint(1, 10, {5, 5}, at::device(kCPU).dtype(scalar_type));
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

#define NO_OUTPUT_UNARY_TEST_ITER_FOR_TYPE(ctype,name)                         \
TEST(TensorIteratorTest, SerialLoopUnaryNoOutput_##name) {                     \
  auto in = random_tensor_for_type(k##name);                                   \
  auto iter = at::TensorIterator();                                            \
  iter.add_input(in);                                                          \
  iter.build();                                                                \
  int64_t acc = 0;                                                             \
  at::native::cpu_serial_kernel(iter, [&](ctype a) -> void { acc++; }); \
  EXPECT_TRUE(acc == in.numel());                                              \
}

#define BINARY_TEST_ITER_FOR_TYPE(ctype,name)                                          \
TEST(TensorIteratorTest, SerialLoopBinary_##name) {                                    \
  Tensor out;                                                                          \
  auto in1 = random_tensor_for_type(k##name);                                          \
  auto in2 = random_tensor_for_type(k##name);                                          \
  auto expected = in1.add(in2);                                                        \
  auto iter = TensorIterator::binary_op(out, in1, in2);                                \
  at::native::cpu_serial_kernel(iter, [=](ctype a, ctype b) -> int { return a + b; }); \
  ASSERT_ANY_THROW(out.equal(expected));                                               \
}

#define NO_OUTPUT_BINARY_TEST_ITER_FOR_TYPE(ctype,name)                          \
TEST(TensorIteratorTest, SerialLoopBinaryNoOutput_##name) {                      \
  auto in1 = random_tensor_for_type(k##name);                                    \
  auto in2 = random_tensor_for_type(k##name);                                    \
  auto iter = at::TensorIterator();                                              \
  iter.add_input(in1);                                                           \
  iter.add_input(in2);                                                           \
  iter.build();                                                                  \
  int64_t acc = 0;                                                               \
  at::native::cpu_serial_kernel(iter, [&](ctype a, ctype b) -> void { acc++; }); \
  EXPECT_TRUE(acc == in1.numel());                                               \
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

#define NO_OUTPUT_POINTWISE_TEST_ITER_FOR_TYPE(ctype,name)                                \
TEST(TensorIteratorTest, SerialLoopPoinwiseNoOutput_##name) {                             \
  auto in1 = random_tensor_for_type(k##name);                                             \
  auto in2 = random_tensor_for_type(k##name);                                             \
  auto in3 = random_tensor_for_type(k##name);                                             \
  auto iter = at::TensorIterator();                                                       \
  iter.add_input(in1);                                                                    \
  iter.add_input(in2);                                                                    \
  iter.add_input(in3);                                                                    \
  iter.build();                                                                           \
  int64_t acc = 0;                                                                        \
  at::native::cpu_serial_kernel(iter, [&](ctype a, ctype b, ctype c) -> void { acc++; }); \
  EXPECT_TRUE(acc == in1.numel());                                                        \
}

// The alternative way to calculate a < b is (b - a).clamp(0).toBool()
// To prevent an overflow in subtraction (b - a) for unsigned types(unit, bool)
// we will convert in to int first
#define COMPARISON_TEST_ITER_FOR_TYPE(ctype,name)                                          \
TEST(TensorIteratorTest, ComparisonLoopBinary_##name) {                                    \
  auto in1 = random_tensor_for_type(k##name);                                              \
  auto in2 = random_tensor_for_type(k##name);                                              \
  Tensor out = at::empty({0}, in1.options().dtype(kBool));                                 \
  Tensor diff;                                                                             \
  if (k##name == kByte || k##name == kBool) {                                              \
    diff = in2.to(kInt).sub(in1.to(kInt));                                                 \
  } else {                                                                                 \
    diff = in2.sub(in1);                                                                   \
  }                                                                                        \
  auto expected = diff.clamp_min(0).to(kBool);                                             \
  auto iter = TensorIterator::comparison_op(out, in1, in2, true);                          \
  at::native::cpu_serial_kernel(iter, [=](ctype a, ctype b) -> bool { return a < b; });    \
  EXPECT_TRUE(out.equal(expected));                                                        \
}

AT_FORALL_SCALAR_TYPES(UNARY_TEST_ITER_FOR_TYPE)
AT_FORALL_SCALAR_TYPES(BINARY_TEST_ITER_FOR_TYPE)
AT_FORALL_SCALAR_TYPES(POINTWISE_TEST_ITER_FOR_TYPE)
AT_FORALL_SCALAR_TYPES(NO_OUTPUT_UNARY_TEST_ITER_FOR_TYPE)
AT_FORALL_SCALAR_TYPES(NO_OUTPUT_BINARY_TEST_ITER_FOR_TYPE)
AT_FORALL_SCALAR_TYPES(NO_OUTPUT_POINTWISE_TEST_ITER_FOR_TYPE)
AT_FORALL_SCALAR_TYPES_AND(Bool, COMPARISON_TEST_ITER_FOR_TYPE)

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

TEST(TensorIteratorTest, InputDType) {
  auto iter = at::TensorIterator();
  iter.add_output(at::ones({1, 1}, at::dtype(at::kBool)));
  iter.add_input(at::ones({1, 1}, at::dtype(at::kFloat)));
  iter.add_input(at::ones({1, 1}, at::dtype(at::kDouble)));
  iter.dont_compute_common_dtype();
  iter.build();
  EXPECT_TRUE(iter.input_dtype() == at::kFloat);
  EXPECT_TRUE(iter.input_dtype(0) == at::kFloat);
  EXPECT_TRUE(iter.input_dtype(1) == at::kDouble);
}

TEST(TensorIteratorTest, ComputeCommonDTypeInputOnly) {
  auto iter = at::TensorIterator();
  iter.add_output(at::ones({1, 1}, at::dtype(at::kBool)));
  iter.add_input(at::ones({1, 1}, at::dtype(at::kFloat)));
  iter.add_input(at::ones({1, 1}, at::dtype(at::kDouble)));
  iter.compute_common_dtype_only_for_inputs();
  iter.build();
  EXPECT_TRUE(iter.dtype(0) == at::kBool);
  EXPECT_TRUE(iter.dtype(1) == at::kDouble);
  EXPECT_TRUE(iter.dtype(2) == at::kDouble);
  EXPECT_TRUE(iter.common_dtype() == at::kDouble);
}

TEST(TensorIteratorTest, DoNotComputeCommonDTypeInputOnly) {
  auto iter = at::TensorIterator();
  iter.add_output(at::ones({1, 1}, at::dtype(at::kLong)));
  iter.add_input(at::ones({1, 1}, at::dtype(at::kFloat)));
  iter.add_input(at::ones({1, 1}, at::dtype(at::kDouble)));
  iter.compute_common_dtype_only_for_inputs();
  iter.dont_compute_common_dtype();
  iter.build();
  EXPECT_TRUE(iter.dtype(0) == at::kLong);
  EXPECT_TRUE(iter.dtype(1) == at::kFloat);
  EXPECT_TRUE(iter.dtype(2) == at::kDouble);
}

TEST(TensorIteratorTest, DoNotComputeCommonDTypeIfOutputIsUndefined) {
  Tensor out;
  auto iter = at::TensorIterator();
  iter.add_output(out);
  iter.add_input(at::ones({1, 1}, at::dtype(at::kDouble)));
  iter.add_input(at::ones({1, 1}, at::dtype(at::kFloat)));
  iter.compute_common_dtype_only_for_inputs();
  ASSERT_ANY_THROW(iter.build());
}

TEST(TensorIteratorTest, FailNonPromotingBinaryOp) {
  Tensor out;
  auto iter = at::TensorIterator();
  iter.add_output(out);
  iter.add_input(at::ones({1,1}, at::dtype(at::kDouble)));
  iter.add_input(at::ones({1,1}, at::dtype(at::kInt)));
  ASSERT_ANY_THROW(iter.build());
}
