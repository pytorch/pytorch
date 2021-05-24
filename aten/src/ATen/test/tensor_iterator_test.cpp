#include <gtest/gtest.h>
#include <thread>

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

using namespace at;

// An operation with a CUDA tensor and CPU scalar should keep the scalar
// on the CPU (and lift it to a parameter).
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// Verifies multiple zero-dim CPU inputs are not coerced to CUDA
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TensorIteratorTest, CPUScalarInputs) {
  if (!at::hasCUDA()) return;
  Tensor out = at::empty({5, 5}, kCUDA);
  auto x = at::ones(1, kCPU).squeeze();
  auto y = at::ones(1, kCPU).squeeze();
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(TensorIterator::binary_op(out, x, y));
}

// Mixing CPU and CUDA tensors should raise an exception (if the CPU tensor isn't zero-dim)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TensorIteratorTest, MixedDevices) {
  if (!at::hasCUDA()) return;
  Tensor out;
  auto x = at::randn({5, 5}, kCUDA);
  auto y = at::ones({5}, kCPU);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
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

#define UNARY_TEST_ITER_FOR_TYPE(ctype,name)                                    \
TEST(TensorIteratorTest, SerialLoopUnary_##name) {                              \
  Tensor out;                                                                   \
  auto in = random_tensor_for_type(k##name);                                    \
  auto expected = in.add(1);                                                    \
  auto iter = TensorIterator::unary_op(out, in);                                \
  at::native::cpu_serial_kernel(iter, [=](ctype a) -> ctype { return a + 1; }); \
  ASSERT_ANY_THROW(out.equal(expected));                                        \
}

#define NO_OUTPUT_UNARY_TEST_ITER_FOR_TYPE(ctype,name)                         \
TEST(TensorIteratorTest, SerialLoopUnaryNoOutput_##name) {                     \
  auto in = random_tensor_for_type(k##name);                                   \
  auto iter = at::TensorIteratorConfig()                                       \
      .add_owned_input(in)                                                           \
      .build();                                                                \
  int64_t acc = 0;                                                             \
  at::native::cpu_serial_kernel(iter, [&](ctype a) -> void { acc++; }); \
  EXPECT_TRUE(acc == in.numel());                                              \
}

#define BINARY_TEST_ITER_FOR_TYPE(ctype,name)                                            \
TEST(TensorIteratorTest, SerialLoopBinary_##name) {                                      \
  Tensor out;                                                                            \
  auto in1 = random_tensor_for_type(k##name);                                            \
  auto in2 = random_tensor_for_type(k##name);                                            \
  auto expected = in1.add(in2);                                                          \
  auto iter = TensorIterator::binary_op(out, in1, in2);                                  \
  at::native::cpu_serial_kernel(iter, [=](ctype a, ctype b) -> ctype { return a + b; }); \
  ASSERT_ANY_THROW(out.equal(expected));                                                 \
}

#define NO_OUTPUT_BINARY_TEST_ITER_FOR_TYPE(ctype,name)                          \
TEST(TensorIteratorTest, SerialLoopBinaryNoOutput_##name) {                      \
  auto in1 = random_tensor_for_type(k##name);                                    \
  auto in2 = random_tensor_for_type(k##name);                                    \
  auto iter = at::TensorIteratorConfig()                                         \
      .add_owned_input(in1)                                                            \
      .add_owned_input(in2)                                                            \
      .build();                                                                  \
  int64_t acc = 0;                                                               \
  at::native::cpu_serial_kernel(iter, [&](ctype a, ctype b) -> void { acc++; }); \
  EXPECT_TRUE(acc == in1.numel());                                               \
}

#define POINTWISE_TEST_ITER_FOR_TYPE(ctype,name)                                                      \
TEST(TensorIteratorTest, SerialLoopPointwise_##name) {                                                \
  Tensor out;                                                                                         \
  auto in1 = random_tensor_for_type(k##name);                                                         \
  auto in2 = random_tensor_for_type(k##name);                                                         \
  auto in3 = random_tensor_for_type(k##name);                                                         \
  auto expected = in1.add(in2).add(in3);                                                              \
  auto iter = at::TensorIteratorConfig()                                                              \
      .add_output(out)                                                                                \
      .add_owned_input(in1)                                                                                 \
      .add_owned_input(in2)                                                                                 \
      .add_owned_input(in3)                                                                                 \
      .build();                                                                                       \
  at::native::cpu_serial_kernel(iter, [=](ctype a, ctype b, ctype c) -> ctype { return a + b + c; }); \
  ASSERT_ANY_THROW(out.equal(expected));                                                              \
}

#define NO_OUTPUT_POINTWISE_TEST_ITER_FOR_TYPE(ctype,name)                                \
TEST(TensorIteratorTest, SerialLoopPoinwiseNoOutput_##name) {                             \
  auto in1 = random_tensor_for_type(k##name);                                             \
  auto in2 = random_tensor_for_type(k##name);                                             \
  auto in3 = random_tensor_for_type(k##name);                                             \
  auto iter = at::TensorIteratorConfig()                                                  \
      .add_owned_input(in1)                                                                     \
      .add_owned_input(in2)                                                                     \
      .add_owned_input(in3)                                                                     \
      .build();                                                                           \
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
  auto iter = TensorIterator::comparison_op(out, in1, in2);                                \
  at::native::cpu_serial_kernel(iter, [=](ctype a, ctype b) -> bool { return a < b; });    \
  EXPECT_TRUE(out.equal(expected));                                                        \
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
AT_FORALL_SCALAR_TYPES(UNARY_TEST_ITER_FOR_TYPE)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
AT_FORALL_SCALAR_TYPES(BINARY_TEST_ITER_FOR_TYPE)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
AT_FORALL_SCALAR_TYPES(POINTWISE_TEST_ITER_FOR_TYPE)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
AT_FORALL_SCALAR_TYPES(NO_OUTPUT_UNARY_TEST_ITER_FOR_TYPE)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
AT_FORALL_SCALAR_TYPES(NO_OUTPUT_BINARY_TEST_ITER_FOR_TYPE)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
AT_FORALL_SCALAR_TYPES(NO_OUTPUT_POINTWISE_TEST_ITER_FOR_TYPE)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
AT_FORALL_SCALAR_TYPES_AND(Bool, COMPARISON_TEST_ITER_FOR_TYPE)

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TensorIteratorTest, SerialLoopSingleThread) {
  std::thread::id thread_id = std::this_thread::get_id();
  Tensor out;
  auto x = at::zeros({50000}, at::TensorOptions(kCPU).dtype(kInt));
  auto iter = TensorIterator::unary_op(out, x);
  at::native::cpu_serial_kernel(iter, [=](int a) -> int {
    std::thread::id lambda_thread_id = std::this_thread::get_id();
    EXPECT_TRUE(lambda_thread_id == thread_id);
    return a + 1;
  });
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TensorIteratorTest, InputDType) {
  auto iter = at::TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_owned_output(at::ones({1, 1}, at::dtype(at::kBool)))
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kFloat)))
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kDouble)))
      .build();
  EXPECT_TRUE(iter.input_dtype() == at::kFloat);
  EXPECT_TRUE(iter.input_dtype(0) == at::kFloat);
  EXPECT_TRUE(iter.input_dtype(1) == at::kDouble);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TensorIteratorTest, ComputeCommonDTypeInputOnly) {
  auto iter = at::TensorIteratorConfig()
      .add_owned_output(at::ones({1, 1}, at::dtype(at::kBool)))
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kFloat)))
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kDouble)))
      .promote_inputs_to_common_dtype(true)
      .build();
  EXPECT_TRUE(iter.dtype(0) == at::kBool);
  EXPECT_TRUE(iter.dtype(1) == at::kDouble);
  EXPECT_TRUE(iter.dtype(2) == at::kDouble);
  EXPECT_TRUE(iter.common_dtype() == at::kDouble);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TensorIteratorTest, DoNotComputeCommonDTypeInputOnly) {
  auto iter = at::TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_owned_output(at::ones({1, 1}, at::dtype(at::kLong)))
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kFloat)))
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kDouble)))
      .build();
  EXPECT_TRUE(iter.dtype(0) == at::kLong);
  EXPECT_TRUE(iter.dtype(1) == at::kFloat);
  EXPECT_TRUE(iter.dtype(2) == at::kDouble);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TensorIteratorTest, FailNonPromotingBinaryOp) {
  Tensor out;
  at::TensorIteratorConfig config;
  config.add_output(out);
  config.add_owned_input(at::ones({1,1}, at::dtype(at::kDouble)));
  config.add_owned_input(at::ones({1,1}, at::dtype(at::kInt)));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(config.build());
}

#define MULTIPLE_OUTPUTS_TEST_ITER_FOR_TYPE(ctype,name)                                             \
TEST(TensorIteratorTest, CpuKernelMultipleOutputs_##name) {                                         \
  auto in1 = random_tensor_for_type(k##name);                                                       \
  auto in2 = random_tensor_for_type(k##name);                                                       \
  Tensor out1 = at::empty({0}, in1.options());                                                      \
  Tensor out2 = at::empty({0}, in1.options());                                                      \
  auto expected1 = in1.add(in2);                                                                    \
  auto expected2 = in1.mul(in2);                                                                    \
  auto iter = at::TensorIteratorConfig()                                                            \
    .add_output(out1)                                                                               \
    .add_output(out2)                                                                               \
    .add_owned_input(in1)                                                                                 \
    .add_owned_input(in2)                                                                                 \
    .build();                                                                                       \
  at::native::cpu_kernel_multiple_outputs(iter, [=](ctype a, ctype b) -> std::tuple<ctype, ctype> { \
    ctype add = a + b;                                                                              \
    ctype mul = a * b;                                                                              \
    return std::tuple<ctype, ctype>(add, mul);                                                      \
  });                                                                                               \
  EXPECT_TRUE(out1.equal(expected1));                                                               \
  EXPECT_TRUE(out2.equal(expected2));                                                               \
}
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
AT_FORALL_SCALAR_TYPES(MULTIPLE_OUTPUTS_TEST_ITER_FOR_TYPE)
