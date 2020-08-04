#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <iostream>
using namespace std;
using namespace at;

class atest : public ::testing::Test {
 protected:
  void SetUp() override {
    x_tensor = tensor({10, -1, 0, 1, -10});
    y_tensor = tensor({-10, 1, 0, -1, 10});
    x_logical = tensor({1, 1, 0, 1, 0});
    y_logical = tensor({0, 1, 0, 1, 1});
    x_float = tensor({2.0, 2.4, 5.6, 7.0, 36.0});
    y_float = tensor({1.0, 1.1, 8.7, 10.0, 24.0});
  }

  Tensor x_tensor, y_tensor;
  Tensor x_logical, y_logical;
  Tensor x_float, y_float;
  const int INT = 1;
  const int FLOAT = 2;
  const int INTFLOAT = 3;
  const int INTBOOL = 5;
  const int INTBOOLFLOAT = 7;
};

namespace BinaryOpsKernel {
const int IntMask = 1; // test dtype = kInt
const int FloatMask = 2; // test dtype = kFloat
const int BoolMask = 4; // test dtype = kBool
} // namespace BinaryOpsKernel

template <typename T, typename... Args>
void unit_binary_ops_test(
    T func,
    const Tensor& x_tensor,
    const Tensor& y_tensor,
    const Tensor& exp,
    ScalarType dtype,
    Args... args) {
  auto out_tensor = empty({5}, dtype);
  func(out_tensor, x_tensor.to(dtype), y_tensor.to(dtype), args...);
  ASSERT_EQ(out_tensor.dtype(), dtype);
  if (dtype == kFloat) {
    ASSERT_TRUE(exp.to(dtype).allclose(out_tensor));
  } else {
    ASSERT_TRUE(exp.to(dtype).equal(out_tensor));
  }
}

/*
  template function for running binary operator test
  - exp: expected output
  - func: function to be tested
  - option: 3 bits,
    - 1st bit: Test op over integer tensors
    - 2nd bit: Test op over float tensors
    - 3rd bit: Test op over boolean tensors
    For example, if function should be tested over integer/boolean but not for
    float, option will be 1 * 1 + 0 * 2 + 1 * 4 = 5. If tested over all the
    type, option should be 7.
*/
template <typename T, typename... Args>
void run_binary_ops_test(
    T func,
    const Tensor& x_tensor,
    const Tensor& y_tensor,
    const Tensor& exp,
    int option,
    Args... args) {
  // Test op over integer tensors
  if (option & BinaryOpsKernel::IntMask) {
    unit_binary_ops_test(func, x_tensor, y_tensor, exp, kInt, args...);
  }

  // Test op over float tensors
  if (option & BinaryOpsKernel::FloatMask) {
    unit_binary_ops_test(func, x_tensor, y_tensor, exp, kFloat, args...);
  }

  // Test op over boolean tensors
  if (option & BinaryOpsKernel::BoolMask) {
    unit_binary_ops_test(func, x_tensor, y_tensor, exp, kBool, args...);
  }
}

void trace() {
  Tensor foo = rand({12, 12});

  // ASSERT foo is 2-dimensional and holds floats.
  auto foo_a = foo.accessor<float, 2>();
  float trace = 0;

  for (int i = 0; i < foo_a.size(0); i++) {
    trace += foo_a[i][i];
  }

  ASSERT_FLOAT_EQ(foo.trace().item<float>(), trace);
}

TEST_F(atest, operators) {
  int a = 0b10101011;
  int b = 0b01111011;

  auto a_tensor = tensor({a});
  auto b_tensor = tensor({b});

  ASSERT_TRUE(tensor({~a}).equal(~a_tensor));
  ASSERT_TRUE(tensor({a | b}).equal(a_tensor | b_tensor));
  ASSERT_TRUE(tensor({a & b}).equal(a_tensor & b_tensor));
  ASSERT_TRUE(tensor({a ^ b}).equal(a_tensor ^ b_tensor));
}

TEST_F(atest, logical_and_operators) {
  auto exp_tensor = tensor({0, 1, 0, 1, 0});
  run_binary_ops_test(
      logical_and_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, logical_or_operators) {
  auto exp_tensor = tensor({1, 1, 0, 1, 1});
  run_binary_ops_test(
      logical_or_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, logical_xor_operators) {
  auto exp_tensor = tensor({1, 0, 0, 0, 1});
  run_binary_ops_test(
      logical_xor_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, lt_operators) {
  auto exp_tensor = tensor({0, 0, 0, 0, 1});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      lt_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, le_operators) {
  auto exp_tensor = tensor({0, 1, 1, 1, 1});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      le_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, gt_operators) {
  auto exp_tensor = tensor({1, 0, 0, 0, 0});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      gt_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, ge_operators) {
  auto exp_tensor = tensor({1, 1, 1, 1, 0});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      ge_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, eq_operators) {
  auto exp_tensor = tensor({0, 1, 1, 1, 0});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      eq_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, ne_operators) {
  auto exp_tensor = tensor({1, 0, 0, 0, 1});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      ne_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

TEST_F(atest, add_operators) {
  auto exp_tensor = tensor({-10, 1, 0, -1, 10});
  run_binary_ops_test(add_out, x_tensor, y_tensor, exp_tensor, INTBOOL, 2);
}

TEST_F(atest, max_operators) {
  auto exp_tensor = tensor({10, 1, 0, 1, 10});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      max_out, x_tensor, y_tensor, exp_tensor, INTBOOLFLOAT);
}

TEST_F(atest, min_operators) {
  auto exp_tensor = tensor({-10, -1, 0, -1, -10});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      min_out, x_tensor, y_tensor, exp_tensor, INTBOOLFLOAT);
}

TEST_F(atest, sigmoid_backward_operator) {
  auto exp_tensor = tensor({-1100, 0, 0, -2, 900});
  // only test with type Float
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      sigmoid_backward_out, x_tensor, y_tensor, exp_tensor, FLOAT);
}

TEST_F(atest, fmod_tensor_operators) {
  auto exp_tensor = tensor({0.0, 0.2, 5.6, 7.0, 12.0});
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      fmod_out, x_float, y_float, exp_tensor, INTFLOAT);
}

// TEST_CASE( "atest", "[]" ) {
TEST_F(atest, atest) {
  manual_seed(123);

  auto foo = rand({12, 6});

  ASSERT_EQ(foo.size(0), 12);
  ASSERT_EQ(foo.size(1), 6);

  foo = foo + foo * 3;
  foo -= 4;

  Scalar a = 4;
  float b = a.to<float>();
  ASSERT_EQ(b, 4);

  foo = ((foo * foo) == (foo.pow(3))).to(kByte);
  foo = 2 + (foo + 1);
  // foo = foo[3];
  auto foo_v = foo.accessor<uint8_t, 2>();

  for (int i = 0; i < foo_v.size(0); i++) {
    for (int j = 0; j < foo_v.size(1); j++) {
      foo_v[i][j]++;
    }
  }

  ASSERT_TRUE(foo.equal(4 * ones({12, 6}, kByte)));

  trace();

  float data[] = {1, 2, 3, 4, 5, 6};

  auto f = from_blob(data, {1, 2, 3});
  auto f_a = f.accessor<float, 3>();

  ASSERT_EQ(f_a[0][0][0], 1.0);
  ASSERT_EQ(f_a[0][1][1], 5.0);

  ASSERT_EQ(f.strides()[0], 6);
  ASSERT_EQ(f.strides()[1], 3);
  ASSERT_EQ(f.strides()[2], 1);
  ASSERT_EQ(f.sizes()[0], 1);
  ASSERT_EQ(f.sizes()[1], 2);
  ASSERT_EQ(f.sizes()[2], 3);

  // TODO(ezyang): maybe do a more precise exception type.
  ASSERT_THROW(f.resize_({3, 4, 5}), std::exception);
  {
    int isgone = 0;
    {
      auto f2 = from_blob(data, {1, 2, 3}, [&](void*) { isgone++; });
    }
    ASSERT_EQ(isgone, 1);
  }
  {
    int isgone = 0;
    Tensor a_view;
    {
      auto f2 = from_blob(data, {1, 2, 3}, [&](void*) { isgone++; });
      a_view = f2.view({3, 2, 1});
    }
    ASSERT_EQ(isgone, 0);
    a_view.reset();
    ASSERT_EQ(isgone, 1);
  }

  if (at::hasCUDA()) {
    int isgone = 0;
    {
      auto base = at::empty({1, 2, 3}, TensorOptions(kCUDA));
      auto f2 = from_blob(base.data_ptr(), {1, 2, 3}, [&](void*) { isgone++; });
    }
    ASSERT_EQ(isgone, 1);

    // Attempt to specify the wrong device in from_blob
    auto t = at::empty({1, 2, 3}, TensorOptions(kCUDA, 0));
    EXPECT_ANY_THROW(from_blob(t.data_ptr(), {1, 2, 3}, at::Device(kCUDA, 1)));

    // Infers the correct device
    auto t_ = from_blob(t.data_ptr(), {1, 2, 3}, kCUDA);
    ASSERT_EQ(t_.device(), at::Device(kCUDA, 0));
  }
}
