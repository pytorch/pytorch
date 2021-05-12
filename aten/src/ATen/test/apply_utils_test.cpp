#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/test/test_assert.h>

#include <iostream>
using namespace std;
using namespace at;

void fill_tensor(int64_t scalar, Tensor& t_) {
  auto t = t_.view(-1);
  for (int64_t i = 0; i < t.numel(); i++) {
    t[i] = (i + 1) * scalar;
  }
}

// This test exercises all sequential applyX functions. Given a shape and two
// transpose dimensions we create 5 tensors (a0, ..., a4) of the given shape and
// transpose the dimension a with b for each tensor. Then we call the applyX
// function on each floating type. a4 is allocated in doubles only,  whereas a0,
// ..., a3 are allocated in the given type. For each applyX function we once
// write the same type as we read (using a0, ..., aX-1) and we once write to
// double (using a4 as a target). We also exercise on a zero_dim and empty
// tensor.
void test(DeprecatedTypeProperties& type, IntArrayRef shape, int64_t a = 0, int64_t b = 1) {
  auto zero_dim = at::empty({}, type);
  zero_dim.fill_(2);
  zero_dim.exp_();
  AT_DISPATCH_FLOATING_TYPES(zero_dim.scalar_type(), "test0", [&] {
    ASSERT(zero_dim.data_ptr<scalar_t>()[0] == std::exp(2));
  });

  auto empty_t = at::empty({0}, type);
  empty_t.fill_(3);
  empty_t.exp_();

  auto a0 = at::empty({0}, type.options());
  auto a1 = at::empty({0}, type.options());
  auto a2 = at::empty({0}, type.options());
  auto a3 = at::empty({0}, type.options());
  auto a4 = at::empty({0}, at::TensorOptions(kCPU).dtype(kDouble));

  std::vector<Tensor> tensors({a0, a1, a2, a3, a4});
  for (size_t i = 0; i < tensors.size(); i++) {
    tensors[i].resize_(shape);
    fill_tensor(i + 1, tensors[i]);
    if (a >= 0 && b >= 0) {
      tensors[i].transpose_(a, b);
    }
  }

  AT_DISPATCH_FLOATING_TYPES(a0.scalar_type(), "test1", [&] {
    CPU_tensor_apply2<scalar_t, scalar_t>(
        a0, a1, [](scalar_t& y, const scalar_t& x) { y = x * x; });
    CPU_tensor_apply2<double, scalar_t>(
        a4, a1, [](double& y, scalar_t x) { y = (double)(x * x); });
    for (int64_t i = 0; i < a0.numel(); i++) {
      auto target = a1.data_ptr<scalar_t>()[i] * a1.data_ptr<scalar_t>()[i];
      ASSERT(a0.data_ptr<scalar_t>()[i] == target);
      ASSERT(a4.data_ptr<double>()[i] == target);
    }
  });

  AT_DISPATCH_FLOATING_TYPES(a0.scalar_type(), "test2", [&] {
    CPU_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        a0, a1, a2, [](scalar_t& y, const scalar_t& x, const scalar_t& z) {
          y = x * x + z;
        });
    CPU_tensor_apply3<double, scalar_t, scalar_t>(
        a4, a1, a2, [](double& y, const scalar_t& x, const scalar_t& z) {
          y = (double)(x * x + z);
        });
    for (int64_t i = 0; i < a0.numel(); i++) {
      auto target = a1.data_ptr<scalar_t>()[i] * a1.data_ptr<scalar_t>()[i];
      target = target + a2.data_ptr<scalar_t>()[i];
      ASSERT(a0.data_ptr<scalar_t>()[i] == target);
      ASSERT(a4.data_ptr<double>()[i] == target);
    }
  });

  AT_DISPATCH_FLOATING_TYPES(a0.scalar_type(), "test3", [&] {
    CPU_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
        a0,
        a1,
        a2,
        a3,
        [](scalar_t& y,
           const scalar_t& x,
           const scalar_t& z,
           const scalar_t& a) { y = x * x + z * a; });
    CPU_tensor_apply4<double, scalar_t, scalar_t, scalar_t>(
        a4,
        a1,
        a2,
        a3,
        [](double& y, const scalar_t& x, const scalar_t& z, const scalar_t& a) {
          y = (double)(x * x + z * a);
        });
    for (int64_t i = 0; i < a0.numel(); i++) {
      auto target = a1.data_ptr<scalar_t>()[i] * a1.data_ptr<scalar_t>()[i];
      target = target + a2.data_ptr<scalar_t>()[i] * a3.data_ptr<scalar_t>()[i];
      ASSERT(a0.data_ptr<scalar_t>()[i] == target);
      ASSERT(a4.data_ptr<double>()[i] == target);
    }
  });
}

// apply utils test 2-dim small contiguous
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ApplyUtilsTest, Contiguous2D) {
  manual_seed(123);
  test(CPU(kDouble), {2, 1}, -1, -1);
}

// apply utils test 2-dim small
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ApplyUtilsTest, Small2D) {
  manual_seed(123);
  test(CPU(kDouble), {2, 1});
}

// apply utils test 2-dim
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ApplyUtilsTest, _2D) {
  manual_seed(123);
  test(CPU(kDouble), {20, 10});
}

// apply utils test 3-dim
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ApplyUtilsTest, _3D) {
  manual_seed(123);
  test(CPU(kDouble), {3, 4, 2});
}

// apply utils test 3-dim medium
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ApplyUtilsTest, Medium3D) {
  manual_seed(123);
  test(CPU(kDouble), {3, 40, 2});
}

// apply utils test 10-dim
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ApplyUtilsTest, _10D) {
  manual_seed(123);
  test(CPU(kDouble), {3, 4, 2, 5, 2, 1, 3, 4, 2, 3});
}
