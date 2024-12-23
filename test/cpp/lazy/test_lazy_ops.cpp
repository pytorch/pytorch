#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <gtest/gtest.h>
#include <test/cpp/lazy/test_lazy_ops_util.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/torch.h>

namespace torch {
namespace lazy {

// Lazy Tensor is disabled in FBCODE until addressing non-virtual methods (e.g.
// sizes) in TensorImpl
#ifndef FBCODE_CAFFE2

namespace {
// This registers the torchscript backend, without which lazy device won't work.
// FIXME: This registers the backend for the whole test binary. We should
// probably do it and undo it in the test fixture below.
static bool inline init_backend() {
  torch::lazy::InitTorchScriptBackend();
  return true;
}
static const bool backend_initialized = init_backend();

} // namespace

class LazyTsTest : public ::testing::Test {
 protected:
  void SetUp() override;

  void TearDown() override;

  static void CommonSetup() {}

  void ExpectCounterNotChanged(
      const std::string& counter_regex,
      const std::unordered_set<std::string>* ignore_set) {}

  void ExpectCounterChanged(
      const std::string& counter_regex,
      const std::unordered_set<std::string>* ignore_set) {}

  void ResetCounters() {}

 private:
  void MakeEndSnapshot() {}
};

class LazyOpsTestBase : public LazyTsTest {
 protected:
  static void SetUpTestCase() {}
};

void LazyTsTest::SetUp() {
  (void)backend_initialized; // avoid unused parameter warning
  at::manual_seed(42);
  torch::lazy::LazyGraphExecutor::Get()->SetRngSeed(
      torch::lazy::BackendDevice(), 42);
}

void LazyTsTest::TearDown() {}

namespace {
using torch::lazy::DebugUtil;

class LazyOpsTest : public LazyOpsTestBase {};

static inline bool IsCuda() {
  return torch::lazy::getBackend()->EagerFallbackDeviceType() == at::kCUDA;
}

static inline at::DeviceType DefaultDevice() {
  return torch::lazy::getBackend()->EagerFallbackDeviceType();
}

} // namespace

TEST_F(LazyOpsTest, TestScalarTensor) {
  torch::Tensor scalar_tensor = torch::scalar_tensor(
      1., torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_scalar_tensor = torch::scalar_tensor(
        1., torch::TensorOptions(torch::kFloat).device(torch::kLazy));
    AllClose(scalar_tensor, lazy_scalar_tensor);
  });
}

TEST_F(LazyOpsTest, TestClone) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = lazy_a.clone();
    AllClose(a, lazy_b);
    lazy_a.add_(1.0);
    AllClose(a, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestTo) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestIsFloatingPoint) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    bool is_float = torch::is_floating_point(a);
    bool lazy_is_float = torch::is_floating_point(lazy_a);
    EXPECT_EQ(is_float, lazy_is_float);
  });
}

TEST_F(LazyOpsTest, TestIsSigned) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    bool is_signed = torch::is_signed(a);
    bool lazy_is_signed = torch::is_signed(lazy_a);
    EXPECT_EQ(is_signed, lazy_is_signed);
  });
}

TEST_F(LazyOpsTest, TestCastByte) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Byte(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Byte(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastChar) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Char(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Char(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastShort) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Short(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Short(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastInt) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Int(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Int(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastLong) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Long(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Long(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCastFloat) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::_cast_Float(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::_cast_Float(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestRetainType) {
  torch::Tensor lazy_a = torch::zeros(
      {2, 2}, torch::TensorOptions(torch::kByte).device(torch::kLazy));
  torch::Tensor lazy_b = torch::ones(
      {2, 2}, torch::TensorOptions(torch::kByte).device(torch::kLazy));
  torch::Tensor lazy_c = lazy_a + lazy_b;
  EXPECT_EQ(lazy_c.scalar_type(), torch::ScalarType::Byte);
}

TEST_F(LazyOpsTest, TestLogicalTypeWithInterop) {
  torch::Tensor query = torch::rand(
      {2, 12, 20, 64},
      torch::TensorOptions(torch::kFloat).device(torch::kLazy));
  torch::Tensor key = torch::rand(
      {2, 12, 64, 20},
      torch::TensorOptions(torch::kFloat).device(torch::kLazy));
  torch::Tensor scores =
      torch::matmul(query, key) /
      torch::scalar_tensor(
          8, torch::TensorOptions(torch::kDouble).device(torch::kLazy));
  torch::Tensor p_attn = torch::softmax(scores, /*dim=*/-1);
  EXPECT_EQ(p_attn.scalar_type(), torch::ScalarType::Float);
}

TEST_F(LazyOpsTest, TestAdd) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddHalf) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kHalf).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kHalf).device(DefaultDevice()));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddMixedPrecision) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kHalf).device(DefaultDevice()));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor c = a.add_(b);
    torch::Tensor lazy_c = lazy_a.add_(lazy_b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddScalar) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar b(1);
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_c = torch::add(lazy_a, b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddScalarInPlace) {
  torch::Scalar b(1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor c = a.add_(b);
    torch::Tensor lazy_c = lazy_a.add_(b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestAddZeroSizeDim) {
  torch::Tensor a = torch::rand(
      {0, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {1, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::add(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSub) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::sub(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::sub(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSubInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor c = a.sub_(b);
    torch::Tensor lazy_c = lazy_a.sub_(lazy_b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSubScalar) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar b(1);
  torch::Tensor c = torch::sub(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_c = torch::sub(lazy_a, b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestSubScalarInPlace) {
  torch::Scalar b(1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor c = a.sub_(b);
    torch::Tensor lazy_c = lazy_a.sub_(b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMul) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::mul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::mul(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor c = a.mul_(b);
    torch::Tensor lazy_c = lazy_a.mul_(lazy_b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulScalar) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar b(3);
  torch::Tensor c = torch::mul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_c = torch::mul(lazy_a, b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMulScalarInPlace) {
  torch::Scalar b(3);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor c = a.mul_(b);
    torch::Tensor lazy_c = lazy_a.mul_(b);
    AllClose(a, lazy_a);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestDiv) {
  for (torch::ScalarType scalar_type1 :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor a = isFloatingType(scalar_type1)
        ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
        : torch::randint(0, 100, {3, 4}, torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 :
         {torch::kFloat,
          torch::kByte,
          torch::kChar,
          torch::kShort,
          torch::kInt,
          torch::kLong}) {
      torch::Tensor b = isFloatingType(scalar_type2)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
          : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type2));
      torch::Tensor c = torch::div(a, b);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        torch::Tensor lazy_b = CopyToDevice(b, device);
        torch::Tensor lazy_c = torch::div(lazy_a, lazy_b);
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivWithRoundingMode) {
  std::optional<std::string_view> rounding_modes[] = {
      "trunc", "floor", std::nullopt};
  for (const auto& rounding_mode : rounding_modes) {
    for (torch::ScalarType scalar_type1 :
         {torch::kFloat,
          torch::kByte,
          torch::kChar,
          torch::kShort,
          torch::kInt,
          torch::kLong}) {
      int lower_bound = (scalar_type1 == torch::kByte) ? 0 : -100;
      torch::Tensor a = isFloatingType(scalar_type1)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
          : torch::randint(
                lower_bound, 50, {3, 4}, torch::TensorOptions(scalar_type1));
      for (torch::ScalarType scalar_type2 :
           {torch::kFloat,
            torch::kByte,
            torch::kChar,
            torch::kShort,
            torch::kInt,
            torch::kLong}) {
        torch::Tensor b = isFloatingType(scalar_type2)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
            : torch::randint(
                  51, 100, {3, 4}, torch::TensorOptions(scalar_type2));
        torch::Tensor c = torch::div(a, b, rounding_mode);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = CopyToDevice(b, device);
          torch::Tensor lazy_c = torch::div(lazy_a, lazy_b, rounding_mode);
          AllClose(c, lazy_c);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestDivInPlace) {
  for (torch::ScalarType scalar_type1 : {torch::kFloat}) {
    torch::Tensor a = isFloatingType(scalar_type1)
        ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
        : torch::randint(0, 100, {3, 4}, torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 : {torch::kFloat}) {
      torch::Tensor b = isFloatingType(scalar_type2)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
          : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type2));
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        torch::Tensor c = a.div_(b);
        torch::Tensor lazy_b = CopyToDevice(b, device);
        torch::Tensor lazy_c = lazy_a.div_(lazy_b);
        ;
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivInPlaceWithRoundingMode) {
  std::optional<std::string_view> rounding_modes[] = {
      "trunc", "floor", std::nullopt};
  for (const auto& rounding_mode : rounding_modes) {
    for (torch::ScalarType scalar_type1 : {torch::kFloat}) {
      torch::Tensor a = isFloatingType(scalar_type1)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
          : torch::randint(
                -100, 100, {3, 4}, torch::TensorOptions(scalar_type1));
      for (torch::ScalarType scalar_type2 : {torch::kFloat}) {
        torch::Tensor b = isFloatingType(scalar_type2)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
            : torch::randint(
                  1, 100, {3, 4}, torch::TensorOptions(scalar_type2));
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor c = a.div_(b, rounding_mode);
          torch::Tensor lazy_b = CopyToDevice(b, device);
          torch::Tensor lazy_c = lazy_a.div_(lazy_b, rounding_mode);
          AllClose(c, lazy_c);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestDivScalar) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              1,
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool is_float : {true, false}) {
      torch::Scalar b = is_float ? torch::Scalar(3.0) : torch::Scalar(3);
      torch::Tensor c = torch::div(a, b);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        torch::Tensor lazy_c = torch::div(lazy_a, b);
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivScalarInPlace) {
  for (torch::ScalarType scalar_type : {torch::kFloat}) {
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              1,
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool is_float : {true, false}) {
      torch::Scalar b = is_float ? torch::Scalar(3.0) : torch::Scalar(3);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        torch::Tensor c = a.div_(b);
        torch::Tensor lazy_c = lazy_a.div_(b);
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDivOut) {
  for (torch::ScalarType scalar_type : {torch::kFloat, torch::kDouble}) {
    torch::Tensor a = torch::rand(
        {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor b = torch::rand(
        {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor c = torch::empty(
        {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::div_out(c, a, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = torch::empty({3, 4}, lazy_b.options());
      torch::div_out(lazy_c, lazy_a, lazy_b);
      AllClose(c, lazy_c);
    });
  }
}

TEST_F(LazyOpsTest, TestRsubScalar) {
  torch::Tensor input = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar other(1.5);
  torch::Scalar alpha(2.5);
  torch::Tensor result = torch::rsub(input, other, alpha);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::rsub(lazy_input, other, alpha);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestNe) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::ne(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::ne(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestNeInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor a_copy = a.clone();
  torch::Tensor b = a.clone();
  b[0] += 1;
  a.ne_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.ne_(lazy_b);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestEq) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::eq(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::eq(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEqInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  b[0] += 1;
  torch::Tensor a_copy = a.clone();
  a.eq_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.eq_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestGe) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::ge(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::ge(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestGeInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.ge_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.ge_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestLe) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::le(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::le(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestLeInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.le_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.le_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestGt) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::add(a.clone(), torch::ones_like(a));
  torch::Tensor c = torch::gt(b, a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::gt(lazy_b, lazy_a);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestGtInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.gt_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.gt_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestLt) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::add(a.clone(), torch::ones_like(a));
  torch::Tensor c = torch::lt(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::lt(lazy_a, lazy_b);
    AllEqual(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestLtInplace) {
  torch::Tensor a = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.lt_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a_copy, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    lazy_a.lt_(lazy_b);
    AllClose(lazy_a, a);
  });
}

TEST_F(LazyOpsTest, TestNeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(0));
  torch::Tensor result = torch::ne(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::ne(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestEqScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::eq(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::eq(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestGeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::ge(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::ge(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestGeScalarInplace) {
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.ge_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.ge_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestLeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::le(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::le(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLeScalarInplace) {
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.le_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.le_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestGtScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(0.5));
  torch::Tensor result = torch::gt(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::gt(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestGtScalarInplace) {
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.gt_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.gt_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestLtScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1.5));
  torch::Tensor result = torch::lt(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::lt(lazy_input, other);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLtScalarInplace) {
  torch::Tensor input = torch::arange(
      -1.,
      1.5,
      0.5,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.lt_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    lazy_input.lt_(other);
    AllClose(lazy_input, input);
  });
}

TEST_F(LazyOpsTest, TestIntegerAdd) {
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  ForEachDevice([&](const torch::Device& device) {
    for (auto type : types) {
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Tensor b =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Scalar one =
          isIntegralType(type, false) ? torch::Scalar(1) : torch::Scalar(1.0);
      torch::Tensor c = torch::add(b, one);

      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = torch::add(lazy_b, one);

      AllEqual(c, lazy_c);
    }
  });
}

TEST_F(LazyOpsTest, TestSVD) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      torch::Tensor a = torch::rand(
          {m, n}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      auto b = torch::svd(a, /*some=*/true, /*compute_uv=*/true);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        auto lazy_b = torch::svd(lazy_a, /*some=*/true, /*compute_uv=*/true);
        // The U and V matrices might have different sign for column vectors, so
        // cannot be compared if not by absolute value.
        AllClose(
            std::get<0>(b).abs(),
            std::get<0>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        torch::Tensor diag = std::get<1>(b);
        torch::Tensor lazy_diag = std::get<1>(lazy_b);
        ASSERT_EQ(diag.sizes(), lazy_diag.sizes());
        AllClose(
            diag,
            lazy_diag,
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        AllClose(
            std::get<2>(b).abs(),
            std::get<2>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestQR) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      torch::Tensor a = torch::rand(
          {m, n}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      auto b = torch::qr(a);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        auto lazy_b = torch::qr(lazy_a);
        AllClose(
            std::get<0>(b).abs(),
            std::get<0>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
        AllClose(
            std::get<1>(b).abs(),
            std::get<1>(lazy_b).abs(),
            /*rtol=*/1e-3,
            /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestCholesky) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (bool upper : {true, false}) {
      torch::Tensor a = torch::rand(
          {3, m, m},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor pd_a =
          torch::matmul(a, torch::transpose(a, 1, 2)) +
          torch::eye(
              m, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      auto b = torch::cholesky(pd_a, upper);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(pd_a, device);
        auto lazy_b = torch::cholesky(lazy_a, upper);
        AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestLogDet) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    torch::Tensor a = torch::rand(
        {3, m, m}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor pd_a = torch::matmul(a, torch::transpose(a, 1, 2)) +
        torch::eye(m,
                   torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor b = torch::logdet(pd_a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(pd_a, device);
      torch::Tensor lazy_b = torch::logdet(lazy_a);
      AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(LazyOpsTest, TestTriangularSolve) {
  static const int dims[] = {4, 7};
  for (bool batched_a : {true, false}) {
    for (bool batched_b : {true, false}) {
      for (auto m : dims) {
        for (auto n : dims) {
          for (bool upper : {true, false}) {
            for (bool transpose : {true, false}) {
              for (bool unitriangular : {true, false}) {
                torch::Tensor a = torch::randn(
                    {m, m},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice()));
                torch::Tensor b = torch::randn(
                    {m, n},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice()));
                a = batched_a ? a.expand({3, m, m}).clone() : a;
                b = batched_b ? b.expand({3, m, n}).clone() : b;
                auto result = torch::triangular_solve(
                    b,
                    a,
                    /*upper=*/upper,
                    /*transpose=*/transpose,
                    /*unitriangular=*/unitriangular);
                ForEachDevice([&](const torch::Device& device) {
                  torch::Tensor lazy_a = CopyToDevice(a, device);
                  torch::Tensor lazy_b = CopyToDevice(b, device);
                  auto lazy_result = torch::triangular_solve(
                      lazy_b,
                      lazy_a,
                      /*upper=*/upper,
                      /*transpose=*/transpose,
                      /*unitriangular=*/unitriangular);
                  AllClose(
                      std::get<0>(result),
                      std::get<0>(lazy_result),
                      /*rtol=*/1e-3,
                      /*atol=*/1e-4);
                  AllClose(
                      std::get<1>(result),
                      std::get<1>(lazy_result),
                      /*rtol=*/1e-3,
                      /*atol=*/1e-4);
                });
              }
            }
          }
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestKthValue) {
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool keepdim : {false, true}) {
        auto b = torch::kthvalue(a, k, dim, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = torch::kthvalue(lazy_a, k, dim, keepdim);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestTopK) {
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool largest : {false, true}) {
        auto b = torch::topk(a, k, dim, largest, /*sorted=*/true);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = torch::topk(lazy_a, k, dim, largest, /*sorted=*/true);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSort) {
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        auto b = torch::sort(a, dim, descending);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = torch::sort(lazy_a, dim, descending);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllEqual(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSortDescWithMinValue) {
  std::vector<int8_t> values{-128, 100};
  torch::Tensor input =
      torch::tensor(values, torch::TensorOptions(torch::kChar));
  auto output = torch::sort(input, /*dim=*/0, /*descending=*/true);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    auto lazy_output = torch::sort(lazy_input, /*dim=*/0, /*descending=*/true);
    AllEqual(std::get<0>(output), std::get<0>(lazy_output));
    AllEqual(std::get<1>(output), std::get<1>(lazy_output));
  });
}

TEST_F(LazyOpsTest, TestArgSort) {
  torch::Tensor a = torch::rand(
      {4, 5, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        torch::Tensor b = torch::argsort(a, dim, descending);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::argsort(lazy_a, dim, descending);
          AllEqual(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMin) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::min(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::min(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMax) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::max(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::max(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestUnaryMin) {
  torch::Tensor input = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::min(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::min(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestUnaryMax) {
  torch::Tensor input = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::max(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::max(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestAll) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor b = torch::all(a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::all(lazy_a);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAllDim) {
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::all(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::all(lazy_a, dim, /*keepdim=*/false);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAllDimKeep) {
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::all(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::all(lazy_a, dim, /*keepdim=*/true);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAmax) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (bool keepdim : {false, true}) {
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor values = torch::amax(input, {dim}, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_values =
            torch::amax(lazy_input, {dim}, /*keepdim=*/keepdim);
        AllClose(values, lazy_values);
      });
    }
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        torch::Tensor values =
            torch::amax(input, {dim1, dim2}, /*keepdim=*/keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_input = CopyToDevice(input, device);
          torch::Tensor lazy_values =
              torch::amax(lazy_input, {dim1, dim2}, /*keepdim=*/keepdim);
          AllClose(values, lazy_values);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("xla::amax", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestAmin) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (bool keepdim : {false, true}) {
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor values = torch::amin(input, {dim}, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_values =
            torch::amin(lazy_input, {dim}, /*keepdim=*/keepdim);
        AllClose(values, lazy_values);
      });
    }
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        torch::Tensor values =
            torch::amin(input, {dim1, dim2}, /*keepdim=*/keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_input = CopyToDevice(input, device);
          torch::Tensor lazy_values =
              torch::amin(lazy_input, {dim1, dim2}, /*keepdim=*/keepdim);
          AllClose(values, lazy_values);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("xla::amin", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestAny) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor b = torch::any(a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::any(lazy_a);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAnyDim) {
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::any(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::any(lazy_a, dim, /*keepdim=*/false);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAnyDimKeep) {
  torch::Tensor a = torch::randint(
      0,
      5,
      {2, 3, 4},
      torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::any(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::any(lazy_a, dim, /*keepdim=*/true);
      EqualValues(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMean) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::mean(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::mean(lazy_a);
    ASSERT_EQ(b.sizes(), lazy_b.sizes());
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestMeanCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::mean(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::mean(lazy_a, torch::kDouble);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestMeanInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::mean(a, {dim});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::mean(lazy_a, {dim});
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMeanInDims) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::mean(a, dims);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::mean(lazy_a, dims);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMeanInDimsKeepCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::mean(a, dims, true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::mean(lazy_a, dims, true, torch::kDouble);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestMeanInDimOut) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::empty(
        {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::mean_out(b, a, {dim});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::empty({4, 4}, lazy_a.options());
      torch::mean_out(lazy_b, lazy_a, {dim});
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestStd) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto unbiased : {true, false}) {
    torch::Tensor b = torch::std(a, unbiased);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::std(lazy_a, unbiased);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestStdInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (auto unbiased : {true, false}) {
    for (auto keepdim : {true, false}) {
      for (int dim = -rank; dim < rank; ++dim) {
        torch::Tensor b = torch::std(a, {dim}, unbiased, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::std(lazy_a, {dim}, unbiased, keepdim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestStdWithCorrection) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // int rank = a.dim();
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& correction : corrections) {
    for (auto keepdim : {true, false}) {
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        torch::Tensor b = torch::std(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::std(lazy_a, dim, correction, keepdim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestStdMeanWithCorrection) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // int rank = a.dim();
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& correction : corrections) {
    for (auto keepdim : {true, false}) {
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        auto b = torch::std_mean(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = torch::std_mean(lazy_a, dim, correction, keepdim);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllClose(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSum) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::sum(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sum(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::sum(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sum(lazy_a, torch::kDouble);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumU8) {
  torch::Tensor a = torch::ones(
      {256}, torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  torch::Tensor b = torch::sum(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sum(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSumInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::sum(a, {dim});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::sum(lazy_a, {dim});
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestSumInDims) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::sum(lazy_a, dims);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestSumInDimsKeep) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::sum(lazy_a, dims, /*keepdim=*/true);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestSumInDimsKeepCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims, /*keepdim=*/true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b =
          torch::sum(lazy_a, dims, /*keepdim=*/true, torch::kDouble);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestVar) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (bool unbiased : {true, false}) {
    torch::Tensor b = torch::var(a, unbiased);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::var(lazy_a, unbiased);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestVarWithDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepDim : {true, false}) {
      for (bool unbiased : {true, false}) {
        torch::Tensor b = torch::var(a, dims, unbiased, keepDim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::var(lazy_a, dims, unbiased, keepDim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestVarWithCorrection) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepDim : {true, false}) {
      for (const auto& correction : corrections) {
        torch::Tensor b = torch::var(a, dim, correction, keepDim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = torch::var(lazy_a, dim, correction, keepDim);
          AllClose(b, lazy_b);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::var", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestVarMeanWithCorrection) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::optional<c10::Scalar> corrections[] = {1, 2, std::nullopt};
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (const auto& correction : corrections) {
      for (auto keepdim : {true, false}) {
        auto b = torch::var_mean(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          auto lazy_b = torch::var_mean(lazy_a, dim, correction, keepdim);
          AllClose(std::get<0>(b), std::get<0>(lazy_b));
          AllClose(std::get<1>(b), std::get<1>(lazy_b));
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxInDim) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = torch::max(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        auto lazy_values_indices =
            torch::max(lazy_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(lazy_values_indices));
        AllEqual(std::get<1>(values_indices), std::get<1>(lazy_values_indices));
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMinInDim) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = torch::min(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        auto lazy_values_indices =
            torch::min(lazy_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(lazy_values_indices));
        AllEqual(std::get<1>(values_indices), std::get<1>(lazy_values_indices));
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNorm) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::norm(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::norm(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestNormInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::norm(a, 2, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::norm(lazy_a, 2, {dim}, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestNormInDims) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::norm(a, 2, dims, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::norm(lazy_a, 2, dims, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestNormInDimsKeep) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::norm(a, 2, dims, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::norm(lazy_a, 2, dims, /*keepdim=*/true);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestNormalTwoTensor) {
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    at::Tensor lazy_mean = CopyToDevice(mean, device);
    at::Tensor lazy_std = CopyToDevice(std, device);
    at::Tensor lazy_normal = at::normal(lazy_mean, lazy_std);
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestNormalDoubleMean) {
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    at::Tensor lazy_std = CopyToDevice(std, device);
    at::Tensor lazy_normal = at::normal(0, lazy_std);
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestNormalDoubleStd) {
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    at::Tensor lazy_mean = CopyToDevice(mean, device);
    at::Tensor lazy_normal = at::normal(lazy_mean, 1);
    double res_mean = lazy_normal.mean().item().toDouble();
    double res_std = lazy_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestNormalInPlace) {
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    at::Tensor lazy_a = CopyToDevice(a, device);
    lazy_a.normal_(/*mean=*/0, /*std=*/1);
    double res_mean = lazy_a.mean().item().toDouble();
    double res_std = lazy_a.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(LazyOpsTest, TestUniformInPlace) {
  const double eps = 1e-3;
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    at::Tensor lazy_a = CopyToDevice(a, device);
    lazy_a.uniform_(/*from=*/0, /*to=*/1);
    at::Tensor cpu_a = ToCpuTensor(lazy_a);
    double res_min = cpu_a.min().item().toDouble();
    double res_max = cpu_a.max().item().toDouble();
    EXPECT_GT(res_min, 0.0 - eps);
    EXPECT_LT(res_max, 1.0 + eps);
  });
}

TEST_F(LazyOpsTest, TestRandomInPlace) {
  for (auto dtype :
       {torch::kFloat,
        torch::kDouble,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    const double eps = 0.2;
    torch::Tensor a = torch::zeros({10, 10, 10}, torch::TensorOptions(dtype));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      lazy_a.random_(/*from=*/0, /*to=*/10);
      double res_mean = lazy_a.sum().item().toDouble() / a.numel();
      double res_min = lazy_a.min().item().toDouble();
      double res_max = lazy_a.max().item().toDouble();
      EXPECT_GT(res_mean, 4.5 - eps);
      EXPECT_LT(res_mean, 4.5 + eps);
      EXPECT_EQ(res_min, 0.0);
      EXPECT_EQ(res_max, 9.0);
    });
  }
}

TEST_F(LazyOpsTest, TestRandomInPlaceDefaultFrom) {
  for (auto dtype :
       {torch::kFloat,
        torch::kDouble,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    const double eps = 0.2;
    torch::Tensor a = torch::zeros({10, 10, 10}, torch::TensorOptions(dtype));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      lazy_a.random_(/*to=*/10);
      double res_mean = lazy_a.sum().item().toDouble() / a.numel();
      double res_min = lazy_a.min().item().toDouble();
      double res_max = lazy_a.max().item().toDouble();
      EXPECT_GT(res_mean, 4.5 - eps);
      EXPECT_LT(res_mean, 4.5 + eps);
      EXPECT_EQ(res_min, 0.0);
      EXPECT_EQ(res_max, 9.0);
    });
  }
}

TEST_F(LazyOpsTest, TestRandomInPlaceDefault) {
  for (auto dtype :
       {torch::kFloat,
        torch::kDouble,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    auto input = torch::zeros({10}, torch::TensorOptions(dtype));
    ForEachDevice([&](const torch::Device& device) {
      auto lazyInput = CopyToDevice(input, device);
      lazyInput.random_();
      auto output = ToCpuTensor(lazyInput);
      EXPECT_TRUE(torch::all(output.ne(input)).item<bool>());
    });
  }
}

TEST_F(LazyOpsTest, TestNormGeneral) {
  torch::Tensor a = torch::randn(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::norm(a, 3.5);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::norm(lazy_a, 3.5);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestNormNuclear) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::norm(a, 1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::norm(lazy_a, 1);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestFrobeniusNormInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::frobenius_norm(a, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b =
          torch::frobenius_norm(lazy_a, {dim}, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestFrobeniusNormInDims) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::frobenius_norm(a, dims, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b =
          torch::frobenius_norm(lazy_a, dims, /*keepdim=*/false);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestGroupNorm) {
  int num_channels = 6;
  torch::Tensor input = torch::rand(
      {20, num_channels, 10, 10},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor bias = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double eps = 1e-05;
  for (int num_groups : {3, 6, 1}) {
    torch::Tensor output = torch::group_norm(
        input,
        num_groups,
        weight,
        bias,
        eps,
        /*cudnn_enabled=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_weight = CopyToDevice(weight, device);
      torch::Tensor lazy_bias = CopyToDevice(bias, device);
      torch::Tensor lazy_output = torch::group_norm(
          lazy_input,
          num_groups,
          lazy_weight,
          lazy_bias,
          eps,
          /*cudnn_enabled=*/false);
      AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
    });
  }
}

TEST_F(LazyOpsTest, TestGroupNormBackward) {
  int num_channels = 6;
  torch::Tensor input = torch::rand(
      {2, num_channels, 5, 5},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  torch::Tensor weight = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  torch::Tensor bias = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  double eps = 1e-05;
  for (bool undef_weight : {true, false}) {
    for (int num_groups : {3, 6, 1}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::group_norm(
            /*input=*/inputs[0],
            num_groups,
            inputs[1],
            inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      torch::Tensor undef;
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {input, undef_weight ? undef : weight, undef_weight ? undef : bias},
            device,
            testfn,
            /*rtol=*/1e-3,
            /*atol=*/1e-3,
            /*derivative_level=*/2);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestInstanceNorm) {
  int batch = 5;
  int num_channels = 20;
  torch::Tensor input = torch::rand(
      {batch, num_channels, 10, 10},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor bias = torch::rand(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor running_mean = torch::zeros(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor running_var = torch::ones(
      {num_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double momentum = 0.1;
  double eps = 1e-05;
  torch::Tensor output = torch::instance_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      /*use_input_stats=*/true,
      momentum,
      eps,
      /*cudnn_enabled=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    torch::Tensor lazy_bias = CopyToDevice(bias, device);
    torch::Tensor lazy_running_mean = CopyToDevice(running_mean, device);
    torch::Tensor lazy_running_var = CopyToDevice(running_var, device);
    torch::Tensor lazy_output = torch::instance_norm(
        lazy_input,
        lazy_weight,
        lazy_bias,
        lazy_running_mean,
        lazy_running_var,
        /*use_input_stats=*/true,
        momentum,
        eps,
        /*cudnn_enabled=*/false);
    AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLayerNorm) {
  torch::Tensor input = torch::rand(
      {20, 10, 10, 10},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double eps = 1e-05;
  torch::Tensor undef;
  for (bool undef_weight : {true, false}) {
    for (int64_t normalized_size : {2, 3}) {
      std::vector<int64_t> normalized_shape(normalized_size, 10);
      torch::Tensor weight = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor bias = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor output = torch::layer_norm(
          input,
          normalized_shape,
          undef_weight ? undef : weight,
          undef_weight ? undef : bias,
          eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        torch::Tensor lazy_bias =
            undef_weight ? undef : CopyToDevice(bias, device);
        torch::Tensor lazy_output = torch::layer_norm(
            lazy_input,
            normalized_shape,
            lazy_weight,
            lazy_bias,
            eps,
            /*cudnn_enabled=*/false);
        AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestLayerNormBackward) {
  torch::Tensor input = torch::rand(
      {2, 3, 3, 3},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  double eps = 1e-05;
  for (bool undef_weight : {true, false}) {
    for (int64_t normalized_size : {2, 3}) {
      std::vector<int64_t> normalized_shape(normalized_size, 3);
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::layer_norm(
            /*input=*/inputs[0],
            normalized_shape,
            inputs[1],
            inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      torch::Tensor weight = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      torch::Tensor bias = torch::rand(
          normalized_shape,
          torch::TensorOptions(torch::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      torch::Tensor undef;
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {input, undef_weight ? undef : weight, undef_weight ? undef : bias},
            device,
            testfn,
            /*rtol=*/1e-3,
            /*atol=*/1e-4,
            /*derivative_level=*/2);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNuclearNorm) {
  torch::Tensor a = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::nuclear_norm(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::nuclear_norm(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestPairwiseDistance) {
  torch::Tensor x1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor x2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double eps = 1e-6;
  for (bool keepdim : {false, true}) {
    for (double p : {1, 2, 3, 4}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::pairwise_distance(x1, x2, p, eps, keepdim);
        torch::Tensor lazy_x1 = CopyToDevice(x1, device);
        torch::Tensor lazy_x2 = CopyToDevice(x2, device);
        torch::Tensor lazy_output =
            torch::pairwise_distance(lazy_x1, lazy_x2, p, eps, keepdim);
        AllClose(output, lazy_output, /*rtol=*/1e-5, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestCosineSimilarity) {
  torch::Tensor x1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor x2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double eps = 1e-8;
  int rank = x1.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::cosine_similarity(x1, x2, dim, eps);
      torch::Tensor lazy_x1 = CopyToDevice(x1, device);
      torch::Tensor lazy_x2 = CopyToDevice(x2, device);
      torch::Tensor lazy_output =
          torch::cosine_similarity(lazy_x1, lazy_x2, dim, eps);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestCosineEmbeddingLoss) {
  torch::Tensor input1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor input2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::cosine_embedding_loss(
            input1, input2, target, margin, reduction);
        torch::Tensor lazy_input1 = CopyToDevice(input1, device);
        torch::Tensor lazy_input2 = CopyToDevice(input2, device);
        torch::Tensor lazy_target = CopyToDevice(target, device);
        torch::Tensor lazy_output = torch::cosine_embedding_loss(
            lazy_input1, lazy_input2, lazy_target, margin, reduction);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestHingeEmbeddingLoss) {
  torch::Tensor input = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::hinge_embedding_loss(input, target, margin, reduction);
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_target = CopyToDevice(target, device);
        torch::Tensor lazy_output = torch::hinge_embedding_loss(
            lazy_input, lazy_target, margin, reduction);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestTripletMarginLoss) {
  torch::Tensor anchor = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor positive = torch::abs(torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  torch::Tensor negative = torch::neg(torch::abs(torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()))));
  double eps = 1e-6;
  for (double margin : {0., 0.2}) {
    for (double p : {1, 2, 3, 4}) {
      for (bool swap : {false, true}) {
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean, torch::Reduction::Sum}) {
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor output = torch::triplet_margin_loss(
                anchor, positive, negative, margin, p, eps, swap, reduction);
            torch::Tensor lazy_anchor = CopyToDevice(anchor, device);
            torch::Tensor lazy_positive = CopyToDevice(positive, device);
            torch::Tensor lazy_negative = CopyToDevice(negative, device);
            torch::Tensor lazy_output = torch::triplet_margin_loss(
                lazy_anchor,
                lazy_positive,
                lazy_negative,
                margin,
                p,
                eps,
                swap,
                reduction);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestBinaryCrossEntropy) {
  int batch = 10;
  int classes = 5;
  torch::Tensor input = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor undef;
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean,
        torch::Reduction::Sum,
        torch::Reduction::None}) {
    for (bool undef_weight : {false, true}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::binary_cross_entropy(
            input, target, undef_weight ? undef : weight, reduction);
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_target = CopyToDevice(target, device);
        torch::Tensor lazy_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        torch::Tensor lazy_output = torch::binary_cross_entropy(
            lazy_input, lazy_target, lazy_weight, reduction);
        AllClose(output, lazy_output, /*rtol=*/1e-4, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMarginRankingLoss) {
  torch::Tensor input1 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor input2 = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::margin_ranking_loss(
            input1, input2, target, margin, reduction);
        torch::Tensor lazy_input1 = CopyToDevice(input1, device);
        torch::Tensor lazy_input2 = CopyToDevice(input2, device);
        torch::Tensor lazy_target = CopyToDevice(target, device);
        torch::Tensor lazy_output = torch::margin_ranking_loss(
            lazy_input1, lazy_input2, lazy_target, margin, reduction);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestBCEWithLogits) {
  int batch = 10;
  int classes = 5;
  torch::Tensor input = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {batch, classes},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {classes}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor pos_weight = torch::rand(
      {classes}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor undef;
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor output = torch::binary_cross_entropy_with_logits(
              input,
              target,
              undef_weight ? undef : weight,
              undef_pos_weight ? undef : pos_weight,
              reduction);
          torch::Tensor lazy_input = CopyToDevice(input, device);
          torch::Tensor lazy_target = CopyToDevice(target, device);
          torch::Tensor lazy_weight =
              undef_weight ? undef : CopyToDevice(weight, device);
          torch::Tensor lazy_pos_weight =
              undef_pos_weight ? undef : CopyToDevice(pos_weight, device);
          torch::Tensor lazy_output = torch::binary_cross_entropy_with_logits(
              lazy_input, lazy_target, lazy_weight, lazy_pos_weight, reduction);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestKlDiv) {
  torch::Tensor input = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (bool log_target : {true, false}) {
    for (torch::Reduction::Reduction reduction :
         {torch::Reduction::Mean, torch::Reduction::Sum}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::kl_div(input, target, reduction, log_target);
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_target = CopyToDevice(target, device);
        torch::Tensor lazy_output =
            torch::kl_div(lazy_input, lazy_target, reduction, log_target);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestProd) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::prod(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::prod(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestProdCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::prod(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::prod(lazy_a, torch::kDouble);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestProdInDim) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::prod(lazy_a, dim);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestProdInDimKeepCast) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim, /*keepdim=*/true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b =
          torch::prod(lazy_a, dim, /*keepdim=*/true, torch::kDouble);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestProdInDimKeep) {
  torch::Tensor a = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::prod(lazy_a, dim, /*keepdim=*/true);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSum) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSumCast) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result =
          torch::cumsum(lazy_input, dim, torch::kDouble);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSumLong) {
  torch::Tensor input = torch::randint(
      1000,
      {4, 3, 4},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumSumCastLong) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kLong);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim, torch::kLong);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProd) {
  torch::Tensor input = torch::rand(
      {4, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumprod(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumprod(lazy_input, dim);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProdCast) {
  torch::Tensor input = torch::mul(
      torch::rand(
          {4, 3, 4},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice())),
      10);
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumprod(input, dim, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result =
          torch::cumprod(lazy_input, dim, torch::kDouble);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProdLong) {
  torch::Tensor input = torch::randint(
      7, {2, 3}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCumProdCastLong) {
  torch::Tensor input =
      torch::rand(
          {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      7;
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kLong);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_result = torch::cumsum(lazy_input, dim, torch::kLong);
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMin) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::argmin(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b =
        torch::argmin(lazy_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMinDim) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmin(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMinDimKeep) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmin(lazy_a, dim, /*keepdim=*/true);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMinSameValue) {
  torch::Tensor a = torch::ones(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::argmin(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::argmin(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMinWrapper) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmin(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMax) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::argmax(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b =
        torch::argmax(lazy_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMaxDim) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmax(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMaxDimKeep) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmax(lazy_a, dim, /*keepdim=*/true);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestArgMaxSameValue) {
  torch::Tensor a = torch::ones(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::argmax(a, std::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b =
        torch::argmax(lazy_a, std::nullopt, /*keepdim=*/false);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestArgMaxWrapper) {
  torch::Tensor a = torch::rand(
      {4, 4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::argmax(lazy_a, dim, /*keepdim=*/false);
      AllEqual(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestAsin) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::asin(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::asin(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAsinh) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::asinh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::asinh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAsinhInPlace) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = torch::asinh_(a);
    torch::Tensor lazy_b = torch::asinh_(lazy_a);
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestSin) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::sin(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sin(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestSinh) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::sinh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sinh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAcos) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::acos(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::acos(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAcosh) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100;
  torch::Tensor b = torch::acosh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::acosh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAcoshInPlace) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = torch::acosh_(a);
    torch::Tensor lazy_b = torch::acosh_(lazy_a);
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestCos) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::cos(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::cos(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestCosh) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::cosh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::cosh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtan) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::atan(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::atan(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtanh) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::atanh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::atanh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtanhInPlace) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = torch::atanh_(a);
    torch::Tensor lazy_b = torch::atanh_(lazy_a);
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestAtan2) {
  torch::Tensor a = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::atan2(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::atan2(lazy_a, lazy_b);
    AllClose(c, lazy_c, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestTan) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::tan(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::tan(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestTanh) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::tanh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::tanh(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestClampMinMax) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar min_val(0.311);
  torch::Scalar max_val(0.409);
  torch::Tensor b = torch::clamp(a, min_val, max_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::clamp(lazy_a, min_val, max_val);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMin) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar min_val(0.311);
  torch::Tensor b = torch::clamp(a, min_val, std::nullopt);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::clamp(lazy_a, min_val, std::nullopt);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMax) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar max_val(0.409);
  torch::Tensor b = torch::clamp(a, std::nullopt, max_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::clamp(lazy_a, std::nullopt, max_val);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMinExplicit) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar min_val(0.311);
  torch::Tensor b = torch::clamp_min(a, min_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::clamp_min(lazy_a, min_val);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMaxExplicit) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar max_val(0.409);
  torch::Tensor b = torch::clamp_max(a, max_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::clamp_max(lazy_a, max_val);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMinExplicitInPlace) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar min_val(0.311);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = torch::clamp_min_(a, min_val);
    torch::Tensor lazy_b = torch::clamp_min_(lazy_a, min_val);
    AllClose(a, lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestClampMaxExplicitInPlace) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar max_val(0.409);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = torch::clamp_max_(a, max_val);
    torch::Tensor lazy_b = torch::clamp_max_(lazy_a, max_val);
    AllClose(a, lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestCeil) {
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::ceil(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::ceil(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestFloor) {
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::floor(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::floor(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestRound) {
  torch::Tensor a = torch::cat(
      {torch::randn(
           {8}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
           100.0,
       // Special case: 0.5, -0.5. lazy::Round impl rounds to -1/1 whereas
       // lazy::RoundToEven properly implements bankers rounding.
       torch::tensor(
           {-0.5, 0.5},
           torch::TensorOptions(torch::kFloat).device(DefaultDevice()))},
      0);
  torch::Tensor b = torch::round(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::round(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestTrunc) {
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::trunc(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::trunc(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestFrac) {
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::frac(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::frac(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestNeg) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::neg(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::neg(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestBitwiseNot) {
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  ForEachDevice([&](const torch::Device& device) {
    for (auto type : types) {
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Tensor b = torch::bitwise_not(a);
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = torch::bitwise_not(lazy_a);
      AllEqual(b, lazy_b);
    }
  });
}

TEST_F(LazyOpsTest, TestBitwiseNotInPlace) {
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  ForEachDevice([&](const torch::Device& device) {
    for (auto type : types) {
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Tensor lazy_a = CopyToDevice(a, device);
      a.bitwise_not_();
      lazy_a.bitwise_not_();
      AllEqual(a, lazy_a);
    }
  });
}

TEST_F(LazyOpsTest, TestSign) {
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b = torch::sign(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sign(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestSignByte) {
  torch::Tensor a = torch::randint(
      256, {2, 2}, torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  torch::Tensor b = torch::sign(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sign(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestAbs) {
  torch::Tensor a = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::abs(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::abs(lazy_a);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestAbsByte) {
  torch::Tensor a = torch::randint(
      256, {2, 2}, torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  torch::Tensor b = torch::abs(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::abs(lazy_a);
    AllEqual(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestEmptyLike) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::empty_like(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::empty_like(lazy_a);
    EXPECT_EQ(b.sizes(), lazy_b.sizes());
  });
}

TEST_F(LazyOpsTest, TestEmptyLikeOptions) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::empty_like(
      a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::empty_like(
        lazy_a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    EXPECT_EQ(b.sizes(), lazy_b.sizes());
  });
}

TEST_F(LazyOpsTest, TestEmpty) {
  torch::Tensor a = torch::zeros(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = torch::empty(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    EXPECT_EQ(a.sizes(), lazy_a.sizes());
  });
}

TEST_F(LazyOpsTest, TestZeroInPlace) {
  torch::Tensor input = torch::ones(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));

  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazyInput = CopyToDevice(input, device);
    auto& output = torch::zero_(input);
    auto& lazyOutput = torch::zero_(lazyInput);
    AllClose(output, lazyOutput);
  });
}

TEST_F(LazyOpsTest, TestZerosLike) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::zeros_like(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::zeros_like(lazy_a);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestZerosLikeOptions) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::zeros_like(
      a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::zeros_like(
        lazy_a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestZeros) {
  torch::Tensor a = torch::zeros(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = torch::zeros(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestOnes) {
  torch::Tensor a = torch::ones(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a =
        torch::ones({2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestOnesLike) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::ones_like(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::ones_like(lazy_a);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestOnesLikeOptions) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::ones_like(
      a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::ones_like(
        lazy_a, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFull) {
  torch::Tensor a = torch::full(
      {2, 2},
      3.1165,
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = torch::full(
        {2, 2}, 3.1165, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFullLike) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::full_like(a, 3.1165);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::full_like(lazy_a, 3.1165);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFullLikeOptions) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::full_like(
      a, 3.1165, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::full_like(
        lazy_a,
        3.1165,
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestARange) {
  for (auto& ranges : std::vector<std::vector<float>>{
           {0.0, 100.0, 0.5}, {0.0, -100.0, -0.5}}) {
    torch::Tensor a = torch::arange(
        ranges[0],
        ranges[1],
        ranges[2],
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = torch::arange(
          ranges[0],
          ranges[1],
          ranges[2],
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(a, lazy_a);
    });
  }
}

TEST_F(LazyOpsTest, TestARangeOut) {
  torch::Tensor a = torch::randn(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto& ranges : std::vector<std::vector<float>>{
           {0.0, 100.0, 0.5}, {0.0, -100.0, -0.5}}) {
    torch::Tensor b = torch::arange_out(a, ranges[0], ranges[1], ranges[2]);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b =
          torch::arange_out(lazy_a, ranges[0], ranges[1], ranges[2]);
      AllClose(b, lazy_b);
    });
  }
}

TEST_F(LazyOpsTest, TestDimARange) {
  torch::Tensor like = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor a = torch::_dim_arange(like, 1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_like = CopyToDevice(like, device);
    torch::Tensor lazy_a = torch::_dim_arange(lazy_like, 1);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestBartlettWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::bartlett_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));

      torch::Tensor lazy_output = torch::bartlett_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, lazy_output, /*rtol=*/1e-5, /*atol=*/1e-7);
    });
  }
}

TEST_F(LazyOpsTest, TestBlackmanWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::blackman_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor lazy_output = torch::blackman_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, lazy_output, /*rtol=*/1e-5, /*atol=*/1e-7);
    });
  }
}

TEST_F(LazyOpsTest, TestHammingWindow) {
  double alpha = 0.54;
  double beta = 0.46;
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::hamming_window(
          window_length,
          periodic,
          alpha,
          beta,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor lazy_output = torch::hamming_window(
          window_length,
          periodic,
          alpha,
          beta,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestHannWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::hann_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor lazy_output = torch::hann_window(
          window_length,
          periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestLogSigmoid) {
  torch::Tensor a = torch::empty(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  a.uniform_(-1.0, 1.0);
  torch::Tensor b = torch::log_sigmoid(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::log_sigmoid(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLogSigmoidForward) {
  torch::Tensor a = torch::empty(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  a.uniform_(-1.0, 1.0);
  auto tuple = torch::log_sigmoid_forward(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    auto lazy_tuple = torch::log_sigmoid_forward(lazy_a);
    AllClose(
        std::get<0>(tuple),
        std::get<0>(lazy_tuple),
        /*rtol=*/1e-3,
        /*atol=*/1e-5);
    AllClose(
        std::get<1>(tuple),
        std::get<1>(lazy_tuple),
        /*rtol=*/1e-3,
        /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLogsumexp) {
  torch::Tensor a = torch::rand(
      {3, 4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepdim : {false, true}) {
      torch::Tensor b = torch::logsumexp(a, dims, keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        torch::Tensor lazy_b = torch::logsumexp(lazy_a, dims, keepdim);
        AllClose(b, lazy_b);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestSiLU) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::silu(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::silu(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterChanged("lazy::silu_out", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestSigmoid) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::sigmoid(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sigmoid(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestMatmul_1x1) {
  torch::Tensor a = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::matmul(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMatmul_2x1) {
  torch::Tensor a = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::matmul(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMatmul_1x2) {
  torch::Tensor a = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::matmul(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMatmul_2x2) {
  torch::Tensor a = torch::rand(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::matmul(lazy_a, lazy_b);
    AllClose(c, lazy_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestMatmulBcast) {
  torch::Tensor a = torch::rand(
      {4, 2, 3, 2, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 1, 4, 3},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::matmul(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestDot) {
  torch::Tensor a = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::dot(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::dot(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestTensorDot) {
  torch::Tensor a = torch::rand(
      {6, 4, 8}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {4, 7, 8}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> dims_a = {1, 2};
  std::vector<int64_t> dims_b = {0, 2};
  torch::Tensor c = torch::tensordot(a, b, dims_a, dims_b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::tensordot(lazy_a, lazy_b, dims_a, dims_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestGer) {
  torch::Tensor a = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::ger(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::ger(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMv) {
  torch::Tensor a = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::mv(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::mv(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestMvOut) {
  torch::Tensor a = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::empty(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::mv_out(c, a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::empty({4}, lazy_b.options());
    torch::mv_out(lazy_c, lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestBatchAddBatchMatMul) {
  torch::Tensor a = torch::rand(
      {3, 6, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3, 6, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::rand(
      {3, 4, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar alpha = 0.5;
  torch::Scalar beta = 1.5;
  torch::Tensor d = torch::baddbmm(a, b, c, beta, alpha);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    torch::Tensor lazy_d = torch::baddbmm(lazy_a, lazy_b, lazy_c, beta, alpha);
    AllClose(d, lazy_d, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestBatchAddBatchMatMulInPlace) {
  torch::Tensor a = torch::rand(
      {3, 6, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3, 6, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::rand(
      {3, 4, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar alpha = 0.5;
  torch::Scalar beta = 1.5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    torch::Tensor d = a.baddbmm_(b, c, beta, alpha);
    torch::Tensor lazy_d = lazy_a.baddbmm_(lazy_b, lazy_c, beta, alpha);
    AllClose(d, lazy_d, /*rtol=*/1e-3, /*atol=*/1e-4);
    AllClose(a, lazy_a, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestBatchMatMul) {
  torch::Tensor a = torch::rand(
      {3, 6, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3, 4, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::bmm(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::bmm(lazy_a, lazy_b);
    AllClose(c, lazy_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestChainMatMul) {
  torch::Tensor a = torch::rand(
      {5, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {4, 6}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::rand(
      {6, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor d = torch::rand(
      {2, 7}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor result = torch::chain_matmul({a, b, c, d});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    torch::Tensor lazy_d = CopyToDevice(d, device);
    torch::Tensor lazy_result =
        torch::chain_matmul({lazy_a, lazy_b, lazy_c, lazy_d});
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestLinear) {
  torch::Tensor input = torch::rand(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor bias = torch::rand(
      {3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor result = torch::linear(input, weight);
  torch::Tensor result_with_bias = torch::linear(input, weight, bias);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    torch::Tensor lazy_bias = CopyToDevice(bias, device);
    torch::Tensor lazy_result = torch::linear(lazy_input, lazy_weight);
    torch::Tensor lazy_result_with_bias =
        torch::linear(lazy_input, lazy_weight, lazy_bias);
    AllClose(result, lazy_result, /*rtol=*/1e-2, /*atol=*/1e-4);
    AllClose(
        result_with_bias,
        lazy_result_with_bias,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestPinverse) {
  torch::Tensor input = torch::rand(
      {4, 6}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor result = torch::pinverse(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::pinverse(lazy_input);
    AllClose(result, lazy_result, /*rtol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestEinsumOuter) {
  torch::Tensor a = torch::rand(
      {5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::string equation = "i,j->ij";
  torch::Tensor c = torch::einsum(equation, {a, b});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::einsum(equation, {lazy_a, lazy_b});
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEinsumOuterBackward) {
  torch::Tensor a = torch::rand(
      {5},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  torch::Tensor b = torch::rand(
      {5},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  std::string equation = "i,j->ij";
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::einsum(equation, inputs);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward({a, b}, device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestEinsumBatchMatMul) {
  torch::Tensor a = torch::rand(
      {3, 2, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3, 5, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::string equation = "bij,bjk->bik";
  torch::Tensor c = torch::einsum(equation, {a, b});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::einsum(equation, {lazy_a, lazy_b});
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEinsumPyTorchLowerBilinear) {
  torch::Tensor a = torch::rand(
      {3, 5, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor l = torch::rand(
      {2, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor r = torch::rand(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::string equation = "bn,anm,bm->ba";
  torch::Tensor c = torch::einsum(equation, {l, a, r});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_l = CopyToDevice(l, device);
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_r = CopyToDevice(r, device);
    torch::Tensor lazy_c = torch::einsum(equation, {lazy_l, lazy_a, lazy_r});
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEinsumPyTorchLowerDiagonal) {
  torch::Tensor input = torch::rand(
      {3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::string equation = "ii->i";
  torch::Tensor result = torch::einsum(equation, {input});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::einsum(equation, {lazy_input});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestEinsumPyTorchLowerBatchDiagonal) {
  torch::Tensor input = torch::rand(
      {4, 3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::string equation = "...ii->...i";
  torch::Tensor result = torch::einsum(equation, {input});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::einsum(equation, {lazy_input});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestEinsumPyTorchLowerBatchPermute) {
  torch::Tensor input = torch::rand(
      {2, 3, 4, 5},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::string equation = "...ij->...ji";
  torch::Tensor result = torch::einsum(equation, {input});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::einsum(equation, {lazy_input});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestEinsumPyTorchLowerRepeatedAxis) {
  torch::Tensor x = torch::rand(
      {2, 3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor y = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::string equation = "ijj,k->ik";
  torch::Tensor result = torch::einsum(equation, {x, y});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_x = CopyToDevice(x, device);
    torch::Tensor lazy_y = CopyToDevice(y, device);
    torch::Tensor lazy_result = torch::einsum(equation, {lazy_x, lazy_y});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBilinear) {
  int batch_size = 16;
  int in1_features = 4;
  int in2_features = 6;
  int out_features = 8;
  torch::Tensor input1 = torch::rand(
      {batch_size, in1_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor input2 = torch::rand(
      {batch_size, in2_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {out_features, in1_features, in2_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor bias = torch::rand(
      {out_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input1 = CopyToDevice(input1, device);
    torch::Tensor lazy_input2 = CopyToDevice(input2, device);
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    torch::Tensor lazy_bias = CopyToDevice(bias, device);
    torch::Tensor result = torch::bilinear(input1, input2, weight, bias);
    torch::Tensor lazy_result =
        torch::bilinear(lazy_input1, lazy_input2, lazy_weight, lazy_bias);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestUpsampleNearest2D) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  torch::Tensor input = torch::rand(
      {batch_size, chans, h, w},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor result = torch::upsample_nearest2d(input, {uh, uw});
    torch::Tensor lazy_result = torch::upsample_nearest2d(lazy_input, {uh, uw});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestUpsampleNearest2DBackward) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::upsample_nearest2d(inputs[0], {uh, uw});
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {batch_size, chans, h, w},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestUpsampleNearest2DWithScale) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int chans = 2;
  double scale_h = 2.5;
  double scale_w = 3.4;
  torch::Tensor input = torch::rand(
      {batch_size, chans, h, w},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor result = torch::upsample_nearest2d(
        input, std::nullopt, at::ArrayRef<double>{scale_h, scale_w});
    torch::Tensor lazy_result = torch::upsample_nearest2d(
        lazy_input, std::nullopt, at::ArrayRef<double>{scale_h, scale_w});
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestUpsampleNearest2DBackwardWithScale) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int chans = 2;
  double scale_h = 2.5;
  double scale_w = 3.4;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::upsample_nearest2d(
        inputs[0], std::nullopt, at::ArrayRef<double>{scale_h, scale_w});
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {batch_size, chans, h, w},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestUpsampleBilinear2D) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  for (bool align_corners : {true, false}) {
    torch::Tensor input = torch::rand(
        {batch_size, chans, h, w},
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor result =
          torch::upsample_bilinear2d(input, {uh, uw}, align_corners);
      torch::Tensor lazy_result =
          torch::upsample_bilinear2d(lazy_input, {uh, uw}, align_corners);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestUpsampleBilinear2DBackward) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  for (bool align_corners : {true, false}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::upsample_bilinear2d(inputs[0], {uh, uw}, align_corners);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {batch_size, chans, h, w},
              torch::TensorOptions(torch::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestAddCMul) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor d = torch::addcmul(a, b, c, 3.1165);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    torch::Tensor lazy_d = torch::addcmul(lazy_a, lazy_b, lazy_c, 3.1165);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestAddCDiv) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c =
      torch::abs(torch::rand(
          {2, 2},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()))) +
      1.0;
  torch::Tensor d = torch::addcdiv(a, b, c, 3.1165);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    torch::Tensor lazy_d = torch::addcdiv(lazy_a, lazy_b, lazy_c, 3.1165);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestAddCDivWithBroadcast) {
  torch::Tensor a = torch::rand(
      {1, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c =
      torch::abs(torch::rand(
          {1, 3},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()))) +
      1.0;
  torch::Tensor d = torch::addcdiv(a, b, c, 3.1165);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    torch::Tensor lazy_d = torch::addcdiv(lazy_a, lazy_b, lazy_c, 3.1165);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestSize) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    for (int dim = -rank; dim < rank; ++dim) {
      EXPECT_EQ(torch::size(input, dim), torch::size(lazy_input, dim));
    }
  });
}

TEST_F(LazyOpsTest, TestSelect) {
  std::vector<int64_t> input_sizes = {14, 24, 8};
  int rank = input_sizes.size();
  for (int dim = -rank; dim < rank; ++dim) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::select(inputs[0], dim, 0);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              input_sizes,
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device,
          testfn);
    });
  };
}

TEST_F(LazyOpsTest, TestBernoulliScalarProb) {
  torch::Tensor input = torch::zeros(
      1000, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::bernoulli(lazy_input, 0.1);
    double frac = lazy_output.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestBernoulliTensorProb) {
  std::vector<float> prob_values(1000, 0.1);
  torch::Tensor input = torch::tensor(
      prob_values, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::bernoulli(lazy_input);
    double frac = lazy_output.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestBernoulliScalarProbInPlace) {
  torch::Tensor input = torch::zeros(
      1000, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    lazy_input.bernoulli_(0.1);
    double frac = lazy_input.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestBernoulliTensorProbInPlace) {
  torch::Tensor input = torch::zeros(
      1000, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor prob = torch::scalar_tensor(
      0.1, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_prob = CopyToDevice(prob, device);
    lazy_input.bernoulli_(lazy_prob);
    double frac = lazy_input.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(LazyOpsTest, TestDropout) {
  torch::Tensor a = torch::rand(
      {17, 21}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::dropout(lazy_a, 0.1, /*train=*/true);
    double prob =
        static_cast<double>(lazy_b.cpu().ne(0.0f).sum().item().toDouble()) /
        a.numel();
    EXPECT_GT(prob, 0.86);
    EXPECT_LT(prob, 0.94);
  });
}

TEST_F(LazyOpsTest, TestDropoutInPlace) {
  torch::Tensor a = torch::rand(
      {17, 21}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::dropout_(lazy_a, 0.1, /*train=*/true);
    double prob =
        static_cast<double>(lazy_a.cpu().ne(0.0f).sum().item().toDouble()) /
        a.numel();
    EXPECT_GT(prob, 0.85);
    EXPECT_LT(prob, 0.94);
  });
}

TEST_F(LazyOpsTest, TestRandperm) {
  unsigned n = 5;
  torch::Tensor shuffle = torch::randperm(
      n, torch::TensorOptions(torch::kLong).device(torch::kLazy));
  torch::Tensor shuffle_cpu = CopyToDevice(shuffle, torch::kCPU);
  std::vector<int64_t> shuffle_data(
      shuffle_cpu.data_ptr<int64_t>(), shuffle_cpu.data_ptr<int64_t>() + n);
  EXPECT_TRUE(
      shuffle_data.size() == n && torch::lazy::IsPermutation(shuffle_data));
}

TEST_F(LazyOpsTest, TestSlice) {
  torch::Tensor a = torch::rand(
      {32, 24, 16},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::slice(a, 1, 0, 16, 1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::slice(lazy_a, 1, 0, 16, 1);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestTake) {
  torch::Tensor a = torch::rand(
      {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::randint(
      16, {5}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Tensor c = torch::take(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::take(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestTakeBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::take(inputs[0], inputs[1]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
             {4, 4},
             torch::TensorOptions(torch::kFloat)
                 .device(DefaultDevice())
                 .requires_grad(true)),
         torch::randint(
             16,
             {5},
             torch::TensorOptions(torch::kLong).device(DefaultDevice()))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestStack) {
  torch::Tensor a = torch::rand(
      {2, 4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::rand(
      {2, 4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = a.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor d = torch::stack({a, b, c}, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = torch::stack({lazy_a, lazy_b, lazy_c}, dim);
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestCat) {
  torch::Tensor a = torch::rand(
      {2, 1, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::rand(
      {2, 3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int dim : {1, -2}) {
    torch::Tensor d = torch::cat({a, b, c}, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = torch::cat({lazy_a, lazy_b, lazy_c}, dim);
      EXPECT_TRUE(d.sizes() == lazy_d.sizes() && d.dtype() == lazy_d.dtype());
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestUnbind) {
  torch::Tensor input = torch::rand(
      {4, 3, 7}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<torch::Tensor> output = torch::unbind(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      std::vector<torch::Tensor> lazy_output = torch::unbind(lazy_input, dim);
      ASSERT_EQ(output.size(), lazy_output.size());
      for (size_t i = 0; i < output.size(); ++i) {
        AllClose(output[i], lazy_output[i]);
      }
    });
  }
}

TEST_F(LazyOpsTest, TestRepeat) {
  std::vector<std::vector<int64_t>> repeats_list = {{4, 2}, {4, 2, 3}};
  std::vector<std::vector<int64_t>> input_size_list = {{3}, {2, 4}};
  for (const auto& repeats : repeats_list) {
    for (const auto& input_size : input_size_list) {
      torch::Tensor input = torch::rand(
          input_size,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor output = input.repeat(repeats);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_output = lazy_input.repeat(repeats);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestGather) {
  torch::Tensor a = torch::rand(
      {3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::empty(
      {3, 3}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      b[i][j] = (i + j) % 3;
    }
  }
  for (bool sparse_grad : {false, true}) {
    torch::Tensor c = torch::gather(a, 1, b, sparse_grad);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = torch::gather(lazy_a, 1, lazy_b, sparse_grad);
      AllClose(c, lazy_c);
    });
  }
}

TEST_F(LazyOpsTest, TestScatter) {
  torch::Tensor a = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::empty(
      {3, 5}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = torch::scatter(lazy_a, dim, lazy_c, lazy_b);
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestScatterR1) {
  torch::Tensor a = torch::rand(
      {5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::empty(
      {2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  c[0] = 1;
  c[1] = 3;
  torch::Tensor d = torch::scatter(a, 0, c, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    torch::Tensor lazy_d = torch::scatter(lazy_a, 0, lazy_c, lazy_b);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestScatterR3) {
  torch::Tensor a = torch::rand(
      {3, 5, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3, 4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::empty(
      {3, 4, 2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 2; k++) {
        c[i][j][k] = (i + j + k) % 4;
      }
    }
  }
  torch::Tensor d = torch::scatter(a, 1, c, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    torch::Tensor lazy_d = torch::scatter(lazy_a, 1, lazy_c, lazy_b);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestScatterBiggerSource) {
  torch::Tensor a = torch::rand(
      {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {8, 8}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::empty(
      {4, 4}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    torch::Tensor d = torch::scatter(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = torch::scatter(lazy_a, dim, lazy_c, lazy_b);
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestScatterScalar) {
  torch::Tensor a = torch::rand(
      {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar b = 1.0f;
  torch::Tensor c = torch::empty(
      {4, 4}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    torch::Tensor d = torch::scatter(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = torch::scatter(lazy_a, dim, lazy_c, b);
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestScatterReduceAdd) {
  torch::Tensor a = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::empty(
      {3, 5}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter(a, dim, c, b, "add");
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = torch::scatter(lazy_a, dim, lazy_c, lazy_b, "add");
      AllClose(d, lazy_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::scatter_out", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestScatterAdd) {
  torch::Tensor a = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::empty(
      {3, 5}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter_add(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = torch::scatter_add(lazy_a, dim, lazy_c, lazy_b);
      AllClose(d, lazy_d);
    });
  }
}

TEST_F(LazyOpsTest, TestScatterAddInPlace) {
  torch::Tensor b = torch::rand(
      {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::empty(
      {4, 4}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor a = torch::rand(
          {4, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor d = a.scatter_add_(dim, c, b);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c = CopyToDevice(c, device);
      torch::Tensor lazy_d = lazy_a.scatter_add_(dim, lazy_c, lazy_b);
      AllClose(d, lazy_d);
      AllClose(a, lazy_a);
    });
  }
}

TEST_F(LazyOpsTest, TestIndexSelect) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (torch::ScalarType index_scalar_type : {torch::kInt, torch::kLong}) {
      torch::Tensor b = torch::empty(
          {2}, torch::TensorOptions(index_scalar_type).device(DefaultDevice()));
      b[0] = 0;
      b[1] = 2;
      for (auto offset : {-2, 0}) {
        torch::Tensor c0 = torch::index_select(a, 0 + offset, b);
        torch::Tensor c1 = torch::index_select(a, 1 + offset, b);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a, device);
          torch::Tensor lazy_b = CopyToDevice(b, device);
          torch::Tensor lazy_c0 =
              torch::index_select(lazy_a, 0 + offset, lazy_b);
          torch::Tensor lazy_c1 =
              torch::index_select(lazy_a, 1 + offset, lazy_b);
          AllEqual(c0, lazy_c0);
          AllEqual(c1, lazy_c1);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestIndexSelectRank0) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor a = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor b = torch::scalar_tensor(
        2, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor c0 = torch::index_select(a, 0, b);
    torch::Tensor c1 = torch::index_select(a, 1, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_a = CopyToDevice(a, device);
      torch::Tensor lazy_b = CopyToDevice(b, device);
      torch::Tensor lazy_c0 = torch::index_select(lazy_a, 0, lazy_b);
      torch::Tensor lazy_c1 = torch::index_select(lazy_a, 1, lazy_b);
      AllEqual(c0, lazy_c0);
      AllEqual(c1, lazy_c1);
    });
  }
}

TEST_F(LazyOpsTest, TestInverse) {
  if (IsCuda()) {
    // TODO(whc) debug failure on cuda, lazy_b comes back transposed
    GTEST_SKIP();
  }
  torch::Tensor a = torch::randn(
      {5, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::inverse(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::inverse(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestIsnan) {
  torch::Tensor a = torch::tensor(
      {1.0, 2.0, std::nan("1"), 4.0},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::isnan(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::isnan(lazy_a);
    AllEqual(b, lazy_b);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::isnan", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestExpand) {
  torch::Tensor a = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.expand({2, 3, 4}, /*implicit=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = lazy_a.expand({2, 3, 4}, /*implicit=*/false);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestExpandBack) {
  torch::Tensor a = torch::rand(
      {3, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = a.expand({3, 4}, /*implicit=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = lazy_a.expand({3, 4}, /*implicit=*/false);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestExpandAs) {
  torch::Tensor a = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::native::expand_as(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::native::expand_as(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestEye) {
  int n = 5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor out = torch::eye(
        n, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_out =
        torch::eye(n, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(out, lazy_out);
  });
}

TEST_F(LazyOpsTest, TestEyeWide) {
  int lines = 3;
  int cols = 5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor out = torch::eye(
        lines,
        cols,
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_out = torch::eye(
        lines, cols, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(out, lazy_out);
  });
}

TEST_F(LazyOpsTest, TestEyeNarrow) {
  int lines = 5;
  int cols = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor out = torch::eye(
        lines,
        cols,
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_out = torch::eye(
        lines, cols, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(out, lazy_out);
  });
}

TEST_F(LazyOpsTest, TestBroadcastTensors) {
  torch::Tensor a = torch::rand(
      {2, 1, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<torch::Tensor> c = torch::broadcast_tensors({a, b});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    std::vector<torch::Tensor> lazy_c =
        torch::broadcast_tensors({lazy_a, lazy_b});
    ASSERT_EQ(c.size(), lazy_c.size());
    for (size_t i = 0; i < c.size(); ++i) {
      AllClose(c[i], lazy_c[i]);
    }
  });
}

TEST_F(LazyOpsTest, TestOneIndex) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor indices = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor result = torch::index(params, {indices});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_params = CopyToDevice(params, device);
      torch::Tensor lazy_indices = CopyToDevice(indices, device);
      torch::Tensor lazy_result = torch::index(lazy_params, {lazy_indices});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestOneIndexTransfer) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor indices = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor result = torch::index(params, {indices});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_params = CopyToDevice(params, device);
      torch::Tensor lazy_result = torch::index(lazy_params, {indices.cpu()});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestNonzero) {
  torch::Tensor a = torch::zeros(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  a[0][1] = 1.0;
  a[1][0] = 2.0;
  a[3][1] = 3.0;
  torch::Tensor b = torch::nonzero(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::nonzero(lazy_a);
    AllClose(b, lazy_b);

    if (DebugUtil::ExperimentEnabled("nonzero")) {
      // If the nonzero support is enabled, we must not see any aten:: calls.
      ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
    }
    ResetCounters();
  });
}

TEST_F(LazyOpsTest, TestMaskedSelect) {
  torch::Tensor a = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::randint(
      0, 2, {5}, torch::TensorOptions(torch::kBool).device(DefaultDevice()));
  torch::Tensor c = torch::masked_select(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::masked_select(lazy_a, lazy_b);
    AllClose(c, lazy_c);

    if (DebugUtil::ExperimentEnabled("masked_select")) {
      // If the masked_select support is enabled, we must not see any aten::
      // calls.
      ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
    }
    ResetCounters();
  });
}

TEST_F(LazyOpsTest, TestMaskedScatter) {
  torch::Tensor a = torch::rand(
      {3, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::randint(
      0, 2, {3, 5}, torch::TensorOptions(torch::kBool).device(DefaultDevice()));
  torch::Tensor c = torch::rand(
      {15}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor d = torch::masked_scatter(a, b, c);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    torch::Tensor lazy_d = torch::masked_scatter(lazy_a, lazy_b, lazy_c);
    AllClose(d, lazy_d);

    if (DebugUtil::ExperimentEnabled("masked_scatter")) {
      // If the masked_select support is enabled, we must not see any aten::
      // calls.
      ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
    }
    ResetCounters();
  });
}

TEST_F(LazyOpsTest, TestMultiIndexHeadNull) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor indices_null;
    torch::Tensor indices_0 = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor indices_1 = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor result =
        torch::index(params, {indices_null, indices_0, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_params = CopyToDevice(params, device);
      torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor lazy_result = torch::index(
          lazy_params, {indices_null, lazy_indices_0, lazy_indices_1});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMultiIndexMiddleNull) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor indices_0 = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor indices_null;
    torch::Tensor indices_1 = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor result =
        torch::index(params, {indices_0, indices_null, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_params = CopyToDevice(params, device);
      torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor lazy_result = torch::index(
          lazy_params, {lazy_indices_0, indices_null, lazy_indices_1});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMultiIndexTailNull) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor indices_0 = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor indices_null;
    torch::Tensor indices_1 = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor result =
        torch::index(params, {indices_0, indices_1, indices_null});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_params = CopyToDevice(params, device);
      torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor lazy_result = torch::index(
          lazy_params, {lazy_indices_0, lazy_indices_1, indices_null});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMultiIndexMiddleBroadcast) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor indices_0 = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor indices_1 = torch::randint(
        -3,
        3,
        {2, 1, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor result = torch::index(params, {indices_0, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_params = CopyToDevice(params, device);
      torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor lazy_result =
          torch::index(lazy_params, {lazy_indices_0, lazy_indices_1});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMultiIndexTailBroadcast) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor indices_0 = torch::randint(
        -3,
        3,
        {2, 1, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor indices_1 = torch::randint(
        -3,
        3,
        {2, 1},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor result = torch::index(params, {indices_0, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_params = CopyToDevice(params, device);
      torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor lazy_result =
          torch::index(lazy_params, {lazy_indices_0, lazy_indices_1});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestMaskIndex) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {2, 2}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {2, 2},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor indices = torch::randint(
        0,
        2,
        {2, 2},
        torch::TensorOptions(torch::kBool).device(DefaultDevice()));
    torch::Tensor result = torch::index(params, {indices});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_params = CopyToDevice(params, device);
      torch::Tensor lazy_indices = CopyToDevice(indices, device);
      torch::Tensor lazy_result = torch::index(lazy_params, {lazy_indices});
      AllEqual(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestOneIndexPut) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor indices = torch::randint(
        -3,
        3,
        {2, 4, 3},
        torch::TensorOptions(torch::kLong).device(DefaultDevice()));
    torch::Tensor values = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices = CopyToDevice(indices, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        torch::Tensor lazy_result = torch::index_put(
            lazy_params, {lazy_indices}, lazy_values, accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestOneIndexPutInPlace) {
  torch::Tensor indices = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor values = torch::ones(
        {3, 5, 6, 7},
        torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor params = isFloatingType(scalar_type)
            ? torch::rand(
                  {4, 3, 5, 6, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  {4, 3, 5, 6, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        torch::Tensor lazy_params = CopyToDevice(params.clone(), device);
        torch::Tensor result =
            torch::index_put_(params, {indices}, values, accumulate);
        torch::Tensor lazy_indices = CopyToDevice(indices, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        torch::Tensor lazy_result = torch::index_put_(
            lazy_params, {lazy_indices}, lazy_values, accumulate);
        AllEqual(result, lazy_result);
        AllEqual(params, lazy_params);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestOneIndexPutTransfer) {
  torch::Tensor indices = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor values = torch::ones(
        {3, 5, 6, 7},
        torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        torch::Tensor lazy_result =
            torch::index_put(lazy_params, {indices}, lazy_values, accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPut) {
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor values = torch::ones(
        {5, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result =
          torch::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        torch::Tensor lazy_result = torch::index_put(
            lazy_params,
            {lazy_indices_0, lazy_indices_1},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPutHeadNull) {
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Tensor indices_null;
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor values = torch::ones(
        {3, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result = torch::index_put(
          params, {indices_null, indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        torch::Tensor lazy_result = torch::index_put(
            lazy_params,
            {indices_null, lazy_indices_0, lazy_indices_1},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPutMiddleNull) {
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Tensor indices_null;
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor values = torch::ones(
        {3, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result = torch::index_put(
          params, {indices_0, indices_null, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        torch::Tensor lazy_result = torch::index_put(
            lazy_params,
            {lazy_indices_0, indices_null, lazy_indices_1},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPutTailNull) {
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Tensor indices_null;
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 3, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor values = torch::ones(
        {3, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result = torch::index_put(
          params, {indices_0, indices_1, indices_null}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        torch::Tensor lazy_result = torch::index_put(
            lazy_params,
            {lazy_indices_0, lazy_indices_1, indices_null},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPutMiddleBroadcast) {
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 1, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor values = torch::ones(
        {5, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result =
          torch::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        torch::Tensor lazy_result = torch::index_put(
            lazy_params,
            {lazy_indices_0, lazy_indices_1},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMultiIndexPutTailBroadcast) {
  torch::Tensor indices_0 = torch::randint(
      -3,
      3,
      {2, 1, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Tensor indices_1 = torch::randint(
      -3,
      3,
      {2, 1},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {4, 3, 5, 6, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor values = torch::ones(
        {5, 6, 7}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result =
          torch::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor lazy_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        torch::Tensor lazy_result = torch::index_put(
            lazy_params,
            {lazy_indices_0, lazy_indices_1},
            lazy_values,
            accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestMaskIndexPut) {
  torch::Tensor indices =
      torch::tensor(
          {0, 1}, torch::TensorOptions(torch::kByte).device(DefaultDevice()))
          .to(torch::kBool);
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor params = isFloatingType(scalar_type)
        ? torch::rand(
              {2, 2}, torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {2, 2},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor values = torch::ones(
        {2}, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_params = CopyToDevice(params, device);
        torch::Tensor lazy_indices = CopyToDevice(indices, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        torch::Tensor lazy_result = torch::index_put(
            lazy_params, {lazy_indices}, lazy_values, accumulate);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexPutImpl) {
  torch::Tensor indices = torch::randint(
      -3,
      3,
      {2, 4, 3},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor values = torch::ones(
        {3, 5, 6, 7},
        torch::TensorOptions(scalar_type).device(DefaultDevice()));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor params = isFloatingType(scalar_type)
            ? torch::rand(
                  {4, 3, 5, 6, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  {4, 3, 5, 6, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        torch::Tensor lazy_params = CopyToDevice(params.clone(), device);
        torch::Tensor result = torch::_index_put_impl_(
            params, {indices}, values, accumulate, /*unsafe=*/true);
        torch::Tensor lazy_indices = CopyToDevice(indices, device);
        torch::Tensor lazy_values = CopyToDevice(values, device);
        torch::Tensor lazy_result = torch::_index_put_impl_(
            lazy_params,
            {lazy_indices},
            lazy_values,
            accumulate,
            /*unsafe=*/true);
        AllEqual(result, lazy_result);
        AllEqual(params, lazy_params);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillWithScalar) {
  torch::Tensor index = torch::tensor(
      {0, 2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Scalar value = 42;
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_base = CopyToDevice(base, device);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_result =
            torch::index_fill(lazy_base, dim, lazy_index, value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillWithScalarInPlace) {
  torch::Tensor index = torch::tensor(
      {0, 2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Scalar value = 42;
  int rank = 3;
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor base = isFloatingType(scalar_type)
            ? torch::rand(
                  {3, 4, 5},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  {3, 4, 5},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
        torch::Tensor result = base.index_fill_(dim, index, value);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_result =
            lazy_base.index_fill_(dim, lazy_index, value);
        AllEqual(result, lazy_result);
        AllEqual(base, lazy_base);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillWithTensor) {
  torch::Tensor index = torch::tensor(
      {0, 2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor value = torch::scalar_tensor(
        42, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_base = CopyToDevice(base, device);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        torch::Tensor lazy_result =
            torch::index_fill(lazy_base, dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillWithTensorInPlace) {
  torch::Tensor index = torch::tensor(
      {0, 2}, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor value = torch::scalar_tensor(
        42, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = 3;
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor base = isFloatingType(scalar_type)
            ? torch::rand(
                  {3, 4, 5},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  {3, 4, 5},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
        torch::Tensor result = base.index_fill_(dim, index, value);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        torch::Tensor lazy_result =
            lazy_base.index_fill_(dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
        AllEqual(base, lazy_base);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexFillRank0) {
  torch::Tensor index = torch::scalar_tensor(
      2, torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {3, 4, 5},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    torch::Tensor value = torch::scalar_tensor(
        42, torch::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_base = CopyToDevice(base, device);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        torch::Tensor lazy_result =
            torch::index_fill(lazy_base, dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexAdd) {
  int index_size = 10;
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (torch::ScalarType index_scalar_type : {torch::kInt, torch::kLong}) {
        torch::Tensor index = torch::randint(
            0,
            base.size(dim),
            {index_size},
            torch::TensorOptions(index_scalar_type).device(DefaultDevice()));
        std::vector<int64_t> value_sizes(
            base.sizes().begin(), base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        torch::Tensor value = isFloatingType(scalar_type)
            ? torch::rand(
                  value_sizes,
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  value_sizes,
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        torch::Tensor result = torch::index_add(base, dim, index, value);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_base = CopyToDevice(base, device);
          torch::Tensor lazy_index = CopyToDevice(index, device);
          torch::Tensor lazy_value = CopyToDevice(value, device);
          torch::Tensor lazy_result =
              torch::index_add(lazy_base, dim, lazy_index, lazy_value);
          AllClose(result, lazy_result);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestIndexAddInPlace) {
  int index_size = 10;
  int rank = 3;
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor base = isFloatingType(scalar_type)
            ? torch::rand(
                  {5, 3, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  {5, 3, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        torch::Tensor index = torch::randint(
            0,
            base.size(dim),
            {index_size},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));
        std::vector<int64_t> value_sizes(
            base.sizes().begin(), base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        torch::Tensor value = isFloatingType(scalar_type)
            ? torch::rand(
                  value_sizes,
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  value_sizes,
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
        torch::Tensor result = base.index_add_(dim, index, value);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        torch::Tensor lazy_result =
            lazy_base.index_add_(dim, lazy_index, lazy_value);
        AllClose(result, lazy_result);
        AllClose(base, lazy_base);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexAddRank0) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor index = torch::randint(
          0,
          base.size(dim),
          at::IntArrayRef{},
          torch::TensorOptions(torch::kLong).device(DefaultDevice()));
      std::vector<int64_t> value_sizes(
          base.sizes().begin(), base.sizes().end());
      int canonical_dim = dim < 0 ? dim + rank : dim;
      value_sizes[canonical_dim] = 1;
      torch::Tensor value = isFloatingType(scalar_type)
          ? torch::rand(
                value_sizes,
                torch::TensorOptions(scalar_type).device(DefaultDevice()))
          : torch::randint(
                100,
                value_sizes,
                torch::TensorOptions(scalar_type).device(DefaultDevice()));
      torch::Tensor result = torch::index_add(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_base = CopyToDevice(base, device);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        torch::Tensor lazy_result =
            torch::index_add(lazy_base, dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexCopy) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor index = torch::randperm(
          base.size(dim),
          torch::TensorOptions(torch::kLong).device(DefaultDevice()));
      torch::Tensor value = isFloatingType(scalar_type)
          ? torch::rand(
                base.sizes(),
                torch::TensorOptions(scalar_type).device(DefaultDevice()))
          : torch::randint(
                100,
                base.sizes(),
                torch::TensorOptions(scalar_type).device(DefaultDevice()));
      torch::Tensor result = torch::index_copy(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_base = CopyToDevice(base, device);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        torch::Tensor lazy_result =
            torch::index_copy(lazy_base, dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexCopyInPlace) {
  if (IsCuda()) {
    GTEST_SKIP();
  }
  int index_size = 10;
  int rank = 3;
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor base = isFloatingType(scalar_type)
            ? torch::rand(
                  {5, 3, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  {5, 3, 7},
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        torch::Tensor index = torch::randint(
            0,
            base.size(dim),
            {index_size},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));
        std::vector<int64_t> value_sizes(
            base.sizes().begin(), base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        torch::Tensor value = isFloatingType(scalar_type)
            ? torch::rand(
                  value_sizes,
                  torch::TensorOptions(scalar_type).device(DefaultDevice()))
            : torch::randint(
                  100,
                  value_sizes,
                  torch::TensorOptions(scalar_type).device(DefaultDevice()));
        torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
        torch::Tensor result = base.index_copy_(dim, index, value);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        torch::Tensor lazy_result =
            lazy_base.index_copy_(dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
        AllEqual(base, lazy_base);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestIndexCopyRank0) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor base = isFloatingType(scalar_type)
        ? torch::rand(
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()))
        : torch::randint(
              100,
              {5, 3, 7},
              torch::TensorOptions(scalar_type).device(DefaultDevice()));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor index = torch::randint(
          0,
          base.size(dim),
          at::IntArrayRef{},
          torch::TensorOptions(torch::kLong).device(DefaultDevice()));
      std::vector<int64_t> value_sizes(
          base.sizes().begin(), base.sizes().end());
      int canonical_dim = dim < 0 ? dim + rank : dim;
      value_sizes[canonical_dim] = 1;
      torch::Tensor value = isFloatingType(scalar_type)
          ? torch::rand(
                value_sizes,
                torch::TensorOptions(scalar_type).device(DefaultDevice()))
          : torch::randint(
                100,
                value_sizes,
                torch::TensorOptions(scalar_type).device(DefaultDevice()));
      torch::Tensor result = torch::index_copy(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_base = CopyToDevice(base, device);
        torch::Tensor lazy_index = CopyToDevice(index, device);
        torch::Tensor lazy_value = CopyToDevice(value, device);
        torch::Tensor lazy_result =
            torch::index_copy(lazy_base, dim, lazy_index, lazy_value);
        AllEqual(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestRelu) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::relu(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::relu(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReluInPlace) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = torch::relu_(input);
    torch::Tensor lazy_output = torch::relu_(lazy_input);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestHardshrink) {
  torch::Tensor input = torch::randn(
      {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::hardshrink(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::hardshrink(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardSigmoid) {
  torch::Tensor input = torch::randn(
      {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::hardsigmoid(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::hardsigmoid(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardSigmoidInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input = torch::randn(
        {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = torch::hardsigmoid_(input);
    torch::Tensor lazy_output = torch::hardsigmoid_(lazy_input);
    AllClose(input, lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardSigmoidBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardsigmoid(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn(
            {10},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestSoftshrink) {
  torch::Tensor input = torch::randn(
      {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::softshrink(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::softshrink(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardtanh) {
  torch::Tensor input = torch::randn(
      {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::hardtanh(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::hardtanh(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestHardtanhInPlace) {
  torch::Tensor input = torch::randn(
      {10}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = torch::hardtanh_(input);
    torch::Tensor lazy_output = torch::hardtanh_(lazy_input);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestLeakyRelu) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double negative_slope = 0.01;
  torch::Tensor output = torch::leaky_relu(input, negative_slope);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::leaky_relu(lazy_input, negative_slope);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestLeakyReluInPlace) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double negative_slope = 0.01;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = torch::leaky_relu_(input, negative_slope);
    torch::Tensor lazy_output = torch::leaky_relu_(lazy_input, negative_slope);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestExp) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::exp(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::exp(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestExpm1) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::expm1(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::expm1(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::log(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::log(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog2) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::log2(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::log2(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog10) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::log10(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::log10(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLog1p) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::log1p(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::log1p(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestErf) {
  torch::Tensor a = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::erf(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::erf(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestErfc) {
  torch::Tensor a = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::erfc(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::erfc(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestErfinv) {
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::erfinv(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::erfinv(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestSqrt) {
  torch::Tensor a = torch::abs(torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  torch::Tensor b = torch::sqrt(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::sqrt(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestRsqrt) {
  torch::Tensor a = torch::abs(torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  torch::Tensor b = torch::rsqrt(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::rsqrt(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestReciprocal) {
  torch::Tensor a = torch::randn(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::reciprocal(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::reciprocal(lazy_a);
    AllClose(b, lazy_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorScalar) {
  torch::Tensor base = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar exponent = 4.09;
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_base = CopyToDevice(base, device);
    torch::Tensor lazy_result = torch::pow(lazy_base, exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorScalarInPlace) {
  torch::Tensor base = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar exponent = 4.09;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
    torch::Tensor result = base.pow_(exponent);
    torch::Tensor lazy_result = lazy_base.pow_(exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(base, lazy_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorTensor) {
  torch::Tensor base = torch::abs(torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  torch::Tensor exponent = torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_base = CopyToDevice(base, device);
    torch::Tensor lazy_exponent = CopyToDevice(exponent, device);
    torch::Tensor lazy_result = torch::pow(lazy_base, lazy_exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorTensorInPlace) {
  torch::Tensor base = torch::abs(torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  torch::Tensor exponent = torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_base = CopyToDevice(base.clone(), device);
    torch::Tensor result = base.pow_(exponent);
    torch::Tensor lazy_exponent = CopyToDevice(exponent, device);
    torch::Tensor lazy_result = lazy_base.pow_(lazy_exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(base, lazy_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowTensorTensorBroadcast) {
  torch::Tensor base = torch::abs(torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  torch::Tensor exponent = torch::rand(
      {4, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_base = CopyToDevice(base, device);
    torch::Tensor lazy_exponent = CopyToDevice(exponent, device);
    torch::Tensor lazy_result = torch::pow(lazy_base, lazy_exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowScalarTensor) {
  torch::Scalar base = 3.5;
  torch::Tensor exponent = torch::rand({4, 2});
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_exponent = CopyToDevice(exponent, device);
    torch::Tensor lazy_result = torch::pow(base, lazy_exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestPowIntExponent) {
  torch::Tensor base = torch::abs(torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())));
  torch::Scalar exponent = 3;
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_base = CopyToDevice(base, device);
    torch::Tensor lazy_result = torch::pow(lazy_base, exponent);
    AllClose(result, lazy_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestFmodScalar) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Scalar divisor = 2.0;
  torch::Tensor b = torch::fmod(a, divisor);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::fmod(lazy_a, divisor);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestFmodScalarInPlace) {
  torch::Scalar divisor = 2.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::rand(
            {2, 2},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
        100.0;
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = a.fmod_(divisor);
    torch::Tensor lazy_b = lazy_a.fmod_(divisor);
    AllClose(b, lazy_b);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestFmodTensor) {
  torch::Tensor a =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      10.0;
  torch::Tensor c = torch::fmod(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::fmod(lazy_a, lazy_b);
    AllClose(c, lazy_c);
  });
}

TEST_F(LazyOpsTest, TestFmodTensorInPlace) {
  torch::Tensor b =
      torch::rand(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      10.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::rand(
            {2, 2},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
        100.0;
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor c = a.fmod_(b);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = lazy_a.fmod_(lazy_b);
    AllClose(c, lazy_c);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestRemainderScalar) {
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Scalar divisor = -2.0;
  torch::Tensor b = torch::remainder(a, divisor);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = torch::remainder(lazy_a, divisor);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestRemainderScalarInPlace) {
  torch::Scalar divisor = -2.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::randn(
            {2, 2},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
        100.0;
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor b = a.remainder_(divisor);
    torch::Tensor lazy_b = lazy_a.remainder_(divisor);
    AllClose(b, lazy_b);
    AllClose(a, lazy_a);
  });
}

TEST_F(LazyOpsTest, TestRemainderTensor) {
  torch::Tensor a =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      100.0;
  torch::Tensor b =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      10.0;
  torch::Tensor c = torch::remainder(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = torch::remainder(lazy_a, lazy_b);
    AllClose(c, lazy_c, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

TEST_F(LazyOpsTest, TestRemainderTensorInPlace) {
  torch::Tensor b =
      torch::randn(
          {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
      10.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::randn(
            {2, 2},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice())) *
        100.0;
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor c = a.remainder_(b);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = lazy_a.remainder_(lazy_b);
    AllClose(c, lazy_c, /*rtol=*/1e-4, /*atol=*/1e-6);
    AllClose(a, lazy_a, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

TEST_F(LazyOpsTest, TestWhere) {
  torch::Tensor a = torch::rand(
      {3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::empty(
      {3, 3}, torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  torch::Tensor d = torch::where(c, a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    torch::Tensor lazy_d = torch::where(lazy_c, lazy_a, lazy_b);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestWhereBroadcast) {
  torch::Tensor a = torch::rand(
      {3, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::zeros(
      {}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::empty(
      {3, 3}, torch::TensorOptions(torch::kByte).device(DefaultDevice()));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  torch::Tensor d = torch::where(c, a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    torch::Tensor lazy_d = torch::where(lazy_c, lazy_a, lazy_b);
    AllClose(d, lazy_d);
  });
}

TEST_F(LazyOpsTest, TestThreshold) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  float threshold = 0.4;
  float value = 20;
  torch::Tensor output = torch::threshold(input, threshold, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::threshold(lazy_input, threshold, value);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestThresholdBackward) {
  float threshold = 0.4;
  float value = 20;

  auto testFunction =
      [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::threshold(inputs[0], threshold, value);
  };

  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 1, 4, 6},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testFunction);
  });
}

TEST_F(LazyOpsTest, TestThresholdInPlace) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = input.clone();
  float threshold = 0.4;
  float value = 20;
  torch::threshold_(output, threshold, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_output = CopyToDevice(input, device);
    torch::threshold_(lazy_output, threshold, value);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestElu) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar alpha = 0.5;
  torch::Scalar scale = 2.5;
  torch::Scalar input_scale = 1.5;
  torch::Tensor output = torch::elu(input, alpha, scale, input_scale);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output =
        torch::elu(lazy_input, alpha, scale, input_scale);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestEluInPlace) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar alpha = 0.5;
  torch::Scalar scale = 2.5;
  torch::Scalar input_scale = 1.5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = torch::elu_(input, alpha, scale, input_scale);
    torch::Tensor lazy_output =
        torch::elu_(lazy_input, alpha, scale, input_scale);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestSelu) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::selu(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::selu(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestSeluInPlace) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = torch::selu_(input);
    torch::Tensor lazy_output = torch::selu_(lazy_input);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestCelu) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar alpha = 2.5;
  torch::Tensor output = torch::celu(input, alpha);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::celu(lazy_input, alpha);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestCeluInPlace) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar alpha = 2.5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = torch::celu_(input, alpha);
    torch::Tensor lazy_output = torch::celu_(lazy_input, alpha);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestGelu) {
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::gelu(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::gelu(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestAddMatMul) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  torch::Tensor input = torch::rand(
      {in_channels, out_channels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {out_channels, labels},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor bias = torch::rand(
      {labels}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test beta != 1. through the CPU interop.
  for (double beta : {1., 2.}) {
    torch::Tensor output = torch::addmm(bias, input, weight, /*beta=*/beta);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_weight = CopyToDevice(weight, device);
      torch::Tensor lazy_bias = CopyToDevice(bias, device);
      torch::Tensor lazy_output =
          torch::addmm(lazy_bias, lazy_input, lazy_weight, /*beta=*/beta);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestEmbedding) {
  torch::Tensor a = torch::rand(
      {32, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor i = torch::randint(
      0,
      31,
      {3, 4},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Tensor b = torch::embedding(
      a,
      i,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_i = CopyToDevice(i, device);
    torch::Tensor lazy_b = torch::embedding(
        lazy_a,
        lazy_i,
        /*padding_idx=*/0,
        /*scale_grad_by_freq=*/false,
        /*sparse=*/false);
    AllClose(b, lazy_b);
  });
}

TEST_F(LazyOpsTest, TestOneHot) {
  int num_classes = 5;
  torch::Tensor input = torch::randint(
      0,
      num_classes,
      {10},
      torch::TensorOptions(torch::kLong).device(DefaultDevice()));
  torch::Tensor output = torch::one_hot(input, num_classes);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::one_hot(lazy_input, num_classes);
    AllEqual(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTranspose) {
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::t(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::t(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTransposeInPlace) {
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = input.t_();
    torch::Tensor lazy_output = lazy_input.t_();
    EXPECT_EQ(lazy_output.sizes(), output.sizes());
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestReshape) {
  torch::Tensor input = torch::rand(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::reshape(input, {-1, 320});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::reshape(lazy_input, {-1, 320});
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestResize) {
  // Testing a resize_() with target size bigger than original size is not
  // possible, as we fill with zeros, while pytorch fills with random garbage.
  torch::Tensor input = torch::rand(
      {2, 2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor saved_input = input.clone();
  input.resize_({3, 3});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(saved_input, device);
    lazy_input.resize_({3, 3});
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestViewResize) {
  torch::Tensor input = torch::zeros(
      {8, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor saved_input = input.clone();
  torch::Tensor output = input.view({4, 4});
  output.resize_({3, 3});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(saved_input, device);
    torch::Tensor lazy_output = lazy_input.view({4, 4});
    lazy_output.resize_({3, 3});
    AllClose(input, lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestView) {
  torch::Tensor input = torch::rand(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = input.view({-1, 320});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = lazy_input.view({-1, 320});
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestViewMod) {
  torch::Tensor input = torch::zeros(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor one = torch::tensor(
      1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = input.view({-1, 320});
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput = torch::zeros(
        {32, 20, 4, 4},
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_input = CopyToDevice(xinput, device);
    torch::Tensor lazy_one = CopyToDevice(one, device);
    torch::Tensor lazy_output = lazy_input.view({-1, 320});
    lazy_output.add_(lazy_one, 1.0);
    lazy_input.add_(lazy_one, 1.0);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestViewModComplex) {
  torch::Tensor input = torch::zeros(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor one = torch::tensor(
      1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  torch::Tensor output2 = input.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput = torch::zeros(
        {32, 20, 4, 4},
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_input = CopyToDevice(xinput, device);
    torch::Tensor lazy_one = CopyToDevice(one, device);
    torch::Tensor lazy_output1 = lazy_input.view({-1, 320});
    lazy_output1.add_(lazy_one, 1.0);
    torch::Tensor lazy_output2 = lazy_input.view({-1, 160});
    lazy_output2.add_(lazy_one, 1.0);
    AllClose(output1, lazy_output1);
    AllClose(output2, lazy_output2);
  });
}

TEST_F(LazyOpsTest, TestViewOfViewMod) {
  torch::Tensor input = torch::zeros(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor one = torch::tensor(
      1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  torch::Tensor output2 = output1.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput = torch::zeros(
        {32, 20, 4, 4},
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_input = CopyToDevice(xinput, device);
    torch::Tensor lazy_one = CopyToDevice(one, device);
    torch::Tensor lazy_output1 = lazy_input.view({-1, 320});
    lazy_output1.add_(lazy_one, 1.0);
    torch::Tensor lazy_output2 = lazy_output1.view({-1, 160});
    lazy_output2.add_(lazy_one, 1.0);
    AllClose(output1, lazy_output1);
    AllClose(output2, lazy_output2);
  });
}

TEST_F(LazyOpsTest, TestViewSqueezeAddInPlace) {
  torch::Tensor input = torch::zeros(
      {2, 3, 1}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> view_size = {2, 3, 1, 1};
  int squeeze_dim = 2;
  torch::Tensor one = torch::tensor(
      1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = input.view(view_size);
    output.squeeze_(squeeze_dim);
    output.add_(one, 1.0);
    torch::Tensor lazy_one = CopyToDevice(one, device);
    torch::Tensor lazy_output = lazy_input.view(view_size);
    lazy_output.squeeze_(squeeze_dim);
    lazy_output.add_(lazy_one, 1.0);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestUnsafeView) {
  torch::Tensor input = torch::rand(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::_unsafe_view(input, {-1, 320});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::_unsafe_view(lazy_input, {-1, 320});
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestNarrow) {
  torch::Tensor a = torch::rand(
      {8, 10, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int64_t dim : {1, -3}) {
    for (int64_t start : {2, -8}) {
      torch::Tensor b = a.narrow(dim, start, 6);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a, device);
        torch::Tensor lazy_b = lazy_a.narrow(dim, start, 6);
        AllClose(b, lazy_b);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowUpdate) {
  for (int64_t dim : {1, -2}) {
    for (int64_t start : {2, -6}) {
      torch::Tensor a = torch::rand(
          {3, 8, 3},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor a_copy = a.clone();
      torch::Tensor b = torch::rand(
          {3, 4, 3},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor c = a.narrow(dim, start, 4);
      c.add_(b, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a_copy, device);
        torch::Tensor lazy_b = CopyToDevice(b, device);
        torch::Tensor lazy_c = lazy_a.narrow(dim, start, 4);
        lazy_c.add_(lazy_b, 1.0);
        AllClose(c, lazy_c);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowUpdateBaseCheck) {
  for (int64_t dim : {0, -2}) {
    for (int64_t start : {2, -6}) {
      torch::Tensor a = torch::zeros(
          {8, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor a_copy = a.clone();
      torch::Tensor b = torch::ones(
          {4, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor c = a.narrow(dim, start, 4);
      c.add_(b, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a_copy, device);
        torch::Tensor lazy_b = CopyToDevice(b, device);
        torch::Tensor lazy_c = lazy_a.narrow(dim, start, 4);
        lazy_c.add_(lazy_b, 1.0);
        AllClose(a, lazy_a);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowUpdateTwoSlices) {
  for (int64_t dim : {0, -2}) {
    for (int64_t start0 : {2, -6}) {
      for (int64_t start1 : {6, -2}) {
        torch::Tensor a = torch::zeros(
            {8, 3},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        torch::Tensor a_copy = a.clone();
        torch::Tensor b = torch::ones(
            {2, 3},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        torch::Tensor c = b + 1;
        torch::Tensor d = a.narrow(dim, start0, 2);
        torch::Tensor e = a.narrow(dim, start1, 2);
        d.add_(b, 1.0);
        e.add_(c, 1.0);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a_copy, device);
          torch::Tensor lazy_b = CopyToDevice(b, device);
          torch::Tensor lazy_c = CopyToDevice(c, device);
          torch::Tensor lazy_d = lazy_a.narrow(dim, start0, 2);
          torch::Tensor lazy_e = lazy_a.narrow(dim, start1, 2);
          lazy_d.add_(lazy_b, 1.0);
          lazy_e.add_(lazy_c, 1.0);
          AllClose(d, lazy_d);
          AllClose(e, lazy_e);
          AllClose(a, lazy_a);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowUpdateView) {
  for (int64_t dim : {0, -3}) {
    for (int64_t start : {2, -6}) {
      torch::Tensor a = torch::rand(
          {8, 2, 3},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor a_copy = a.clone();
      torch::Tensor b = torch::rand(
          {4, 6}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor c = a.narrow(dim, start, 4);
      torch::Tensor d = c.view({4, 6});
      d.add_(b, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_a = CopyToDevice(a_copy, device);
        torch::Tensor lazy_b = CopyToDevice(b, device);
        torch::Tensor lazy_c = lazy_a.narrow(dim, start, 4);
        torch::Tensor lazy_d = lazy_c.view({4, 6});
        lazy_d.add_(lazy_b, 1.0);
        AllClose(d, lazy_d);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowInNarrowUpdate) {
  for (int64_t dim : {1, -2}) {
    for (int64_t start0 : {1, -7}) {
      for (int64_t start1 : {1, -5}) {
        torch::Tensor a = torch::rand(
            {3, 8, 3},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        torch::Tensor a_copy = a.clone();
        torch::Tensor b = torch::rand(
            {3, 2, 3},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        torch::Tensor c = a.narrow(dim, start0, 6);
        torch::Tensor d = c.narrow(dim, start1, 2);
        d.add_(b, 1.0);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor lazy_a = CopyToDevice(a_copy, device);
          torch::Tensor lazy_b = CopyToDevice(b, device);
          torch::Tensor lazy_c = lazy_a.narrow(dim, start0, 6);
          torch::Tensor lazy_d = lazy_c.narrow(dim, start1, 2);
          lazy_d.add_(lazy_b, 1.0);
          AllClose(a, lazy_a);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNarrowCopy) {
  for (int64_t dim : {1, -3}) {
    for (int64_t start : {2, -8}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor input = torch::rand(
            {8, 10, 4, 4},
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor result = input.narrow_copy(dim, start, 6);
        input.add_(1);
        torch::Tensor lazy_result = lazy_input.narrow_copy(dim, start, 6);
        lazy_input.add_(1);
        AllClose(result, lazy_result);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestViewAs) {
  torch::Tensor input = torch::rand(
      {32, 20, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor empty = torch::empty({32, 320});
  torch::Tensor output = input.view_as(empty);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_empty = CopyToDevice(empty, device);
    torch::Tensor lazy_output = lazy_input.view_as(lazy_empty);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestLogSoftmax) {
  torch::Tensor input = torch::rand(
      {5, 3, 4, 2},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::log_softmax(input, dim);
      torch::Tensor lazy_output = torch::log_softmax(lazy_input, dim);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestLogSoftmaxCast) {
  torch::Tensor input = torch::rand(
      {5, 3, 4, 2},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::log_softmax(input, dim, torch::kDouble);
      torch::Tensor lazy_output =
          torch::log_softmax(lazy_input, dim, torch::kDouble);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestLogSoftmaxWrapper) {
  torch::Tensor input = torch::rand(
      {10, 2, 6, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output =
          torch::_log_softmax(input, dim, /*half_to_float=*/false);
      torch::Tensor lazy_output =
          torch::_log_softmax(lazy_input, dim, /*half_to_float=*/false);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestSoftmax) {
  torch::Tensor input = torch::rand(
      {10, 2, 6, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::softmax(input, dim);
      torch::Tensor lazy_output = torch::softmax(lazy_input, dim);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestSoftmaxCast) {
  torch::Tensor input = torch::rand(
      {10, 2, 6, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::softmax(input, dim, torch::kDouble);
      torch::Tensor lazy_output =
          torch::softmax(lazy_input, dim, torch::kDouble);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestSoftmaxWrapper) {
  torch::Tensor input = torch::rand(
      {10, 2, 6, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output =
          torch::_softmax(input, dim, /*half_to_float=*/false);
      torch::Tensor lazy_output =
          torch::_softmax(lazy_input, dim, /*half_to_float=*/false);
      AllClose(output, lazy_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(LazyOpsTest, TestSoftplus) {
  torch::Tensor input = torch::rand(
      {2, 1, 4, 6},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::softplus(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::softplus(lazy_input);
    AllClose(output, lazy_output, /*rtol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestMaxPool1D) {
  torch::Tensor input = torch::rand(
      {1, 16, 56}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool1d(
              input,
              /*kernel_size=*/{kernel_size},
              /*stride=*/{stride},
              /*padding=*/{padding},
              /*dilation=*/{dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::max_pool1d(
                lazy_input,
                /*kernel_size=*/{kernel_size},
                /*stride=*/{stride},
                /*padding=*/{padding},
                /*dilation=*/{dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2D) {
  torch::Tensor input = torch::rand(
      {1, 4, 14, 14},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::max_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2DWithIndices) {
  torch::Tensor input = torch::rand(
      {1, 4, 14, 14},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          auto outputs = torch::max_pool2d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            auto lazy_outputs = torch::max_pool2d_with_indices(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(std::get<0>(outputs), std::get<0>(lazy_outputs));
            AllClose(std::get<1>(outputs), std::get<1>(lazy_outputs));
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2DNonSquare) {
  torch::Tensor input = torch::rand(
      {1, 4, 14, 14},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::max_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1},
                /*stride=*/{stride, stride + 1},
                /*padding=*/{padding, padding + 1},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3D) {
  torch::Tensor input = torch::rand(
      {1, 1, 8, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DWithIndices) {
  torch::Tensor input = torch::rand(
      {1, 1, 8, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          auto outputs = torch::max_pool3d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            auto lazy_outputs = torch::max_pool3d_with_indices(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);

            AllClose(std::get<0>(outputs), std::get<0>(lazy_outputs));
            AllClose(std::get<1>(outputs), std::get<1>(lazy_outputs));
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DIncompleteAttributes) {
  torch::Tensor input = torch::rand(
      {1, 1, 8, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{},
              /*padding=*/{padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{},
                /*padding=*/{padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DNonSquare) {
  torch::Tensor input = torch::rand(
      {1, 1, 8, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
              /*stride=*/{stride, stride + 1, stride},
              /*padding=*/{padding, padding + 1, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
                /*stride=*/{stride, stride + 1, stride},
                /*padding=*/{padding, padding + 1, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2DNoBatch) {
  torch::Tensor input = torch::rand(
      {4, 14, 14}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::max_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DNoBatch) {
  torch::Tensor input = torch::rand(
      {1, 8, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::max_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool1D) {
  torch::Tensor input = torch::rand(
      {4, 1, 28}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool1d(
              input,
              /*kernel_size=*/{kernel_size},
              /*stride=*/{stride},
              /*padding=*/{padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::avg_pool1d(
                lazy_input,
                /*kernel_size=*/{kernel_size},
                /*stride=*/{stride},
                /*padding=*/{padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool2D) {
  torch::Tensor input = torch::rand(
      {2, 1, 14, 14},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            // torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::avg_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output.to(torch::kCPU));
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool2DNonSquare) {
  torch::Tensor input = torch::rand(
      {2, 1, 14, 14},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::avg_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1},
                /*stride=*/{stride, stride + 1},
                /*padding=*/{padding, padding + 1},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3D) {
  torch::Tensor input = torch::rand(
      {1, 1, 7, 7, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DIncompleteAttributes) {
  torch::Tensor input = torch::rand(
      {1, 1, 7, 7, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{},
              /*padding=*/{padding, padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DNonSquare) {
  torch::Tensor input = torch::rand(
      {1, 1, 7, 7, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
              /*stride=*/{stride, stride + 1, stride},
              /*padding=*/{padding, padding + 1, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
                /*stride=*/{stride, stride + 1, stride},
                /*padding=*/{padding, padding + 1, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool2DNoBatch) {
  torch::Tensor input = torch::rand(
      {1, 7, 7}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool2d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::avg_pool2d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DNoBatch) {
  torch::Tensor input = torch::rand(
      {1, 7, 7, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool3d(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output = torch::avg_pool3d(
                lazy_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2D) {
  torch::Tensor input = torch::rand(
      {4, 1, 28, 28},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int64_t output_size : {7, 4}) {
    torch::Tensor output =
        torch::adaptive_avg_pool2d(input, {output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output =
          torch::adaptive_avg_pool2d(lazy_input, {output_size, output_size});
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool3D) {
  torch::Tensor input = torch::rand(
      {9, 4, 56, 28, 28},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int64_t output_size : {7, 4}) {
    torch::Tensor output = torch::adaptive_avg_pool3d(
        input, {output_size, output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::adaptive_avg_pool3d(
          lazy_input, {output_size, output_size, output_size});
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool3DNoBatch) {
  torch::Tensor input = torch::rand(
      {3, 56, 28, 28},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int64_t output_size : {7, 4}) {
    torch::Tensor output = torch::adaptive_avg_pool3d(
        input, {output_size, output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::adaptive_avg_pool3d(
          lazy_input, {output_size, output_size, output_size});
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2DNoBatch) {
  torch::Tensor input = torch::rand(
      {1, 56, 56}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int64_t output_size : {7, 8}) {
    torch::Tensor output =
        torch::adaptive_avg_pool2d(input, {output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output =
          torch::adaptive_avg_pool2d(lazy_input, {output_size, output_size});
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestMaxUnpool2D) {
  int kernel_size = 2;
  torch::Tensor input = torch::rand(
      {2, 2, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool2d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size({input.size(2), input.size(3)});
          at::Tensor utensor = torch::max_unpool2d(
              output,
              indices,
              output_size,
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding});

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_output = CopyToDevice(output, device);
            torch::Tensor lazy_indices = CopyToDevice(indices, device);
            at::Tensor lazy_utensor = torch::max_unpool2d(
                lazy_output,
                lazy_indices,
                output_size,
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding});
            AllClose(utensor, lazy_utensor);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxUnpool3D) {
  int kernel_size = 2;
  torch::Tensor input = torch::rand(
      {1, 1, 4, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool3d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size(
              {input.size(2), input.size(3), input.size(4)});
          at::Tensor utensor = torch::max_unpool3d(
              output,
              indices,
              output_size,
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding});

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_output = CopyToDevice(output, device);
            torch::Tensor lazy_indices = CopyToDevice(indices, device);
            at::Tensor lazy_utensor = torch::max_unpool3d(
                lazy_output,
                lazy_indices,
                output_size,
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding});
            AllClose(utensor, lazy_utensor);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLoss) {
  int batch = 6;
  int classes = 2;
  // TODO(asuhan): Fix the torch::kDouble case.
  for (auto dtype : {torch::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input = torch::rand(
            {batch, classes},
            torch::TensorOptions(dtype).device(DefaultDevice()));
        torch::Tensor target = torch::randint(
            std::min(ignore_index, 0),
            classes,
            {batch},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));
        torch::Tensor weight;
        if (def_weight) {
          weight = torch::rand(
              {classes}, torch::TensorOptions(dtype).device(DefaultDevice()));
        }
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean,
              torch::Reduction::Sum,
              torch::Reduction::None}) {
          torch::Tensor output = torch::nll_loss(
              /*self=*/input,
              /*target=*/target,
              /*weight=*/weight,
              /*reduction=*/reduction,
              /*ignore_index=*/ignore_index);

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_target = CopyToDevice(target, device);
            torch::Tensor lazy_weight =
                def_weight ? CopyToDevice(weight, device) : torch::Tensor();
            torch::Tensor lazy_output = torch::nll_loss(
                /*self=*/lazy_input,
                /*target=*/lazy_target,
                /*weight=*/lazy_weight,
                /*reduction=*/reduction,
                /*ignore_index=*/ignore_index);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLoss2d) {
  int batch = 6;
  int classes = 2;
  int height = 3;
  int width = 3;
  // TODO(asuhan): Fix the torch::kDouble case.
  for (auto dtype : {torch::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input = torch::rand(
            {batch, classes, height, width},
            torch::TensorOptions(dtype).device(DefaultDevice()));
        torch::Tensor target = torch::randint(
            std::min(ignore_index, 0),
            classes,
            {batch, height, width},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));
        torch::Tensor weight;
        if (def_weight) {
          weight = torch::rand(
              {classes}, torch::TensorOptions(dtype).device(DefaultDevice()));
        }
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean,
              torch::Reduction::Sum,
              torch::Reduction::None}) {
          torch::Tensor output = torch::nll_loss2d(
              /*self=*/input,
              /*target=*/target,
              /*weight=*/weight,
              /*reduction=*/reduction,
              /*ignore_index=*/ignore_index);

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_target = CopyToDevice(target, device);
            torch::Tensor lazy_weight =
                def_weight ? CopyToDevice(weight, device) : torch::Tensor();
            torch::Tensor lazy_output = torch::nll_loss2d(
                /*self=*/lazy_input,
                /*target=*/lazy_target,
                /*weight=*/lazy_weight,
                /*reduction=*/reduction,
                /*ignore_index=*/ignore_index);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSmoothL1Loss) {
  torch::Tensor input = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    for (double beta : {0.25, 1.}) {
      torch::Tensor output =
          torch::smooth_l1_loss(input, target, reduction, beta);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_target = CopyToDevice(target, device);
        torch::Tensor lazy_output =
            torch::smooth_l1_loss(lazy_input, lazy_target, reduction, beta);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestL1Loss) {
  torch::Tensor input = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    torch::Tensor output = torch::l1_loss(input, target, reduction);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_target = CopyToDevice(target, device);
      torch::Tensor lazy_output =
          torch::l1_loss(lazy_input, lazy_target, reduction);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestL1LossBackward) {
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::l1_loss(inputs[0], inputs[1], reduction);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
               {2, 4},
               torch::TensorOptions(torch::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           torch::rand(
               {2, 4},
               torch::TensorOptions(torch::kFloat).device(DefaultDevice()))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestMseLoss) {
  torch::Tensor input = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor target = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    torch::Tensor output = torch::mse_loss(input, target, reduction);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_target = CopyToDevice(target, device);
      torch::Tensor lazy_output =
          torch::mse_loss(lazy_input, lazy_target, reduction);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestMseLossBackward) {
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::mse_loss(inputs[0], inputs[1], reduction);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
               {2, 4},
               torch::TensorOptions(torch::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           torch::rand(
               {2, 4},
               torch::TensorOptions(torch::kFloat).device(DefaultDevice()))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestBatchNorm1D) {
  int num_features = 3;
  torch::Tensor input = torch::rand(
      {2, num_features, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor bias = torch::rand(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor running_mean = torch::zeros(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor running_var = torch::ones(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double momentum = 0.1;
  double eps = 0.5;
  torch::Tensor undef;
  for (bool training : {true, false}) {
    for (bool undef_weight_bias : {false, true}) {
      torch::Tensor output = torch::batch_norm(
          /*input=*/input,
          /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean,
          /*running_var=*/running_var,
          /*training=*/training,
          /*momentum=*/momentum,
          /*eps=*/eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_weight =
            undef_weight_bias ? undef : CopyToDevice(weight, device);
        torch::Tensor lazy_bias =
            undef_weight_bias ? undef : CopyToDevice(bias, device);
        torch::Tensor lazy_running_mean = CopyToDevice(running_mean, device);
        torch::Tensor lazy_running_var = CopyToDevice(running_var, device);
        torch::Tensor lazy_output = torch::batch_norm(
            /*input=*/lazy_input,
            /*weight=*/lazy_weight,
            /*bias=*/lazy_bias,
            /*running_mean=*/lazy_running_mean,
            /*running_var=*/lazy_running_var,
            /*training=*/training,
            /*momentum=*/momentum,
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
        AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestBatchNorm2D) {
  int num_features = 3;
  torch::Tensor input = torch::rand(
      {2, num_features, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor bias = torch::rand(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor running_mean = torch::zeros(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor running_var = torch::ones(
      {num_features},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  double momentum = 0.1;
  double eps = 0.5;
  torch::Tensor undef;
  for (bool training : {true, false}) {
    for (bool undef_weight_bias : {false, true}) {
      torch::Tensor output = torch::batch_norm(
          /*input=*/input,
          /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean,
          /*running_var=*/running_var,
          /*training=*/training,
          /*momentum=*/momentum,
          /*eps=*/eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_weight =
            undef_weight_bias ? undef : CopyToDevice(weight, device);
        torch::Tensor lazy_bias =
            undef_weight_bias ? undef : CopyToDevice(bias, device);
        torch::Tensor lazy_running_mean = CopyToDevice(running_mean, device);
        torch::Tensor lazy_running_var = CopyToDevice(running_var, device);
        torch::Tensor lazy_output = torch::batch_norm(
            /*input=*/lazy_input,
            /*weight=*/lazy_weight,
            /*bias=*/lazy_bias,
            /*running_mean=*/lazy_running_mean,
            /*running_var=*/lazy_running_var,
            /*training=*/training,
            /*momentum=*/momentum,
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
        AllClose(output, lazy_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestDim) {
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    EXPECT_EQ(input.dim(), lazy_input.dim());
  });
}

TEST_F(LazyOpsTest, TestContiguous) {
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::native::contiguous(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::native::contiguous(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestSqueezeAll) {
  torch::Tensor input = torch::rand(
      {2, 1, 3, 1},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::squeeze(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::squeeze(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestSqueezeAllInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input = torch::rand(
        {2, 1, 3, 1},
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = input.squeeze_();
    torch::Tensor lazy_output = lazy_input.squeeze_();
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
    ASSERT_EQ(input.dim(), lazy_input.dim());
    for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
      ASSERT_EQ(input.size(dim_idx), lazy_input.size(dim_idx));
    }
  });
}

TEST_F(LazyOpsTest, TestSqueezeOne) {
  torch::Tensor input = torch::rand(
      {2, 1, 3, 1},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor output = torch::squeeze(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::squeeze(lazy_input, dim);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestSqueezeOneInPlace) {
  int rank = 4;
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input = torch::rand(
          {2, 1, 3, 1},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor output = input.squeeze_(dim);
      torch::Tensor lazy_output = lazy_input.squeeze_(dim);
      AllClose(output, lazy_output);
      AllClose(input, lazy_input);
      ASSERT_EQ(input.dim(), lazy_input.dim());
      for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
        ASSERT_EQ(input.size(dim_idx), lazy_input.size(dim_idx));
      }
    });
  }
}

TEST_F(LazyOpsTest, TestUnsqueeze) {
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor output = torch::unsqueeze(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::unsqueeze(lazy_input, dim);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestUnsqueezeInPlace) {
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor output = input.unsqueeze_(dim);
      torch::Tensor lazy_output = lazy_input.unsqueeze_(dim);
      AllClose(output, lazy_output);
      AllClose(input, lazy_input);
      ASSERT_EQ(input.dim(), lazy_input.dim());
      for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
        ASSERT_EQ(input.size(dim_idx), lazy_input.size(dim_idx));
      }
    });
  }
}

TEST_F(LazyOpsTest, TestMaskedFill) {
  torch::Tensor input = torch::rand(
      {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor mask = torch::randint(
      0, 2, {2, 3}, torch::TensorOptions(torch::kBool).device(DefaultDevice()));
  torch::Scalar value(42);
  torch::Tensor result = torch::masked_fill(input, mask, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_mask = CopyToDevice(mask, device);
    torch::Tensor lazy_result =
        torch::masked_fill(lazy_input, lazy_mask, value);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestMaskedFillInPlace) {
  torch::Scalar value(42);
  torch::Tensor mask = torch::randint(
      0, 2, {2, 3}, torch::TensorOptions(torch::kBool).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input = torch::rand(
        {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_mask = CopyToDevice(mask, device);
    torch::Tensor result = input.masked_fill_(mask, value);
    torch::Tensor lazy_result = lazy_input.masked_fill_(lazy_mask, value);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestMaskedFillBroadcast) {
  torch::Tensor input = torch::rand(
      {2, 5, 4, 3},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor mask = torch::randint(
      0, 2, {4, 1}, torch::TensorOptions(torch::kBool).device(DefaultDevice()));
  torch::Scalar value(42);
  torch::Tensor result = torch::masked_fill(input, mask, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_mask = CopyToDevice(mask, device);
    torch::Tensor lazy_result =
        torch::masked_fill(lazy_input, lazy_mask, value);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestFill) {
  torch::Scalar value(42);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input = torch::empty(
        {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor result = torch::fill_(input, value);
    torch::Tensor lazy_result = torch::fill_(lazy_input, value);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestFillWithRank0) {
  torch::Tensor value = torch::scalar_tensor(42);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input = torch::empty(
        {2, 3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor result = torch::fill_(input, value);
    torch::Tensor lazy_value = CopyToDevice(value, device);
    torch::Tensor lazy_result = torch::fill_(lazy_input, value);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestPermute) {
  torch::Tensor input = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  int rank = input.dim();
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(
            dims_permutation.begin(),
            dims_permutation.end(),
            [rank](int64_t& dim) { dim -= rank; });
      }
      torch::Tensor output = input.permute(dims_permutation);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_output = lazy_input.permute(dims_permutation);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestPermuteMod) {
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  std::vector<int64_t> input_sizes = {2, 3, 4};
  int rank = input_sizes.size();
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(
            dims_permutation.begin(),
            dims_permutation.end(),
            [rank](int64_t& dim) { dim -= rank; });
      }
      torch::Tensor input = torch::zeros(
          input_sizes,
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor one = torch::tensor(
          1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor output = input.permute(dims_permutation);
      output.add_(one, 1.0);
      input.add_(one, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xinput = torch::zeros(
            input_sizes,
            torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        torch::Tensor lazy_input = CopyToDevice(xinput, device);
        torch::Tensor lazy_one = CopyToDevice(one, device);
        torch::Tensor lazy_output = lazy_input.permute(dims_permutation);
        lazy_output.add_(lazy_one, 1.0);
        lazy_input.add_(lazy_one, 1.0);
        AllClose(output, lazy_output);
        AllClose(input, lazy_input);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestFlip) {
  torch::Tensor input = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<std::vector<int64_t>> dim_powerset = {
      {0}, {1}, {2}, {0, 1}, {1, 2}, {2, 0}, {0, 1, 2}};
  for (std::vector<int64_t> flip_dims : dim_powerset) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(
            flip_dims.begin(), flip_dims.end(), [](int64_t& dim) { dim -= 3; });
      }
      torch::Tensor output = torch::flip(input, flip_dims);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        torch::Tensor lazy_output = torch::flip(lazy_input, flip_dims);
        AllClose(output, lazy_output);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestPixelShuffle) {
  torch::Tensor input = torch::rand(
      {5, 18, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int upscale_factor = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = torch::pixel_shuffle(input, upscale_factor);
    torch::Tensor lazy_output =
        torch::pixel_shuffle(lazy_input, upscale_factor);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestSumToSize) {
  torch::Tensor input = torch::rand(
      {4, 6, 3, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> out_size = {4, 1, 1, 7};
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = input.sum_to_size(out_size);
    torch::Tensor lazy_output = lazy_input.sum_to_size(out_size);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTransposeDims) {
  torch::Tensor input = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int dim0 = 0;
  int dim1 = 2;
  torch::Tensor output = torch::transpose(input, dim0, dim1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::transpose(lazy_input, dim0, dim1);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTransposeDimsMod) {
  std::vector<int64_t> input_sizes = {2, 3, 4};
  int dim0 = 0;
  int dim1 = 2;
  torch::Tensor input = torch::zeros(
      input_sizes, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor one = torch::tensor(
      1.0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::transpose(input, dim0, dim1);
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput = torch::zeros(
        input_sizes,
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor lazy_input = CopyToDevice(xinput, device);
    torch::Tensor lazy_one = CopyToDevice(one, device);
    torch::Tensor lazy_output = torch::transpose(lazy_input, dim0, dim1);
    lazy_output.add_(lazy_one, 1.0);
    lazy_input.add_(lazy_one, 1.0);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestTransposeDimsInPlace) {
  torch::Tensor input = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int dim0 = 0;
  int dim1 = 2;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output = input.transpose_(dim0, dim1);
    torch::Tensor lazy_output = lazy_input.transpose_(dim0, dim1);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestSplit) {
  torch::Tensor input = torch::rand(
      {7, 8, 9}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int split_size : {2, 3}) {
    for (int dim = -rank; dim < rank; ++dim) {
      std::vector<torch::Tensor> outputs = torch::split(input, split_size, dim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_input = CopyToDevice(input, device);
        std::vector<torch::Tensor> lazy_outputs =
            torch::split(lazy_input, split_size, dim);
        ASSERT_EQ(outputs.size(), lazy_outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
          AllClose(outputs[i], lazy_outputs[i]);
        }
      });
    }
  }
}

TEST_F(LazyOpsTest, TestSplitEmpty) {
  torch::Tensor input = torch::rand(
      {0}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int split_size = 0;
  int dim = 0;
  std::vector<torch::Tensor> outputs = torch::split(input, split_size, dim);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    std::vector<torch::Tensor> lazy_outputs =
        torch::split(lazy_input, split_size, dim);
    ASSERT_EQ(outputs.size(), lazy_outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      AllClose(outputs[i], lazy_outputs[i]);
    }
  });
}

TEST_F(LazyOpsTest, TestSplitWithSizes) {
  torch::Tensor input = torch::rand(
      {15, 15, 15},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<torch::Tensor> outputs =
        torch::split_with_sizes(input, {4, 5, 6}, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      std::vector<torch::Tensor> lazy_outputs =
          torch::split_with_sizes(lazy_input, {4, 5, 6}, dim);
      ASSERT_EQ(outputs.size(), lazy_outputs.size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        AllClose(outputs[i], lazy_outputs[i]);
      }
    });
  }
}

TEST_F(LazyOpsTest, TestCrossImplicitDim) {
  std::vector<std::vector<int64_t>> dim_sizes = {
      {4, 5, 3}, {4, 3, 5}, {3, 4, 5}};
  for (auto dim_size : dim_sizes) {
    torch::Tensor input = torch::rand(
        dim_size, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor other = torch::rand(
        dim_size, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    torch::Tensor result = torch::cross(input, other);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_other = CopyToDevice(other, device);
      torch::Tensor lazy_result = torch::cross(lazy_input, lazy_other);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCrossExplicitDim) {
  std::vector<int64_t> dim_size = {3, 3};
  torch::Tensor input = torch::rand(
      dim_size, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor other = torch::rand(
      dim_size, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  int rank = dim_size.size();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cross(input, other, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_other = CopyToDevice(other, device);
      torch::Tensor lazy_result = torch::cross(lazy_input, lazy_other, dim);
      AllClose(result, lazy_result);
    });
  }
}

TEST_F(LazyOpsTest, TestCrossZeroDim) {
  torch::Tensor input = torch::rand(
      {0, 1, 3, 0},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor result = torch::cross(input, input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::cross(lazy_input, lazy_input);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestTriu) {
  int size = 5;
  torch::Tensor input = torch::rand(
      {size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::triu(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::triu(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTriuNonSquare) {
  int size = 5;
  torch::Tensor input = torch::rand(
      {size, size + 1},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::triu(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::triu(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTriuBatch) {
  int size = 5;
  int batch_size = 3;
  torch::Tensor input = torch::rand(
      {batch_size, size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::triu(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::triu(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTril) {
  int size = 5;
  torch::Tensor input = torch::rand(
      {size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::tril(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::tril(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTrilNonSquare) {
  int size = 5;
  torch::Tensor input = torch::rand(
      {size, size + 1},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::tril(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::tril(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTrilBatch) {
  int size = 5;
  int batch_size = 3;
  torch::Tensor input = torch::rand(
      {batch_size, size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::tril(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::tril(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestTriuInPlace) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input = torch::rand(
          {size, size},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor output = input.triu_(diagonal);
      torch::Tensor lazy_output = lazy_input.triu_(diagonal);
      AllClose(output, lazy_output);
      AllClose(input, lazy_input);
    });
  }
}

TEST_F(LazyOpsTest, TestTrilInPlace) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input = torch::rand(
          {size, size},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor output = input.tril_(diagonal);
      torch::Tensor lazy_output = lazy_input.tril_(diagonal);
      AllClose(output, lazy_output);
      AllClose(input, lazy_input);
    });
  }
}

TEST_F(LazyOpsTest, TestTrace) {
  int n = 5;
  torch::Tensor input = torch::rand(
      {n, n}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::trace(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::trace(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTraceWide) {
  int lines = 3;
  int cols = 5;
  torch::Tensor input = torch::rand(
      {lines, cols},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::trace(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::trace(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestTraceNarrow) {
  int lines = 5;
  int cols = 3;
  torch::Tensor input = torch::rand(
      {lines, cols},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor output = torch::trace(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::trace(lazy_input);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestDiagRank1) {
  int size = 7;
  torch::Tensor input = torch::rand(
      {size}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -2 * size; diagonal <= 2 * size; ++diagonal) {
    torch::Tensor output = torch::diag(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::diag(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagRank2) {
  int size = 7;
  torch::Tensor input = torch::rand(
      {size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::diag(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::diag(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagFlat) {
  torch::Tensor input = torch::rand(
      {4, 3, 6, 7},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int diagonal = -10; diagonal < 10; ++diagonal) {
    torch::Tensor output = torch::diagflat(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::diagflat(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagonal) {
  int size = 5;
  torch::Tensor input = torch::rand(
      {size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::diagonal(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::diagonal(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagonalUpdate) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    auto input = torch::rand(
        {size, size},
        torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
    auto input_clone = input.clone();
    auto output = torch::diagonal(input, diagonal);
    output.add_(1);

    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input_clone, device);
      torch::Tensor lazy_output = torch::diagonal(lazy_input, diagonal);
      lazy_output.add_(1);

      AllClose(output, lazy_output);
      AllClose(input, lazy_input);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagonalNonSquare) {
  int size = 5;
  torch::Tensor input = torch::rand(
      {size, size + 1},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::diagonal(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output = torch::diagonal(lazy_input, diagonal);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestDiagonalBatch) {
  int size = 5;
  int batch_size = 3;
  int dim1 = 1;
  int dim2 = 2;
  torch::Tensor input = torch::rand(
      {batch_size, size, size},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output =
        torch::diagonal(input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor lazy_input = CopyToDevice(input, device);
      torch::Tensor lazy_output =
          torch::diagonal(lazy_input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
      AllClose(output, lazy_output);
    });
  }
}

TEST_F(LazyOpsTest, TestFlatten) {
  torch::Tensor input = torch::rand({4, 7, 5, 3});
  int rank = input.dim();
  for (int pos_start_dim = 0; pos_start_dim < rank; ++pos_start_dim) {
    for (int pos_end_dim = pos_start_dim; pos_end_dim < rank; ++pos_end_dim) {
      for (bool negative_start_dim : {false, true}) {
        for (bool negative_end_dim : {false, true}) {
          int start_dim =
              negative_start_dim ? pos_start_dim - rank : pos_start_dim;
          int end_dim = negative_end_dim ? pos_end_dim - rank : pos_end_dim;
          torch::Tensor output = torch::flatten(input, start_dim, end_dim);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor lazy_input = CopyToDevice(input, device);
            torch::Tensor lazy_output =
                torch::flatten(lazy_input, start_dim, end_dim);
            AllClose(output, lazy_output);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestLogicalAnd) {
  for (torch::ScalarType scalar_type1 :
       {torch::kFloat,
        torch::kByte,
        torch::kChar,
        torch::kShort,
        torch::kInt,
        torch::kLong}) {
    torch::Tensor lhs = isFloatingType(scalar_type1)
        ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
        : torch::randint(0, 100, {3, 4}, torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 :
         {torch::kFloat,
          torch::kByte,
          torch::kChar,
          torch::kShort,
          torch::kInt,
          torch::kLong}) {
      torch::Tensor rhs = isFloatingType(scalar_type2)
          ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
          : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type2));
      torch::Tensor result = torch::logical_and(lhs, rhs);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
        torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
        torch::Tensor lazy_result = torch::logical_and(lazy_lhs, lazy_rhs);
        AllEqual(result, lazy_result);
      });
    }
  }

  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("xla::logical_and_out", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestBitwiseAnd) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    torch::Tensor lazy_result = lazy_lhs.__and__(lazy_rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseAndInPlace) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__iand__(rhs);
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    torch::Tensor lazy_result = lazy_lhs.__iand__(lazy_rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseAndScalar) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  torch::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor lazy_result = lazy_lhs.__and__(rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseAndScalarInPlace) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__iand__(rhs);
    torch::Tensor lazy_result = lazy_lhs.__iand__(rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseAndPromotion) {
  torch::Tensor input = torch::rand(
      {4, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor view = input.reshape(-1);
  torch::Tensor result = torch::__and__(view.gt(0), view.ne(0));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_view = lazy_input.reshape(-1);
    torch::Tensor lazy_result =
        torch::__and__(lazy_view.gt(0), lazy_view.ne(0));
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseOr) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    torch::Tensor lazy_result = lazy_lhs.__or__(lazy_rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseOrInPlace) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ior__(rhs);
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    torch::Tensor lazy_result = lazy_lhs.__ior__(lazy_rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseOrScalar) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  torch::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor lazy_result = lazy_lhs.__or__(rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseOrScalarInPlace) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ior__(rhs);
    torch::Tensor lazy_result = lazy_lhs.__ior__(rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseXor) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    torch::Tensor lazy_result = lazy_lhs.__xor__(lazy_rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseXorInPlace) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ixor__(rhs);
    torch::Tensor lazy_rhs = CopyToDevice(rhs, device);
    torch::Tensor lazy_result = lazy_lhs.__ixor__(lazy_rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestBitwiseXorScalar) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  torch::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor lazy_result = lazy_lhs.__xor__(rhs);
    AllEqual(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestBitwiseXorScalarInPlace) {
  torch::Tensor lhs = torch::randint(
      0,
      std::numeric_limits<int32_t>::max(),
      {4, 2},
      torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ixor__(rhs);
    torch::Tensor lazy_result = lazy_lhs.__ixor__(rhs);
    AllEqual(result, lazy_result);
    AllEqual(lhs, lazy_lhs);
  });
}

TEST_F(LazyOpsTest, TestLshift) {
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor shift_amount = torch::randint(
      16,
      input.sizes(),
      torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor result = torch::__lshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor lazy_result =
        torch::__lshift__(lazy_input, lazy_shift_amount);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLshiftInPlace) {
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor shift_amount = torch::randint(
        16,
        input.sizes(),
        torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
    torch::Tensor result = input.__ilshift__(shift_amount);
    torch::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor lazy_result = lazy_input.__ilshift__(lazy_shift_amount);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestLshiftScalar) {
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Scalar shift_amount = 3;
  torch::Tensor result = torch::__lshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::__lshift__(lazy_input, shift_amount);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestLshiftScalarInPlace) {
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Scalar shift_amount = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor result = input.__ilshift__(shift_amount);
    torch::Tensor lazy_result = lazy_input.__ilshift__(shift_amount);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestRshift) {
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor shift_amount = torch::randint(
      16,
      input.sizes(),
      torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor result = torch::__rshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor lazy_result =
        torch::__rshift__(lazy_input, lazy_shift_amount);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestRshiftInPlace) {
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor shift_amount = torch::randint(
        16,
        input.sizes(),
        torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
    torch::Tensor result = input.__irshift__(shift_amount);
    torch::Tensor lazy_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor lazy_result = lazy_input.__irshift__(lazy_shift_amount);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestRshiftScalar) {
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Scalar shift_amount = 3;
  torch::Tensor result = torch::__rshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_result = torch::__rshift__(lazy_input, shift_amount);
    AllClose(result, lazy_result);
  });
}

TEST_F(LazyOpsTest, TestRshiftScalarInPlace) {
  torch::Tensor input = torch::ones(
      {4, 2}, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Scalar shift_amount = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor result = input.__irshift__(shift_amount);
    torch::Tensor lazy_result = lazy_input.__irshift__(shift_amount);
    AllClose(result, lazy_result);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestMeshgrid) {
  torch::Tensor a = torch::rand(
      {3}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor b = torch::rand(
      {2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor c = torch::rand(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  auto d = torch::meshgrid({a, b, c});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_a = CopyToDevice(a, device);
    torch::Tensor lazy_b = CopyToDevice(b, device);
    torch::Tensor lazy_c = CopyToDevice(c, device);
    auto lazy_d = torch::meshgrid({lazy_a, lazy_b, lazy_c});
    EXPECT_EQ(d.size(), lazy_d.size());
    for (size_t i = 0; i < d.size(); ++i) {
      AllClose(d[i], lazy_d[i]);
    }
  });
}

TEST_F(LazyOpsTest, TestConstantPad) {
  torch::Tensor input = torch::rand(
      {4, 2, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 2, 3, 4, 5, 6};
  float pad_value = 5;
  torch::Tensor output = torch::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output =
        torch::constant_pad_nd(lazy_input, pad, pad_value);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestConstantPadIncomplete) {
  torch::Tensor input = torch::rand(
      {4, 2, 5}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 2};
  float pad_value = 5;
  torch::Tensor output = torch::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output =
        torch::constant_pad_nd(lazy_input, pad, pad_value);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReflectionPad2dRank3) {
  torch::Tensor input = torch::rand(
      {2, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{2, 2, 2, 2};
  torch::Tensor output = torch::reflection_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::reflection_pad2d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReflectionPad2dRank4) {
  torch::Tensor input = torch::rand(
      {2, 2, 3, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{2, 2, 2, 2};
  torch::Tensor output = torch::reflection_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::reflection_pad2d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReflectionPad2dBackward) {
  std::vector<int64_t> pad{2, 3, 1, 2};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::reflection_pad2d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {1, 2, 4, 4},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad1d) {
  torch::Tensor input = torch::rand(
      {1, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 2};
  torch::Tensor output = torch::replication_pad1d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::replication_pad1d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad1dZeroPad) {
  torch::Tensor input = torch::rand(
      {1, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 0};
  torch::Tensor output = torch::replication_pad1d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::replication_pad1d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad1dBackward) {
  std::vector<int64_t> pad{2, 3};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::replication_pad1d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 4},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad2d) {
  torch::Tensor input = torch::rand(
      {1, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 2, 2, 1};
  torch::Tensor output = torch::replication_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::replication_pad2d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad2dZeroPad) {
  torch::Tensor input = torch::rand(
      {1, 3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> pad{1, 0, 0, 1};
  torch::Tensor output = torch::replication_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::replication_pad2d(lazy_input, pad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestReplicationPad2dBackward) {
  std::vector<int64_t> pad{2, 3, 1, 1};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::replication_pad2d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 3, 4},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestAsStrided) {
  torch::Tensor input = torch::rand(
      {128, 320}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> size = {128, 20, 4, 4};
  std::vector<int64_t> stride = {320, 16, 4, 1};
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output =
        torch::as_strided(lazy_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestAsStridedInPlace) {
  torch::Tensor input = torch::rand(
      {128, 320}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> size = {128, 20, 4, 4};
  std::vector<int64_t> stride = {320, 16, 4, 1};
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor output =
        torch::as_strided_(input, /*size=*/size, /*stride=*/stride);
    torch::Tensor lazy_output =
        torch::as_strided_(lazy_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, lazy_output);
    AllClose(input, lazy_input);
  });
}

TEST_F(LazyOpsTest, TestAsStridedWithOffset) {
  torch::Tensor input = torch::rand(
      {4, 8, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> size = {4, 4, 2};
  std::vector<int64_t> stride = {8, 2, 1};
  int64_t storage_offset = 4;
  torch::Tensor output = torch::as_strided(
      input,
      /*size=*/size,
      /*stride=*/stride,
      /*storage_offset=*/storage_offset);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input, device);
    torch::Tensor lazy_output = torch::as_strided(
        lazy_input,
        /*size=*/size,
        /*stride=*/stride,
        /*storage_offset=*/storage_offset);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestAsStridedWithInplaceCopy) {
  torch::Tensor grad = torch::ones(
      {4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  std::vector<int64_t> size = {4};
  std::vector<int64_t> stride = {1};
  torch::Tensor output = torch::zeros({4}, grad.options());
  output.as_strided(size, stride).copy_(grad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_grad = CopyToDevice(grad, device);
    torch::Tensor lazy_output = torch::zeros({4}, lazy_grad.options());
    lazy_output.as_strided(size, stride).copy_(lazy_grad);
    AllClose(output, lazy_output);
  });
}

TEST_F(LazyOpsTest, TestEmptyStrided) {
  std::vector<int64_t> size = {4, 4, 2};
  std::vector<int64_t> stride = {8, 2, 1};
  torch::Tensor output = torch::empty_strided(/*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_output =
        torch::empty_strided(/*size=*/size, /*stride=*/stride);
    EXPECT_EQ(output.sizes(), lazy_output.sizes());
    EXPECT_EQ(output.strides(), lazy_output.strides());
  });
}

TEST_F(LazyOpsTest, TestAvgPool2DBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::avg_pool2d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {torch::rand(
                    {1, 1, 7, 7},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::avg_pool3d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {torch::rand(
                    {1, 1, 7, 7, 7},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool2DNoBatchBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::avg_pool2d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {torch::rand(
                    {1, 7, 7},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAvgPool3DNoBatchBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::avg_pool3d(
                inputs[0],
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {torch::rand(
                    {1, 7, 7, 7},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true))},
                device,
                testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool3DNoBatchBackward) {
  if (IsCuda()) {
    GTEST_SKIP();
  }
  for (int64_t output_size : {7, 4}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool3d(
          inputs[0], {output_size, output_size, output_size});
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {1, 56, 28, 28},
              torch::TensorOptions(torch::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool3DBackward) {
  if (IsCuda()) {
    GTEST_SKIP();
  }
  for (int64_t output_size : {7, 4}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool3d(
          inputs[0], {output_size, output_size, output_size});
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {4, 1, 56, 28, 28},
              torch::TensorOptions(torch::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2DBackward) {
  for (int64_t output_size : {7, 8}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {4, 1, 56, 56},
              torch::TensorOptions(torch::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestAdaptiveAvgPool2DNoBatchBackward) {
  for (int64_t output_size : {7, 8}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {1, 56, 56},
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestConv2D) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 3; ++dilation) {
          for (int groups :
               {1, 2, 4}) { // covers normal, grouped, depthwise conv.
            ForEachDevice([&](const torch::Device& device) {
              torch::Tensor input = torch::rand(
                  {1, in_channels, 7, 7},
                  torch::TensorOptions(torch::kDouble).device(DefaultDevice()));
              torch::Tensor weight = torch::rand(
                  {out_channels,
                   in_channels / groups,
                   kernel_size,
                   kernel_size},
                  torch::TensorOptions(torch::kDouble).device(DefaultDevice()));
              torch::Tensor bias = with_bias
                  ? torch::rand(
                        {out_channels},
                        torch::TensorOptions(torch::kDouble)
                            .device(DefaultDevice()))
                  : torch::Tensor();

              torch::Tensor lazy_input = CopyToDevice(input, device);
              torch::Tensor lazy_weight = CopyToDevice(weight, device);
              torch::Tensor lazy_bias =
                  with_bias ? CopyToDevice(bias, device) : torch::Tensor();

              torch::Tensor output = torch::conv2d(
                  input,
                  weight,
                  bias,
                  /*stride=*/{stride, stride},
                  /*padding=*/{padding, padding},
                  /*dilation=*/{dilation, dilation},
                  groups);
              torch::Tensor lazy_output = torch::conv2d(
                  lazy_input,
                  lazy_weight,
                  lazy_bias,
                  /*stride=*/{stride, stride},
                  /*padding=*/{padding, padding},
                  /*dilation=*/{dilation, dilation},
                  groups);
              AllClose(output, lazy_output);
            });
          }
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestConv2DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 3; ++dilation) {
          for (int groups :
               {1, 2, 4}) { // covers normal, grouped, depthwise conv.
            auto testfn =
                [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
              return torch::conv2d(
                  inputs[0],
                  inputs[1],
                  inputs[2],
                  /*stride=*/{stride, stride},
                  /*padding=*/{padding, padding},
                  /*dilation=*/{dilation, dilation},
                  groups);
            };

            ForEachDevice([&](const torch::Device& device) {
              torch::Tensor bias = with_bias
                  ? torch::rand(
                        {out_channels},
                        torch::TensorOptions(torch::kDouble)
                            .device(DefaultDevice()))
                  : torch::Tensor();
              TestBackward(
                  {torch::rand(
                       {1, in_channels, 7, 7},
                       torch::TensorOptions(torch::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   torch::rand(
                       {out_channels,
                        in_channels / groups,
                        kernel_size,
                        kernel_size},
                       torch::TensorOptions(torch::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   bias},
                  device,
                  testfn);
            });
          }
        };
      }
    }
  }
}

TEST_F(LazyOpsTest, TestTransposedConv2DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (int dilation = 1; dilation <= 2; ++dilation) {
        for (int output_padding = 0;
             output_padding < std::max(stride, dilation);
             ++output_padding) {
          for (bool with_bias : {true, false}) {
            for (int groups :
                 {1, 2, 4}) { // covers normal, grouped, depthwise conv.
              auto testfn = [&](const std::vector<torch::Tensor>& inputs)
                  -> torch::Tensor {
                return torch::conv_transpose2d(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    /*stride=*/{stride, stride + 1},
                    /*padding=*/{padding, padding + 1},
                    /*output_padding=*/output_padding,
                    /*groups=*/groups,
                    /*dilation=*/{dilation, dilation + 1});
              };
              ForEachDevice([&](const torch::Device& device) {
                torch::Tensor input = torch::rand(
                    {4, out_channels, 7, 7},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true));
                torch::Tensor weight = torch::rand(
                    {out_channels,
                     in_channels / groups,
                     kernel_size,
                     kernel_size},
                    torch::TensorOptions(torch::kFloat)
                        .device(DefaultDevice())
                        .requires_grad(true));
                torch::Tensor bias = with_bias
                    ? torch::rand(
                          {in_channels},
                          torch::TensorOptions(torch::kFloat)
                              .device(DefaultDevice())
                              .requires_grad(true))
                    : torch::Tensor();
                TestBackward(
                    {input, weight, bias},
                    device,
                    testfn,
                    /*rtol=*/1e-5,
                    /*atol=*/1e-5);
              });
            }
          };
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestConv3DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 1; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          for (int groups :
               {1, 2, 4}) { // covers normal, grouped, depthwise conv.
            auto testfn =
                [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
              return torch::conv3d(
                  inputs[0],
                  inputs[1],
                  inputs[2],
                  /*stride=*/{stride, stride, stride},
                  /*padding=*/{padding, padding, padding},
                  /*dilation=*/{dilation, dilation, dilation},
                  groups);
            };

            ForEachDevice([&](const torch::Device& device) {
              torch::Tensor bias = with_bias
                  ? torch::rand(
                        {out_channels},
                        torch::TensorOptions(torch::kDouble)
                            .device(DefaultDevice()))
                  : torch::Tensor();
              TestBackward(
                  {torch::rand(
                       {4, in_channels, 7, 7, 7},
                       torch::TensorOptions(torch::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   torch::rand(
                       {out_channels,
                        in_channels / groups,
                        kernel_size,
                        kernel_size,
                        kernel_size},
                       torch::TensorOptions(torch::kDouble)
                           .device(DefaultDevice())
                           .requires_grad(true)),
                   bias},
                  device,
                  testfn);
            });
          }
        };
      }
    }
  }
}

TEST_F(LazyOpsTest, TestTransposedConv3DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (int dilation = 1; dilation <= 2; ++dilation) {
        for (int output_padding = 0;
             output_padding < std::max(stride, dilation);
             ++output_padding) {
          for (bool with_bias : {true, false}) {
            for (int groups :
                 {1, 2, 4}) { // covers normal, grouped, depthwise conv.
              auto testfn = [&](const std::vector<torch::Tensor>& inputs)
                  -> torch::Tensor {
                return torch::conv_transpose3d(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    /*stride=*/{stride, stride + 1, stride},
                    /*padding=*/{padding, padding + 1, stride},
                    /*output_padding=*/output_padding,
                    /*groups=*/groups,
                    /*dilation=*/{dilation, dilation + 1, dilation});
              };
              ForEachDevice([&](const torch::Device& device) {
                torch::Tensor input = torch::rand(
                    {4, out_channels, 7, 7, 7},
                    torch::TensorOptions(torch::kDouble)
                        .device(DefaultDevice())
                        .requires_grad(true));
                torch::Tensor weight = torch::rand(
                    {out_channels,
                     in_channels / groups,
                     kernel_size,
                     kernel_size,
                     kernel_size},
                    torch::TensorOptions(torch::kDouble)
                        .device(DefaultDevice())
                        .requires_grad(true));
                torch::Tensor bias = with_bias
                    ? torch::rand(
                          {in_channels},
                          torch::TensorOptions(torch::kDouble)
                              .device(DefaultDevice())
                              .requires_grad(true))
                    : torch::Tensor();
                TestBackward({input, weight, bias}, device, testfn);
              });
            }
          };
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2DBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::max_pool2d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {1, 2, 8, 8},
                  torch::TensorOptions(torch::kFloat)
                      .device(DefaultDevice())
                      .requires_grad(true))},
              device,
              testfn);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::max_pool3d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{1, 1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {1, 2, 4, 4, 4},
                  torch::TensorOptions(torch::kFloat)
                      .device(DefaultDevice())
                      .requires_grad(true))},
              device,
              testfn);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool2DNoBatchBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::max_pool2d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {2, 8, 8},
                  torch::TensorOptions(torch::kFloat)
                      .device(DefaultDevice())
                      .requires_grad(true))},
              device,
              testfn);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxPool3DNoBatchBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::max_pool3d(
              inputs[0],
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{1, 1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {2, 4, 4, 4},
                  torch::TensorOptions(torch::kFloat)
                      .device(DefaultDevice())
                      .requires_grad(true))},
              device,
              testfn);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxUnpool2DBackward) {
  int kernel_size = 2;
  torch::Tensor input = torch::rand(
      {2, 2, 8, 8},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool2d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size({input.size(2), input.size(3)});
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::max_unpool2d(
                inputs[0],
                inputs[1],
                output_size,
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding});
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {output.requires_grad_(true), indices}, device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestMaxUnpool3DBackward) {
  int kernel_size = 2;
  torch::Tensor input = torch::rand(
      {1, 1, 4, 4, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool3d_with_indices(
              input,
              /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size(
              {input.size(2), input.size(3), input.size(4)});
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::max_unpool3d(
                inputs[0],
                inputs[1],
                output_size,
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding});
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {output.requires_grad_(true), indices}, device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestTanhBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::tanh(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 2},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn,
        /*rtol=*/1e-3,
        /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestSigmoidBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::sigmoid(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 2},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestLogSigmoidBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::log_sigmoid(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 2},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn,
        /*rtol=*/1e-3,
        /*atol=*/1e-5);
  });
}

TEST_F(LazyOpsTest, TestLogSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::log_softmax(inputs[0], dim);
    };

    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {5, 3, 4, 2},
              torch::TensorOptions(torch::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn,
          /*rtol=*/1e-3,
          /*atol=*/1e-4);
    });
  }
}

TEST_F(LazyOpsTest, TestSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::softmax(inputs[0], dim);
    };

    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {5, 3, 4, 2},
              torch::TensorOptions(torch::kFloat)
                  .device(DefaultDevice())
                  .requires_grad(true))},
          device,
          testfn,
          /*rtol=*/1e-3,
          /*atol=*/1e-4);
    });
  }
}

TEST_F(LazyOpsTest, TestSoftplusBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::softplus(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 1, 4, 6},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn,
        /*rtol=*/1e-4);
  });
}

TEST_F(LazyOpsTest, TestReluBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::relu(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 1, 4, 6},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestRreluBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::rrelu(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 1, 4, 6},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestHardshrinkBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardshrink(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn(
            {100},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestSoftshrinkBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::softshrink(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn(
            {100},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestHardtanhBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardtanh(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn(
            {100},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestEluBackward) {
  torch::Scalar alpha = 0.5;
  torch::Scalar scale = 2.5;
  torch::Scalar input_scale = 1.5;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::elu(inputs[0], alpha, scale, input_scale);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 1, 4, 6},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestGeluBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::gelu(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 3},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
  ExpectCounterChanged("lazy::gelu_backward", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLeakyReluBackward) {
  double negative_slope = 0.01;
  auto testfn = [=](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::leaky_relu(inputs[0], negative_slope);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 1, 4, 6},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestTransposeBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::t(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {2, 3},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestAddMatMulBackward) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  // Test beta != 1. through the CPU interop.
  for (double beta : {1., 2.}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::addmm(inputs[0], inputs[1], inputs[2], /*beta=*/beta);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
               {labels},
               torch::TensorOptions(torch::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           torch::rand(
               {in_channels, out_channels},
               torch::TensorOptions(torch::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true)),
           torch::rand(
               {out_channels, labels},
               torch::TensorOptions(torch::kFloat)
                   .device(DefaultDevice())
                   .requires_grad(true))},
          device,
          testfn);
    });
  }
}

TEST_F(LazyOpsTest, TestBinaryCrossEntropyBackward) {
  int batch = 6;
  int classes = 2;
  // TODO(asuhan): Fix the torch::kDouble case.
  for (auto dtype : {torch::kFloat}) {
    for (bool def_weight : {false, true}) {
      torch::Tensor input = torch::rand(
          {batch, classes}, torch::TensorOptions(dtype).requires_grad(true));
      torch::Tensor target =
          torch::rand({batch, classes}, torch::TensorOptions(dtype));
      torch::Tensor weight;
      if (def_weight) {
        weight = torch::rand({batch, classes}, torch::TensorOptions(dtype));
      }
      for (torch::Reduction::Reduction reduction :
           {torch::Reduction::Mean,
            torch::Reduction::Sum,
            torch::Reduction::None}) {
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::binary_cross_entropy(
              /*self=*/inputs[0],
              /*target=*/inputs[1],
              /*weight=*/inputs[2],
              /*reduction=*/reduction);
        };
        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {input, target, weight},
              device,
              testfn,
              /*rtol=*/1e-4,
              /*atol=*/1e-7);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLossBackward) {
  int batch = 6;
  int classes = 2;
  // TODO(asuhan): Fix the torch::kDouble case.
  for (auto dtype : {torch::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input = torch::rand(
            {batch, classes},
            torch::TensorOptions(dtype)
                .device(DefaultDevice())
                .requires_grad(true));
        torch::Tensor target = torch::randint(
            std::min(ignore_index, 0),
            classes,
            {batch},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));
        torch::Tensor weight;
        if (def_weight) {
          weight = torch::rand(
              {classes}, torch::TensorOptions(dtype).device(DefaultDevice()));
        }
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean,
              torch::Reduction::Sum,
              torch::Reduction::None}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::nll_loss(
                /*self=*/inputs[0],
                /*target=*/inputs[1],
                /*weight=*/inputs[2],
                /*reduction=*/reduction,
                /*ignore_index=*/ignore_index);
          };
          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {input, target, weight},
                device,
                testfn,
                /*rtol=*/1e-5,
                /*atol=*/1e-8);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestNllLoss2dBackward) {
  int batch = 6;
  int classes = 2;
  int height = 3;
  int width = 3;
  // TODO(asuhan): Fix the torch::kDouble case.
  for (auto dtype : {torch::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input = torch::rand(
            {batch, classes, height, width},
            torch::TensorOptions(dtype)
                .device(DefaultDevice())
                .requires_grad(true));
        torch::Tensor target = torch::randint(
            std::min(ignore_index, 0),
            classes,
            {batch, height, width},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));
        torch::Tensor weight;
        if (def_weight) {
          weight = torch::rand(
              {classes}, torch::TensorOptions(dtype).device(DefaultDevice()));
        }
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean,
              torch::Reduction::Sum,
              torch::Reduction::None}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::nll_loss2d(
                /*self=*/inputs[0],
                /*target=*/inputs[1],
                /*weight=*/inputs[2],
                /*reduction=*/reduction,
                /*ignore_index=*/ignore_index);
          };
          ForEachDevice([&](const torch::Device& device) {
            TestBackward(
                {input, target, weight},
                device,
                testfn,
                /*rtol=*/1e-5,
                /*atol=*/1e-8);
          });
        }
      }
    }
  }
}

TEST_F(LazyOpsTest, TestSmoothL1LossBackward) {
  torch::Tensor input = torch::randn(
      {2, 4},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  torch::Tensor target = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    for (double beta : {0.25, 1.}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::smooth_l1_loss(
            /*input=*/inputs[0],
            /*target=*/inputs[1],
            /*reduction=*/reduction,
            /*beta=*/beta);
      };
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {input, target},
            device,
            testfn,
            /*rtol=*/1e-5,
            /*atol=*/1e-8);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestViewBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return inputs[0].view({-1, 320});
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand(
            {32, 20, 4, 4},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true))},
        device,
        testfn);
  });
}

TEST_F(LazyOpsTest, TestBatchNorm2DBackward) {
  double momentum = 0.1;
  double eps = 0.5;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::batch_norm(
        /*input=*/inputs[0],
        /*weight=*/inputs[1],
        /*bias=*/inputs[2],
        /*running_mean=*/inputs[3],
        /*running_var=*/inputs[4],
        /*training=*/true,
        /*momentum=*/momentum,
        /*eps=*/eps,
        /*cudnn_enabled=*/false);
  };
  int num_features = 3;
  torch::Tensor undef;
  for (bool undef_weight_bias : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input = torch::rand(
          {2, num_features, 4, 4},
          torch::TensorOptions(torch::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      torch::Tensor weight = undef_weight_bias
          ? undef
          : torch::rand(
                {num_features},
                torch::TensorOptions(torch::kFloat)
                    .device(DefaultDevice())
                    .requires_grad(true));
      torch::Tensor bias = undef_weight_bias
          ? undef
          : torch::rand(
                {num_features},
                torch::TensorOptions(torch::kFloat)
                    .device(DefaultDevice())
                    .requires_grad(true));
      torch::Tensor running_mean = torch::zeros(
          {num_features},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor running_var = torch::ones(
          {num_features},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      TestBackward(
          {input, weight, bias, running_mean, running_var},
          device,
          testfn,
          /*rtol=*/1e-3,
          /*atol=*/1e-4);
    });
  }
}

TEST_F(LazyOpsTest, TestBatchNorm3DBackward) {
  double momentum = 0.1;
  double eps = 0.5;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::batch_norm(
        /*input=*/inputs[0],
        /*weight=*/inputs[1],
        /*bias=*/inputs[2],
        /*running_mean=*/inputs[3],
        /*running_var=*/inputs[4],
        /*training=*/true,
        /*momentum=*/momentum,
        /*eps=*/eps,
        /*cudnn_enabled=*/false);
  };
  int num_features = 3;
  torch::Tensor undef;
  for (bool undef_weight_bias : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input = torch::rand(
          {2, num_features, 4, 4, 2},
          torch::TensorOptions(torch::kFloat)
              .device(DefaultDevice())
              .requires_grad(true));
      torch::Tensor weight = undef_weight_bias
          ? undef
          : torch::rand(
                {num_features},
                torch::TensorOptions(torch::kFloat)
                    .device(DefaultDevice())
                    .requires_grad(true));
      torch::Tensor bias = undef_weight_bias
          ? undef
          : torch::rand(
                {num_features},
                torch::TensorOptions(torch::kFloat)
                    .device(DefaultDevice())
                    .requires_grad(true));
      torch::Tensor running_mean = torch::zeros(
          {num_features},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      torch::Tensor running_var = torch::ones(
          {num_features},
          torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
      TestBackward(
          {input, weight, bias, running_mean, running_var},
          device,
          testfn,
          /*rtol=*/1e-3,
          /*atol=*/1e-3);
    });
  }
}

TEST_F(LazyOpsTest, TestBCEWithLogitsBackward) {
  int batch = 10;
  int classes = 5;
  torch::Tensor undef;
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None,
        torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::binary_cross_entropy_with_logits(
          /*input=*/inputs[0],
          /*target=*/inputs[1],
          /*weight=*/inputs[2],
          /*pos_weight=*/inputs[3],
          /*reduction=*/reduction);
    };
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        torch::Tensor input = torch::rand(
            {batch, classes},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true));
        torch::Tensor target = torch::rand(
            {batch, classes},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true));
        torch::Tensor weight = undef_weight
            ? undef
            : torch::rand(
                  {classes},
                  torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        torch::Tensor pos_weight = undef_pos_weight
            ? undef
            : torch::rand(
                  {classes},
                  torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {input, target, weight, pos_weight},
              device,
              testfn,
              /*rtol=*/1e-3,
              /*atol=*/1e-5);
        });
      }
    }
  }
}

TEST_F(LazyOpsTest, TestKlDivBackward) {
  torch::Tensor input = torch::rand(
      {4, 3},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  torch::Tensor target = torch::rand(
      {4, 3},
      torch::TensorOptions(torch::kFloat)
          .device(DefaultDevice())
          .requires_grad(true));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean,
        torch::Reduction::Sum,
        torch::Reduction::None}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::kl_div(/*self=*/inputs[0], /*target=*/inputs[1], reduction);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {input, target},
          device,
          testfn,
          /*rtol=*/1e-4,
          /*atol=*/1e-5);
    });
  }
}

TEST_F(LazyOpsTest, TestEmbeddingBackward) {
  int num_weights = 32;
  for (int padding_idx = -1; padding_idx < num_weights; ++padding_idx) {
    for (bool scale_grad_by_freq : {false, true}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::embedding(
            inputs[0],
            inputs[1],
            /*padding_idx=*/padding_idx,
            /*scale_grad_by_freq=*/scale_grad_by_freq,
            /*sparse=*/false);
      };
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor weight = torch::rand(
            {num_weights, 7},
            torch::TensorOptions(torch::kFloat)
                .device(DefaultDevice())
                .requires_grad(true));
        torch::Tensor indices = torch::randint(
            num_weights,
            {3, 9, 4},
            torch::TensorOptions(torch::kLong).device(DefaultDevice()));
        TestBackward(
            {weight, indices},
            device,
            testfn,
            /*rtol=*/1e-5,
            /*atol=*/1e-8);
      });
    }
  }
}

TEST_F(LazyOpsTest, TestAmpForeachNonFiniteCheckAndUnscale) {
  if (IsCuda()) {
    // TODO(whc) debug failure on cuda
    GTEST_SKIP();
  }

  torch::Tensor grads0 = torch::tensor(
      {1, 2, 3, 4},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor grads1 = torch::tensor(
      {1.0, 2.0, std::nan("1"), 4.0},
      torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor inv_scale = torch::scalar_tensor(
      0.2, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor found_inf = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor grads_output0 = grads0 * inv_scale;
  torch::Tensor found_inf_output0 = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor found_inf_output1 = torch::scalar_tensor(
      1, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ForEachDevice([&](const torch::Device& device) {
    if (grads0.device() == at::kCPU) {
      GTEST_SKIP();
    }
    torch::Tensor lazy_grads0 = CopyToDevice(grads0, device);
    torch::Tensor lazy_inv_scale = CopyToDevice(inv_scale, device);
    torch::Tensor lazy_found_inf = CopyToDevice(found_inf, device);
    torch::_amp_foreach_non_finite_check_and_unscale_(
        lazy_grads0, lazy_found_inf, lazy_inv_scale);
    AllClose(grads_output0, lazy_grads0, /*rtol=*/1e-2, /*atol=*/1e-4);
    AllEqual(found_inf_output0, lazy_found_inf);

    torch::Tensor lazy_grads1 = CopyToDevice(grads1, device);
    torch::_amp_foreach_non_finite_check_and_unscale_(
        lazy_grads1, lazy_found_inf, lazy_inv_scale);
    AllEqual(found_inf_output1, lazy_found_inf);
  });
}

TEST_F(LazyOpsTest, TestAmpUpdateScale) {
  torch::Tensor growth_tracker = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor current_scale = torch::scalar_tensor(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor found_inf = torch::scalar_tensor(
      1, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor not_found_inf = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  float scale_growth_factor = 2.0;
  float scale_backoff_factor = 0.5;
  int growth_interval = 3;

  torch::Tensor growth_tracker_result0 = torch::scalar_tensor(
      1, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor current_scale_result0 = torch::scalar_tensor(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor growth_tracker_result1 = torch::scalar_tensor(
      2, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor current_scale_result1 = torch::scalar_tensor(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor growth_tracker_result2 = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor current_scale_result2 = torch::scalar_tensor(
      8, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor growth_tracker_result3 = torch::scalar_tensor(
      0, torch::TensorOptions(torch::kInt32).device(DefaultDevice()));
  torch::Tensor current_scale_result3 = torch::scalar_tensor(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));

  ForEachDevice([&](const torch::Device& device) {
    if (growth_tracker.device() == at::kCPU) {
      GTEST_SKIP();
    }
    torch::Tensor lazy_growth_tracker = CopyToDevice(growth_tracker, device);
    torch::Tensor lazy_current_scale = CopyToDevice(current_scale, device);
    torch::Tensor lazy_found_inf = CopyToDevice(found_inf, device);
    torch::Tensor lazy_not_found_inf = CopyToDevice(not_found_inf, device);

    torch::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_not_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    AllClose(
        current_scale_result0,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    AllEqual(growth_tracker_result0, lazy_growth_tracker);

    torch::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_not_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    AllClose(
        current_scale_result1,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    AllEqual(growth_tracker_result1, lazy_growth_tracker);

    // torch::_amp_update_scale_ returns the reference of current_scale
    lazy_current_scale = torch::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_not_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    AllClose(
        current_scale_result2,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    AllEqual(growth_tracker_result2, lazy_growth_tracker);

    lazy_current_scale = torch::_amp_update_scale_(
        lazy_current_scale,
        lazy_growth_tracker,
        lazy_found_inf,
        scale_growth_factor,
        scale_backoff_factor,
        growth_interval);
    AllClose(
        current_scale_result3,
        lazy_current_scale,
        /*rtol=*/1e-2,
        /*atol=*/1e-4);
    AllEqual(growth_tracker_result3, lazy_growth_tracker);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::_amp_update_scale_", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestEarlySyncLiveTensors) {
  torch::Tensor scalar_tensor = torch::scalar_tensor(
      1., torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar scalar1 = scalar_tensor.item();
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_scalar_tensor = CopyToDevice(scalar_tensor, device);
    torch::Scalar scalar2 = lazy_scalar_tensor.item();
    ASSERT_EQ(scalar1.to<float>(), scalar2.to<float>());
  });
  if (DebugUtil::ExperimentEnabled("early_sync")) {
    ExpectCounterChanged("EarlySyncLiveTensorsCount", GetIgnoredCounters());
  } else {
    ExpectCounterNotChanged("EarlySyncLiveTensorsCount", GetIgnoredCounters());
  }
  ExpectCounterChanged("aten::_local_scalar_dense", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerp) {
  torch::Tensor start = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor res = torch::lerp(start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_start = CopyToDevice(start, device);
    torch::Tensor lazy_end = CopyToDevice(end, device);
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    torch::Tensor lazy_res = torch::lerp(lazy_start, lazy_end, lazy_weight);
    AllClose(res, lazy_res);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpScalar) {
  torch::Tensor start = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar weight = torch::Scalar(3.0);
  torch::Tensor res = torch::lerp(start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_start = CopyToDevice(start, device);
    torch::Tensor lazy_end = CopyToDevice(end, device);
    torch::Tensor lazy_res = torch::lerp(lazy_start, lazy_end, weight);
    AllClose(res, lazy_res);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpInplace) {
  torch::Tensor input = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor input_copy = input.clone();
  input.lerp_(end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    torch::Tensor lazy_end = CopyToDevice(end, device);
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    lazy_input.lerp_(lazy_end, lazy_weight);
    AllClose(lazy_input, input);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpScalarInplace) {
  torch::Tensor input = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar weight = torch::Scalar(3.0);
  torch::Tensor input_copy = input.clone();
  input.lerp_(end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_input = CopyToDevice(input_copy, device);
    torch::Tensor lazy_end = CopyToDevice(end, device);
    lazy_input.lerp_(lazy_end, weight);
    AllClose(lazy_input, input);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpOut) {
  torch::Tensor start = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor weight = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor res = torch::empty(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  ;
  torch::lerp_out(res, start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_start = CopyToDevice(start, device);
    torch::Tensor lazy_end = CopyToDevice(end, device);
    torch::Tensor lazy_weight = CopyToDevice(weight, device);
    torch::Tensor lazy_res = torch::empty({3, 4}, lazy_start.options());
    torch::lerp_out(lazy_res, lazy_start, lazy_end, lazy_weight);
    AllClose(res, lazy_res);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, TestLerpScalarOut) {
  torch::Tensor start = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Tensor end = torch::rand(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::Scalar weight = torch::Scalar(3.0);
  torch::Tensor res = torch::empty(
      {3, 4}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  torch::lerp_out(res, start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor lazy_start = CopyToDevice(start, device);
    torch::Tensor lazy_end = CopyToDevice(end, device);
    torch::Tensor lazy_res = torch::empty({3, 4}, lazy_start.options());
    torch::lerp_out(lazy_res, lazy_start, lazy_end, weight);
    AllClose(res, lazy_res);
  });
  ExpectCounterNotChanged("aten::.*", GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", GetIgnoredCounters());
}

TEST_F(LazyOpsTest, IsAliasOf) {
  auto a = torch::empty(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  auto b = torch::empty(
      4, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));

  ForEachDevice([&](const torch::Device& device) {
    auto lazy_a = CopyToDevice(a, device);
    auto lazy_b = CopyToDevice(b, device);
    EXPECT_EQ(!a.is_alias_of(b), !lazy_a.is_alias_of(lazy_b));

    auto c = a.view({2, 2});
    auto lazy_c = lazy_a.view({2, 2});
    EXPECT_EQ(a.is_alias_of(c), lazy_a.is_alias_of(lazy_c));

    auto d = c.view({1, 4});
    auto lazy_d = lazy_c.view({1, 4});
    EXPECT_EQ(d.is_alias_of(c), lazy_d.is_alias_of(lazy_c));
    EXPECT_EQ(d.is_alias_of(a), lazy_d.is_alias_of(lazy_a));
  });
}

#endif // FBCODE_CAFFE2

} // namespace lazy
} // namespace torch
