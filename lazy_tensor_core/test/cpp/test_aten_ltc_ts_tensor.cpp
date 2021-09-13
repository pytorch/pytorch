#include <gtest/gtest.h>
#include <torch/torch.h>

#include <iostream>

#include "cpp_test_util.h"
#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensor_core/csrc/ts_backend/backend_impl.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_computation_client.h"
#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensors/permutation_util.h"
#include "torch_ltc_ts_test.h"

namespace torch_lazy_tensors {
namespace cpp_test {
namespace {

class AtenLtcTsTensorTest : public AtenLtcTsTensorTestBase {};

bool IsCuda() {
  return lazy_tensors::compiler::TSComputationClient::HardwareDeviceType() ==
         at::kCUDA;
}

compiler::BackendRegistrar g_registrar(compiler::GetTSBackendImpl());

}  // namespace

TEST_F(AtenLtcTsTensorTest, TestScalarTensor) {
  torch::Tensor scalar_tensor =
      torch::scalar_tensor(1., torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_scalar_tensor = torch::scalar_tensor(
        1., torch::TensorOptions(torch::kFloat).device(torch::kLazy));
    AllClose(scalar_tensor, xla_scalar_tensor);
  });
}

TEST_F(AtenLtcTsTensorTest, TestClone) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = xla_a.clone();
    AllClose(a, xla_b);
    xla_a.add_(1.0);
    AllClose(a, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTo) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestIsFloatingPoint) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    bool is_float = torch::is_floating_point(a);
    bool xla_is_float = torch::is_floating_point(xla_a);
    EXPECT_EQ(is_float, xla_is_float);
  });
}

TEST_F(AtenLtcTsTensorTest, TestIsSigned) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    bool is_signed = torch::is_signed(a);
    bool xla_is_signed = torch::is_signed(xla_a);
    EXPECT_EQ(is_signed, xla_is_signed);
  });
}

TEST_F(AtenLtcTsTensorTest, TestCastByte) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Byte(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Byte(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestCastChar) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Char(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Char(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestCastShort) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Short(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Short(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestCastInt) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Int(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Int(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestCastLong) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Long(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Long(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestCastFloat) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::_cast_Float(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::_cast_Float(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRetainType) {
  torch::Tensor xla_a = torch::zeros(
      {2, 2}, torch::TensorOptions(torch::kByte).device(torch::kLazy));
  torch::Tensor xla_b = torch::ones(
      {2, 2}, torch::TensorOptions(torch::kByte).device(torch::kLazy));
  torch::Tensor xla_c = xla_a + xla_b;
  EXPECT_EQ(xla_c.scalar_type(), torch::ScalarType::Byte);
}

TEST_F(AtenLtcTsTensorTest, TestLogicalTypeWithInterop) {
  torch::Tensor query =
      torch::rand({2, 12, 20, 64},
                  torch::TensorOptions(torch::kFloat).device(torch::kLazy));
  torch::Tensor key =
      torch::rand({2, 12, 64, 20},
                  torch::TensorOptions(torch::kFloat).device(torch::kLazy));
  torch::Tensor scores =
      torch::matmul(query, key) /
      torch::scalar_tensor(
          8, torch::TensorOptions(torch::kDouble).device(torch::kLazy));
  torch::Tensor p_attn = torch::softmax(scores, /*dim=*/-1);
  EXPECT_EQ(p_attn.scalar_type(), torch::ScalarType::Float);
}

TEST_F(AtenLtcTsTensorTest, TestAdd) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::add(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAddHalf) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kHalf));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kHalf));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::add(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAddMixedPrecision) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kHalf));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::add(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAddInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor c = a.add_(b);
    torch::Tensor xla_c = xla_a.add_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAddScalar) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar b(1);
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_c = torch::add(xla_a, b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAddScalarInPlace) {
  torch::Scalar b(1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor c = a.add_(b);
    torch::Tensor xla_c = xla_a.add_(b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAddZeroSizeDim) {
  torch::Tensor a = torch::rand({0, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({1, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::add(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::add(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSub) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::sub(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::sub(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSubInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor c = a.sub_(b);
    torch::Tensor xla_c = xla_a.sub_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSubScalar) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar b(1);
  torch::Tensor c = torch::sub(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_c = torch::sub(xla_a, b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSubScalarInPlace) {
  torch::Scalar b(1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor c = a.sub_(b);
    torch::Tensor xla_c = xla_a.sub_(b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMul) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::mul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::mul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMulInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor c = a.mul_(b);
    torch::Tensor xla_c = xla_a.mul_(xla_b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMulScalar) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar b(3);
  torch::Tensor c = torch::mul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_c = torch::mul(xla_a, b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMulScalarInPlace) {
  torch::Scalar b(3);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor c = a.mul_(b);
    torch::Tensor xla_c = xla_a.mul_(b);
    AllClose(a, xla_a);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestDiv) {
  for (torch::ScalarType scalar_type1 :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type1)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
            : torch::randint(0, 100, {3, 4},
                             torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 :
         {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
          torch::kLong}) {
      torch::Tensor b =
          isFloatingType(scalar_type2)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
              : torch::randint(1, 100, {3, 4},
                               torch::TensorOptions(scalar_type2));
      torch::Tensor c = torch::div(a, b);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = torch::div(xla_a, xla_b);
        AllClose(c, xla_c);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestDivWithRoundingMode) {
  c10::optional<c10::string_view> rounding_modes[] = {"trunc", "floor",
                                                      c10::nullopt};
  for (const auto& rounding_mode : rounding_modes) {
    for (torch::ScalarType scalar_type1 :
         {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
          torch::kLong}) {
      int lower_bound = (scalar_type1 == torch::kByte) ? 0 : -100;
      torch::Tensor a =
          isFloatingType(scalar_type1)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
              : torch::randint(lower_bound, 50, {3, 4},
                               torch::TensorOptions(scalar_type1));
      for (torch::ScalarType scalar_type2 :
           {torch::kFloat, torch::kByte, torch::kChar, torch::kShort,
            torch::kInt, torch::kLong}) {
        torch::Tensor b =
            isFloatingType(scalar_type2)
                ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
                : torch::randint(51, 100, {3, 4},
                                 torch::TensorOptions(scalar_type2));
        torch::Tensor c = torch::div(a, b, rounding_mode);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = CopyToDevice(b, device);
          torch::Tensor xla_c = torch::div(xla_a, xla_b, rounding_mode);
          AllClose(c, xla_c);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestDivInPlace) {
  for (torch::ScalarType scalar_type1 : {torch::kFloat}) {
    torch::Tensor a =
        isFloatingType(scalar_type1)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
            : torch::randint(0, 100, {3, 4},
                             torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 : {torch::kFloat}) {
      torch::Tensor b =
          isFloatingType(scalar_type2)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
              : torch::randint(1, 100, {3, 4},
                               torch::TensorOptions(scalar_type2));
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor c = a.div_(b);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = xla_a.div_(xla_b);
        ;
        AllClose(c, xla_c);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestDivInPlaceWithRoundingMode) {
  c10::optional<c10::string_view> rounding_modes[] = {"trunc", "floor",
                                                      c10::nullopt};
  for (const auto& rounding_mode : rounding_modes) {
    for (torch::ScalarType scalar_type1 : {torch::kFloat}) {
      torch::Tensor a =
          isFloatingType(scalar_type1)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
              : torch::randint(-100, 100, {3, 4},
                               torch::TensorOptions(scalar_type1));
      for (torch::ScalarType scalar_type2 : {torch::kFloat}) {
        torch::Tensor b =
            isFloatingType(scalar_type2)
                ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
                : torch::randint(1, 100, {3, 4},
                                 torch::TensorOptions(scalar_type2));
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor c = a.div_(b, rounding_mode);
          torch::Tensor xla_b = CopyToDevice(b, device);
          torch::Tensor xla_c = xla_a.div_(xla_b, rounding_mode);
          AllClose(c, xla_c);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestDivScalar) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type));
    for (bool is_float : {true, false}) {
      torch::Scalar b = is_float ? torch::Scalar(3.0) : torch::Scalar(3);
      torch::Tensor c = torch::div(a, b);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_c = torch::div(xla_a, b);
        AllClose(c, xla_c);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestDivScalarInPlace) {
  for (torch::ScalarType scalar_type : {torch::kFloat}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(1, 100, {3, 4}, torch::TensorOptions(scalar_type));
    for (bool is_float : {true, false}) {
      torch::Scalar b = is_float ? torch::Scalar(3.0) : torch::Scalar(3);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor c = a.div_(b);
        torch::Tensor xla_c = xla_a.div_(b);
        AllClose(c, xla_c);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestDivOut) {
  for (torch::ScalarType scalar_type : {torch::kFloat, torch::kDouble}) {
    torch::Tensor a = torch::rand({3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor b = torch::rand({3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor c = torch::empty({3, 4}, torch::TensorOptions(scalar_type));
    torch::div_out(c, a, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = torch::empty({3, 4}, xla_b.options());
      torch::div_out(xla_c, xla_a, xla_b);
      AllClose(c, xla_c);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestRsubScalar) {
  torch::Tensor input =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar other(1.5);
  torch::Scalar alpha(2.5);
  torch::Tensor result = torch::rsub(input, other, alpha);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::rsub(xla_input, other, alpha);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestNe) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::ne(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::ne(xla_a, xla_b);
    AllEqual(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestNeInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor a_copy = a.clone();
  torch::Tensor b = a.clone();
  b[0] += 1;
  a.ne_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.ne_(xla_b);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEq) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::eq(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::eq(xla_a, xla_b);
    AllEqual(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEqInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  b[0] += 1;
  torch::Tensor a_copy = a.clone();
  a.eq_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.eq_(xla_b);
    AllClose(xla_a, a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestGe) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::ge(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::ge(xla_a, xla_b);
    AllEqual(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestGeInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.ge_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.ge_(xla_b);
    AllClose(xla_a, a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLe) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  torch::Tensor c = torch::le(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::le(xla_a, xla_b);
    AllEqual(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLeInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.le_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.le_(xla_b);
    AllClose(xla_a, a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestGt) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::add(a.clone(), torch::ones_like(a));
  torch::Tensor c = torch::gt(b, a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::gt(xla_b, xla_a);
    AllEqual(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestGtInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.gt_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.gt_(xla_b);
    AllClose(xla_a, a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLt) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::add(a.clone(), torch::ones_like(a));
  torch::Tensor c = torch::lt(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::lt(xla_a, xla_b);
    AllEqual(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLtInplace) {
  torch::Tensor a = torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.clone();
  b[0] += 1;
  b[1] -= 1;
  torch::Tensor a_copy = a.clone();
  a.lt_(b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a_copy, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    xla_a.lt_(xla_b);
    AllClose(xla_a, a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestNeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(0));
  torch::Tensor result = torch::ne(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::ne(xla_input, other);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEqScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::eq(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::eq(xla_input, other);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestGeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::ge(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::ge(xla_input, other);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestGeScalarInplace) {
  torch::Tensor input =
      torch::arange(-1., 1.5, 0.5, torch::TensorOptions(torch::kFloat));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.ge_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    xla_input.ge_(other);
    AllClose(xla_input, input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLeScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1));
  torch::Tensor result = torch::le(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::le(xla_input, other);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLeScalarInplace) {
  torch::Tensor input =
      torch::arange(-1., 1.5, 0.5, torch::TensorOptions(torch::kFloat));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.le_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    xla_input.le_(other);
    AllClose(xla_input, input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestGtScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(0.5));
  torch::Tensor result = torch::gt(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::gt(xla_input, other);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestGtScalarInplace) {
  torch::Tensor input =
      torch::arange(-1., 1.5, 0.5, torch::TensorOptions(torch::kFloat));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.gt_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    xla_input.gt_(other);
    AllClose(xla_input, input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLtScalar) {
  torch::Tensor input = torch::ones({2, 3});
  torch::Scalar other(float(1.5));
  torch::Tensor result = torch::lt(input, other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::lt(xla_input, other);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLtScalarInplace) {
  torch::Tensor input =
      torch::arange(-1., 1.5, 0.5, torch::TensorOptions(torch::kFloat));
  torch::Scalar other(float(0));
  torch::Tensor input_copy = input.clone();
  input.lt_(other);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    xla_input.lt_(other);
    AllClose(xla_input, input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestIntegerAdd) {
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  ForEachDevice([&](const torch::Device& device) {
    for (auto type : types) {
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Tensor b =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Scalar one =
          isIntegralType(type) ? torch::Scalar(1) : torch::Scalar(1.0);
      torch::Tensor c = torch::add(b, one);

      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = torch::add(xla_b, one);

      AllEqual(c, xla_c);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, DISABLED_TestSVD) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      torch::Tensor a =
          torch::rand({m, n}, torch::TensorOptions(torch::kFloat));
      auto b = torch::svd(a, /*some=*/true, /*compute_uv=*/true);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        auto xla_b = torch::svd(xla_a, /*some=*/true, /*compute_uv=*/true);
        // The U and V matrices might have different sign for column vectors, so
        // cannot be compared if not by absolute value.
        AllClose(std::get<0>(b).abs(), std::get<0>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
        torch::Tensor diag = std::get<1>(b);
        torch::Tensor xla_diag = std::get<1>(xla_b);
        ASSERT_EQ(diag.sizes(), xla_diag.sizes());
        AllClose(diag, xla_diag, /*rtol=*/1e-3,
                 /*atol=*/1e-4);
        AllClose(std::get<2>(b).abs(), std::get<2>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, DISABLED_TestQR) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (auto n : dims) {
      torch::Tensor a =
          torch::rand({m, n}, torch::TensorOptions(torch::kFloat));
      auto b = torch::qr(a);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        auto xla_b = torch::qr(xla_a);
        AllClose(std::get<0>(b).abs(), std::get<0>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
        AllClose(std::get<1>(b).abs(), std::get<1>(xla_b).abs(), /*rtol=*/1e-3,
                 /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, DISABLED_TestSymEig) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (bool eigenvectors : {true, false}) {
      for (bool upper : {true, false}) {
        torch::Tensor a =
            torch::rand({m, m}, torch::TensorOptions(torch::kFloat));
        torch::Tensor sym_a = a.mm(a.t());
        auto b = torch::symeig(sym_a, eigenvectors, upper);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(sym_a, device);
          auto xla_b = torch::symeig(xla_a, eigenvectors, upper);
          AllClose(std::get<0>(b), std::get<0>(xla_b), /*rtol=*/3e-2,
                   /*atol=*/1e-2);
          if (eigenvectors) {
            AllClose(std::get<1>(b).abs(), std::get<1>(xla_b).abs(),
                     /*rtol=*/3e-2,
                     /*atol=*/1e-2);
          } else {
            EXPECT_EQ(std::get<1>(b).sizes(), std::get<1>(xla_b).sizes());
          }
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, DISABLED_TestCholesky) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    for (bool upper : {true, false}) {
      torch::Tensor a =
          torch::rand({3, m, m}, torch::TensorOptions(torch::kFloat));
      torch::Tensor pd_a = torch::matmul(a, torch::transpose(a, 1, 2)) +
                           torch::eye(m, torch::TensorOptions(torch::kFloat));
      auto b = torch::cholesky(pd_a, upper);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(pd_a, device);
        auto xla_b = torch::cholesky(xla_a, upper);
        AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-4);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, DISABLED_TestLogDet) {
  static const int dims[] = {4, 7};
  for (auto m : dims) {
    torch::Tensor a =
        torch::rand({3, m, m}, torch::TensorOptions(torch::kFloat));
    torch::Tensor pd_a = torch::matmul(a, torch::transpose(a, 1, 2)) +
                         torch::eye(m, torch::TensorOptions(torch::kFloat));
    torch::Tensor b = torch::logdet(pd_a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(pd_a, device);
      torch::Tensor xla_b = torch::logdet(xla_a);
      AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, DISABLED_TestTriangularSolve) {
  static const int dims[] = {4, 7};
  for (bool batched_a : {true, false}) {
    for (bool batched_b : {true, false}) {
      for (auto m : dims) {
        for (auto n : dims) {
          for (bool upper : {true, false}) {
            for (bool transpose : {true, false}) {
              for (bool unitriangular : {true, false}) {
                torch::Tensor a =
                    torch::randn({m, m}, torch::TensorOptions(torch::kFloat));
                torch::Tensor b =
                    torch::randn({m, n}, torch::TensorOptions(torch::kFloat));
                a = batched_a ? a.expand({3, m, m}).clone() : a;
                b = batched_b ? b.expand({3, m, n}).clone() : b;
                auto result = torch::triangular_solve(
                    b, a, /*upper=*/upper, /*transpose=*/transpose,
                    /*unitriangular=*/unitriangular);
                ForEachDevice([&](const torch::Device& device) {
                  torch::Tensor xla_a = CopyToDevice(a, device);
                  torch::Tensor xla_b = CopyToDevice(b, device);
                  auto xla_result = torch::triangular_solve(
                      xla_b, xla_a, /*upper=*/upper, /*transpose=*/transpose,
                      /*unitriangular=*/unitriangular);
                  AllClose(std::get<0>(result), std::get<0>(xla_result),
                           /*rtol=*/1e-3, /*atol=*/1e-4);
                  AllClose(std::get<1>(result), std::get<1>(xla_result),
                           /*rtol=*/1e-3, /*atol=*/1e-4);
                });
              }
            }
          }
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestKthValue) {
  torch::Tensor a = torch::rand({4, 5, 3}, torch::TensorOptions(torch::kFloat));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool keepdim : {false, true}) {
        auto b = torch::kthvalue(a, k, dim, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          auto xla_b = torch::kthvalue(xla_a, k, dim, keepdim);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllEqual(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestTopK) {
  torch::Tensor a = torch::rand({4, 5, 3}, torch::TensorOptions(torch::kFloat));
  for (int k = 1; k <= 3; ++k) {
    int rank = a.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (bool largest : {false, true}) {
        auto b = torch::topk(a, k, dim, largest, /*sorted=*/true);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          auto xla_b = torch::topk(xla_a, k, dim, largest, /*sorted=*/true);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllEqual(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestSort) {
  torch::Tensor a = torch::rand({4, 5, 3}, torch::TensorOptions(torch::kFloat));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        auto b = torch::sort(a, dim, descending);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          auto xla_b = torch::sort(xla_a, dim, descending);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllEqual(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestSortDescWithMinValue) {
  std::vector<int8_t> values{-128, 100};
  torch::Tensor input =
      torch::tensor(values, torch::TensorOptions(torch::kChar));
  auto output = torch::sort(input, /*dim=*/0, /*descending=*/true);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    auto xla_output = torch::sort(xla_input, /*dim=*/0, /*descending=*/true);
    AllEqual(std::get<0>(output), std::get<0>(xla_output));
    AllEqual(std::get<1>(output), std::get<1>(xla_output));
  });
}

TEST_F(AtenLtcTsTensorTest, TestArgSort) {
  torch::Tensor a = torch::rand({4, 5, 3}, torch::TensorOptions(torch::kFloat));
  for (int k = 1; k <= 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      for (bool descending : {false, true}) {
        torch::Tensor b = torch::argsort(a, dim, descending);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = torch::argsort(xla_a, dim, descending);
          AllEqual(b, xla_b);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMin) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::min(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::min(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMax) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::max(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::max(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestUnaryMin) {
  torch::Tensor input =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::min(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::min(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestUnaryMax) {
  torch::Tensor input =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::max(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::max(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAll) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor b = torch::all(a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::all(xla_a);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAllDim) {
  torch::Tensor a =
      torch::randint(0, 5, {2, 3, 4}, torch::TensorOptions(torch::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::all(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::all(xla_a, dim, /*keepdim=*/false);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAllDimKeep) {
  torch::Tensor a =
      torch::randint(0, 5, {2, 3, 4}, torch::TensorOptions(torch::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::all(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::all(xla_a, dim, /*keepdim=*/true);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAmax) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (bool keepdim : {false, true}) {
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor values = torch::amax(input, {dim}, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_values =
            torch::amax(xla_input, {dim}, /*keepdim=*/keepdim);
        AllClose(values, xla_values);
      });
    }
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        torch::Tensor values =
            torch::amax(input, {dim1, dim2}, /*keepdim=*/keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_input = CopyToDevice(input, device);
          torch::Tensor xla_values =
              torch::amax(xla_input, {dim1, dim2}, /*keepdim=*/keepdim);
          AllClose(values, xla_values);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::amax", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestAmin) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (bool keepdim : {false, true}) {
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor values = torch::amin(input, {dim}, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_values =
            torch::amin(xla_input, {dim}, /*keepdim=*/keepdim);
        AllClose(values, xla_values);
      });
    }
    for (int dim1 = -rank; dim1 < rank; ++dim1) {
      for (int dim2 = -rank; dim2 < rank; ++dim2) {
        if ((dim1 == dim2) || (dim1 == rank + dim2) || (dim2 == rank + dim1))
          continue;
        torch::Tensor values =
            torch::amin(input, {dim1, dim2}, /*keepdim=*/keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_input = CopyToDevice(input, device);
          torch::Tensor xla_values =
              torch::amin(xla_input, {dim1, dim2}, /*keepdim=*/keepdim);
          AllClose(values, xla_values);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::amin", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestAny) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor b = torch::any(a);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::any(xla_a);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAnyDim) {
  torch::Tensor a =
      torch::randint(0, 5, {2, 3, 4}, torch::TensorOptions(torch::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::any(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::any(xla_a, dim, /*keepdim=*/false);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAnyDimKeep) {
  torch::Tensor a =
      torch::randint(0, 5, {2, 3, 4}, torch::TensorOptions(torch::kByte));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::any(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::any(xla_a, dim, /*keepdim=*/true);
      EqualValues(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMean) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::mean(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::mean(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMeanCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::mean(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::mean(xla_a, torch::kDouble);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMeanInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::mean(a, {dim});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::mean(xla_a, {dim});
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMeanInDims) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::mean(a, dims);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::mean(xla_a, dims);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMeanInDimsKeepCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::mean(a, dims, true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::mean(xla_a, dims, true, torch::kDouble);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestStd) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto unbiased : {true, false}) {
    torch::Tensor b = torch::std(a, unbiased);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::std(xla_a, unbiased);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestStdInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (auto unbiased : {true, false}) {
    for (auto keepdim : {true, false}) {
      for (int dim = -rank; dim < rank; ++dim) {
        torch::Tensor b = torch::std(a, {dim}, unbiased, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = torch::std(xla_a, {dim}, unbiased, keepdim);
          AllClose(b, xla_b);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestStdWithCorrection) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  c10::optional<int64_t> corrections[] = {1, 2, c10::nullopt};
  for (const auto& correction : corrections) {
    for (auto keepdim : {true, false}) {
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        torch::Tensor b = torch::std(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = torch::std(xla_a, dim, correction, keepdim);
          AllClose(b, xla_b);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestStdMeanWithCorrection) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  c10::optional<int64_t> corrections[] = {1, 2, c10::nullopt};
  for (const auto& correction : corrections) {
    for (auto keepdim : {true, false}) {
      for (const auto& dim :
           std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
        auto b = torch::std_mean(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          auto xla_b = torch::std_mean(xla_a, dim, correction, keepdim);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllClose(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestSum) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::sum(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sum(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSumCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::sum(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sum(xla_a, torch::kDouble);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSumU8) {
  torch::Tensor a = torch::ones({256}, torch::TensorOptions(torch::kByte));
  torch::Tensor b = torch::sum(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sum(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSumInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::sum(a, {dim});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::sum(xla_a, {dim});
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestSumInDims) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::sum(xla_a, dims);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestSumInDimsKeep) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::sum(xla_a, dims, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestSumInDimsKeepCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    torch::Tensor b = torch::sum(a, dims, /*keepdim=*/true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::sum(xla_a, dims, /*keepdim=*/true, torch::kDouble);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestVar) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (bool unbiased : {true, false}) {
    torch::Tensor b = torch::var(a, unbiased);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::var(xla_a, unbiased);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestVarWithDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepDim : {true, false}) {
      for (bool unbiased : {true, false}) {
        torch::Tensor b = torch::var(a, dims, unbiased, keepDim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = torch::var(xla_a, dims, unbiased, keepDim);
          AllClose(b, xla_b);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestVarWithCorrection) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  c10::optional<int64_t> corrections[] = {1, 2, c10::nullopt};
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepDim : {true, false}) {
      for (const auto& correction : corrections) {
        torch::Tensor b = torch::var(a, dim, correction, keepDim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          torch::Tensor xla_b = torch::var(xla_a, dim, correction, keepDim);
          AllClose(b, xla_b);
        });
      }
    }
  }
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("lazy::var", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestVarMeanWithCorrection) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  c10::optional<int64_t> corrections[] = {1, 2, c10::nullopt};
  for (const auto& dim : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (const auto& correction : corrections) {
      for (auto keepdim : {true, false}) {
        auto b = torch::var_mean(a, dim, correction, keepdim);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a, device);
          auto xla_b = torch::var_mean(xla_a, dim, correction, keepdim);
          AllClose(std::get<0>(b), std::get<0>(xla_b));
          AllClose(std::get<1>(b), std::get<1>(xla_b));
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxInDim) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = torch::max(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        auto xla_values_indices =
            torch::max(xla_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(xla_values_indices));
        AllEqual(std::get<1>(values_indices), std::get<1>(xla_values_indices));
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMinInDim) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    for (bool keepdim : {false, true}) {
      auto values_indices = torch::min(input, dim, /*keepdim=*/keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        auto xla_values_indices =
            torch::min(xla_input, dim, /*keepdim=*/keepdim);
        AllClose(std::get<0>(values_indices), std::get<0>(xla_values_indices));
        AllEqual(std::get<1>(values_indices), std::get<1>(xla_values_indices));
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestNorm) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::norm(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::norm(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestNormInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::norm(a, 2, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::norm(xla_a, 2, {dim}, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestNormInDims) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::norm(a, 2, dims, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::norm(xla_a, 2, dims, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestNormInDimsKeep) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::norm(a, 2, dims, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::norm(xla_a, 2, dims, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestNormalTwoTensor) {
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_mean = bridge::CreateLtcTensor(mean, device);
    at::Tensor xla_std = bridge::CreateLtcTensor(std, device);
    at::Tensor xla_normal = at::normal(xla_mean, xla_std);
    double res_mean = xla_normal.mean().item().toDouble();
    double res_std = xla_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(AtenLtcTsTensorTest, TestNormalDoubleMean) {
  at::Tensor std = at::ones({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_std = bridge::CreateLtcTensor(std, device);
    at::Tensor xla_normal = at::normal(0, xla_std);
    double res_mean = xla_normal.mean().item().toDouble();
    double res_std = xla_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(AtenLtcTsTensorTest, TestNormalDoubleStd) {
  at::Tensor mean = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_mean = bridge::CreateLtcTensor(mean, device);
    at::Tensor xla_normal = at::normal(xla_mean, 1);
    double res_mean = xla_normal.mean().item().toDouble();
    double res_std = xla_normal.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(AtenLtcTsTensorTest, TestNormalInPlace) {
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateLtcTensor(a, device);
    xla_a.normal_(/*mean=*/0, /*std=*/1);
    double res_mean = xla_a.mean().item().toDouble();
    double res_std = xla_a.std().item().toDouble();
    EXPECT_GT(res_mean, -0.06);
    EXPECT_LT(res_mean, 0.06);
    EXPECT_GT(res_std, 0.94);
    EXPECT_LT(res_std, 1.06);
  });
}

TEST_F(AtenLtcTsTensorTest, TestUniformInPlace) {
  const double eps = 1e-3;
  at::Tensor a = at::zeros({10, 10, 10}, at::dtype(at::kFloat));
  ForEachDevice([&](const Device& device) {
    at::Tensor xla_a = bridge::CreateLtcTensor(a, device);
    xla_a.uniform_(/*from=*/0, /*to=*/1);
    at::Tensor cpu_a = ToCpuTensor(xla_a);
    double res_min = cpu_a.min().item().toDouble();
    double res_max = cpu_a.max().item().toDouble();
    EXPECT_GT(res_min, 0.0 - eps);
    EXPECT_LT(res_max, 1.0 + eps);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRandomInPlace) {
  for (auto dtype : {torch::kFloat, torch::kDouble, torch::kByte, torch::kChar,
                     torch::kShort, torch::kInt, torch::kLong}) {
    const double eps = 0.15;
    torch::Tensor a = torch::zeros({10, 10, 10}, torch::TensorOptions(dtype));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      xla_a.random_(/*from=*/0, /*to=*/10);
      double res_mean = xla_a.sum().item().toDouble() / a.numel();
      double res_min = xla_a.min().item().toDouble();
      double res_max = xla_a.max().item().toDouble();
      EXPECT_GT(res_mean, 4.5 - eps);
      EXPECT_LT(res_mean, 4.5 + eps);
      EXPECT_EQ(res_min, 0.0);
      EXPECT_EQ(res_max, 9.0);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestRandomInPlaceDefaultFrom) {
  for (auto dtype : {torch::kFloat, torch::kDouble, torch::kByte, torch::kChar,
                     torch::kShort, torch::kInt, torch::kLong}) {
    const double eps = 0.15;
    torch::Tensor a = torch::zeros({10, 10, 10}, torch::TensorOptions(dtype));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      xla_a.random_(/*to=*/10);
      double res_mean = xla_a.sum().item().toDouble() / a.numel();
      double res_min = xla_a.min().item().toDouble();
      double res_max = xla_a.max().item().toDouble();
      EXPECT_GT(res_mean, 4.5 - eps);
      EXPECT_LT(res_mean, 4.5 + eps);
      EXPECT_EQ(res_min, 0.0);
      EXPECT_EQ(res_max, 9.0);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestNormGeneral) {
  torch::Tensor a =
      torch::randn({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::norm(a, 3.5);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::norm(xla_a, 3.5);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestNormNuclear) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::norm(a, 1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::norm(xla_a, 1);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFrobeniusNorm) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::frobenius_norm(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::frobenius_norm(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFrobeniusNormInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::frobenius_norm(a, {dim}, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::frobenius_norm(xla_a, {dim}, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestFrobeniusNormInDims) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2}, {-2, -1}}) {
    torch::Tensor b = torch::frobenius_norm(a, dims, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::frobenius_norm(xla_a, dims, /*keepdim=*/false);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestGroupNorm) {
  int num_channels = 6;
  torch::Tensor input = torch::rand({20, num_channels, 10, 10},
                                    torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({num_channels}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({num_channels}, torch::TensorOptions(torch::kFloat));
  double eps = 1e-05;
  for (int num_groups : {3, 6, 1}) {
    torch::Tensor output =
        torch::group_norm(input, num_groups, weight, bias, eps,
                          /*cudnn_enabled=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_weight = CopyToDevice(weight, device);
      torch::Tensor xla_bias = CopyToDevice(bias, device);
      torch::Tensor xla_output =
          torch::group_norm(xla_input, num_groups, xla_weight, xla_bias, eps,
                            /*cudnn_enabled=*/false);
      AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestGroupNormBackward) {
  int num_channels = 6;
  torch::Tensor input =
      torch::rand({2, num_channels, 5, 5},
                  torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor weight = torch::rand(
      {num_channels}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor bias = torch::rand(
      {num_channels}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  double eps = 1e-05;
  for (bool undef_weight : {true, false}) {
    for (int num_groups : {3, 6, 1}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::group_norm(
            /*input=*/inputs[0], num_groups, inputs[1], inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      torch::Tensor undef;
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {input, undef_weight ? undef : weight, undef_weight ? undef : bias},
            device, testfn,
            /*rtol=*/1e-3, /*atol=*/1e-3,
            /*derivative_level=*/2);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestInstanceNorm) {
  int batch = 5;
  int num_channels = 20;
  torch::Tensor input = torch::rand({batch, num_channels, 10, 10},
                                    torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({num_channels}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({num_channels}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_mean =
      torch::zeros({num_channels}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_var =
      torch::ones({num_channels}, torch::TensorOptions(torch::kFloat));
  double momentum = 0.1;
  double eps = 1e-05;
  torch::Tensor output = torch::instance_norm(
      input, weight, bias, running_mean, running_var,
      /*use_input_stats=*/true, momentum, eps, /*cudnn_enabled=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    torch::Tensor xla_bias = CopyToDevice(bias, device);
    torch::Tensor xla_running_mean = CopyToDevice(running_mean, device);
    torch::Tensor xla_running_var = CopyToDevice(running_var, device);
    torch::Tensor xla_output = torch::instance_norm(
        xla_input, xla_weight, xla_bias, xla_running_mean, xla_running_var,
        /*use_input_stats=*/true, momentum, eps, /*cudnn_enabled=*/false);
    AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLayerNorm) {
  torch::Tensor input =
      torch::rand({20, 10, 10, 10}, torch::TensorOptions(torch::kFloat));
  double eps = 1e-05;
  torch::Tensor undef;
  for (bool undef_weight : {true, false}) {
    for (int64_t normalized_size : {2, 3}) {
      std::vector<int64_t> normalized_shape(normalized_size, 10);
      torch::Tensor weight =
          torch::rand(normalized_shape, torch::TensorOptions(torch::kFloat));
      torch::Tensor bias =
          torch::rand(normalized_shape, torch::TensorOptions(torch::kFloat));
      torch::Tensor output = torch::layer_norm(input, normalized_shape,
                                               undef_weight ? undef : weight,
                                               undef_weight ? undef : bias, eps,
                                               /*cudnn_enabled=*/false);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        torch::Tensor xla_bias =
            undef_weight ? undef : CopyToDevice(bias, device);
        torch::Tensor xla_output = torch::layer_norm(
            xla_input, normalized_shape, xla_weight, xla_bias, eps,
            /*cudnn_enabled=*/false);
        AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestLayerNormBackward) {
  torch::Tensor input = torch::rand(
      {2, 3, 3, 3}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  double eps = 1e-05;
  for (bool undef_weight : {true, false}) {
    for (int64_t normalized_size : {2, 3}) {
      std::vector<int64_t> normalized_shape(normalized_size, 3);
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::layer_norm(
            /*input=*/inputs[0], normalized_shape, inputs[1], inputs[2],
            /*eps=*/eps,
            /*cudnn_enabled=*/false);
      };
      torch::Tensor weight =
          torch::rand(normalized_shape,
                      torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor bias =
          torch::rand(normalized_shape,
                      torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor undef;
      ForEachDevice([&](const torch::Device& device) {
        TestBackward(
            {input, undef_weight ? undef : weight, undef_weight ? undef : bias},
            device, testfn,
            /*rtol=*/1e-3, /*atol=*/1e-4, /*derivative_level=*/2);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, DISABLED_TestNuclearNorm) {
  torch::Tensor a = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::nuclear_norm(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::nuclear_norm(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestPairwiseDistance) {
  torch::Tensor x1 = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor x2 = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  double eps = 1e-6;
  for (bool keepdim : {false, true}) {
    for (double p : {1, 2, 3, 4}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::pairwise_distance(x1, x2, p, eps, keepdim);
        torch::Tensor xla_x1 = CopyToDevice(x1, device);
        torch::Tensor xla_x2 = CopyToDevice(x2, device);
        torch::Tensor xla_output =
            torch::pairwise_distance(xla_x1, xla_x2, p, eps, keepdim);
        AllClose(output, xla_output, /*rtol=*/1e-5, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestCosineSimilarity) {
  torch::Tensor x1 = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor x2 = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  double eps = 1e-8;
  int rank = x1.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::cosine_similarity(x1, x2, dim, eps);
      torch::Tensor xla_x1 = CopyToDevice(x1, device);
      torch::Tensor xla_x2 = CopyToDevice(x2, device);
      torch::Tensor xla_output =
          torch::cosine_similarity(xla_x1, xla_x2, dim, eps);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCosineEmbeddingLoss) {
  torch::Tensor input1 =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor input2 =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::cosine_embedding_loss(
            input1, input2, target, margin, reduction);
        torch::Tensor xla_input1 = CopyToDevice(input1, device);
        torch::Tensor xla_input2 = CopyToDevice(input2, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_output = torch::cosine_embedding_loss(
            xla_input1, xla_input2, xla_target, margin, reduction);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestHingeEmbeddingLoss) {
  torch::Tensor input =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::hinge_embedding_loss(input, target, margin, reduction);
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_output = torch::hinge_embedding_loss(
            xla_input, xla_target, margin, reduction);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestTripletMarginLoss) {
  torch::Tensor anchor =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor positive =
      torch::abs(torch::rand({4, 3}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor negative = torch::neg(
      torch::abs(torch::rand({4, 3}, torch::TensorOptions(torch::kFloat))));
  double eps = 1e-6;
  for (double margin : {0., 0.2}) {
    for (double p : {1, 2, 3, 4}) {
      for (bool swap : {false, true}) {
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean, torch::Reduction::Sum}) {
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor output = torch::triplet_margin_loss(
                anchor, positive, negative, margin, p, eps, swap, reduction);
            torch::Tensor xla_anchor = CopyToDevice(anchor, device);
            torch::Tensor xla_positive = CopyToDevice(positive, device);
            torch::Tensor xla_negative = CopyToDevice(negative, device);
            torch::Tensor xla_output = torch::triplet_margin_loss(
                xla_anchor, xla_positive, xla_negative, margin, p, eps, swap,
                reduction);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestBinaryCrossEntropy) {
  int batch = 10;
  int classes = 5;
  torch::Tensor input =
      torch::rand({batch, classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::rand({batch, classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({batch, classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor undef;
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum,
        torch::Reduction::None}) {
    for (bool undef_weight : {false, true}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::binary_cross_entropy(
            input, target, undef_weight ? undef : weight, reduction);
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_weight =
            undef_weight ? undef : CopyToDevice(weight, device);
        torch::Tensor xla_output = torch::binary_cross_entropy(
            xla_input, xla_target, xla_weight, reduction);
        AllClose(output, xla_output, /*rtol=*/1e-4, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMarginRankingLoss) {
  torch::Tensor input1 =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor input2 =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (double margin : {0., 0.2}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output = torch::margin_ranking_loss(
            input1, input2, target, margin, reduction);
        torch::Tensor xla_input1 = CopyToDevice(input1, device);
        torch::Tensor xla_input2 = CopyToDevice(input2, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_output = torch::margin_ranking_loss(
            xla_input1, xla_input2, xla_target, margin, reduction);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestBCEWithLogits) {
  int batch = 10;
  int classes = 5;
  torch::Tensor input =
      torch::rand({batch, classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::rand({batch, classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor pos_weight =
      torch::rand({classes}, torch::TensorOptions(torch::kFloat));
  torch::Tensor undef;
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor output = torch::binary_cross_entropy_with_logits(
              input, target, undef_weight ? undef : weight,
              undef_pos_weight ? undef : pos_weight, reduction);
          torch::Tensor xla_input = CopyToDevice(input, device);
          torch::Tensor xla_target = CopyToDevice(target, device);
          torch::Tensor xla_weight =
              undef_weight ? undef : CopyToDevice(weight, device);
          torch::Tensor xla_pos_weight =
              undef_pos_weight ? undef : CopyToDevice(pos_weight, device);
          torch::Tensor xla_output = torch::binary_cross_entropy_with_logits(
              xla_input, xla_target, xla_weight, xla_pos_weight, reduction);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestKlDiv) {
  torch::Tensor input =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  for (bool log_target : {true, false}) {
    for (torch::Reduction::Reduction reduction :
         {torch::Reduction::Mean, torch::Reduction::Sum}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor output =
            torch::kl_div(input, target, reduction, log_target);
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_output =
            torch::kl_div(xla_input, xla_target, reduction, log_target);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestProd) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::prod(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::prod(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestProdCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::prod(a, torch::kDouble);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::prod(xla_a, torch::kDouble);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestProdInDim) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::prod(xla_a, dim);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestProdInDimKeepCast) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim, /*keepdim=*/true, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::prod(xla_a, dim, /*keepdim=*/true, torch::kDouble);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestProdInDimKeep) {
  torch::Tensor a = torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor b = torch::prod(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::prod(xla_a, dim, /*keepdim=*/true);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCumSum) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCumSumCast) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim, torch::kDouble);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCumSumLong) {
  torch::Tensor input =
      torch::randint(1000, {4, 3, 4}, torch::TensorOptions(torch::kLong));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim);
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCumSumCastLong) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kLong);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim, torch::kLong);
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCumProd) {
  torch::Tensor input =
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumprod(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumprod(xla_input, dim);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCumProdCast) {
  torch::Tensor input = torch::mul(
      torch::rand({4, 3, 4}, torch::TensorOptions(torch::kFloat)), 10);
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumprod(input, dim, torch::kDouble);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumprod(xla_input, dim, torch::kDouble);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCumProdLong) {
  torch::Tensor input =
      torch::randint(7, {2, 3}, torch::TensorOptions(torch::kLong));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim);
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCumProdCastLong) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat)) * 7;
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cumsum(input, dim, torch::kLong);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_result = torch::cumsum(xla_input, dim, torch::kLong);
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestArgMin) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::argmin(a, c10::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::argmin(xla_a, c10::nullopt, /*keepdim=*/false);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestArgMinDim) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmin(xla_a, dim, /*keepdim=*/false);
      AllEqual(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestArgMinDimKeep) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmin(xla_a, dim, /*keepdim=*/true);
      AllEqual(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestArgMinSameValue) {
  torch::Tensor a = torch::ones({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::argmin(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::argmin(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestArgMinWrapper) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmin(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmin(xla_a, dim, /*keepdim=*/false);
      AllEqual(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestArgMax) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::argmax(a, c10::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::argmax(xla_a, c10::nullopt, /*keepdim=*/false);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestArgMaxDim) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmax(xla_a, dim, /*keepdim=*/false);
      AllEqual(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestArgMaxDimKeep) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/true);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmax(xla_a, dim, /*keepdim=*/true);
      AllEqual(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestArgMaxSameValue) {
  torch::Tensor a = torch::ones({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::argmax(a, c10::nullopt, /*keepdim=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::argmax(xla_a, c10::nullopt, /*keepdim=*/false);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestArgMaxWrapper) {
  torch::Tensor a = torch::rand({4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor b = torch::argmax(a, dim, /*keepdim=*/false);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::argmax(xla_a, dim, /*keepdim=*/false);
      AllEqual(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAsin) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::asin(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::asin(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAsinh) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::asinh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::asinh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAsinhInPlace) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::asinh_(a);
    torch::Tensor xla_b = torch::asinh_(xla_a);
    AllClose(a, xla_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSin) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::sin(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sin(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSinh) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::sinh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sinh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAcos) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::acos(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::acos(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAcosh) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100;
  torch::Tensor b = torch::acosh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::acosh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAcoshInPlace) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::acosh_(a);
    torch::Tensor xla_b = torch::acosh_(xla_a);
    AllClose(a, xla_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestCos) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::cos(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::cos(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestCosh) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::cosh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::cosh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAtan) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::atan(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::atan(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAtanh) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::atanh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::atanh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAtanhInPlace) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::atanh_(a);
    torch::Tensor xla_b = torch::atanh_(xla_a);
    AllClose(a, xla_a, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAtan2) {
  torch::Tensor a = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::atan2(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::atan2(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTan) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::tan(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::tan(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTanh) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::tanh(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::tanh(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestClampMinMax) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar min_val(0.311);
  torch::Scalar max_val(0.409);
  torch::Tensor b = torch::clamp(a, min_val, max_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::clamp(xla_a, min_val, max_val);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestClampMin) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar min_val(0.311);
  torch::Tensor b = torch::clamp(a, min_val, c10::nullopt);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::clamp(xla_a, min_val, c10::nullopt);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestClampMax) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar max_val(0.409);
  torch::Tensor b = torch::clamp(a, c10::nullopt, max_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::clamp(xla_a, c10::nullopt, max_val);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestClampMinExplicit) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar min_val(0.311);
  torch::Tensor b = torch::clamp_min(a, min_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::clamp_min(xla_a, min_val);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestClampMaxExplicit) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar max_val(0.409);
  torch::Tensor b = torch::clamp_max(a, max_val);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::clamp_max(xla_a, max_val);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestClampMinExplicitInPlace) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar min_val(0.311);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::clamp_min_(a, min_val);
    torch::Tensor xla_b = torch::clamp_min_(xla_a, min_val);
    AllClose(a, xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestClampMaxExplicitInPlace) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar max_val(0.409);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = torch::clamp_max_(a, max_val);
    torch::Tensor xla_b = torch::clamp_max_(xla_a, max_val);
    AllClose(a, xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestCeil) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::ceil(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::ceil(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFloor) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::floor(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::floor(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRound) {
  torch::Tensor a = torch::cat(
      {torch::randn({8}, torch::TensorOptions(torch::kFloat)) * 100.0,
       // Special case: 0.5, -0.5. lazy::Round impl rounds to -1/1 whereas
       // lazy::RoundToEven properly implements bankers rounding.
       torch::tensor({-0.5, 0.5}, torch::TensorOptions(torch::kFloat))},
      0);
  torch::Tensor b = torch::round(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::round(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTrunc) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::trunc(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::trunc(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFrac) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::frac(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::frac(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestNeg) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::neg(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::neg(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseNot) {
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  ForEachDevice([&](const torch::Device& device) {
    for (auto type : types) {
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Tensor b = torch::bitwise_not(a);
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = torch::bitwise_not(xla_a);
      AllEqual(b, xla_b);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseNotInPlace) {
  std::vector<torch::ScalarType> types(
      {torch::kByte, torch::kChar, torch::kShort, torch::kInt, torch::kLong});

  ForEachDevice([&](const torch::Device& device) {
    for (auto type : types) {
      torch::Tensor a =
          torch::randint(0, 63, {2, 2}, torch::TensorOptions(type));
      torch::Tensor xla_a = CopyToDevice(a, device);
      a.bitwise_not_();
      xla_a.bitwise_not_();
      AllEqual(a, xla_a);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestSign) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b = torch::sign(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sign(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSignByte) {
  torch::Tensor a =
      torch::randint(256, {2, 2}, torch::TensorOptions(torch::kByte));
  torch::Tensor b = torch::sign(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sign(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAbs) {
  torch::Tensor a = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::abs(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::abs(xla_a);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAbsByte) {
  torch::Tensor a =
      torch::randint(256, {2, 2}, torch::TensorOptions(torch::kByte));
  torch::Tensor b = torch::abs(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::abs(xla_a);
    AllEqual(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEmptyLike) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::empty_like(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::empty_like(xla_a);
    EXPECT_EQ(b.sizes(), xla_b.sizes());
  });
}

TEST_F(AtenLtcTsTensorTest, TestEmptyLikeOptions) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::empty_like(a, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b =
        torch::empty_like(xla_a, torch::TensorOptions(torch::kFloat));
    EXPECT_EQ(b.sizes(), xla_b.sizes());
  });
}

TEST_F(AtenLtcTsTensorTest, TestEmpty) {
  torch::Tensor a = torch::zeros({2, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = torch::empty(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    EXPECT_EQ(a.sizes(), xla_a.sizes());
  });
}

TEST_F(AtenLtcTsTensorTest, TestZerosLike) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::zeros_like(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::zeros_like(xla_a);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestZerosLikeOptions) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::zeros_like(a, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b =
        torch::zeros_like(xla_a, torch::TensorOptions(torch::kFloat));
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestZeros) {
  torch::Tensor a = torch::zeros({2, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = torch::zeros(
        {2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestOnes) {
  torch::Tensor a = torch::ones({2, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a =
        torch::ones({2, 2}, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestOnesLike) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::ones_like(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::ones_like(xla_a);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestOnesLikeOptions) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::ones_like(a, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b =
        torch::ones_like(xla_a, torch::TensorOptions(torch::kFloat));
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFull) {
  torch::Tensor a =
      torch::full({2, 2}, 3.1165, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = torch::full(
        {2, 2}, 3.1165, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFullLike) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::full_like(a, 3.1165);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::full_like(xla_a, 3.1165);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFullLikeOptions) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b =
      torch::full_like(a, 3.1165, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b =
        torch::full_like(xla_a, 3.1165, torch::TensorOptions(torch::kFloat));
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestARange) {
  for (auto& ranges : std::vector<std::vector<float>>{{0.0, 100.0, 0.5},
                                                      {0.0, -100.0, -0.5}}) {
    torch::Tensor a = torch::arange(ranges[0], ranges[1], ranges[2],
                                    torch::TensorOptions(torch::kFloat));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a =
          torch::arange(ranges[0], ranges[1], ranges[2],
                        torch::TensorOptions(torch::kFloat).device(device));
      AllClose(a, xla_a);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestARangeOut) {
  torch::Tensor a = torch::randn({4}, torch::TensorOptions(torch::kFloat));
  for (auto& ranges : std::vector<std::vector<float>>{{0.0, 100.0, 0.5},
                                                      {0.0, -100.0, -0.5}}) {
    torch::Tensor b = torch::arange_out(a, ranges[0], ranges[1], ranges[2]);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b =
          torch::arange_out(xla_a, ranges[0], ranges[1], ranges[2]);
      AllClose(b, xla_b);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestDimARange) {
  torch::Tensor like = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor a = torch::_dim_arange(like, 1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_like = CopyToDevice(like, device);
    torch::Tensor xla_a = torch::_dim_arange(xla_like, 1);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBartlettWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::bartlett_window(
          window_length, periodic, torch::TensorOptions(torch::kFloat));

      torch::Tensor xla_output = torch::bartlett_window(
          window_length, periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, xla_output, /*rtol=*/1e-5, /*atol=*/1e-7);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestBlackmanWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::blackman_window(
          window_length, periodic, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_output = torch::blackman_window(
          window_length, periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, xla_output, /*rtol=*/1e-5, /*atol=*/1e-7);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestHammingWindow) {
  double alpha = 0.54;
  double beta = 0.46;
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output =
          torch::hamming_window(window_length, periodic, alpha, beta,
                                torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_output = torch::hamming_window(
          window_length, periodic, alpha, beta,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestHannWindow) {
  int window_length = 10;
  for (bool periodic : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor output = torch::hann_window(
          window_length, periodic, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_output = torch::hann_window(
          window_length, periodic,
          torch::TensorOptions(torch::kFloat).device(device));
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestLogSigmoid) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::log_sigmoid(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::log_sigmoid(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLogsumexp) {
  torch::Tensor a = torch::rand({3, 4, 3}, torch::TensorOptions(torch::kFloat));
  for (auto dims : std::vector<std::vector<int64_t>>{{0, 1}, {-3, -2}}) {
    for (bool keepdim : {false, true}) {
      torch::Tensor b = torch::logsumexp(a, dims, keepdim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_b = torch::logsumexp(xla_a, dims, keepdim);
        AllClose(b, xla_b);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestSiLU) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::silu(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::silu(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
  ExpectCounterChanged("lazy::silu_out", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestSigmoid) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::sigmoid(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sigmoid(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMatmul_1x1) {
  torch::Tensor a = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMatmul_2x1) {
  torch::Tensor a = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMatmul_1x2) {
  torch::Tensor a = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMatmul_2x2) {
  torch::Tensor a = torch::rand({2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::matmul(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMatmulBcast) {
  torch::Tensor a =
      torch::rand({4, 2, 3, 2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b =
      torch::rand({2, 1, 4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::matmul(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::matmul(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestDot) {
  torch::Tensor a = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::dot(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::dot(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTensorDot) {
  torch::Tensor a = torch::rand({6, 4, 8}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4, 7, 8}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> dims_a = {1, 2};
  std::vector<int64_t> dims_b = {0, 2};
  torch::Tensor c = torch::tensordot(a, b, dims_a, dims_b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::tensordot(xla_a, xla_b, dims_a, dims_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestGer) {
  torch::Tensor a = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::ger(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::ger(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMv) {
  torch::Tensor a = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::mv(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::mv(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMvOut) {
  torch::Tensor a = torch::rand({4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({4}, torch::TensorOptions(torch::kFloat));
  torch::mv_out(c, a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::empty({4}, xla_b.options());
    torch::mv_out(xla_c, xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBatchAddBatchMatMul) {
  torch::Tensor a = torch::rand({3, 6, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 6, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({3, 4, 5}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 0.5;
  torch::Scalar beta = 1.5;
  torch::Tensor d = torch::baddbmm(a, b, c, beta, alpha);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::baddbmm(xla_a, xla_b, xla_c, beta, alpha);
    AllClose(d, xla_d, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBatchAddBatchMatMulInPlace) {
  torch::Tensor a = torch::rand({3, 6, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 6, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({3, 4, 5}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 0.5;
  torch::Scalar beta = 1.5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor d = a.baddbmm_(b, c, beta, alpha);
    torch::Tensor xla_d = xla_a.baddbmm_(xla_b, xla_c, beta, alpha);
    AllClose(d, xla_d, /*rtol=*/1e-3, /*atol=*/1e-4);
    AllClose(a, xla_a, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBatchMatMul) {
  torch::Tensor a = torch::rand({3, 6, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 4, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::bmm(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::bmm(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenLtcTsTensorTest, TestChainMatMul) {
  torch::Tensor a = torch::rand({5, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({6, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor d = torch::rand({2, 7}, torch::TensorOptions(torch::kFloat));
  torch::Tensor result = torch::chain_matmul({a, b, c, d});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = CopyToDevice(d, device);
    torch::Tensor xla_result =
        torch::chain_matmul({xla_a, xla_b, xla_c, xla_d});
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLinear) {
  torch::Tensor input =
      torch::rand({2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias = torch::rand({3});
  torch::Tensor result = torch::linear(input, weight);
  torch::Tensor result_with_bias = torch::linear(input, weight, bias);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    torch::Tensor xla_bias = CopyToDevice(bias, device);
    torch::Tensor xla_result = torch::linear(xla_input, xla_weight);
    torch::Tensor xla_result_with_bias =
        torch::linear(xla_input, xla_weight, xla_bias);
    AllClose(result, xla_result, /*rtol=*/1e-2, /*atol=*/1e-4);
    AllClose(result_with_bias, xla_result_with_bias, /*rtol=*/1e-2,
             /*atol=*/1e-4);
  });
}

TEST_F(AtenLtcTsTensorTest, TestPinverse) {
  torch::Tensor input =
      torch::rand({4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor result = torch::pinverse(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::pinverse(xla_input);
    AllClose(result, xla_result, /*rtol=*/1e-4);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEinsumOuter) {
  torch::Tensor a = torch::rand({5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({5}, torch::TensorOptions(torch::kFloat));
  std::string equation = "i,j->ij";
  torch::Tensor c = torch::einsum(equation, {a, b});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::einsum(equation, {xla_a, xla_b});
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEinsumOuterBackward) {
  torch::Tensor a =
      torch::rand({5}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor b =
      torch::rand({5}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  std::string equation = "i,j->ij";
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::einsum(equation, inputs);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward({a, b}, device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEinsumBatchMatMul) {
  torch::Tensor a = torch::rand({3, 2, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5, 4}, torch::TensorOptions(torch::kFloat));
  std::string equation = "bij,bjk->bik";
  torch::Tensor c = torch::einsum(equation, {a, b});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::einsum(equation, {xla_a, xla_b});
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEinsumPyTorchLowerBilinear) {
  torch::Tensor a = torch::rand({3, 5, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor l = torch::rand({2, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor r = torch::rand({2, 4}, torch::TensorOptions(torch::kFloat));
  std::string equation = "bn,anm,bm->ba";
  torch::Tensor c = torch::einsum(equation, {l, a, r});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_l = CopyToDevice(l, device);
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_r = CopyToDevice(r, device);
    torch::Tensor xla_c = torch::einsum(equation, {xla_l, xla_a, xla_r});
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEinsumPyTorchLowerDiagonal) {
  torch::Tensor input =
      torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  std::string equation = "ii->i";
  torch::Tensor result = torch::einsum(equation, {input});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEinsumPyTorchLowerBatchDiagonal) {
  torch::Tensor input =
      torch::rand({4, 3, 3}, torch::TensorOptions(torch::kFloat));
  std::string equation = "...ii->...i";
  torch::Tensor result = torch::einsum(equation, {input});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEinsumPyTorchLowerBatchPermute) {
  torch::Tensor input =
      torch::rand({2, 3, 4, 5}, torch::TensorOptions(torch::kFloat));
  std::string equation = "...ij->...ji";
  torch::Tensor result = torch::einsum(equation, {input});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::einsum(equation, {xla_input});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEinsumPyTorchLowerRepeatedAxis) {
  torch::Tensor x = torch::rand({2, 3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor y = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  std::string equation = "ijj,k->ik";
  torch::Tensor result = torch::einsum(equation, {x, y});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_x = CopyToDevice(x, device);
    torch::Tensor xla_y = CopyToDevice(y, device);
    torch::Tensor xla_result = torch::einsum(equation, {xla_x, xla_y});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBilinear) {
  int batch_size = 16;
  int in1_features = 4;
  int in2_features = 6;
  int out_features = 8;
  torch::Tensor input1 = torch::rand({batch_size, in1_features},
                                     torch::TensorOptions(torch::kFloat));
  torch::Tensor input2 = torch::rand({batch_size, in2_features},
                                     torch::TensorOptions(torch::kFloat));
  torch::Tensor weight = torch::rand({out_features, in1_features, in2_features},
                                     torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({out_features}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input1 = CopyToDevice(input1, device);
    torch::Tensor xla_input2 = CopyToDevice(input2, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    torch::Tensor xla_bias = CopyToDevice(bias, device);
    torch::Tensor result = torch::bilinear(input1, input2, weight, bias);
    torch::Tensor xla_result =
        torch::bilinear(xla_input1, xla_input2, xla_weight, xla_bias);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestUpsampleNearest2D) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  torch::Tensor input = torch::rand({batch_size, chans, h, w},
                                    torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor result = torch::upsample_nearest2d(input, {uh, uw});
    torch::Tensor xla_result = torch::upsample_nearest2d(xla_input, {uh, uw});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestUpsampleNearest2DBackward) {
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
        {torch::rand({batch_size, chans, h, w},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestUpsampleNearest2DWithScale) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int chans = 2;
  double scale_h = 2.5;
  double scale_w = 3.4;
  torch::Tensor input = torch::rand({batch_size, chans, h, w},
                                    torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor result = torch::upsample_nearest2d(
        input, c10::nullopt, at::ArrayRef<double>{scale_h, scale_w});
    torch::Tensor xla_result = torch::upsample_nearest2d(
        xla_input, c10::nullopt, at::ArrayRef<double>{scale_h, scale_w});
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestUpsampleNearest2DBackwardWithScale) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int chans = 2;
  double scale_h = 2.5;
  double scale_w = 3.4;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::upsample_nearest2d(inputs[0], c10::nullopt,
                                     at::ArrayRef<double>{scale_h, scale_w});
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({batch_size, chans, h, w},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestUpsampleBilinear2D) {
  int batch_size = 2;
  int h = 5;
  int w = 5;
  int uh = 8;
  int uw = 8;
  int chans = 2;
  for (bool align_corners : {true, false}) {
    torch::Tensor input = torch::rand({batch_size, chans, h, w},
                                      torch::TensorOptions(torch::kFloat));
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor result =
          torch::upsample_bilinear2d(input, {uh, uw}, align_corners);
      torch::Tensor xla_result =
          torch::upsample_bilinear2d(xla_input, {uh, uw}, align_corners);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestUpsampleBilinear2DBackward) {
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
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAddCMul) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor d = torch::addcmul(a, b, c, 3.1165);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::addcmul(xla_a, xla_b, xla_c, 3.1165);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAddCDiv) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c =
      torch::abs(torch::rand({2, 2}, torch::TensorOptions(torch::kFloat))) +
      1.0;
  torch::Tensor d = torch::addcdiv(a, b, c, 3.1165);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::addcdiv(xla_a, xla_b, xla_c, 3.1165);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAddCDivWithBroadcast) {
  torch::Tensor a = torch::rand({1, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 1}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c =
      torch::abs(torch::rand({1, 3}, torch::TensorOptions(torch::kFloat))) +
      1.0;
  torch::Tensor d = torch::addcdiv(a, b, c, 3.1165);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::addcdiv(xla_a, xla_b, xla_c, 3.1165);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSize) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    for (int dim = -rank; dim < rank; ++dim) {
      EXPECT_EQ(torch::size(input, dim), torch::size(xla_input, dim));
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestSelect) {
  std::vector<int64_t> input_sizes = {14, 24, 8};
  int rank = input_sizes.size();
  for (int dim = -rank; dim < rank; ++dim) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::select(inputs[0], dim, 0);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward({torch::rand(input_sizes, torch::TensorOptions(torch::kFloat)
                                                 .requires_grad(true))},
                   device, testfn);
    });
  };
}

TEST_F(AtenLtcTsTensorTest, TestBernoulliScalarProb) {
  torch::Tensor input = torch::zeros(1000, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::bernoulli(xla_input, 0.1);
    double frac = xla_output.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBernoulliTensorProb) {
  std::vector<float> prob_values(1000, 0.1);
  torch::Tensor input =
      torch::tensor(prob_values, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::bernoulli(xla_input);
    double frac = xla_output.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBernoulliScalarProbInPlace) {
  torch::Tensor input = torch::zeros(1000, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    xla_input.bernoulli_(0.1);
    double frac = xla_input.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBernoulliTensorProbInPlace) {
  torch::Tensor input = torch::zeros(1000, torch::TensorOptions(torch::kFloat));
  torch::Tensor prob =
      torch::scalar_tensor(0.1, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_prob = CopyToDevice(prob, device);
    xla_input.bernoulli_(xla_prob);
    double frac = xla_input.sum().item().toDouble() / input.numel();
    EXPECT_GT(frac, 0.06);
    EXPECT_LT(frac, 0.14);
  });
}

TEST_F(AtenLtcTsTensorTest, TestDropout) {
  torch::Tensor a = torch::rand({17, 21}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::dropout(xla_a, 0.1, /*train=*/true);
    double prob =
        static_cast<double>(xla_b.cpu().ne(0.0f).sum().item().toDouble()) /
        a.numel();
    EXPECT_GT(prob, 0.86);
    EXPECT_LT(prob, 0.94);
  });
}

TEST_F(AtenLtcTsTensorTest, TestDropoutInPlace) {
  torch::Tensor a = torch::rand({17, 21}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::dropout_(xla_a, 0.1, /*train=*/true);
    double prob =
        static_cast<double>(xla_a.cpu().ne(0.0f).sum().item().toDouble()) /
        a.numel();
    EXPECT_GT(prob, 0.85);
    EXPECT_LT(prob, 0.94);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRandperm) {
  int n = 5;
  torch::Tensor shuffle = torch::randperm(
      n, torch::TensorOptions(torch::kLong).device(torch::kLazy));
  torch::Tensor shuffle_cpu = CopyToDevice(shuffle, torch::kCPU);
  std::vector<lazy_tensors::int64> shuffle_data(
      shuffle_cpu.data_ptr<int64_t>(), shuffle_cpu.data_ptr<int64_t>() + n);
  EXPECT_TRUE(shuffle_data.size() == n &&
              lazy_tensors::IsPermutation(shuffle_data));
}

TEST_F(AtenLtcTsTensorTest, TestSlice) {
  torch::Tensor a =
      torch::rand({32, 24, 16}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::slice(a, 1, 0, 16, 1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::slice(xla_a, 1, 0, 16, 1);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTake) {
  torch::Tensor a = torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::randint(16, {5}, torch::TensorOptions(torch::kLong));
  torch::Tensor c = torch::take(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::take(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTakeBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::take(inputs[0], inputs[1]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({4, 4},
                     torch::TensorOptions(torch::kFloat).requires_grad(true)),
         torch::randint(16, {5}, torch::TensorOptions(torch::kLong))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestStack) {
  torch::Tensor a = torch::rand({2, 4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({2, 4, 3}, torch::TensorOptions(torch::kFloat));
  int rank = a.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor d = torch::stack({a, b, c}, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::stack({xla_a, xla_b, xla_c}, dim);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCat) {
  torch::Tensor a = torch::rand({2, 1, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({2, 3, 3}, torch::TensorOptions(torch::kFloat));
  for (int dim : {1, -2}) {
    torch::Tensor d = torch::cat({a, b, c}, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::cat({xla_a, xla_b, xla_c}, dim);
      EXPECT_TRUE(d.sizes() == xla_d.sizes() && d.dtype() == xla_d.dtype());
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestUnbind) {
  torch::Tensor input =
      torch::rand({4, 3, 7}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<torch::Tensor> output = torch::unbind(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      std::vector<torch::Tensor> xla_output = torch::unbind(xla_input, dim);
      ASSERT_EQ(output.size(), xla_output.size());
      for (size_t i = 0; i < output.size(); ++i) {
        AllClose(output[i], xla_output[i]);
      }
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestRepeat) {
  std::vector<std::vector<int64_t>> repeats_list = {{4, 2}, {4, 2, 3}};
  std::vector<std::vector<int64_t>> input_size_list = {{3}, {2, 4}};
  for (const auto& repeats : repeats_list) {
    for (const auto& input_size : input_size_list) {
      torch::Tensor input =
          torch::rand(input_size, torch::TensorOptions(torch::kFloat));
      torch::Tensor output = input.repeat(repeats);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_output = xla_input.repeat(repeats);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestGather) {
  torch::Tensor a = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::empty({3, 3}, torch::TensorOptions(torch::kLong));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      b[i][j] = (i + j) % 3;
    }
  }
  for (bool sparse_grad : {false, true}) {
    torch::Tensor c = torch::gather(a, 1, b, sparse_grad);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = torch::gather(xla_a, 1, xla_b, sparse_grad);
      AllClose(c, xla_c);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestScatter) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::scatter(xla_a, dim, xla_c, xla_b);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestScatterR1) {
  torch::Tensor a = torch::rand({5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({2}, torch::TensorOptions(torch::kLong));
  c[0] = 1;
  c[1] = 3;
  torch::Tensor d = torch::scatter(a, 0, c, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::scatter(xla_a, 0, xla_c, xla_b);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenLtcTsTensorTest, TestScatterR3) {
  torch::Tensor a = torch::rand({3, 5, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 4, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 4, 2}, torch::TensorOptions(torch::kLong));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 2; k++) {
        c[i][j][k] = (i + j + k) % 4;
      }
    }
  }
  torch::Tensor d = torch::scatter(a, 1, c, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::scatter(xla_a, 1, xla_c, xla_b);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenLtcTsTensorTest, TestScatterBiggerSource) {
  torch::Tensor a = torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({8, 8}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({4, 4}, torch::TensorOptions(torch::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    torch::Tensor d = torch::scatter(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::scatter(xla_a, dim, xla_c, xla_b);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestScatterScalar) {
  torch::Tensor a = torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Scalar b = 1.0f;
  torch::Tensor c = torch::empty({4, 4}, torch::TensorOptions(torch::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    torch::Tensor d = torch::scatter(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::scatter(xla_a, dim, xla_c, b);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestScatterReduceAdd) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter(a, dim, c, b, "add");
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::scatter(xla_a, dim, xla_c, xla_b, "add");
      AllClose(d, xla_d);
    });
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("lazy::scatter_out", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestScatterAdd) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 5}, torch::TensorOptions(torch::kLong));
  for (int dim = 0; dim < 2; ++dim) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 5; j++) {
        c[i][j] = (i + j) % c.sizes()[dim];
      }
    }
    torch::Tensor d = torch::scatter_add(a, dim, c, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = torch::scatter_add(xla_a, dim, xla_c, xla_b);
      AllClose(d, xla_d);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestScatterAddInPlace) {
  torch::Tensor b = torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({4, 4}, torch::TensorOptions(torch::kLong));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      c[i][j] = (i + j) % 4;
    }
  }
  for (int dim = 0; dim < 2; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor a =
          torch::rand({4, 4}, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor d = a.scatter_add_(dim, c, b);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c = CopyToDevice(c, device);
      torch::Tensor xla_d = xla_a.scatter_add_(dim, xla_c, xla_b);
      AllClose(d, xla_d);
      AllClose(a, xla_a);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexSelect) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4}, torch::TensorOptions(scalar_type));
    for (torch::ScalarType index_scalar_type : {torch::kInt, torch::kLong}) {
      torch::Tensor b =
          torch::empty({2}, torch::TensorOptions(index_scalar_type));
      b[0] = 0;
      b[1] = 2;
      torch::Tensor c0 = torch::index_select(a, 0, b);
      torch::Tensor c1 = torch::index_select(a, 1, b);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c0 = torch::index_select(xla_a, 0, xla_b);
        torch::Tensor xla_c1 = torch::index_select(xla_a, 1, xla_b);
        AllEqual(c0, xla_c0);
        AllEqual(c1, xla_c1);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexSelectRank0) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor a =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4}, torch::TensorOptions(scalar_type));
    torch::Tensor b =
        torch::scalar_tensor(2, torch::TensorOptions(torch::kLong));
    torch::Tensor c0 = torch::index_select(a, 0, b);
    torch::Tensor c1 = torch::index_select(a, 1, b);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_a = CopyToDevice(a, device);
      torch::Tensor xla_b = CopyToDevice(b, device);
      torch::Tensor xla_c0 = torch::index_select(xla_a, 0, xla_b);
      torch::Tensor xla_c1 = torch::index_select(xla_a, 1, xla_b);
      AllEqual(c0, xla_c0);
      AllEqual(c1, xla_c1);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestInverse) {
  torch::Tensor a = torch::randn({5, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::inverse(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::inverse(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-4);
  });
}

TEST_F(AtenLtcTsTensorTest, TestIsnan) {
  torch::Tensor a = torch::tensor({1.0, 2.0, std::nan("1"), 4.0},
                                  torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::isnan(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::isnan(xla_a);
    AllEqual(b, xla_b);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("lazy::isnan", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestExpand) {
  torch::Tensor a = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.expand({2, 3, 4}, /*implicit=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = xla_a.expand({2, 3, 4}, /*implicit=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestExpandBack) {
  torch::Tensor a = torch::rand({3, 1}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = a.expand({3, 4}, /*implicit=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = xla_a.expand({3, 4}, /*implicit=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestExpandAs) {
  torch::Tensor a = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::native::expand_as(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::native::expand_as(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEye) {
  int n = 5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor out = torch::eye(n, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_out =
        torch::eye(n, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(out, xla_out);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEyeWide) {
  int lines = 3;
  int cols = 5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor out =
        torch::eye(lines, cols, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_out = torch::eye(
        lines, cols, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(out, xla_out);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEyeNarrow) {
  int lines = 5;
  int cols = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor out =
        torch::eye(lines, cols, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_out = torch::eye(
        lines, cols, torch::TensorOptions(torch::kFloat).device(device));
    AllClose(out, xla_out);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBroadcastTensors) {
  torch::Tensor a = torch::rand({2, 1, 1}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2, 1}, torch::TensorOptions(torch::kFloat));
  std::vector<torch::Tensor> c = torch::broadcast_tensors({a, b});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    std::vector<torch::Tensor> xla_c = torch::broadcast_tensors({xla_a, xla_b});
    ASSERT_EQ(c.size(), xla_c.size());
    for (size_t i = 0; i < c.size(); ++i) {
      AllClose(c[i], xla_c[i]);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestOneIndex) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result = torch::index(params, {indices});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices = CopyToDevice(indices, device);
      torch::Tensor xla_result = torch::index(xla_params, {xla_indices});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestOneIndexTransfer) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result = torch::index(params, {indices});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_result = torch::index(xla_params, {indices});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestNonzero) {
  torch::Tensor a = torch::zeros({4, 2}, torch::TensorOptions(torch::kFloat));
  a[0][1] = 1.0;
  a[1][0] = 2.0;
  a[3][1] = 3.0;
  torch::Tensor b = torch::nonzero(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::nonzero(xla_a);
    AllClose(b, xla_b);

    if (DebugUtil::ExperimentEnabled("nonzero") &&
        bridge::AtenDeviceToLtcDevice(device).hw_type == DeviceType::TPU) {
      // If the nonzero support is enabled, we must not see any aten:: calls.
      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    }
    ResetCounters();
  });
}

TEST_F(AtenLtcTsTensorTest, TestMaskedSelect) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b =
      torch::randint(0, 2, {5}, torch::TensorOptions(torch::kBool));
  torch::Tensor c = torch::masked_select(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::masked_select(xla_a, xla_b);
    AllClose(c, xla_c);

    if (DebugUtil::ExperimentEnabled("masked_select") &&
        bridge::AtenDeviceToLtcDevice(device).hw_type == DeviceType::TPU) {
      // If the masked_select support is enabled, we must not see any aten::
      // calls.
      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    }
    ResetCounters();
  });
}

TEST_F(AtenLtcTsTensorTest, TestMaskedScatter) {
  torch::Tensor a = torch::rand({3, 5}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b =
      torch::randint(0, 2, {3, 5}, torch::TensorOptions(torch::kBool));
  torch::Tensor c = torch::rand({15}, torch::TensorOptions(torch::kFloat));
  torch::Tensor d = torch::masked_scatter(a, b, c);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::masked_scatter(xla_a, xla_b, xla_c);
    AllClose(d, xla_d);

    if (DebugUtil::ExperimentEnabled("masked_scatter") &&
        bridge::AtenDeviceToLtcDevice(device).hw_type == DeviceType::TPU) {
      // If the masked_select support is enabled, we must not see any aten::
      // calls.
      ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
    }
    ResetCounters();
  });
}

TEST_F(AtenLtcTsTensorTest, TestMultiIndexHeadNull) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices_null;
    torch::Tensor indices_0 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor indices_1 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result =
        torch::index(params, {indices_null, indices_0, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor xla_result = torch::index(
          xla_params, {indices_null, xla_indices_0, xla_indices_1});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMultiIndexMiddleNull) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices_0 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor indices_null;
    torch::Tensor indices_1 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result =
        torch::index(params, {indices_0, indices_null, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor xla_result = torch::index(
          xla_params, {xla_indices_0, indices_null, xla_indices_1});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMultiIndexTailNull) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices_0 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor indices_null;
    torch::Tensor indices_1 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result =
        torch::index(params, {indices_0, indices_1, indices_null});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor xla_result = torch::index(
          xla_params, {xla_indices_0, xla_indices_1, indices_null});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMultiIndexMiddleBroadcast) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices_0 =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor indices_1 =
        torch::randint(-3, 3, {2, 1, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor result = torch::index(params, {indices_0, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor xla_result =
          torch::index(xla_params, {xla_indices_0, xla_indices_1});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMultiIndexTailBroadcast) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices_0 =
        torch::randint(-3, 3, {2, 1, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor indices_1 =
        torch::randint(-3, 3, {2, 1}, torch::TensorOptions(torch::kLong));
    torch::Tensor result = torch::index(params, {indices_0, indices_1});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
      torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
      torch::Tensor xla_result =
          torch::index(xla_params, {xla_indices_0, xla_indices_1});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaskIndex) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({2, 2}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {2, 2}, torch::TensorOptions(scalar_type));
    torch::Tensor indices =
        torch::randint(0, 2, {2, 2}, torch::TensorOptions(torch::kBool));
    torch::Tensor result = torch::index(params, {indices});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_params = CopyToDevice(params, device);
      torch::Tensor xla_indices = CopyToDevice(indices, device);
      torch::Tensor xla_result = torch::index(xla_params, {xla_indices});
      AllEqual(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestOneIndexPut) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor indices =
        torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
    torch::Tensor values =
        isFloatingType(scalar_type)
            ? torch::rand({3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices = CopyToDevice(indices, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result =
            torch::index_put(xla_params, {xla_indices}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestOneIndexPutInPlace) {
  torch::Tensor indices =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor values =
        torch::ones({3, 5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor params =
            isFloatingType(scalar_type)
                ? torch::rand({4, 3, 5, 6, 7},
                              torch::TensorOptions(scalar_type))
                : torch::randint(100, {4, 3, 5, 6, 7},
                                 torch::TensorOptions(scalar_type));
        torch::Tensor xla_params = CopyToDevice(params.clone(), device);
        torch::Tensor result =
            torch::index_put_(params, {indices}, values, accumulate);
        torch::Tensor xla_indices = CopyToDevice(indices, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put_(xla_params, {xla_indices},
                                                     xla_values, accumulate);
        AllEqual(result, xla_result);
        AllEqual(params, xla_params);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestOneIndexPutTransfer) {
  torch::Tensor indices =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({3, 5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result =
            torch::index_put(xla_params, {indices}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMultiIndexPut) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result =
          torch::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {xla_indices_0, xla_indices_1}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMultiIndexPutHeadNull) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_null;
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 3, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 3, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({3, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result = torch::index_put(
          params, {indices_null, indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {indices_null, xla_indices_0, xla_indices_1},
            xla_values, accumulate);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMultiIndexPutMiddleNull) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_null;
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 3, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 3, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({3, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result = torch::index_put(
          params, {indices_0, indices_null, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {xla_indices_0, indices_null, xla_indices_1},
            xla_values, accumulate);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMultiIndexPutTailNull) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_null;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 3, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 3, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({3, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result = torch::index_put(
          params, {indices_0, indices_1, indices_null}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {xla_indices_0, xla_indices_1, indices_null},
            xla_values, accumulate);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMultiIndexPutMiddleBroadcast) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 1, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result =
          torch::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {xla_indices_0, xla_indices_1}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMultiIndexPutTailBroadcast) {
  torch::Tensor indices_0 =
      torch::randint(-3, 3, {2, 1, 3}, torch::TensorOptions(torch::kLong));
  torch::Tensor indices_1 =
      torch::randint(-3, 3, {2, 1}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({4, 3, 5, 6, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {4, 3, 5, 6, 7},
                             torch::TensorOptions(scalar_type));
    torch::Tensor values =
        torch::ones({5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      torch::Tensor result =
          torch::index_put(params, {indices_0, indices_1}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices_0 = CopyToDevice(indices_0, device);
        torch::Tensor xla_indices_1 = CopyToDevice(indices_1, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::index_put(
            xla_params, {xla_indices_0, xla_indices_1}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaskIndexPut) {
  torch::Tensor indices =
      torch::tensor({0, 1}, torch::TensorOptions(torch::kByte))
          .to(torch::kBool);
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor params =
        isFloatingType(scalar_type)
            ? torch::rand({2, 2}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {2, 2}, torch::TensorOptions(scalar_type));
    torch::Tensor values = torch::ones({2}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      torch::Tensor result =
          torch::index_put(params, {indices}, values, accumulate);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_params = CopyToDevice(params, device);
        torch::Tensor xla_indices = CopyToDevice(indices, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result =
            torch::index_put(xla_params, {xla_indices}, xla_values, accumulate);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexPutImpl) {
  torch::Tensor indices =
      torch::randint(-3, 3, {2, 4, 3}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor values =
        torch::ones({3, 5, 6, 7}, torch::TensorOptions(scalar_type));
    for (bool accumulate : {false, true}) {
      if (accumulate && IsCuda()) {
        GTEST_SKIP();
      }
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor params =
            isFloatingType(scalar_type)
                ? torch::rand({4, 3, 5, 6, 7},
                              torch::TensorOptions(scalar_type))
                : torch::randint(100, {4, 3, 5, 6, 7},
                                 torch::TensorOptions(scalar_type));
        torch::Tensor xla_params = CopyToDevice(params.clone(), device);
        torch::Tensor result = torch::_index_put_impl_(
            params, {indices}, values, accumulate, /*unsafe=*/true);
        torch::Tensor xla_indices = CopyToDevice(indices, device);
        torch::Tensor xla_values = CopyToDevice(values, device);
        torch::Tensor xla_result = torch::_index_put_impl_(
            xla_params, {xla_indices}, xla_values, accumulate, /*unsafe=*/true);
        AllEqual(result, xla_result);
        AllEqual(params, xla_params);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexFillWithScalar) {
  torch::Tensor index =
      torch::tensor({0, 2}, torch::TensorOptions(torch::kLong));
  torch::Scalar value = 42;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4, 5}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4, 5}, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_result =
            torch::index_fill(xla_base, dim, xla_index, value);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexFillWithScalarInPlace) {
  torch::Tensor index =
      torch::tensor({0, 2}, torch::TensorOptions(torch::kLong));
  torch::Scalar value = 42;
  int rank = 3;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor base =
            isFloatingType(scalar_type)
                ? torch::rand({3, 4, 5}, torch::TensorOptions(scalar_type))
                : torch::randint(100, {3, 4, 5},
                                 torch::TensorOptions(scalar_type));
        torch::Tensor xla_base = CopyToDevice(base.clone(), device);
        torch::Tensor result = base.index_fill_(dim, index, value);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_result = xla_base.index_fill_(dim, xla_index, value);
        AllEqual(result, xla_result);
        AllEqual(base, xla_base);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexFillWithTensor) {
  torch::Tensor index =
      torch::tensor({0, 2}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4, 5}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4, 5}, torch::TensorOptions(scalar_type));
    torch::Tensor value =
        torch::scalar_tensor(42, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            torch::index_fill(xla_base, dim, xla_index, xla_value);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexFillWithTensorInPlace) {
  torch::Tensor index =
      torch::tensor({0, 2}, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor value =
        torch::scalar_tensor(42, torch::TensorOptions(scalar_type));
    int rank = 3;
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor base =
            isFloatingType(scalar_type)
                ? torch::rand({3, 4, 5}, torch::TensorOptions(scalar_type))
                : torch::randint(100, {3, 4, 5},
                                 torch::TensorOptions(scalar_type));
        torch::Tensor xla_base = CopyToDevice(base.clone(), device);
        torch::Tensor result = base.index_fill_(dim, index, value);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            xla_base.index_fill_(dim, xla_index, xla_value);
        AllEqual(result, xla_result);
        AllEqual(base, xla_base);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexFillRank0) {
  torch::Tensor index =
      torch::scalar_tensor(2, torch::TensorOptions(torch::kLong));
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({3, 4, 5}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {3, 4, 5}, torch::TensorOptions(scalar_type));
    torch::Tensor value =
        torch::scalar_tensor(42, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor result = torch::index_fill(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            torch::index_fill(xla_base, dim, xla_index, xla_value);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexAdd) {
  int index_size = 10;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {5, 3, 7}, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      for (torch::ScalarType index_scalar_type : {torch::kInt, torch::kLong}) {
        torch::Tensor index =
            torch::randint(0, base.size(dim), {index_size},
                           torch::TensorOptions(index_scalar_type));
        std::vector<int64_t> value_sizes(base.sizes().begin(),
                                         base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        torch::Tensor value =
            isFloatingType(scalar_type)
                ? torch::rand(value_sizes, torch::TensorOptions(scalar_type))
                : torch::randint(100, value_sizes,
                                 torch::TensorOptions(scalar_type));
        torch::Tensor result = torch::index_add(base, dim, index, value);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_base = CopyToDevice(base, device);
          torch::Tensor xla_index = CopyToDevice(index, device);
          torch::Tensor xla_value = CopyToDevice(value, device);
          torch::Tensor xla_result =
              torch::index_add(xla_base, dim, xla_index, xla_value);
          AllClose(result, xla_result);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexAddInPlace) {
  int index_size = 10;
  int rank = 3;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor base =
            isFloatingType(scalar_type)
                ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
                : torch::randint(100, {5, 3, 7},
                                 torch::TensorOptions(scalar_type));
        torch::Tensor index =
            torch::randint(0, base.size(dim), {index_size},
                           torch::TensorOptions(torch::kLong));
        std::vector<int64_t> value_sizes(base.sizes().begin(),
                                         base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        torch::Tensor value =
            isFloatingType(scalar_type)
                ? torch::rand(value_sizes, torch::TensorOptions(scalar_type))
                : torch::randint(100, value_sizes,
                                 torch::TensorOptions(scalar_type));
        torch::Tensor xla_base = CopyToDevice(base.clone(), device);
        torch::Tensor result = base.index_add_(dim, index, value);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            xla_base.index_add_(dim, xla_index, xla_value);
        AllClose(result, xla_result);
        AllClose(base, xla_base);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexAddRank0) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {5, 3, 7}, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor index = torch::randint(0, base.size(dim), at::IntArrayRef{},
                                           torch::TensorOptions(torch::kLong));
      std::vector<int64_t> value_sizes(base.sizes().begin(),
                                       base.sizes().end());
      int canonical_dim = dim < 0 ? dim + rank : dim;
      value_sizes[canonical_dim] = 1;
      torch::Tensor value =
          isFloatingType(scalar_type)
              ? torch::rand(value_sizes, torch::TensorOptions(scalar_type))
              : torch::randint(100, value_sizes,
                               torch::TensorOptions(scalar_type));
      torch::Tensor result = torch::index_add(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            torch::index_add(xla_base, dim, xla_index, xla_value);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexCopy) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {5, 3, 7}, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor index =
          torch::randperm(base.size(dim), torch::TensorOptions(torch::kLong));
      torch::Tensor value =
          isFloatingType(scalar_type)
              ? torch::rand(base.sizes(), torch::TensorOptions(scalar_type))
              : torch::randint(100, base.sizes(),
                               torch::TensorOptions(scalar_type));
      torch::Tensor result = torch::index_copy(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            torch::index_copy(xla_base, dim, xla_index, xla_value);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexCopyInPlace) {
  if (IsCuda()) {
    GTEST_SKIP();
  }
  int index_size = 10;
  int rank = 3;
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    for (int dim = -rank; dim < rank; ++dim) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor base =
            isFloatingType(scalar_type)
                ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
                : torch::randint(100, {5, 3, 7},
                                 torch::TensorOptions(scalar_type));
        torch::Tensor index =
            torch::randint(0, base.size(dim), {index_size},
                           torch::TensorOptions(torch::kLong));
        std::vector<int64_t> value_sizes(base.sizes().begin(),
                                         base.sizes().end());
        int canonical_dim = dim < 0 ? dim + rank : dim;
        value_sizes[canonical_dim] = index_size;
        torch::Tensor value =
            isFloatingType(scalar_type)
                ? torch::rand(value_sizes, torch::TensorOptions(scalar_type))
                : torch::randint(100, value_sizes,
                                 torch::TensorOptions(scalar_type));
        torch::Tensor xla_base = CopyToDevice(base.clone(), device);
        torch::Tensor result = base.index_copy_(dim, index, value);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            xla_base.index_copy_(dim, xla_index, xla_value);
        AllEqual(result, xla_result);
        AllEqual(base, xla_base);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestIndexCopyRank0) {
  for (torch::ScalarType scalar_type :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor base =
        isFloatingType(scalar_type)
            ? torch::rand({5, 3, 7}, torch::TensorOptions(scalar_type))
            : torch::randint(100, {5, 3, 7}, torch::TensorOptions(scalar_type));
    int rank = base.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor index = torch::randint(0, base.size(dim), at::IntArrayRef{},
                                           torch::TensorOptions(torch::kLong));
      std::vector<int64_t> value_sizes(base.sizes().begin(),
                                       base.sizes().end());
      int canonical_dim = dim < 0 ? dim + rank : dim;
      value_sizes[canonical_dim] = 1;
      torch::Tensor value =
          isFloatingType(scalar_type)
              ? torch::rand(value_sizes, torch::TensorOptions(scalar_type))
              : torch::randint(100, value_sizes,
                               torch::TensorOptions(scalar_type));
      torch::Tensor result = torch::index_copy(base, dim, index, value);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_base = CopyToDevice(base, device);
        torch::Tensor xla_index = CopyToDevice(index, device);
        torch::Tensor xla_value = CopyToDevice(value, device);
        torch::Tensor xla_result =
            torch::index_copy(xla_base, dim, xla_index, xla_value);
        AllEqual(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestRelu) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::relu(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::relu(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReluInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::relu_(input);
    torch::Tensor xla_output = torch::relu_(xla_input);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestHardshrink) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::hardshrink(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::hardshrink(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestHardSigmoid) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::hardsigmoid(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::hardsigmoid(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestHardSigmoidInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input =
        torch::randn({10}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::hardsigmoid_(input);
    torch::Tensor xla_output = torch::hardsigmoid_(xla_input);
    AllClose(input, xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestHardSigmoidBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardsigmoid(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn({10},
                      torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSoftshrink) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::softshrink(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::softshrink(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestHardtanh) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::hardtanh(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::hardtanh(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestHardtanhInPlace) {
  torch::Tensor input = torch::randn({10}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::hardtanh_(input);
    torch::Tensor xla_output = torch::hardtanh_(xla_input);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLeakyRelu) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  double negative_slope = 0.01;
  torch::Tensor output = torch::leaky_relu(input, negative_slope);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::leaky_relu(xla_input, negative_slope);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLeakyReluInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  double negative_slope = 0.01;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::leaky_relu_(input, negative_slope);
    torch::Tensor xla_output = torch::leaky_relu_(xla_input, negative_slope);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestExp) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::exp(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::exp(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestExpm1) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::expm1(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::expm1(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLog) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::log(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::log(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLog2) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::log2(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::log2(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLog10) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::log10(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::log10(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLog1p) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::log1p(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::log1p(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestErf) {
  torch::Tensor a = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::erf(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::erf(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestErfc) {
  torch::Tensor a = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::erfc(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::erfc(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestErfinv) {
  torch::Tensor a = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::erfinv(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::erfinv(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSqrt) {
  torch::Tensor a =
      torch::abs(torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor b = torch::sqrt(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::sqrt(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRsqrt) {
  torch::Tensor a =
      torch::abs(torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor b = torch::rsqrt(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::rsqrt(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReciprocal) {
  torch::Tensor a = torch::randn({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::reciprocal(a);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::reciprocal(xla_a);
    AllClose(b, xla_b, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestPowTensorScalar) {
  torch::Tensor base = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar exponent = 4.09;
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base, device);
    torch::Tensor xla_result = torch::pow(xla_base, exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestPowTensorScalarInPlace) {
  torch::Tensor base = torch::rand({2, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar exponent = 4.09;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base.clone(), device);
    torch::Tensor result = base.pow_(exponent);
    torch::Tensor xla_result = xla_base.pow_(exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(base, xla_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestPowTensorTensor) {
  torch::Tensor base =
      torch::abs(torch::rand({4, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor exponent = torch::rand({4, 2});
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base, device);
    torch::Tensor xla_exponent = CopyToDevice(exponent, device);
    torch::Tensor xla_result = torch::pow(xla_base, xla_exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestPowTensorTensorInPlace) {
  torch::Tensor base =
      torch::abs(torch::rand({4, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor exponent = torch::rand({4, 2});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base.clone(), device);
    torch::Tensor result = base.pow_(exponent);
    torch::Tensor xla_exponent = CopyToDevice(exponent, device);
    torch::Tensor xla_result = xla_base.pow_(xla_exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
    AllClose(base, xla_base, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestPowTensorTensorBroadcast) {
  torch::Tensor base =
      torch::abs(torch::rand({4, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Tensor exponent = torch::rand({4, 1});
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base, device);
    torch::Tensor xla_exponent = CopyToDevice(exponent, device);
    torch::Tensor xla_result = torch::pow(xla_base, xla_exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestPowScalarTensor) {
  torch::Scalar base = 3.5;
  torch::Tensor exponent = torch::rand({4, 2});
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_exponent = CopyToDevice(exponent, device);
    torch::Tensor xla_result = torch::pow(base, xla_exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestPowIntExponent) {
  torch::Tensor base =
      torch::abs(torch::rand({4, 2}, torch::TensorOptions(torch::kFloat)));
  torch::Scalar exponent = 3;
  torch::Tensor result = torch::pow(base, exponent);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_base = CopyToDevice(base, device);
    torch::Tensor xla_result = torch::pow(xla_base, exponent);
    AllClose(result, xla_result, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFmodScalar) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Scalar divisor = 2.0;
  torch::Tensor b = torch::fmod(a, divisor);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::fmod(xla_a, divisor);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFmodScalarInPlace) {
  torch::Scalar divisor = 2.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = a.fmod_(divisor);
    torch::Tensor xla_b = xla_a.fmod_(divisor);
    AllClose(b, xla_b);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFmodTensor) {
  torch::Tensor a =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 10.0;
  torch::Tensor c = torch::fmod(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::fmod(xla_a, xla_b);
    AllClose(c, xla_c);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFmodTensorInPlace) {
  torch::Tensor b =
      torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 10.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::rand({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor c = a.fmod_(b);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = xla_a.fmod_(xla_b);
    AllClose(c, xla_c);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRemainderScalar) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Scalar divisor = -2.0;
  torch::Tensor b = torch::remainder(a, divisor);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = torch::remainder(xla_a, divisor);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRemainderScalarInPlace) {
  torch::Scalar divisor = -2.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor b = a.remainder_(divisor);
    torch::Tensor xla_b = xla_a.remainder_(divisor);
    AllClose(b, xla_b);
    AllClose(a, xla_a);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRemainderTensor) {
  torch::Tensor a =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
  torch::Tensor b =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 10.0;
  torch::Tensor c = torch::remainder(a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = torch::remainder(xla_a, xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRemainderTensorInPlace) {
  torch::Tensor b =
      torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 10.0;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor a =
        torch::randn({2, 2}, torch::TensorOptions(torch::kFloat)) * 100.0;
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor c = a.remainder_(b);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = xla_a.remainder_(xla_b);
    AllClose(c, xla_c, /*rtol=*/1e-4, /*atol=*/1e-6);
    AllClose(a, xla_a, /*rtol=*/1e-4, /*atol=*/1e-6);
  });
}

TEST_F(AtenLtcTsTensorTest, TestWhere) {
  torch::Tensor a = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 3}, torch::TensorOptions(torch::kByte));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  torch::Tensor d = torch::where(c, a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::where(xla_c, xla_a, xla_b);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenLtcTsTensorTest, TestWhereBroadcast) {
  torch::Tensor a = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::zeros({}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 3}, torch::TensorOptions(torch::kByte));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  torch::Tensor d = torch::where(c, a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::where(xla_c, xla_a, xla_b);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenLtcTsTensorTest, TestWhereAutograd) {
  torch::Tensor a = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({3, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::empty({3, 3}, torch::TensorOptions(torch::kByte));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = i == j;
    }
  }
  torch::Tensor d = torch::_s_where(c, a, b);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    torch::Tensor xla_d = torch::_s_where(xla_c, xla_a, xla_b);
    AllClose(d, xla_d);
  });
}

TEST_F(AtenLtcTsTensorTest, TestThreshold) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  float threshold = 0.4;
  float value = 20;
  torch::Tensor output = torch::threshold(input, threshold, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::threshold(xla_input, threshold, value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestThresholdInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = input.clone();
  float threshold = 0.4;
  float value = 20;
  torch::threshold_(output, threshold, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_output = CopyToDevice(input, device);
    torch::threshold_(xla_output, threshold, value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestElu) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 0.5;
  torch::Scalar scale = 2.5;
  torch::Scalar input_scale = 1.5;
  torch::Tensor output = torch::elu(input, alpha, scale, input_scale);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::elu(xla_input, alpha, scale, input_scale);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEluInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 0.5;
  torch::Scalar scale = 2.5;
  torch::Scalar input_scale = 1.5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::elu_(input, alpha, scale, input_scale);
    torch::Tensor xla_output =
        torch::elu_(xla_input, alpha, scale, input_scale);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSelu) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::selu(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::selu(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSeluInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::selu_(input);
    torch::Tensor xla_output = torch::selu_(xla_input);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestCelu) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 2.5;
  torch::Tensor output = torch::celu(input, alpha);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::celu(xla_input, alpha);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestCeluInPlace) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Scalar alpha = 2.5;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::celu_(input, alpha);
    torch::Tensor xla_output = torch::celu_(xla_input, alpha);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestGelu) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::gelu(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::gelu(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAddMatMul) {
  int in_channels = 32;
  int out_channels = 320;
  int labels = 50;
  torch::Tensor input = torch::rand({in_channels, out_channels},
                                    torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({out_channels, labels}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({labels}, torch::TensorOptions(torch::kFloat));
  // Test beta != 1. through the CPU interop.
  for (double beta : {1., 2.}) {
    torch::Tensor output = torch::addmm(bias, input, weight, /*beta=*/beta);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_weight = CopyToDevice(weight, device);
      torch::Tensor xla_bias = CopyToDevice(bias, device);
      torch::Tensor xla_output =
          torch::addmm(xla_bias, xla_input, xla_weight, /*beta=*/beta);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestEmbedding) {
  torch::Tensor a = torch::rand({32, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor i =
      torch::randint(0, 31, {3, 4}, torch::TensorOptions(torch::kLong));
  torch::Tensor b =
      torch::embedding(a, i, /*padding_idx=*/0, /*scale_grad_by_freq=*/false,
                       /*sparse=*/false);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_i = CopyToDevice(i, device);
    torch::Tensor xla_b = torch::embedding(xla_a, xla_i, /*padding_idx=*/0,
                                           /*scale_grad_by_freq=*/false,
                                           /*sparse=*/false);
    AllClose(b, xla_b);
  });
}

TEST_F(AtenLtcTsTensorTest, TestOneHot) {
  int num_classes = 5;
  torch::Tensor input =
      torch::randint(0, num_classes, {10}, torch::TensorOptions(torch::kLong));
  torch::Tensor output = torch::one_hot(input, num_classes);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::one_hot(xla_input, num_classes);
    AllEqual(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTranspose) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::t(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::t(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTransposeInPlace) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = input.t_();
    torch::Tensor xla_output = xla_input.t_();
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReshape) {
  torch::Tensor input =
      torch::rand({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::reshape(input, {-1, 320});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::reshape(xla_input, {-1, 320});
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestResize) {
  // Testing a resize_() with target size bigger than original size is not
  // possible, as we fill with zeros, while pytorch fills with random garbage.
  torch::Tensor input =
      torch::rand({2, 2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor saved_input = input.clone();
  input.resize_({3, 3});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(saved_input, device);
    xla_input.resize_({3, 3});
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestViewResize) {
  torch::Tensor input =
      torch::zeros({8, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor saved_input = input.clone();
  torch::Tensor output = input.view({4, 4});
  output.resize_({3, 3});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(saved_input, device);
    torch::Tensor xla_output = xla_input.view({4, 4});
    xla_output.resize_({3, 3});
    AllClose(input, xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestView) {
  torch::Tensor input =
      torch::rand({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = input.view({-1, 320});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = xla_input.view({-1, 320});
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestViewMod) {
  torch::Tensor input =
      torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor one = torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = input.view({-1, 320});
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput =
        torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(xinput, device);
    torch::Tensor xla_one = CopyToDevice(one, device);
    torch::Tensor xla_output = xla_input.view({-1, 320});
    xla_output.add_(xla_one, 1.0);
    xla_input.add_(xla_one, 1.0);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestViewModComplex) {
  torch::Tensor input =
      torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor one = torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
  torch::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  torch::Tensor output2 = input.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput =
        torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(xinput, device);
    torch::Tensor xla_one = CopyToDevice(one, device);
    torch::Tensor xla_output1 = xla_input.view({-1, 320});
    xla_output1.add_(xla_one, 1.0);
    torch::Tensor xla_output2 = xla_input.view({-1, 160});
    xla_output2.add_(xla_one, 1.0);
    AllClose(output1, xla_output1);
    AllClose(output2, xla_output2);
  });
}

TEST_F(AtenLtcTsTensorTest, TestViewOfViewMod) {
  torch::Tensor input =
      torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor one = torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
  torch::Tensor output1 = input.view({-1, 320});
  output1.add_(one, 1.0);
  torch::Tensor output2 = output1.view({-1, 160});
  output2.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput =
        torch::zeros({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(xinput, device);
    torch::Tensor xla_one = CopyToDevice(one, device);
    torch::Tensor xla_output1 = xla_input.view({-1, 320});
    xla_output1.add_(xla_one, 1.0);
    torch::Tensor xla_output2 = xla_output1.view({-1, 160});
    xla_output2.add_(xla_one, 1.0);
    AllClose(output1, xla_output1);
    AllClose(output2, xla_output2);
  });
}

TEST_F(AtenLtcTsTensorTest, TestViewSqueezeAddInPlace) {
  torch::Tensor input =
      torch::zeros({2, 3, 1}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> view_size = {2, 3, 1, 1};
  int squeeze_dim = 2;
  torch::Tensor one = torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = input.view(view_size);
    output.squeeze_(squeeze_dim);
    output.add_(one, 1.0);
    torch::Tensor xla_one = CopyToDevice(one, device);
    torch::Tensor xla_output = xla_input.view(view_size);
    xla_output.squeeze_(squeeze_dim);
    xla_output.add_(xla_one, 1.0);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestUnsafeView) {
  torch::Tensor input =
      torch::rand({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::_unsafe_view(input, {-1, 320});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::_unsafe_view(xla_input, {-1, 320});
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestNarrow) {
  torch::Tensor a =
      torch::rand({8, 10, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (lazy_tensors::int64 dim : {1, -3}) {
    for (lazy_tensors::int64 start : {2, -8}) {
      torch::Tensor b = a.narrow(dim, start, 6);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a, device);
        torch::Tensor xla_b = xla_a.narrow(dim, start, 6);
        AllClose(b, xla_b);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestNarrowUpdate) {
  for (lazy_tensors::int64 dim : {1, -2}) {
    for (lazy_tensors::int64 start : {2, -6}) {
      torch::Tensor a =
          torch::rand({3, 8, 3}, torch::TensorOptions(torch::kFloat));
      torch::Tensor a_copy = a.clone();
      torch::Tensor b =
          torch::rand({3, 4, 3}, torch::TensorOptions(torch::kFloat));
      torch::Tensor c = a.narrow(dim, start, 4);
      c.add_(b, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a_copy, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = xla_a.narrow(dim, start, 4);
        xla_c.add_(xla_b, 1.0);
        AllClose(c, xla_c);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestNarrowUpdateBaseCheck) {
  for (lazy_tensors::int64 dim : {0, -2}) {
    for (lazy_tensors::int64 start : {2, -6}) {
      torch::Tensor a =
          torch::zeros({8, 3}, torch::TensorOptions(torch::kFloat));
      torch::Tensor a_copy = a.clone();
      torch::Tensor b =
          torch::ones({4, 3}, torch::TensorOptions(torch::kFloat));
      torch::Tensor c = a.narrow(dim, start, 4);
      c.add_(b, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a_copy, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = xla_a.narrow(dim, start, 4);
        xla_c.add_(xla_b, 1.0);
        AllClose(a, xla_a);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestNarrowUpdateTwoSlices) {
  for (lazy_tensors::int64 dim : {0, -2}) {
    for (lazy_tensors::int64 start0 : {2, -6}) {
      for (lazy_tensors::int64 start1 : {6, -2}) {
        torch::Tensor a =
            torch::zeros({8, 3}, torch::TensorOptions(torch::kFloat));
        torch::Tensor a_copy = a.clone();
        torch::Tensor b =
            torch::ones({2, 3}, torch::TensorOptions(torch::kFloat));
        torch::Tensor c = b + 1;
        torch::Tensor d = a.narrow(dim, start0, 2);
        torch::Tensor e = a.narrow(dim, start1, 2);
        d.add_(b, 1.0);
        e.add_(c, 1.0);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a_copy, device);
          torch::Tensor xla_b = CopyToDevice(b, device);
          torch::Tensor xla_c = CopyToDevice(c, device);
          torch::Tensor xla_d = xla_a.narrow(dim, start0, 2);
          torch::Tensor xla_e = xla_a.narrow(dim, start1, 2);
          xla_d.add_(xla_b, 1.0);
          xla_e.add_(xla_c, 1.0);
          AllClose(d, xla_d);
          AllClose(e, xla_e);
          AllClose(a, xla_a);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestNarrowUpdateView) {
  for (lazy_tensors::int64 dim : {0, -3}) {
    for (lazy_tensors::int64 start : {2, -6}) {
      torch::Tensor a =
          torch::rand({8, 2, 3}, torch::TensorOptions(torch::kFloat));
      torch::Tensor a_copy = a.clone();
      torch::Tensor b =
          torch::rand({4, 6}, torch::TensorOptions(torch::kFloat));
      torch::Tensor c = a.narrow(dim, start, 4);
      torch::Tensor d = c.view({4, 6});
      d.add_(b, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_a = CopyToDevice(a_copy, device);
        torch::Tensor xla_b = CopyToDevice(b, device);
        torch::Tensor xla_c = xla_a.narrow(dim, start, 4);
        torch::Tensor xla_d = xla_c.view({4, 6});
        xla_d.add_(xla_b, 1.0);
        AllClose(d, xla_d);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestNarrowInNarrowUpdate) {
  for (lazy_tensors::int64 dim : {1, -2}) {
    for (lazy_tensors::int64 start0 : {1, -7}) {
      for (lazy_tensors::int64 start1 : {1, -5}) {
        torch::Tensor a =
            torch::rand({3, 8, 3}, torch::TensorOptions(torch::kFloat));
        torch::Tensor a_copy = a.clone();
        torch::Tensor b =
            torch::rand({3, 2, 3}, torch::TensorOptions(torch::kFloat));
        torch::Tensor c = a.narrow(dim, start0, 6);
        torch::Tensor d = c.narrow(dim, start1, 2);
        d.add_(b, 1.0);
        ForEachDevice([&](const torch::Device& device) {
          torch::Tensor xla_a = CopyToDevice(a_copy, device);
          torch::Tensor xla_b = CopyToDevice(b, device);
          torch::Tensor xla_c = xla_a.narrow(dim, start0, 6);
          torch::Tensor xla_d = xla_c.narrow(dim, start1, 2);
          xla_d.add_(xla_b, 1.0);
          AllClose(a, xla_a);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestNarrowCopy) {
  for (lazy_tensors::int64 dim : {1, -3}) {
    for (lazy_tensors::int64 start : {2, -8}) {
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor input =
            torch::rand({8, 10, 4, 4}, torch::TensorOptions(torch::kFloat));
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor result = input.narrow_copy(dim, start, 6);
        input.add_(1);
        torch::Tensor xla_result = xla_input.narrow_copy(dim, start, 6);
        xla_input.add_(1);
        AllClose(result, xla_result);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestViewAs) {
  torch::Tensor input =
      torch::rand({32, 20, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor empty = torch::empty({32, 320});
  torch::Tensor output = input.view_as(empty);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_empty = CopyToDevice(empty, device);
    torch::Tensor xla_output = xla_input.view_as(xla_empty);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLogSoftmax) {
  torch::Tensor input =
      torch::rand({5, 3, 4, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::log_softmax(input, dim);
      torch::Tensor xla_output = torch::log_softmax(xla_input, dim);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestLogSoftmaxCast) {
  torch::Tensor input =
      torch::rand({5, 3, 4, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::log_softmax(input, dim, torch::kDouble);
      torch::Tensor xla_output =
          torch::log_softmax(xla_input, dim, torch::kDouble);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestLogSoftmaxWrapper) {
  torch::Tensor input =
      torch::rand({10, 2, 6, 4}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output =
          torch::_log_softmax(input, dim, /*half_to_float=*/false);
      torch::Tensor xla_output =
          torch::_log_softmax(xla_input, dim, /*half_to_float=*/false);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestSoftmax) {
  torch::Tensor input =
      torch::rand({10, 2, 6, 4}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::softmax(input, dim);
      torch::Tensor xla_output = torch::softmax(xla_input, dim);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestSoftmaxCast) {
  torch::Tensor input =
      torch::rand({10, 2, 6, 4}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output = torch::softmax(input, dim, torch::kDouble);
      torch::Tensor xla_output = torch::softmax(xla_input, dim, torch::kDouble);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestSoftmaxWrapper) {
  torch::Tensor input =
      torch::rand({10, 2, 6, 4}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    int rank = input.dim();
    for (int dim = -rank; dim < rank; ++dim) {
      torch::Tensor output =
          torch::_softmax(input, dim, /*half_to_float=*/false);
      torch::Tensor xla_output =
          torch::_softmax(xla_input, dim, /*half_to_float=*/false);
      AllClose(output, xla_output, /*rtol=*/1e-3);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestSoftplus) {
  torch::Tensor input =
      torch::rand({2, 1, 4, 6}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::softplus(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::softplus(xla_input);
    AllClose(output, xla_output, /*rtol=*/1e-4);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool1D) {
  torch::Tensor input =
      torch::rand({1, 16, 56}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output =
              torch::max_pool1d(input, /*kernel_size=*/{kernel_size},
                                /*stride=*/{stride},
                                /*padding=*/{padding}, /*dilation=*/{dilation},
                                /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::max_pool1d(xla_input,
                                  /*kernel_size=*/{kernel_size},
                                  /*stride=*/{stride},
                                  /*padding=*/{padding},
                                  /*dilation=*/{dilation},
                                  /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool2D) {
  torch::Tensor input =
      torch::rand({1, 4, 14, 14}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::max_pool2d(xla_input,
                                  /*kernel_size=*/{kernel_size, kernel_size},
                                  /*stride=*/{stride, stride},
                                  /*padding=*/{padding, padding},
                                  /*dilation=*/{dilation, dilation},
                                  /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool2DWithIndices) {
  torch::Tensor input =
      torch::rand({1, 4, 14, 14}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          auto outputs = torch::max_pool2d_with_indices(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            auto xla_outputs = torch::max_pool2d_with_indices(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size},
                /*stride=*/{stride, stride},
                /*padding=*/{padding, padding},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(std::get<0>(outputs), std::get<0>(xla_outputs));
            AllClose(std::get<1>(outputs), std::get<1>(xla_outputs));
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool2DNonSquare) {
  torch::Tensor input =
      torch::rand({1, 4, 14, 14}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1},
              /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::max_pool2d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1},
                /*stride=*/{stride, stride + 1},
                /*padding=*/{padding, padding + 1},
                /*dilation=*/{dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool3D) {
  torch::Tensor input =
      torch::rand({1, 1, 8, 8, 8}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::max_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool3DWithIndices) {
  torch::Tensor input =
      torch::rand({1, 1, 8, 8, 8}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          auto outputs = torch::max_pool3d_with_indices(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            auto xla_outputs = torch::max_pool3d_with_indices(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);

            AllClose(std::get<0>(outputs), std::get<0>(xla_outputs));
            AllClose(std::get<1>(outputs), std::get<1>(xla_outputs));
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool3DIncompleteAttributes) {
  torch::Tensor input =
      torch::rand({1, 1, 8, 8, 8}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{},
              /*padding=*/{padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::max_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{},
                /*padding=*/{padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool3DNonSquare) {
  torch::Tensor input =
      torch::rand({1, 1, 8, 8, 8}, torch::TensorOptions(torch::kFloat));
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
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::max_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
                /*stride=*/{stride, stride + 1, stride},
                /*padding=*/{padding, padding + 1, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool2DNoBatch) {
  torch::Tensor input =
      torch::rand({4, 14, 14}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::max_pool2d(xla_input,
                                  /*kernel_size=*/{kernel_size, kernel_size},
                                  /*stride=*/{stride, stride},
                                  /*padding=*/{padding, padding},
                                  /*dilation=*/{dilation, dilation},
                                  /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool3DNoBatch) {
  torch::Tensor input =
      torch::rand({1, 8, 8, 8}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output = torch::max_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::max_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*dilation=*/{dilation, dilation, dilation},
                /*ceil_mode=*/ceil_mode);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool1D) {
  torch::Tensor input =
      torch::rand({4, 1, 28}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output =
              torch::avg_pool1d(input, /*kernel_size=*/{kernel_size},
                                /*stride=*/{stride},
                                /*padding=*/{padding}, /*ceil_mode=*/ceil_mode,
                                /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::avg_pool1d(xla_input,
                                  /*kernel_size=*/{kernel_size},
                                  /*stride=*/{stride},
                                  /*padding=*/{padding},
                                  /*ceil_mode=*/ceil_mode,
                                  /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool2D) {
  torch::Tensor input =
      torch::rand({2, 1, 14, 14}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            // torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::avg_pool2d(xla_input,
                                  /*kernel_size=*/{kernel_size, kernel_size},
                                  /*stride=*/{stride, stride},
                                  /*padding=*/{padding, padding},
                                  /*ceil_mode=*/ceil_mode,
                                  /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output.to(torch::kCPU));
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool2DNonSquare) {
  torch::Tensor input =
      torch::rand({2, 1, 14, 14}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 4;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size + 1},
              /*stride=*/{stride, stride + 1},
              /*padding=*/{padding, padding + 1}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::avg_pool2d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1},
                /*stride=*/{stride, stride + 1},
                /*padding=*/{padding, padding + 1},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool3D) {
  torch::Tensor input =
      torch::rand({1, 1, 7, 7, 7}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::avg_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool3DIncompleteAttributes) {
  torch::Tensor input =
      torch::rand({1, 1, 7, 7, 7}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{},
              /*padding=*/{padding, padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::avg_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool3DNonSquare) {
  torch::Tensor input =
      torch::rand({1, 1, 7, 7, 7}, torch::TensorOptions(torch::kFloat));
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
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::avg_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size + 1, kernel_size},
                /*stride=*/{stride, stride + 1, stride},
                /*padding=*/{padding, padding + 1, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool2DNoBatch) {
  torch::Tensor input =
      torch::rand({1, 7, 7}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool2d(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::avg_pool2d(xla_input,
                                  /*kernel_size=*/{kernel_size, kernel_size},
                                  /*stride=*/{stride, stride},
                                  /*padding=*/{padding, padding},
                                  /*ceil_mode=*/ceil_mode,
                                  /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool3DNoBatch) {
  torch::Tensor input =
      torch::rand({1, 7, 7, 7}, torch::TensorOptions(torch::kFloat));
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          torch::Tensor output = torch::avg_pool3d(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding}, /*ceil_mode=*/ceil_mode,
              /*count_include_pad=*/count_include_pad);
          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output = torch::avg_pool3d(
                xla_input,
                /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
                /*stride=*/{stride, stride, stride},
                /*padding=*/{padding, padding, padding},
                /*ceil_mode=*/ceil_mode,
                /*count_include_pad=*/count_include_pad);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAdaptiveAvgPool2D) {
  torch::Tensor input =
      torch::rand({4, 1, 28, 28}, torch::TensorOptions(torch::kFloat));
  for (int64_t output_size : {7, 4}) {
    torch::Tensor output =
        torch::adaptive_avg_pool2d(input, {output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output =
          torch::adaptive_avg_pool2d(xla_input, {output_size, output_size});
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAdaptiveAvgPool3D) {
  torch::Tensor input =
      torch::rand({9, 4, 56, 28, 28}, torch::TensorOptions(torch::kFloat));
  for (int64_t output_size : {7, 4}) {
    torch::Tensor output = torch::adaptive_avg_pool3d(
        input, {output_size, output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::adaptive_avg_pool3d(
          xla_input, {output_size, output_size, output_size});
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAdaptiveAvgPool3DNoBatch) {
  torch::Tensor input =
      torch::rand({3, 56, 28, 28}, torch::TensorOptions(torch::kFloat));
  for (int64_t output_size : {7, 4}) {
    torch::Tensor output = torch::adaptive_avg_pool3d(
        input, {output_size, output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::adaptive_avg_pool3d(
          xla_input, {output_size, output_size, output_size});
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAdaptiveAvgPool2DNoBatch) {
  torch::Tensor input =
      torch::rand({1, 56, 56}, torch::TensorOptions(torch::kFloat));
  for (int64_t output_size : {7, 8}) {
    torch::Tensor output =
        torch::adaptive_avg_pool2d(input, {output_size, output_size});
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output =
          torch::adaptive_avg_pool2d(xla_input, {output_size, output_size});
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxUnpool2D) {
  int kernel_size = 2;
  torch::Tensor input =
      torch::rand({2, 2, 8, 8}, torch::TensorOptions(torch::kFloat));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool2d_with_indices(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size({input.size(2), input.size(3)});
          at::Tensor utensor =
              torch::max_unpool2d(output, indices, output_size);

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_output = CopyToDevice(output, device);
            torch::Tensor xla_indices = CopyToDevice(indices, device);
            at::Tensor xla_utensor =
                torch::max_unpool2d(xla_output, xla_indices, output_size);
            AllClose(utensor, xla_utensor);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxUnpool3D) {
  int kernel_size = 2;
  torch::Tensor input =
      torch::rand({1, 1, 4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        // Test dilation through the CPU interop.
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool3d_with_indices(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size(
              {input.size(2), input.size(3), input.size(4)});
          at::Tensor utensor = torch::max_unpool3d(
              output, indices, output_size, /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding});

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_output = CopyToDevice(output, device);
            torch::Tensor xla_indices = CopyToDevice(indices, device);
            at::Tensor xla_utensor =
                torch::max_unpool3d(xla_output, xla_indices, output_size,
                                    /*stride=*/{stride, stride, stride},
                                    /*padding=*/{padding, padding, padding});
            AllClose(utensor, xla_utensor);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestNllLoss) {
  int batch = 6;
  int classes = 2;
  // TODO(asuhan): Fix the torch::kDouble case.
  for (auto dtype : {torch::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input =
            torch::rand({batch, classes}, torch::TensorOptions(dtype));
        torch::Tensor target =
            torch::randint(std::min(ignore_index, 0), classes, {batch},
                           torch::TensorOptions(torch::kLong));
        torch::Tensor weight;
        if (def_weight) {
          weight = torch::rand({classes}, torch::TensorOptions(dtype));
        }
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean, torch::Reduction::Sum,
              torch::Reduction::None}) {
          torch::Tensor output =
              torch::nll_loss(/*self=*/input, /*target=*/target,
                              /*weight=*/weight,
                              /*reduction=*/reduction,
                              /*ignore_index=*/ignore_index);

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_target = CopyToDevice(target, device);
            torch::Tensor xla_weight =
                def_weight ? CopyToDevice(weight, device) : torch::Tensor();
            torch::Tensor xla_output = torch::nll_loss(
                /*self=*/xla_input, /*target=*/xla_target,
                /*weight=*/xla_weight,
                /*reduction=*/reduction, /*ignore_index=*/ignore_index);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestNllLoss2d) {
  int batch = 6;
  int classes = 2;
  int height = 3;
  int width = 3;
  // TODO(asuhan): Fix the torch::kDouble case.
  for (auto dtype : {torch::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input = torch::rand({batch, classes, height, width},
                                          torch::TensorOptions(dtype));
        torch::Tensor target = torch::randint(
            std::min(ignore_index, 0), classes, {batch, height, width},
            torch::TensorOptions(torch::kLong));
        torch::Tensor weight;
        if (def_weight) {
          weight = torch::rand({classes}, torch::TensorOptions(dtype));
        }
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean, torch::Reduction::Sum,
              torch::Reduction::None}) {
          torch::Tensor output =
              torch::nll_loss2d(/*self=*/input, /*target=*/target,
                                /*weight=*/weight,
                                /*reduction=*/reduction,
                                /*ignore_index=*/ignore_index);

          ForEachDevice([&](const torch::Device& device) {
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_target = CopyToDevice(target, device);
            torch::Tensor xla_weight =
                def_weight ? CopyToDevice(weight, device) : torch::Tensor();
            torch::Tensor xla_output = torch::nll_loss2d(
                /*self=*/xla_input, /*target=*/xla_target,
                /*weight=*/xla_weight,
                /*reduction=*/reduction, /*ignore_index=*/ignore_index);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestSmoothL1Loss) {
  torch::Tensor input =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    for (double beta : {0.25, 1.}) {
      torch::Tensor output =
          torch::smooth_l1_loss(input, target, reduction, beta);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_target = CopyToDevice(target, device);
        torch::Tensor xla_output =
            torch::smooth_l1_loss(xla_input, xla_target, reduction, beta);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestL1Loss) {
  torch::Tensor input =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    torch::Tensor output = torch::l1_loss(input, target, reduction);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_target = CopyToDevice(target, device);
      torch::Tensor xla_output =
          torch::l1_loss(xla_input, xla_target, reduction);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestL1LossBackward) {
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::l1_loss(inputs[0], inputs[1], reduction);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand({2, 4},
                       torch::TensorOptions(torch::kFloat).requires_grad(true)),
           torch::rand({2, 4}, torch::TensorOptions(torch::kFloat))},
          device, testfn);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMseLoss) {
  torch::Tensor input =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor target =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    torch::Tensor output = torch::mse_loss(input, target, reduction);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_target = CopyToDevice(target, device);
      torch::Tensor xla_output =
          torch::mse_loss(xla_input, xla_target, reduction);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMseLossBackward) {
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::mse_loss(inputs[0], inputs[1], reduction);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand({2, 4},
                       torch::TensorOptions(torch::kFloat).requires_grad(true)),
           torch::rand({2, 4}, torch::TensorOptions(torch::kFloat))},
          device, testfn);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestBatchNorm1D) {
  int num_features = 3;
  torch::Tensor input =
      torch::rand({2, num_features, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_mean =
      torch::zeros({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_var =
      torch::ones({num_features}, torch::TensorOptions(torch::kFloat));
  double momentum = 0.1;
  double eps = 0.5;
  torch::Tensor undef;
  for (bool training : {true, false}) {
    for (bool undef_weight_bias : {false, true}) {
      torch::Tensor output = torch::batch_norm(
          /*input=*/input, /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean, /*running_var=*/running_var,
          /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_weight =
            undef_weight_bias ? undef : CopyToDevice(weight, device);
        torch::Tensor xla_bias =
            undef_weight_bias ? undef : CopyToDevice(bias, device);
        torch::Tensor xla_running_mean = CopyToDevice(running_mean, device);
        torch::Tensor xla_running_var = CopyToDevice(running_var, device);
        torch::Tensor xla_output = torch::batch_norm(
            /*input=*/xla_input, /*weight=*/xla_weight, /*bias=*/xla_bias,
            /*running_mean=*/xla_running_mean, /*running_var=*/xla_running_var,
            /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
            /*cudnn_enabled=*/false);
        AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestBatchNorm2D) {
  int num_features = 3;
  torch::Tensor input =
      torch::rand({2, num_features, 4, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor bias =
      torch::rand({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_mean =
      torch::zeros({num_features}, torch::TensorOptions(torch::kFloat));
  torch::Tensor running_var =
      torch::ones({num_features}, torch::TensorOptions(torch::kFloat));
  double momentum = 0.1;
  double eps = 0.5;
  torch::Tensor undef;
  for (bool training : {true, false}) {
    for (bool undef_weight_bias : {false, true}) {
      torch::Tensor output = torch::batch_norm(
          /*input=*/input, /*weight=*/undef_weight_bias ? undef : weight,
          /*bias=*/undef_weight_bias ? undef : bias,
          /*running_mean=*/running_mean, /*running_var=*/running_var,
          /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
          /*cudnn_enabled=*/false);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_weight =
            undef_weight_bias ? undef : CopyToDevice(weight, device);
        torch::Tensor xla_bias =
            undef_weight_bias ? undef : CopyToDevice(bias, device);
        torch::Tensor xla_running_mean = CopyToDevice(running_mean, device);
        torch::Tensor xla_running_var = CopyToDevice(running_var, device);
        torch::Tensor xla_output = torch::batch_norm(
            /*input=*/xla_input, /*weight=*/xla_weight, /*bias=*/xla_bias,
            /*running_mean=*/xla_running_mean, /*running_var=*/xla_running_var,
            /*training=*/training, /*momentum=*/momentum, /*eps=*/eps,
            /*cudnn_enabled=*/false);
        AllClose(output, xla_output, /*rtol=*/1e-3, /*atol=*/1e-5);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestDim) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    EXPECT_EQ(input.dim(), xla_input.dim());
  });
}

TEST_F(AtenLtcTsTensorTest, TestContiguous) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::native::contiguous(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::native::contiguous(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSqueezeAll) {
  torch::Tensor input =
      torch::rand({2, 1, 3, 1}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::squeeze(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::squeeze(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSqueezeAllInPlace) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input =
        torch::rand({2, 1, 3, 1}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = input.squeeze_();
    torch::Tensor xla_output = xla_input.squeeze_();
    AllClose(output, xla_output);
    AllClose(input, xla_input);
    ASSERT_EQ(input.dim(), xla_input.dim());
    for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
      ASSERT_EQ(input.size(dim_idx), xla_input.size(dim_idx));
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestSqueezeOne) {
  torch::Tensor input =
      torch::rand({2, 1, 3, 1}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor output = torch::squeeze(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::squeeze(xla_input, dim);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestSqueezeOneInPlace) {
  int rank = 4;
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input =
          torch::rand({2, 1, 3, 1}, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor output = input.squeeze_(dim);
      torch::Tensor xla_output = xla_input.squeeze_(dim);
      AllClose(output, xla_output);
      AllClose(input, xla_input);
      ASSERT_EQ(input.dim(), xla_input.dim());
      for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
        ASSERT_EQ(input.size(dim_idx), xla_input.size(dim_idx));
      }
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestUnsqueeze) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor output = torch::unsqueeze(input, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::unsqueeze(xla_input, dim);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestUnsqueezeInPlace) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim() + 1;
  for (int dim = -rank; dim < rank; ++dim) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor output = input.unsqueeze_(dim);
      torch::Tensor xla_output = xla_input.unsqueeze_(dim);
      AllClose(output, xla_output);
      AllClose(input, xla_input);
      ASSERT_EQ(input.dim(), xla_input.dim());
      for (int64_t dim_idx = 0; dim_idx < input.dim(); ++dim_idx) {
        ASSERT_EQ(input.size(dim_idx), xla_input.size(dim_idx));
      }
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaskedFill) {
  torch::Tensor input =
      torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor mask =
      torch::randint(0, 2, {2, 3}, torch::TensorOptions(torch::kBool));
  torch::Scalar value(42);
  torch::Tensor result = torch::masked_fill(input, mask, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_mask = CopyToDevice(mask, device);
    torch::Tensor xla_result = torch::masked_fill(xla_input, xla_mask, value);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMaskedFillInPlace) {
  torch::Scalar value(42);
  torch::Tensor mask =
      torch::randint(0, 2, {2, 3}, torch::TensorOptions(torch::kBool));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input =
        torch::rand({2, 3}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_mask = CopyToDevice(mask, device);
    torch::Tensor result = input.masked_fill_(mask, value);
    torch::Tensor xla_result = xla_input.masked_fill_(xla_mask, value);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMaskedFillBroadcast) {
  torch::Tensor input =
      torch::rand({2, 5, 4, 3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor mask =
      torch::randint(0, 2, {4, 1}, torch::TensorOptions(torch::kBool));
  torch::Scalar value(42);
  torch::Tensor result = torch::masked_fill(input, mask, value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_mask = CopyToDevice(mask, device);
    torch::Tensor xla_result = torch::masked_fill(xla_input, xla_mask, value);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFill) {
  torch::Scalar value(42);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input =
        torch::empty({2, 3}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor result = torch::fill_(input, value);
    torch::Tensor xla_result = torch::fill_(xla_input, value);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestFillWithRank0) {
  torch::Tensor value = torch::scalar_tensor(42);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor input =
        torch::empty({2, 3}, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor result = torch::fill_(input, value);
    torch::Tensor xla_value = CopyToDevice(value, device);
    torch::Tensor xla_result = torch::fill_(xla_input, value);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestPermute) {
  torch::Tensor input =
      torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  int rank = input.dim();
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(dims_permutation.begin(), dims_permutation.end(),
                      [rank](int64_t& dim) { dim -= rank; });
      }
      torch::Tensor output = input.permute(dims_permutation);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_output = xla_input.permute(dims_permutation);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestPermuteMod) {
  std::vector<std::vector<int64_t>> dims_permutations = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  std::vector<int64_t> input_sizes = {2, 3, 4};
  int rank = input_sizes.size();
  for (std::vector<int64_t> dims_permutation : dims_permutations) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(dims_permutation.begin(), dims_permutation.end(),
                      [rank](int64_t& dim) { dim -= rank; });
      }
      torch::Tensor input =
          torch::zeros(input_sizes, torch::TensorOptions(torch::kFloat));
      torch::Tensor one =
          torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
      torch::Tensor output = input.permute(dims_permutation);
      output.add_(one, 1.0);
      input.add_(one, 1.0);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xinput =
            torch::zeros(input_sizes, torch::TensorOptions(torch::kFloat));
        torch::Tensor xla_input = CopyToDevice(xinput, device);
        torch::Tensor xla_one = CopyToDevice(one, device);
        torch::Tensor xla_output = xla_input.permute(dims_permutation);
        xla_output.add_(xla_one, 1.0);
        xla_input.add_(xla_one, 1.0);
        AllClose(output, xla_output);
        AllClose(input, xla_input);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestFlip) {
  torch::Tensor input =
      torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<std::vector<int64_t>> dim_powerset = {
      {0}, {1}, {2}, {0, 1}, {1, 2}, {2, 0}, {0, 1, 2}};
  for (std::vector<int64_t> flip_dims : dim_powerset) {
    for (bool negative_dims : {false, true}) {
      if (negative_dims) {
        std::for_each(flip_dims.begin(), flip_dims.end(),
                      [](int64_t& dim) { dim -= 3; });
      }
      torch::Tensor output = torch::flip(input, flip_dims);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        torch::Tensor xla_output = torch::flip(xla_input, flip_dims);
        AllClose(output, xla_output);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestPixelShuffle) {
  torch::Tensor input =
      torch::rand({5, 18, 4, 4}, torch::TensorOptions(torch::kFloat));
  int upscale_factor = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = torch::pixel_shuffle(input, upscale_factor);
    torch::Tensor xla_output = torch::pixel_shuffle(xla_input, upscale_factor);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSumToSize) {
  torch::Tensor input =
      torch::rand({4, 6, 3, 7}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> out_size = {4, 1, 1, 7};
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = input.sum_to_size(out_size);
    torch::Tensor xla_output = xla_input.sum_to_size(out_size);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTransposeDims) {
  torch::Tensor input =
      torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  int dim0 = 0;
  int dim1 = 2;
  torch::Tensor output = torch::transpose(input, dim0, dim1);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::transpose(xla_input, dim0, dim1);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTransposeDimsMod) {
  std::vector<int64_t> input_sizes = {2, 3, 4};
  int dim0 = 0;
  int dim1 = 2;
  torch::Tensor input =
      torch::zeros(input_sizes, torch::TensorOptions(torch::kFloat));
  torch::Tensor one = torch::tensor(1.0, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::transpose(input, dim0, dim1);
  output.add_(one, 1.0);
  input.add_(one, 1.0);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xinput =
        torch::zeros(input_sizes, torch::TensorOptions(torch::kFloat));
    torch::Tensor xla_input = CopyToDevice(xinput, device);
    torch::Tensor xla_one = CopyToDevice(one, device);
    torch::Tensor xla_output = torch::transpose(xla_input, dim0, dim1);
    xla_output.add_(xla_one, 1.0);
    xla_input.add_(xla_one, 1.0);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTransposeDimsInPlace) {
  torch::Tensor input =
      torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  int dim0 = 0;
  int dim1 = 2;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output = input.transpose_(dim0, dim1);
    torch::Tensor xla_output = xla_input.transpose_(dim0, dim1);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSplit) {
  torch::Tensor input =
      torch::rand({7, 8, 9}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int split_size : {2, 3}) {
    for (int dim = -rank; dim < rank; ++dim) {
      std::vector<torch::Tensor> outputs = torch::split(input, split_size, dim);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_input = CopyToDevice(input, device);
        std::vector<torch::Tensor> xla_outputs =
            torch::split(xla_input, split_size, dim);
        ASSERT_EQ(outputs.size(), xla_outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
          AllClose(outputs[i], xla_outputs[i]);
        }
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestSplitEmpty) {
  torch::Tensor input = torch::rand({0}, torch::TensorOptions(torch::kFloat));
  int split_size = 0;
  int dim = 0;
  std::vector<torch::Tensor> outputs = torch::split(input, split_size, dim);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    std::vector<torch::Tensor> xla_outputs =
        torch::split(xla_input, split_size, dim);
    ASSERT_EQ(outputs.size(), xla_outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      AllClose(outputs[i], xla_outputs[i]);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestSplitWithSizes) {
  torch::Tensor input =
      torch::rand({15, 15, 15}, torch::TensorOptions(torch::kFloat));
  int rank = input.dim();
  for (int dim = -rank; dim < rank; ++dim) {
    std::vector<torch::Tensor> outputs =
        torch::split_with_sizes(input, {4, 5, 6}, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      std::vector<torch::Tensor> xla_outputs =
          torch::split_with_sizes(xla_input, {4, 5, 6}, dim);
      ASSERT_EQ(outputs.size(), xla_outputs.size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        AllClose(outputs[i], xla_outputs[i]);
      }
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCrossImplicitDim) {
  std::vector<std::vector<int64_t>> dim_sizes = {
      {4, 5, 3}, {4, 3, 5}, {3, 4, 5}};
  for (auto dim_size : dim_sizes) {
    torch::Tensor input =
        torch::rand(dim_size, torch::TensorOptions(torch::kFloat));
    torch::Tensor other =
        torch::rand(dim_size, torch::TensorOptions(torch::kFloat));
    torch::Tensor result = torch::cross(input, other);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_other = CopyToDevice(other, device);
      torch::Tensor xla_result = torch::cross(xla_input, xla_other);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCrossExplicitDim) {
  std::vector<int64_t> dim_size = {3, 3};
  torch::Tensor input =
      torch::rand(dim_size, torch::TensorOptions(torch::kFloat));
  torch::Tensor other =
      torch::rand(dim_size, torch::TensorOptions(torch::kFloat));
  int rank = dim_size.size();
  for (int dim = -rank; dim < rank; ++dim) {
    torch::Tensor result = torch::cross(input, other, dim);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_other = CopyToDevice(other, device);
      torch::Tensor xla_result = torch::cross(xla_input, xla_other, dim);
      AllClose(result, xla_result);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestCrossZeroDim) {
  torch::Tensor input =
      torch::rand({0, 1, 3, 0}, torch::TensorOptions(torch::kFloat));
  torch::Tensor result = torch::cross(input, input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::cross(xla_input, xla_input);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTriu) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::triu(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestTriuNonSquare) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size + 1}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::triu(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestTriuBatch) {
  int size = 5;
  int batch_size = 3;
  torch::Tensor input = torch::rand({batch_size, size, size},
                                    torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::triu(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::triu(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestTril) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::tril(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestTrilNonSquare) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size + 1}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::tril(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestTrilBatch) {
  int size = 5;
  int batch_size = 3;
  torch::Tensor input = torch::rand({batch_size, size, size},
                                    torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::tril(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::tril(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestTriuInPlace) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input =
          torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor output = input.triu_(diagonal);
      torch::Tensor xla_output = xla_input.triu_(diagonal);
      AllClose(output, xla_output);
      AllClose(input, xla_input);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestTrilInPlace) {
  int size = 5;
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input =
          torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor output = input.tril_(diagonal);
      torch::Tensor xla_output = xla_input.tril_(diagonal);
      AllClose(output, xla_output);
      AllClose(input, xla_input);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestTrace) {
  int n = 5;
  torch::Tensor input =
      torch::rand({n, n}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::trace(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::trace(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTraceWide) {
  int lines = 3;
  int cols = 5;
  torch::Tensor input =
      torch::rand({lines, cols}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::trace(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::trace(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTraceNarrow) {
  int lines = 5;
  int cols = 3;
  torch::Tensor input =
      torch::rand({lines, cols}, torch::TensorOptions(torch::kFloat));
  torch::Tensor output = torch::trace(input);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::trace(xla_input);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestDiagRank1) {
  int size = 7;
  torch::Tensor input =
      torch::rand({size}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -2 * size; diagonal <= 2 * size; ++diagonal) {
    torch::Tensor output = torch::diag(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::diag(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestDiagRank2) {
  int size = 7;
  torch::Tensor input =
      torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::diag(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::diag(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestDiagFlat) {
  torch::Tensor input =
      torch::rand({4, 3, 6, 7}, torch::TensorOptions(torch::kFloat));
  for (int diagonal = -10; diagonal < 10; ++diagonal) {
    torch::Tensor output = torch::diagflat(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::diagflat(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestDiagonal) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::diagonal(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::diagonal(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestDiagonalNonSquare) {
  int size = 5;
  torch::Tensor input =
      torch::rand({size, size + 1}, torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output = torch::diagonal(input, diagonal);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output = torch::diagonal(xla_input, diagonal);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestDiagonalBatch) {
  int size = 5;
  int batch_size = 3;
  int dim1 = 1;
  int dim2 = 2;
  torch::Tensor input = torch::rand({batch_size, size, size},
                                    torch::TensorOptions(torch::kFloat));
  // Test all diagonals and out of bounds (must be no-op).
  for (int diagonal = -size; diagonal <= size; ++diagonal) {
    torch::Tensor output =
        torch::diagonal(input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor xla_input = CopyToDevice(input, device);
      torch::Tensor xla_output =
          torch::diagonal(xla_input, diagonal, /*dim1=*/dim1, /*dim1=*/dim2);
      AllClose(output, xla_output);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestFlatten) {
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
            torch::Tensor xla_input = CopyToDevice(input, device);
            torch::Tensor xla_output =
                torch::flatten(xla_input, start_dim, end_dim);
            AllClose(output, xla_output);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestLogicalAnd) {
  for (torch::ScalarType scalar_type1 :
       {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
        torch::kLong}) {
    torch::Tensor lhs =
        isFloatingType(scalar_type1)
            ? torch::rand({3, 4}, torch::TensorOptions(scalar_type1))
            : torch::randint(0, 100, {3, 4},
                             torch::TensorOptions(scalar_type1));
    for (torch::ScalarType scalar_type2 :
         {torch::kFloat, torch::kByte, torch::kChar, torch::kShort, torch::kInt,
          torch::kLong}) {
      torch::Tensor rhs =
          isFloatingType(scalar_type2)
              ? torch::rand({3, 4}, torch::TensorOptions(scalar_type2))
              : torch::randint(1, 100, {3, 4},
                               torch::TensorOptions(scalar_type2));
      torch::Tensor result = torch::logical_and(lhs, rhs);
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor xla_lhs = CopyToDevice(lhs, device);
        torch::Tensor xla_rhs = CopyToDevice(rhs, device);
        torch::Tensor xla_result = torch::logical_and(xla_lhs, xla_rhs);
        AllEqual(result, xla_result);
      });
    }
  }

  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("xla::logical_and_out", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseAnd) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__and__(xla_rhs);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseAndInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__iand__(rhs);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__iand__(xla_rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseAndScalar) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  torch::Tensor result = lhs.__and__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_result = xla_lhs.__and__(rhs);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseAndScalarInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__iand__(rhs);
    torch::Tensor xla_result = xla_lhs.__iand__(rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseAndPromotion) {
  torch::Tensor input =
      torch::rand({4, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor view = input.reshape(-1);
  torch::Tensor result = torch::__and__(view.gt(0), view.ne(0));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_view = xla_input.reshape(-1);
    torch::Tensor xla_result = torch::__and__(xla_view.gt(0), xla_view.ne(0));
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseOr) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__or__(xla_rhs);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseOrInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ior__(rhs);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__ior__(xla_rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseOrScalar) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  torch::Tensor result = lhs.__or__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_result = xla_lhs.__or__(rhs);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseOrScalarInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ior__(rhs);
    torch::Tensor xla_result = xla_lhs.__ior__(rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseXor) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__xor__(xla_rhs);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseXorInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Tensor rhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ixor__(rhs);
    torch::Tensor xla_rhs = CopyToDevice(rhs, device);
    torch::Tensor xla_result = xla_lhs.__ixor__(xla_rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseXorScalar) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  torch::Tensor result = lhs.__xor__(rhs);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor xla_result = xla_lhs.__xor__(rhs);
    AllEqual(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBitwiseXorScalarInPlace) {
  torch::Tensor lhs = torch::randint(0, std::numeric_limits<int32_t>::max(),
                                     {4, 2}, torch::TensorOptions(torch::kInt));
  torch::Scalar rhs(123456789);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_lhs = CopyToDevice(lhs, device);
    torch::Tensor result = lhs.__ixor__(rhs);
    torch::Tensor xla_result = xla_lhs.__ixor__(rhs);
    AllEqual(result, xla_result);
    AllEqual(lhs, xla_lhs);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLshift) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor shift_amount = torch::randint(16, input.sizes());
  torch::Tensor result = torch::__lshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor xla_result = torch::__lshift__(xla_input, xla_shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLshiftInPlace) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor shift_amount = torch::randint(16, input.sizes());
    torch::Tensor result = input.__ilshift__(shift_amount);
    torch::Tensor xla_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor xla_result = xla_input.__ilshift__(xla_shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLshiftScalar) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar shift_amount = 3;
  torch::Tensor result = torch::__lshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::__lshift__(xla_input, shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLshiftScalarInPlace) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar shift_amount = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor result = input.__ilshift__(shift_amount);
    torch::Tensor xla_result = xla_input.__ilshift__(shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRshift) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor shift_amount = torch::randint(16, input.sizes());
  torch::Tensor result = torch::__rshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor xla_result = torch::__rshift__(xla_input, xla_shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRshiftInPlace) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor shift_amount = torch::randint(16, input.sizes());
    torch::Tensor result = input.__irshift__(shift_amount);
    torch::Tensor xla_shift_amount = CopyToDevice(shift_amount, device);
    torch::Tensor xla_result = xla_input.__irshift__(xla_shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRshiftScalar) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar shift_amount = 3;
  torch::Tensor result = torch::__rshift__(input, shift_amount);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_result = torch::__rshift__(xla_input, shift_amount);
    AllClose(result, xla_result);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRshiftScalarInPlace) {
  torch::Tensor input =
      torch::ones({4, 2}, torch::TensorOptions(torch::kFloat));
  torch::Scalar shift_amount = 3;
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor result = input.__irshift__(shift_amount);
    torch::Tensor xla_result = xla_input.__irshift__(shift_amount);
    AllClose(result, xla_result);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestMeshgrid) {
  torch::Tensor a = torch::rand({3}, torch::TensorOptions(torch::kFloat));
  torch::Tensor b = torch::rand({2}, torch::TensorOptions(torch::kFloat));
  torch::Tensor c = torch::rand({4}, torch::TensorOptions(torch::kFloat));
  auto d = torch::meshgrid({a, b, c});
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_a = CopyToDevice(a, device);
    torch::Tensor xla_b = CopyToDevice(b, device);
    torch::Tensor xla_c = CopyToDevice(c, device);
    auto xla_d = torch::meshgrid({xla_a, xla_b, xla_c});
    EXPECT_EQ(d.size(), xla_d.size());
    for (size_t i = 0; i < d.size(); ++i) {
      AllClose(d[i], xla_d[i]);
    }
  });
}

TEST_F(AtenLtcTsTensorTest, TestConstantPad) {
  torch::Tensor input =
      torch::rand({4, 2, 5}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 2, 3, 4, 5, 6};
  float pad_value = 5;
  torch::Tensor output = torch::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::constant_pad_nd(xla_input, pad, pad_value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestConstantPadIncomplete) {
  torch::Tensor input =
      torch::rand({4, 2, 5}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 2};
  float pad_value = 5;
  torch::Tensor output = torch::constant_pad_nd(input, pad, pad_value);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::constant_pad_nd(xla_input, pad, pad_value);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReflectionPad2dRank3) {
  torch::Tensor input =
      torch::rand({2, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{2, 2, 2, 2};
  torch::Tensor output = torch::reflection_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::reflection_pad2d(xla_input, pad);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReflectionPad2dRank4) {
  torch::Tensor input =
      torch::rand({2, 2, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{2, 2, 2, 2};
  torch::Tensor output = torch::reflection_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::reflection_pad2d(xla_input, pad);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReflectionPad2dBackward) {
  std::vector<int64_t> pad{2, 3, 1, 2};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::reflection_pad2d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({1, 2, 4, 4},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReplicationPad1d) {
  torch::Tensor input =
      torch::rand({1, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 2};
  torch::Tensor output = torch::replication_pad1d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::replication_pad1d(xla_input, pad);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReplicationPad1dZeroPad) {
  torch::Tensor input =
      torch::rand({1, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 0};
  torch::Tensor output = torch::replication_pad1d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::replication_pad1d(xla_input, pad);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReplicationPad1dBackward) {
  std::vector<int64_t> pad{2, 3};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::replication_pad1d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 4},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReplicationPad2d) {
  torch::Tensor input =
      torch::rand({1, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 2, 2, 1};
  torch::Tensor output = torch::replication_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::replication_pad2d(xla_input, pad);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReplicationPad2dZeroPad) {
  torch::Tensor input =
      torch::rand({1, 3, 4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> pad{1, 0, 0, 1};
  torch::Tensor output = torch::replication_pad2d(input, pad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output = torch::replication_pad2d(xla_input, pad);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReplicationPad2dBackward) {
  std::vector<int64_t> pad{2, 3, 1, 1};
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::replication_pad2d(inputs[0], pad);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 3, 4},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAsStrided) {
  torch::Tensor input =
      torch::rand({128, 320}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {128, 20, 4, 4};
  std::vector<int64_t> stride = {320, 16, 4, 1};
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::as_strided(xla_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAsStridedInPlace) {
  torch::Tensor input =
      torch::rand({128, 320}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {128, 20, 4, 4};
  std::vector<int64_t> stride = {320, 16, 4, 1};
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor output =
        torch::as_strided_(input, /*size=*/size, /*stride=*/stride);
    torch::Tensor xla_output =
        torch::as_strided_(xla_input, /*size=*/size, /*stride=*/stride);
    AllClose(output, xla_output);
    AllClose(input, xla_input);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAsStridedWithOffset) {
  torch::Tensor input =
      torch::rand({4, 8, 2}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {4, 4, 2};
  std::vector<int64_t> stride = {8, 2, 1};
  int64_t storage_offset = 4;
  torch::Tensor output =
      torch::as_strided(input, /*size=*/size, /*stride=*/stride,
                        /*storage_offset=*/storage_offset);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input, device);
    torch::Tensor xla_output =
        torch::as_strided(xla_input, /*size=*/size, /*stride=*/stride,
                          /*storage_offset=*/storage_offset);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAsStridedWithInplaceCopy) {
  torch::Tensor grad = torch::ones({4}, torch::TensorOptions(torch::kFloat));
  std::vector<int64_t> size = {4};
  std::vector<int64_t> stride = {1};
  torch::Tensor output = torch::zeros({4}, grad.options());
  output.as_strided(size, stride).copy_(grad);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_grad = CopyToDevice(grad, device);
    torch::Tensor xla_output = torch::zeros({4}, xla_grad.options());
    xla_output.as_strided(size, stride).copy_(xla_grad);
    AllClose(output, xla_output);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEmptyStrided) {
  std::vector<int64_t> size = {4, 4, 2};
  std::vector<int64_t> stride = {8, 2, 1};
  torch::Tensor output = torch::empty_strided(/*size=*/size, /*stride=*/stride);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_output =
        torch::empty_strided(/*size=*/size, /*stride=*/stride);
    EXPECT_EQ(output.sizes(), xla_output.sizes());
    EXPECT_EQ(output.strides(), xla_output.strides());
  });
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool2DBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::avg_pool2d(inputs[0],
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
                    torch::TensorOptions(torch::kFloat).requires_grad(true))},
                device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool3DBackward) {
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
                    torch::TensorOptions(torch::kFloat).requires_grad(true))},
                device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool2DNoBatchBackward) {
  int kernel_size = 2;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (bool count_include_pad : {true, false}) {
        // Test ceil_mode=true through the CPU interop.
        for (bool ceil_mode : {false, true}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::avg_pool2d(inputs[0],
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
                    torch::TensorOptions(torch::kFloat).requires_grad(true))},
                device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAvgPool3DNoBatchBackward) {
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
                    torch::TensorOptions(torch::kFloat).requires_grad(true))},
                device, testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAdaptiveAvgPool3DNoBatchBackward) {
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
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAdaptiveAvgPool3DBackward) {
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
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAdaptiveAvgPool2DBackward) {
  for (int64_t output_size : {7, 8}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {4, 1, 56, 56},
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestAdaptiveAvgPool2DNoBatchBackward) {
  for (int64_t output_size : {7, 8}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::adaptive_avg_pool2d(inputs[0], {output_size, output_size});
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward({torch::rand({1, 56, 56}, torch::TensorOptions(torch::kFloat)
                                                 .requires_grad(true))},
                   device, testfn);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestConv2DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 0; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 3; ++dilation) {
          for (int groups :
               {1, 2, 4}) {  // covers normal, grouped, depthwise conv.
            auto testfn =
                [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
              return torch::conv2d(inputs[0], inputs[1], inputs[2],
                                   /*stride=*/{stride, stride},
                                   /*padding=*/{padding, padding},
                                   /*dilation=*/{dilation, dilation}, groups);
            };

            ForEachDevice([&](const torch::Device& device) {
              torch::Tensor bias =
                  with_bias ? torch::rand({out_channels},
                                          torch::TensorOptions(torch::kDouble))
                            : torch::Tensor();
              TestBackward({torch::rand({1, in_channels, 7, 7},
                                        torch::TensorOptions(torch::kDouble)
                                            .requires_grad(true)),
                            torch::rand({out_channels, in_channels / groups,
                                         kernel_size, kernel_size},
                                        torch::TensorOptions(torch::kDouble)
                                            .requires_grad(true)),
                            bias},
                           device, testfn);
            });
          }
        };
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestTransposedConv2DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (int dilation = 1; dilation <= 2; ++dilation) {
        for (int output_padding = 0;
             output_padding < std::max(stride, dilation); ++output_padding) {
          for (bool with_bias : {true, false}) {
            for (int groups :
                 {1, 2, 4}) {  // covers normal, grouped, depthwise conv.
              auto testfn = [&](const std::vector<torch::Tensor>& inputs)
                  -> torch::Tensor {
                return torch::conv_transpose2d(
                    inputs[0], inputs[1], inputs[2],
                    /*stride=*/{stride, stride + 1},
                    /*padding=*/{padding, padding + 1},
                    /*output_padding=*/output_padding,
                    /*groups=*/groups,
                    /*dilation=*/{dilation, dilation + 1});
              };
              ForEachDevice([&](const torch::Device& device) {
                torch::Tensor input = torch::rand(
                    {4, out_channels, 7, 7},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
                torch::Tensor weight = torch::rand(
                    {out_channels, in_channels / groups, kernel_size,
                     kernel_size},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
                torch::Tensor bias =
                    with_bias ? torch::rand({in_channels},
                                            torch::TensorOptions(torch::kFloat)
                                                .requires_grad(true))
                              : torch::Tensor();
                TestBackward({input, weight, bias}, device, testfn,
                             /*rtol=*/1e-5, /*atol=*/1e-5);
              });
            }
          };
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestConv3DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 3; ++stride) {
    for (int padding = 1; padding <= 2; ++padding) {
      for (bool with_bias : {true, false}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          for (int groups :
               {1, 2, 4}) {  // covers normal, grouped, depthwise conv.
            auto testfn =
                [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
              return torch::conv3d(inputs[0], inputs[1], inputs[2],
                                   /*stride=*/{stride, stride, stride},
                                   /*padding=*/{padding, padding, padding},
                                   /*dilation=*/{dilation, dilation, dilation},
                                   groups);
            };

            ForEachDevice([&](const torch::Device& device) {
              torch::Tensor bias =
                  with_bias ? torch::rand({out_channels},
                                          torch::TensorOptions(torch::kDouble))
                            : torch::Tensor();
              TestBackward({torch::rand({4, in_channels, 7, 7, 7},
                                        torch::TensorOptions(torch::kDouble)
                                            .requires_grad(true)),
                            torch::rand({out_channels, in_channels / groups,
                                         kernel_size, kernel_size, kernel_size},
                                        torch::TensorOptions(torch::kDouble)
                                            .requires_grad(true)),
                            bias},
                           device, testfn);
            });
          }
        };
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestTransposedConv3DBackward) {
  int in_channels = 4;
  int out_channels = 4;
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      for (int dilation = 1; dilation <= 2; ++dilation) {
        for (int output_padding = 0;
             output_padding < std::max(stride, dilation); ++output_padding) {
          for (bool with_bias : {true, false}) {
            for (int groups :
                 {1, 2, 4}) {  // covers normal, grouped, depthwise conv.
              auto testfn = [&](const std::vector<torch::Tensor>& inputs)
                  -> torch::Tensor {
                return torch::conv_transpose3d(
                    inputs[0], inputs[1], inputs[2],
                    /*stride=*/{stride, stride + 1, stride},
                    /*padding=*/{padding, padding + 1, stride},
                    /*output_padding=*/output_padding,
                    /*groups=*/groups,
                    /*dilation=*/{dilation, dilation + 1, dilation});
              };
              ForEachDevice([&](const torch::Device& device) {
                torch::Tensor input = torch::rand(
                    {4, out_channels, 7, 7, 7},
                    torch::TensorOptions(torch::kDouble).requires_grad(true));
                torch::Tensor weight = torch::rand(
                    {out_channels, in_channels / groups, kernel_size,
                     kernel_size, kernel_size},
                    torch::TensorOptions(torch::kDouble).requires_grad(true));
                torch::Tensor bias =
                    with_bias ? torch::rand({in_channels},
                                            torch::TensorOptions(torch::kDouble)
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

TEST_F(AtenLtcTsTensorTest, TestMaxPool2DBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::max_pool2d(
              inputs[0], /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {1, 2, 8, 8},
                  torch::TensorOptions(torch::kFloat).requires_grad(true))},
              device, testfn);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool3DBackward) {
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
              /*padding=*/{padding, padding, padding}, /*dilation=*/{1, 1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {1, 2, 4, 4, 4},
                  torch::TensorOptions(torch::kFloat).requires_grad(true))},
              device, testfn);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool2DNoBatchBackward) {
  int kernel_size = 3;
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::max_pool2d(
              inputs[0], /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {2, 8, 8},
                  torch::TensorOptions(torch::kFloat).requires_grad(true))},
              device, testfn);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxPool3DNoBatchBackward) {
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
              /*padding=*/{padding, padding, padding}, /*dilation=*/{1, 1, 1},
              /*ceil_mode=*/ceil_mode);
        };

        ForEachDevice([&](const torch::Device& device) {
          TestBackward(
              {torch::rand(
                  {2, 4, 4, 4},
                  torch::TensorOptions(torch::kFloat).requires_grad(true))},
              device, testfn);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxUnpool2DBackward) {
  int kernel_size = 2;
  torch::Tensor input =
      torch::rand({2, 2, 8, 8}, torch::TensorOptions(torch::kFloat));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool2d_with_indices(
              input, /*kernel_size=*/{kernel_size, kernel_size},
              /*stride=*/{stride, stride},
              /*padding=*/{padding, padding}, /*dilation=*/{dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size({input.size(2), input.size(3)});
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::max_unpool2d(inputs[0], inputs[1], output_size);
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward({output.requires_grad_(true), indices}, device,
                         testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestMaxUnpool3DBackward) {
  int kernel_size = 2;
  torch::Tensor input =
      torch::rand({1, 1, 4, 4, 4}, torch::TensorOptions(torch::kFloat));
  for (int stride = 1; stride <= 2; ++stride) {
    for (int padding = 0; padding <= 1; ++padding) {
      // Test ceil_mode=true through the CPU interop.
      for (bool ceil_mode : {false, true}) {
        for (int dilation = 1; dilation <= 2; ++dilation) {
          torch::Tensor output;
          torch::Tensor indices;
          std::tie(output, indices) = torch::max_pool3d_with_indices(
              input, /*kernel_size=*/{kernel_size, kernel_size, kernel_size},
              /*stride=*/{stride, stride, stride},
              /*padding=*/{padding, padding, padding},
              /*dilation=*/{dilation, dilation, dilation},
              /*ceil_mode=*/ceil_mode);

          std::vector<int64_t> output_size(
              {input.size(2), input.size(3), input.size(4)});
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::max_unpool3d(inputs[0], inputs[1], output_size,
                                       /*stride=*/{stride, stride, stride},
                                       /*padding=*/{padding, padding, padding});
          };

          ForEachDevice([&](const torch::Device& device) {
            TestBackward({output.requires_grad_(true), indices}, device,
                         testfn);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestTanhBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::tanh(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 2},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSigmoidBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::sigmoid(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 2},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLogSigmoidBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::log_sigmoid(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 2},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn, /*rtol=*/1e-3, /*atol=*/1e-5);
  });
}

TEST_F(AtenLtcTsTensorTest, TestLogSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::log_softmax(inputs[0], dim);
    };

    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {5, 3, 4, 2},
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestSoftmaxBackward) {
  for (int dim = -4; dim < 4; ++dim) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::softmax(inputs[0], dim);
    };

    ForEachDevice([&](const torch::Device& device) {
      TestBackward(
          {torch::rand(
              {5, 3, 4, 2},
              torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn, /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestSoftplusBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::softplus(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 1, 4, 6},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn, /*rtol=*/1e-4);
  });
}

TEST_F(AtenLtcTsTensorTest, TestReluBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::relu(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 1, 4, 6},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestRreluBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::rrelu(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 1, 4, 6},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestHardshrinkBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardshrink(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn({100},
                      torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestSoftshrinkBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::softshrink(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn({100},
                      torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestHardtanhBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::hardtanh(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::randn({100},
                      torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestEluBackward) {
  torch::Scalar alpha = 0.5;
  torch::Scalar scale = 2.5;
  torch::Scalar input_scale = 1.5;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::elu(inputs[0], alpha, scale, input_scale);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 1, 4, 6},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestGeluBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::gelu(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 3},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
  ExpectCounterChanged("lazy::gelu_backward", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestLeakyReluBackward) {
  double negative_slope = 0.01;
  auto testfn = [=](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::leaky_relu(inputs[0], negative_slope);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 1, 4, 6},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestTransposeBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::t(inputs[0]);
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({2, 3},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAddMatMulBackward) {
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
          {torch::rand({labels},
                       torch::TensorOptions(torch::kFloat).requires_grad(true)),
           torch::rand({in_channels, out_channels},
                       torch::TensorOptions(torch::kFloat).requires_grad(true)),
           torch::rand(
               {out_channels, labels},
               torch::TensorOptions(torch::kFloat).requires_grad(true))},
          device, testfn);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestBinaryCrossEntropyBackward) {
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
           {torch::Reduction::Mean, torch::Reduction::Sum,
            torch::Reduction::None}) {
        auto testfn =
            [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
          return torch::binary_cross_entropy(
              /*self=*/inputs[0], /*target=*/inputs[1],
              /*weight=*/inputs[2],
              /*reduction=*/reduction);
        };
        ForEachDevice([&](const torch::Device& device) {
          TestBackward({input, target, weight}, device, testfn, /*rtol=*/1e-4,
                       /*atol=*/1e-7);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestNllLossBackward) {
  int batch = 6;
  int classes = 2;
  // TODO(asuhan): Fix the torch::kDouble case.
  for (auto dtype : {torch::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input = torch::rand(
            {batch, classes}, torch::TensorOptions(dtype).requires_grad(true));
        torch::Tensor target =
            torch::randint(std::min(ignore_index, 0), classes, {batch},
                           torch::TensorOptions(torch::kLong));
        torch::Tensor weight;
        if (def_weight) {
          weight = torch::rand({classes}, torch::TensorOptions(dtype));
        }
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean, torch::Reduction::Sum,
              torch::Reduction::None}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::nll_loss(
                /*self=*/inputs[0], /*target=*/inputs[1],
                /*weight=*/inputs[2],
                /*reduction=*/reduction, /*ignore_index=*/ignore_index);
          };
          ForEachDevice([&](const torch::Device& device) {
            TestBackward({input, target, weight}, device, testfn, /*rtol=*/1e-5,
                         /*atol=*/1e-8);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestNllLoss2dBackward) {
  int batch = 6;
  int classes = 2;
  int height = 3;
  int width = 3;
  // TODO(asuhan): Fix the torch::kDouble case.
  for (auto dtype : {torch::kFloat}) {
    for (int ignore_index : {-1, 0, 1, 5}) {
      for (bool def_weight : {false, true}) {
        torch::Tensor input =
            torch::rand({batch, classes, height, width},
                        torch::TensorOptions(dtype).requires_grad(true));
        torch::Tensor target = torch::randint(
            std::min(ignore_index, 0), classes, {batch, height, width},
            torch::TensorOptions(torch::kLong));
        torch::Tensor weight;
        if (def_weight) {
          weight = torch::rand({classes}, torch::TensorOptions(dtype));
        }
        for (torch::Reduction::Reduction reduction :
             {torch::Reduction::Mean, torch::Reduction::Sum,
              torch::Reduction::None}) {
          auto testfn =
              [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
            return torch::nll_loss2d(
                /*self=*/inputs[0], /*target=*/inputs[1],
                /*weight=*/inputs[2],
                /*reduction=*/reduction, /*ignore_index=*/ignore_index);
          };
          ForEachDevice([&](const torch::Device& device) {
            TestBackward({input, target, weight}, device, testfn, /*rtol=*/1e-5,
                         /*atol=*/1e-8);
          });
        }
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestSmoothL1LossBackward) {
  torch::Tensor input = torch::randn(
      {2, 4}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor target =
      torch::randn({2, 4}, torch::TensorOptions(torch::kFloat));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    for (double beta : {0.25, 1.}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::smooth_l1_loss(/*input=*/inputs[0], /*target=*/inputs[1],
                                     /*reduction=*/reduction, /*beta=*/beta);
      };
      ForEachDevice([&](const torch::Device& device) {
        TestBackward({input, target}, device, testfn, /*rtol=*/1e-5,
                     /*atol=*/1e-8);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestViewBackward) {
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return inputs[0].view({-1, 320});
  };
  ForEachDevice([&](const torch::Device& device) {
    TestBackward(
        {torch::rand({32, 20, 4, 4},
                     torch::TensorOptions(torch::kFloat).requires_grad(true))},
        device, testfn);
  });
}

TEST_F(AtenLtcTsTensorTest, TestBatchNorm2DBackward) {
  double momentum = 0.1;
  double eps = 0.5;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::batch_norm(
        /*input=*/inputs[0], /*weight=*/inputs[1], /*bias=*/inputs[2],
        /*running_mean=*/inputs[3], /*running_var=*/inputs[4],
        /*training=*/true, /*momentum=*/momentum, /*eps=*/eps,
        /*cudnn_enabled=*/false);
  };
  int num_features = 3;
  torch::Tensor undef;
  for (bool undef_weight_bias : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input =
          torch::rand({2, num_features, 4, 4},
                      torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor weight =
          undef_weight_bias
              ? undef
              : torch::rand(
                    {num_features},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor bias =
          undef_weight_bias
              ? undef
              : torch::rand(
                    {num_features},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor running_mean =
          torch::zeros({num_features}, torch::TensorOptions(torch::kFloat));
      torch::Tensor running_var =
          torch::ones({num_features}, torch::TensorOptions(torch::kFloat));
      TestBackward({input, weight, bias, running_mean, running_var}, device,
                   testfn,
                   /*rtol=*/1e-3, /*atol=*/1e-4);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestBatchNorm3DBackward) {
  double momentum = 0.1;
  double eps = 0.5;
  auto testfn = [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
    return torch::batch_norm(
        /*input=*/inputs[0], /*weight=*/inputs[1], /*bias=*/inputs[2],
        /*running_mean=*/inputs[3], /*running_var=*/inputs[4],
        /*training=*/true, /*momentum=*/momentum, /*eps=*/eps,
        /*cudnn_enabled=*/false);
  };
  int num_features = 3;
  torch::Tensor undef;
  for (bool undef_weight_bias : {false, true}) {
    ForEachDevice([&](const torch::Device& device) {
      torch::Tensor input =
          torch::rand({2, num_features, 4, 4, 2},
                      torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor weight =
          undef_weight_bias
              ? undef
              : torch::rand(
                    {num_features},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor bias =
          undef_weight_bias
              ? undef
              : torch::rand(
                    {num_features},
                    torch::TensorOptions(torch::kFloat).requires_grad(true));
      torch::Tensor running_mean =
          torch::zeros({num_features}, torch::TensorOptions(torch::kFloat));
      torch::Tensor running_var =
          torch::ones({num_features}, torch::TensorOptions(torch::kFloat));
      TestBackward({input, weight, bias, running_mean, running_var}, device,
                   testfn,
                   /*rtol=*/1e-3, /*atol=*/1e-3);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestBCEWithLogitsBackward) {
  int batch = 10;
  int classes = 5;
  torch::Tensor undef;
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::None, torch::Reduction::Mean,
        torch::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::binary_cross_entropy_with_logits(
          /*input=*/inputs[0], /*target=*/inputs[1], /*weight=*/inputs[2],
          /*pos_weight=*/inputs[3],
          /*reduction=*/reduction);
    };
    for (bool undef_weight : {false, true}) {
      for (bool undef_pos_weight : {false, true}) {
        torch::Tensor input = torch::rand(
            {batch, classes},
            torch::TensorOptions(torch::kFloat).requires_grad(true));
        torch::Tensor target = torch::rand(
            {batch, classes},
            torch::TensorOptions(torch::kFloat).requires_grad(true));
        torch::Tensor weight =
            undef_weight
                ? undef
                : torch::rand({classes}, torch::TensorOptions(torch::kFloat));
        torch::Tensor pos_weight =
            undef_pos_weight
                ? undef
                : torch::rand({classes}, torch::TensorOptions(torch::kFloat));
        ForEachDevice([&](const torch::Device& device) {
          TestBackward({input, target, weight, pos_weight}, device, testfn,
                       /*rtol=*/1e-3, /*atol=*/1e-5);
        });
      }
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestKlDivBackward) {
  torch::Tensor input = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  torch::Tensor target = torch::rand(
      {4, 3}, torch::TensorOptions(torch::kFloat).requires_grad(true));
  for (torch::Reduction::Reduction reduction :
       {torch::Reduction::Mean, torch::Reduction::Sum}) {
    auto testfn =
        [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
      return torch::kl_div(/*self=*/inputs[0], /*target=*/inputs[1], reduction);
    };
    ForEachDevice([&](const torch::Device& device) {
      TestBackward({input, target}, device, testfn, /*rtol=*/1e-4,
                   /*atol=*/1e-5);
    });
  }
}

TEST_F(AtenLtcTsTensorTest, TestEmbeddingBackward) {
  int num_weights = 32;
  for (int padding_idx = -1; padding_idx < num_weights; ++padding_idx) {
    for (bool scale_grad_by_freq : {false, true}) {
      auto testfn =
          [&](const std::vector<torch::Tensor>& inputs) -> torch::Tensor {
        return torch::embedding(inputs[0], inputs[1],
                                /*padding_idx=*/padding_idx,
                                /*scale_grad_by_freq=*/scale_grad_by_freq,
                                /*sparse=*/false);
      };
      ForEachDevice([&](const torch::Device& device) {
        torch::Tensor weight = torch::rand(
            {num_weights, 7},
            torch::TensorOptions(torch::kFloat).requires_grad(true));
        torch::Tensor indices = torch::randint(
            num_weights, {3, 9, 4}, torch::TensorOptions(torch::kLong));
        TestBackward({weight, indices}, device, testfn, /*rtol=*/1e-5,
                     /*atol=*/1e-8);
      });
    }
  }
}

TEST_F(AtenLtcTsTensorTest, TestAmpForeachNonFiniteCheckAndUnscale) {
  DeviceType hw_type = GetDefaultDevice()->hw_type;
  if (hw_type != DeviceType::GPU && hw_type != DeviceType::CPU) {
    return;
  }
  torch::Tensor grads0 =
      torch::tensor({1, 2, 3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor grads1 = torch::tensor({1.0, 2.0, std::nan("1"), 4.0},
                                       torch::TensorOptions(torch::kFloat));
  torch::Tensor inv_scale =
      torch::scalar_tensor(0.2, torch::TensorOptions(torch::kFloat));
  torch::Tensor found_inf =
      torch::scalar_tensor(0, torch::TensorOptions(torch::kFloat));
  torch::Tensor grads_output0 = grads0 * inv_scale;
  torch::Tensor found_inf_output0 =
      torch::scalar_tensor(0, torch::TensorOptions(torch::kFloat));
  torch::Tensor found_inf_output1 =
      torch::scalar_tensor(1, torch::TensorOptions(torch::kFloat));
  ForEachDevice([&](const torch::Device& device) {
    if (grads0.device() == at::kCPU) {
      GTEST_SKIP();
    }
    torch::Tensor xla_grads0 = CopyToDevice(grads0, device);
    torch::Tensor xla_inv_scale = CopyToDevice(inv_scale, device);
    torch::Tensor xla_found_inf = CopyToDevice(found_inf, device);
    torch::_amp_foreach_non_finite_check_and_unscale_(xla_grads0, xla_found_inf,
                                                      xla_inv_scale);
    AllClose(grads_output0, xla_grads0, /*rtol=*/1e-2, /*atol=*/1e-4);
    AllEqual(found_inf_output0, xla_found_inf);

    torch::Tensor xla_grads1 = CopyToDevice(grads1, device);
    torch::_amp_foreach_non_finite_check_and_unscale_(xla_grads1, xla_found_inf,
                                                      xla_inv_scale);
    AllEqual(found_inf_output1, xla_found_inf);
  });
}

TEST_F(AtenLtcTsTensorTest, TestAmpUpdateScale) {
  DeviceType hw_type = GetDefaultDevice()->hw_type;
  if (hw_type != DeviceType::GPU && hw_type != DeviceType::CPU) {
    return;
  }
  torch::Tensor growth_tracker =
      torch::scalar_tensor(0, torch::TensorOptions(torch::kInt32));
  torch::Tensor current_scale =
      torch::scalar_tensor(4, torch::TensorOptions(torch::kFloat));
  torch::Tensor found_inf =
      torch::scalar_tensor(1, torch::TensorOptions(torch::kFloat));
  torch::Tensor not_found_inf =
      torch::scalar_tensor(0, torch::TensorOptions(torch::kFloat));
  float scale_growth_factor = 2.0;
  float scale_backoff_factor = 0.5;
  int growth_interval = 3;

  torch::Tensor growth_tracker_result0 =
      torch::scalar_tensor(1, torch::TensorOptions(torch::kInt32));
  torch::Tensor current_scale_result0 =
      torch::scalar_tensor(4, torch::TensorOptions(torch::kFloat));
  torch::Tensor growth_tracker_result1 =
      torch::scalar_tensor(2, torch::TensorOptions(torch::kInt32));
  torch::Tensor current_scale_result1 =
      torch::scalar_tensor(4, torch::TensorOptions(torch::kFloat));
  torch::Tensor growth_tracker_result2 =
      torch::scalar_tensor(0, torch::TensorOptions(torch::kInt32));
  torch::Tensor current_scale_result2 =
      torch::scalar_tensor(8, torch::TensorOptions(torch::kFloat));
  torch::Tensor growth_tracker_result3 =
      torch::scalar_tensor(0, torch::TensorOptions(torch::kInt32));
  torch::Tensor current_scale_result3 =
      torch::scalar_tensor(4, torch::TensorOptions(torch::kFloat));

  ForEachDevice([&](const torch::Device& device) {
    if (growth_tracker.device() == at::kCPU) {
      GTEST_SKIP();
    }
    torch::Tensor xla_growth_tracker = CopyToDevice(growth_tracker, device);
    torch::Tensor xla_current_scale = CopyToDevice(current_scale, device);
    torch::Tensor xla_found_inf = CopyToDevice(found_inf, device);
    torch::Tensor xla_not_found_inf = CopyToDevice(not_found_inf, device);

    torch::_amp_update_scale_(xla_current_scale, xla_growth_tracker,
                              xla_not_found_inf, scale_growth_factor,
                              scale_backoff_factor, growth_interval);
    AllClose(current_scale_result0, xla_current_scale, /*rtol=*/1e-2,
             /*atol=*/1e-4);
    AllEqual(growth_tracker_result0, xla_growth_tracker);

    torch::_amp_update_scale_(xla_current_scale, xla_growth_tracker,
                              xla_not_found_inf, scale_growth_factor,
                              scale_backoff_factor, growth_interval);
    AllClose(current_scale_result1, xla_current_scale, /*rtol=*/1e-2,
             /*atol=*/1e-4);
    AllEqual(growth_tracker_result1, xla_growth_tracker);

    // torch::_amp_update_scale_ returns the reference of current_scale
    xla_current_scale = torch::_amp_update_scale_(
        xla_current_scale, xla_growth_tracker, xla_not_found_inf,
        scale_growth_factor, scale_backoff_factor, growth_interval);
    AllClose(current_scale_result2, xla_current_scale, /*rtol=*/1e-2,
             /*atol=*/1e-4);
    AllEqual(growth_tracker_result2, xla_growth_tracker);

    xla_current_scale = torch::_amp_update_scale_(
        xla_current_scale, xla_growth_tracker, xla_found_inf,
        scale_growth_factor, scale_backoff_factor, growth_interval);
    AllClose(current_scale_result3, xla_current_scale, /*rtol=*/1e-2,
             /*atol=*/1e-4);
    AllEqual(growth_tracker_result3, xla_growth_tracker);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("lazy::_amp_update_scale_",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestEarlySyncLiveTensors) {
  torch::Tensor scalar_tensor =
      torch::scalar_tensor(1., torch::TensorOptions(torch::kFloat));
  torch::Scalar scalar1 = scalar_tensor.item();
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_scalar_tensor = CopyToDevice(scalar_tensor, device);
    torch::Scalar scalar2 = xla_scalar_tensor.item();
    ASSERT_EQ(scalar1.to<float>(), scalar2.to<float>());
  });
  if (DebugUtil::ExperimentEnabled("early_sync")) {
    ExpectCounterChanged("EarlySyncLiveTensorsCount",
                         cpp_test::GetIgnoredCounters());
  } else {
    ExpectCounterNotChanged("EarlySyncLiveTensorsCount",
                            cpp_test::GetIgnoredCounters());
  }
  ExpectCounterChanged("aten::_local_scalar_dense",
                       cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestLerp) {
  torch::Tensor start =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor res = torch::lerp(start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_start = CopyToDevice(start, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    torch::Tensor xla_res = torch::lerp(xla_start, xla_end, xla_weight);
    AllClose(res, xla_res);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestLerpScalar) {
  torch::Tensor start =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Scalar weight = torch::Scalar(3.0);
  torch::Tensor res = torch::lerp(start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_start = CopyToDevice(start, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    torch::Tensor xla_res = torch::lerp(xla_start, xla_end, weight);
    AllClose(res, xla_res);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestLerpInplace) {
  torch::Tensor input =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor input_copy = input.clone();
  input.lerp_(end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    xla_input.lerp_(xla_end, xla_weight);
    AllClose(xla_input, input);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestLerpScalarInplace) {
  torch::Tensor input =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Scalar weight = torch::Scalar(3.0);
  torch::Tensor input_copy = input.clone();
  input.lerp_(end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_input = CopyToDevice(input_copy, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    xla_input.lerp_(xla_end, weight);
    AllClose(xla_input, input);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestLerpOut) {
  torch::Tensor start =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor weight =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor res = torch::empty({3, 4}, torch::TensorOptions(torch::kFloat));
  ;
  torch::lerp_out(res, start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_start = CopyToDevice(start, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    torch::Tensor xla_weight = CopyToDevice(weight, device);
    torch::Tensor xla_res = torch::empty({3, 4}, xla_start.options());
    torch::lerp_out(xla_res, xla_start, xla_end, xla_weight);
    AllClose(res, xla_res);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", cpp_test::GetIgnoredCounters());
}

TEST_F(AtenLtcTsTensorTest, TestLerpScalarOut) {
  torch::Tensor start =
      torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Tensor end = torch::rand({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::Scalar weight = torch::Scalar(3.0);
  torch::Tensor res = torch::empty({3, 4}, torch::TensorOptions(torch::kFloat));
  torch::lerp_out(res, start, end, weight);
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor xla_start = CopyToDevice(start, device);
    torch::Tensor xla_end = CopyToDevice(end, device);
    torch::Tensor xla_res = torch::empty({3, 4}, xla_start.options());
    torch::lerp_out(xla_res, xla_start, xla_end, weight);
    AllClose(res, xla_res);
  });
  ExpectCounterNotChanged("aten::.*", cpp_test::GetIgnoredCounters());
  ExpectCounterChanged("lazy::lerp", cpp_test::GetIgnoredCounters());
}

}  // namespace cpp_test
}  // namespace torch_lazy_tensors
