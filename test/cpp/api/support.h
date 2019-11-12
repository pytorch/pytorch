#pragma once

#include <test/cpp/common/support.h>

#include <gtest/gtest.h>

#include <ATen/Dispatch.h>
#include <ATen/TensorIndexing.h>
#include <torch/nn/cloneable.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <string>
#include <utility>

namespace torch {

// yf225 TODO: I need to make a separate PR just for this function
// See https://github.com/pytorch/pytorch/pull/28918#discussion_r342137228 for how to do it
/*
def isinf(tensor):
    if tensor.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        return torch.zeros_like(tensor, dtype=torch.bool)
    return tensor.abs() == inf
*/
inline torch::Tensor isinf(const torch::Tensor& tensor) {
  std::vector<c10::ScalarType> integral_types = {torch::kUInt8, torch::kInt8, torch::kInt16, torch::kInt32, torch::kInt64};
  for (const auto& dtype : integral_types) {
    if (tensor.dtype() == dtype) {
      return torch::zeros_like(tensor, torch::kBool);
    }
  }
  Tensor ret;
  AT_DISPATCH_ALL_TYPES_AND3(
      torch::kBool,
      torch::kHalf,
      torch::kBFloat16,
      tensor.scalar_type(),
      "torch_isinf", [&] {
    ret = (tensor.abs() == torch::full_like(tensor, std::numeric_limits<scalar_t>::infinity()));
  });
  return ret;
}

} // namespace torch

namespace torch {
namespace test {

// Lets you use a container without making a new class,
// for experimental implementations
class SimpleContainer : public nn::Cloneable<SimpleContainer> {
 public:
  void reset() override {}

  template <typename ModuleHolder>
  ModuleHolder add(
      ModuleHolder module_holder,
      std::string name = std::string()) {
    return Module::register_module(std::move(name), module_holder);
  }
};

struct SeedingFixture : public ::testing::Test {
  SeedingFixture() {
    torch::manual_seed(0);
  }
};

struct CerrRedirect {
  CerrRedirect(std::streambuf * new_buffer) : prev_buffer(std::cerr.rdbuf(new_buffer)) {}

  ~CerrRedirect( ) {
    std::cerr.rdbuf(prev_buffer);
  }

private:
  std::streambuf * prev_buffer;
};

inline bool pointer_equal(at::Tensor first, at::Tensor second) {
  return first.data_ptr<float>() == second.data_ptr<float>();
}

inline void assert_equal(at::Tensor a, at::Tensor b, bool allow_inf=false) {
  ASSERT_TRUE(a.sizes() == b.sizes());
  if (a.numel() > 0) {
    if (a.device().type() == torch::kCPU && (a.dtype() == torch::kFloat16 || a.dtype() == torch::kBFloat16)) {
      // CPU half and bfloat16 tensors don't have the methods we need below
      a = a.to(torch::kFloat32);
    }
    b = b.to(a);

    if ((a.dtype() == torch::kBool) != (b.dtype() == torch::kBool)) {
      TORCH_CHECK(false, "Was expecting both tensors to be bool type.");
    } else {
      if (a.dtype() == torch::kBool && b.dtype() == torch::kBool) {
        // we want to respect precision but as bool doesn't support substraction,
        // boolean tensor has to be converted to int
        a = a.to(torch::kInt);
        b = b.to(torch::kInt);
      }

      auto diff = a - b;
      if (a.is_floating_point()) {
        // check that NaNs are in the same locations
        auto nan_mask = torch::isnan(a);
        ASSERT_TRUE(torch::equal(nan_mask, torch::isnan(b)));
        diff(nan_mask) = 0;
        // inf check if allow_inf=true
        if (allow_inf) {
          auto inf_mask = torch::isinf(a);
          auto inf_sign = inf_mask.sign();
          ASSERT_TRUE(torch::equal(inf_sign, torch::isinf(b).sign()));
          diff(inf_mask) = 0;
        }
      }
      // TODO: implement abs on CharTensor (int8)
      if (diff.is_signed() && diff.scalar_type() != torch::kInt8) {
        diff = diff.abs();
      }
      auto max_err = diff.max().item<double>();
      ASSERT_LE(max_err, 1e-5);
    }
  }
}

// yf225 TODO: do we actually need this function?
#define TENSOR(T, S) \
inline void assert_equal(const c10::ArrayRef<T>& first, const c10::ArrayRef<T>& second) { \
  ASSERT_TRUE(first.size() == second.size()); \
}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

inline void assert_not_equal(at::Tensor x, at::Tensor y) {
  if (x.sizes() != y.sizes()) {
    return;
  }
  ASSERT_GT(x.numel(), 0);
  y = y.type_as(x);
  y = x.is_cuda() ? y.to({torch::kCUDA, x.get_device()}) : y.cpu();
  auto nan_mask = x != x;
  if (torch::equal(nan_mask, y != y)) {
    auto diff = x - y;
    if (diff.is_signed()) {
      diff = diff.abs();
    }
    diff(nan_mask) = 0;
    // Use `item()` to work around:
    // https://github.com/pytorch/pytorch/issues/22301
    auto max_err = diff.max().item<double>();
    ASSERT_GE(max_err, 1e-5);
  }
}

inline void assert_is_not(const at::Tensor& first, const at::Tensor& second) {
  ASSERT_FALSE(first.unsafeGetTensorImpl() == second.unsafeGetTensorImpl());
}

template <typename T>
bool exactly_equal(at::Tensor left, T right) {
  return left.item<T>() == right;
}

template <typename T>
bool almost_equal(at::Tensor left, T right, T tolerance = 1e-4) {
  return std::abs(left.item<T>() - right) < tolerance;
}

inline int count_substr_occurrences(const std::string& str, const std::string& substr) {
  int count = 0;
  size_t pos = str.find(substr);

  while (pos != std::string::npos) {
    count++;
    pos = str.find(substr, pos + substr.size());
  }

  return count;
}

// A RAII, thread local (!) guard that changes default dtype upon
// construction, and sets it back to the original dtype upon destruction.
struct AutoDefaultDtypeMode {
  AutoDefaultDtypeMode(c10::ScalarType default_dtype) : prev_default_dtype(torch::typeMetaToScalarType(torch::get_default_dtype())) {
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(default_dtype));
  }
  ~AutoDefaultDtypeMode() {
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(prev_default_dtype));
  }
  c10::ScalarType prev_default_dtype;
};

} // namespace test
} // namespace torch
