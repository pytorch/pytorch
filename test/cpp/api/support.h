#pragma once

#include <test/cpp/common/support.h>

#include <gtest/gtest.h>

#include <c10/util/Exception.h>
#include <ATen/TensorIndexing.h>
#include <torch/nn/cloneable.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <string>
#include <utility>

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

struct WarningCapture : public WarningHandler {
  WarningCapture() : prev_(Warning::get_warning_handler()) {
    Warning::set_warning_handler(this);
  }

  ~WarningCapture() {
    Warning::set_warning_handler(prev_);
  }

  const std::vector<std::string>& messages() {
    return messages_;
  }

  std::string str() {
    return c10::Join("\n", messages_);
  }

  void process(const SourceLocation& source_location, const std::string& msg, const bool /*verbatim*/)
      override {
    messages_.push_back(msg);
  }

 private:
  WarningHandler* prev_;
  std::vector<std::string> messages_;
};

inline bool pointer_equal(at::Tensor first, at::Tensor second) {
  return first.data_ptr() == second.data_ptr();
}

// This mirrors the `isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)` branch
// in `TestCase.assertEqual` in torch/testing/_internal/common_utils.py
inline void assert_tensor_equal(at::Tensor a, at::Tensor b, bool allow_inf=false) {
  ASSERT_TRUE(a.sizes() == b.sizes());
  if (a.numel() > 0) {
    if (a.device().type() == torch::kCPU && (a.scalar_type() == torch::kFloat16 || a.scalar_type() == torch::kBFloat16)) {
      // CPU half and bfloat16 tensors don't have the methods we need below
      a = a.to(torch::kFloat32);
    }
    if (a.device().type() == torch::kCUDA && a.scalar_type() == torch::kBFloat16) {
      // CUDA bfloat16 tensors don't have the methods we need below
      a = a.to(torch::kFloat32);
    }
    b = b.to(a);

    if ((a.scalar_type() == torch::kBool) != (b.scalar_type() == torch::kBool)) {
      TORCH_CHECK(false, "Was expecting both tensors to be bool type.");
    } else {
      if (a.scalar_type() == torch::kBool && b.scalar_type() == torch::kBool) {
        // we want to respect precision but as bool doesn't support subtraction,
        // boolean tensor has to be converted to int
        a = a.to(torch::kInt);
        b = b.to(torch::kInt);
      }

      auto diff = a - b;
      if (a.is_floating_point()) {
        // check that NaNs are in the same locations
        auto nan_mask = torch::isnan(a);
        ASSERT_TRUE(torch::equal(nan_mask, torch::isnan(b)));
        diff.index_put_({nan_mask}, 0);
        // inf check if allow_inf=true
        if (allow_inf) {
          auto inf_mask = torch::isinf(a);
          auto inf_sign = inf_mask.sign();
          ASSERT_TRUE(torch::equal(inf_sign, torch::isinf(b).sign()));
          diff.index_put_({inf_mask}, 0);
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

// This mirrors the `isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)` branch
// in `TestCase.assertNotEqual` in torch/testing/_internal/common_utils.py
inline void assert_tensor_not_equal(at::Tensor x, at::Tensor y) {
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
    diff.index_put_({nan_mask}, 0);
    // Use `item()` to work around:
    // https://github.com/pytorch/pytorch/issues/22301
    auto max_err = diff.max().item<double>();
    ASSERT_GE(max_err, 1e-5);
  }
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
//
// Usage of this guard is synchronized across threads, so that at any given time,
// only one guard can take effect.
struct AutoDefaultDtypeMode {
  static std::mutex default_dtype_mutex;

  AutoDefaultDtypeMode(c10::ScalarType default_dtype) : prev_default_dtype(torch::typeMetaToScalarType(torch::get_default_dtype())) {
    default_dtype_mutex.lock();
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(default_dtype));
  }
  ~AutoDefaultDtypeMode() {
    default_dtype_mutex.unlock();
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(prev_default_dtype));
  }
  c10::ScalarType prev_default_dtype;
};

} // namespace test
} // namespace torch
