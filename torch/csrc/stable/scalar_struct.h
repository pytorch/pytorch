#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/complex.h>
#include <torch/headeronly/util/shim_utils.h>

#include <limits>
#include <memory>
#include <type_traits>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

// ABI stable scalar value. Copies share an immutable handle.
// Minimum compatible version: PyTorch 2.16.
class Scalar {
 private:
  std::shared_ptr<AtenScalarOpaque> scalar_;

 public:
  // Takes ownership of the handle.
  explicit Scalar(AtenScalarHandle scalar)
      : scalar_(scalar, [](AtenScalarHandle value) {
          STABLE_TORCH_ERROR_CODE_CHECK(torch_delete_scalar(value));
        }) {}

  Scalar() : Scalar(int64_t{0}) {}
  Scalar(double value);
  Scalar(bool value);
  Scalar(c10::complex<double> value);

  template <
      typename T,
      std::enable_if_t<
          std::is_integral_v<T> && !std::is_same_v<T, bool>,
          int> = 0>
  Scalar(T value) {
    if constexpr (std::is_unsigned_v<T>) {
      STD_TORCH_CHECK(
          value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
          "Scalar integer is out of range for int64_t");
    }
    AtenScalarHandle scalar = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_new_scalar_int(static_cast<int64_t>(value), &scalar));
    *this = Scalar(scalar);
  }

  // Returns a borrowed handle.
  AtenScalarHandle get() const {
    return scalar_.get();
  }

  torch::headeronly::ScalarType type() const;
  bool isBoolean() const;
  bool isIntegral(bool includeBool) const;
  bool isFloatingPoint() const;
  bool isComplex() const;

  int64_t toLong() const;
  double toDouble() const;
  bool toBool() const;
  c10::complex<double> toComplexDouble() const;
};

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable)
