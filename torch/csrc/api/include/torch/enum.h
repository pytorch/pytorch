#pragma once

#include <string>
#include <variant>

#include <ATen/core/Reduction.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>

#define TORCH_ENUM_DECLARE(name)                                      \
  namespace torch {                                                   \
  namespace enumtype {                                                \
  /*                                                                  \
    NOTE: We need to provide the default constructor for each struct, \
    otherwise Clang 3.8 would complain:                               \
    ```                                                               \
    error: default initialization of an object of const type 'const   \
    enumtype::Enum1' without a user-provided default constructor      \
    ```                                                               \
  */                                                                  \
  struct k##name {                                                    \
    k##name() {}                                                      \
  };                                                                  \
  }                                                                   \
  TORCH_API extern const enumtype::k##name k##name;                   \
  }

#define TORCH_ENUM_DEFINE(name)    \
  namespace torch {                \
  const enumtype::k##name k##name; \
  }

#define TORCH_ENUM_PRETTY_PRINT(name)                                         \
  std::string operator()(const enumtype::k##name& v [[maybe_unused]]) const { \
    std::string k("k");                                                       \
    return k + #name;                                                         \
  }

// NOTE: Backstory on why we need the following two macros:
//
// Consider the following options class:
//
// ```
// struct TORCH_API SomeOptions {
//   typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
//   reduction_t; SomeOptions(reduction_t reduction = torch::kMean) :
//   reduction_(reduction) {}
//
//   TORCH_ARG(reduction_t, reduction);
// };
// ```
//
// and the functional that uses it:
//
// ```
// Tensor some_functional(
//     const Tensor& input,
//     SomeOptions options = {}) {
//   ...
// }
// ```
//
// Normally, we would expect this to work:
//
// `F::some_functional(input, torch::kNone)`
//
// However, it throws the following error instead:
//
// ```
// error: could not convert `torch::kNone` from `const torch::enumtype::kNone`
// to `torch::nn::SomeOptions`
// ```
//
// To get around this problem, we explicitly provide the following constructors
// for `SomeOptions`:
//
// ```
// SomeOptions(torch::enumtype::kNone reduction) : reduction_(torch::kNone) {}
// SomeOptions(torch::enumtype::kMean reduction) : reduction_(torch::kMean) {}
// SomeOptions(torch::enumtype::kSum reduction) : reduction_(torch::kSum) {}
// ```
//
// so that the conversion from `torch::kNone` to `SomeOptions` would work.
//
// Note that we also provide the default constructor `SomeOptions() {}`, so that
// `SomeOptions options = {}` can work.
#define TORCH_OPTIONS_CTOR_VARIANT_ARG3(                                       \
    OPTIONS_NAME, ARG_NAME, TYPE1, TYPE2, TYPE3)                               \
  OPTIONS_NAME() = default;                                                    \
  OPTIONS_NAME(torch::enumtype::TYPE1 ARG_NAME) : ARG_NAME##_(torch::TYPE1) {} \
  OPTIONS_NAME(torch::enumtype::TYPE2 ARG_NAME) : ARG_NAME##_(torch::TYPE2) {} \
  OPTIONS_NAME(torch::enumtype::TYPE3 ARG_NAME) : ARG_NAME##_(torch::TYPE3) {}

#define TORCH_OPTIONS_CTOR_VARIANT_ARG4(                                       \
    OPTIONS_NAME, ARG_NAME, TYPE1, TYPE2, TYPE3, TYPE4)                        \
  OPTIONS_NAME() = default;                                                    \
  OPTIONS_NAME(torch::enumtype::TYPE1 ARG_NAME) : ARG_NAME##_(torch::TYPE1) {} \
  OPTIONS_NAME(torch::enumtype::TYPE2 ARG_NAME) : ARG_NAME##_(torch::TYPE2) {} \
  OPTIONS_NAME(torch::enumtype::TYPE3 ARG_NAME) : ARG_NAME##_(torch::TYPE3) {} \
  OPTIONS_NAME(torch::enumtype::TYPE4 ARG_NAME) : ARG_NAME##_(torch::TYPE4) {}

TORCH_ENUM_DECLARE(Linear)
TORCH_ENUM_DECLARE(Conv1D)
TORCH_ENUM_DECLARE(Conv2D)
TORCH_ENUM_DECLARE(Conv3D)
TORCH_ENUM_DECLARE(ConvTranspose1D)
TORCH_ENUM_DECLARE(ConvTranspose2D)
TORCH_ENUM_DECLARE(ConvTranspose3D)
TORCH_ENUM_DECLARE(Sigmoid)
TORCH_ENUM_DECLARE(Tanh)
TORCH_ENUM_DECLARE(ReLU)
TORCH_ENUM_DECLARE(GELU)
TORCH_ENUM_DECLARE(SiLU)
TORCH_ENUM_DECLARE(Mish)
TORCH_ENUM_DECLARE(LeakyReLU)
TORCH_ENUM_DECLARE(FanIn)
TORCH_ENUM_DECLARE(FanOut)
TORCH_ENUM_DECLARE(Constant)
TORCH_ENUM_DECLARE(Reflect)
TORCH_ENUM_DECLARE(Replicate)
TORCH_ENUM_DECLARE(Circular)
TORCH_ENUM_DECLARE(Nearest)
TORCH_ENUM_DECLARE(Bilinear)
TORCH_ENUM_DECLARE(Bicubic)
TORCH_ENUM_DECLARE(Trilinear)
TORCH_ENUM_DECLARE(Area)
TORCH_ENUM_DECLARE(NearestExact)
TORCH_ENUM_DECLARE(Sum)
TORCH_ENUM_DECLARE(Mean)
TORCH_ENUM_DECLARE(Max)
TORCH_ENUM_DECLARE(None)
TORCH_ENUM_DECLARE(BatchMean)
TORCH_ENUM_DECLARE(Zeros)
TORCH_ENUM_DECLARE(Border)
TORCH_ENUM_DECLARE(Reflection)
TORCH_ENUM_DECLARE(RNN_TANH)
TORCH_ENUM_DECLARE(RNN_RELU)
TORCH_ENUM_DECLARE(LSTM)
TORCH_ENUM_DECLARE(GRU)
TORCH_ENUM_DECLARE(Valid)
TORCH_ENUM_DECLARE(Same)

namespace torch {
namespace enumtype {

struct _compute_enum_name {
  TORCH_ENUM_PRETTY_PRINT(Linear)
  TORCH_ENUM_PRETTY_PRINT(Conv1D)
  TORCH_ENUM_PRETTY_PRINT(Conv2D)
  TORCH_ENUM_PRETTY_PRINT(Conv3D)
  TORCH_ENUM_PRETTY_PRINT(ConvTranspose1D)
  TORCH_ENUM_PRETTY_PRINT(ConvTranspose2D)
  TORCH_ENUM_PRETTY_PRINT(ConvTranspose3D)
  TORCH_ENUM_PRETTY_PRINT(Sigmoid)
  TORCH_ENUM_PRETTY_PRINT(Tanh)
  TORCH_ENUM_PRETTY_PRINT(ReLU)
  TORCH_ENUM_PRETTY_PRINT(GELU)
  TORCH_ENUM_PRETTY_PRINT(SiLU)
  TORCH_ENUM_PRETTY_PRINT(Mish)
  TORCH_ENUM_PRETTY_PRINT(LeakyReLU)
  TORCH_ENUM_PRETTY_PRINT(FanIn)
  TORCH_ENUM_PRETTY_PRINT(FanOut)
  TORCH_ENUM_PRETTY_PRINT(Constant)
  TORCH_ENUM_PRETTY_PRINT(Reflect)
  TORCH_ENUM_PRETTY_PRINT(Replicate)
  TORCH_ENUM_PRETTY_PRINT(Circular)
  TORCH_ENUM_PRETTY_PRINT(Nearest)
  TORCH_ENUM_PRETTY_PRINT(Bilinear)
  TORCH_ENUM_PRETTY_PRINT(Bicubic)
  TORCH_ENUM_PRETTY_PRINT(Trilinear)
  TORCH_ENUM_PRETTY_PRINT(Area)
  TORCH_ENUM_PRETTY_PRINT(NearestExact)
  TORCH_ENUM_PRETTY_PRINT(Sum)
  TORCH_ENUM_PRETTY_PRINT(Mean)
  TORCH_ENUM_PRETTY_PRINT(Max)
  TORCH_ENUM_PRETTY_PRINT(None)
  TORCH_ENUM_PRETTY_PRINT(BatchMean)
  TORCH_ENUM_PRETTY_PRINT(Zeros)
  TORCH_ENUM_PRETTY_PRINT(Border)
  TORCH_ENUM_PRETTY_PRINT(Reflection)
  TORCH_ENUM_PRETTY_PRINT(RNN_TANH)
  TORCH_ENUM_PRETTY_PRINT(RNN_RELU)
  TORCH_ENUM_PRETTY_PRINT(LSTM)
  TORCH_ENUM_PRETTY_PRINT(GRU)
  TORCH_ENUM_PRETTY_PRINT(Valid)
  TORCH_ENUM_PRETTY_PRINT(Same)
};

template <typename V>
std::string get_enum_name(V variant_enum) {
  return std::visit(enumtype::_compute_enum_name{}, variant_enum);
}

template <typename V>
at::Reduction::Reduction reduction_get_enum(V variant_enum) {
  if (std::holds_alternative<enumtype::kNone>(variant_enum)) {
    return at::Reduction::None;
  } else if (std::holds_alternative<enumtype::kMean>(variant_enum)) {
    return at::Reduction::Mean;
  } else if (std::holds_alternative<enumtype::kSum>(variant_enum)) {
    return at::Reduction::Sum;
  } else {
    TORCH_CHECK(
        false,
        get_enum_name(variant_enum),
        " is not a valid value for reduction");
    return at::Reduction::END;
  }
}

} // namespace enumtype
} // namespace torch
