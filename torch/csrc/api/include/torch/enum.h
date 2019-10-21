#pragma once

#include <string>

#include <ATen/core/Reduction.h>
#include <c10/util/Exception.h>
#include <c10/util/variant.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#define TORCH_ENUM_DECLARE(name) \
namespace torch { \
namespace enumtype { \
  /*
    NOTE: We need to provide the default constructor for each struct,
    otherwise Clang 3.8 would complain:
    ```
    error: default initialization of an object of const type 'const enumtype::Enum1'
    without a user-provided default constructor
    ```
  */ \
  struct k##name { k##name() {} }; \
} \
TORCH_API extern const enumtype::k##name k##name; \
}

#define TORCH_ENUM_DEFINE(name) \
namespace torch { \
const enumtype::k##name k##name; \
}

#define TORCH_ENUM_PRETTY_PRINT(name) \
const char* operator()(const enumtype::k##name& v) const { \
  return #name; \
}

#define TORCH_OPTIONS_CTOR_VARIANT_ARG3(OPTIONS_NAME, ARG_NAME, TYPE1, TYPE2, TYPE3) \
OPTIONS_NAME(torch::enumtype::TYPE1 ARG_NAME) : ARG_NAME##_(torch::TYPE1) {} \
OPTIONS_NAME(torch::enumtype::TYPE2 ARG_NAME) : ARG_NAME##_(torch::TYPE2) {} \
OPTIONS_NAME(torch::enumtype::TYPE3 ARG_NAME) : ARG_NAME##_(torch::TYPE3) {}

#define TORCH_OPTIONS_CTOR_VARIANT_ARG4(OPTIONS_NAME, ARG_NAME, TYPE1, TYPE2, TYPE3, TYPE4) \
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
TORCH_ENUM_DECLARE(LeakyReLU)
TORCH_ENUM_DECLARE(FanIn)
TORCH_ENUM_DECLARE(FanOut)
TORCH_ENUM_DECLARE(None)
TORCH_ENUM_DECLARE(Mean)
TORCH_ENUM_DECLARE(Sum)
TORCH_ENUM_DECLARE(BatchMean)

namespace torch {
namespace enumtype {

struct enum_name {
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
  TORCH_ENUM_PRETTY_PRINT(LeakyReLU)
  TORCH_ENUM_PRETTY_PRINT(FanIn)
  TORCH_ENUM_PRETTY_PRINT(FanOut)
  TORCH_ENUM_PRETTY_PRINT(None)
  TORCH_ENUM_PRETTY_PRINT(Mean)
  TORCH_ENUM_PRETTY_PRINT(Sum)
  TORCH_ENUM_PRETTY_PRINT(BatchMean)
};

struct _reduction_get_enum {
  at::Reduction::Reduction operator()(const enumtype::kNone& v) const {
    return at::Reduction::None;
  }
  at::Reduction::Reduction operator()(const enumtype::kMean& v) const {
    return at::Reduction::Mean;
  }
  at::Reduction::Reduction operator()(const enumtype::kSum& v) const {
    return at::Reduction::Sum;
  }
  at::Reduction::Reduction operator()(const enumtype::kBatchMean& v) const {
    TORCH_CHECK(false, "Unsupported reduction enum");
    return at::Reduction::END;
  }
};

} // namespace enumtype
} // namespace torch
