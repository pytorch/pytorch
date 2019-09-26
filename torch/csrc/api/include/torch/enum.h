#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#define TORCH_ENUM_DECLARE(name) \
namespace torch { \
namespace enumtype { \
  struct k##name {}; \
} \
TORCH_API extern const enumtype::k##name k##name; \
}

#define TORCH_ENUM_DEFINE(name) \
namespace torch { \
const enumtype::k##name k##name; \
}

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
