#pragma once
#include <torch/csrc/Export.h>
namespace torch::jit {
TORCH_API void setDisableAliasDb(bool o);
TORCH_API bool getDisableAliasDb();
} // namespace torch::jit
