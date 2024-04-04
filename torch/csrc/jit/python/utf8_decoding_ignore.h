#pragma once
#include <torch/csrc/Export.h>
namespace torch::jit {
TORCH_API void setUTF8DecodingIgnore(bool o);
TORCH_API bool getUTF8DecodingIgnore();
} // namespace torch::jit
