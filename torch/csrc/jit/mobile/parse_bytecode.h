#pragma once
#include <torch/csrc/jit/mobile/function.h>

namespace torch {
namespace jit {
namespace mobile {
using c10::IValue;
TORCH_API void parseInstructions(
    const std::string& function_name,
    const std::vector<IValue>& ins_list,
    std::vector<IValue>& debug_handles_m_tuple,
    mobile::Function* function);
TORCH_API void parseConstants(const std::vector<IValue>& consts_list, mobile::Function* function);
TORCH_API void parseTypes(const std::vector<IValue>& types_list, mobile::Function* function);
TORCH_API void parseRegisterSize(size_t rsize, mobile::Function* function);
} // namespace mobile
} // namespace jit
} // namespace torch
