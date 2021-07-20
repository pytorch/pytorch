#pragma once
#include <torch/csrc/jit/mobile/function.h>

namespace torch {
namespace jit {
namespace mobile {
using c10::IValue;
TORCH_API void parseInstructions(
    const std::string& function_name,
    const IValue& codeTable,
    const IValue& debug_handles_element,
    mobile::Function* function);

TORCH_API void parseConstants(
    const IValue& codeTable,
    mobile::Function* function);

TORCH_API void parseTypes(const IValue& codeTable, mobile::Function* function);

TORCH_API void parseRegisterSize(
    const IValue& codeTable,
    mobile::Function* function);
} // namespace mobile
} // namespace jit
} // namespace torch
