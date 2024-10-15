#pragma once
#include <torch/csrc/jit/mobile/function.h>

namespace torch::jit::mobile {
using c10::IValue;
TORCH_API void parseInstructions(
    const std::string& function_name,
    c10::ivalue::TupleElements&& ins_list,
    c10::ivalue::TupleElements& debug_handles_m_tuple,
    mobile::Function* function);
TORCH_API void parseConstants(
    const c10::ivalue::TupleElements& consts_list,
    mobile::Function* function);
TORCH_API void parseTypes(
    const c10::ivalue::TupleElements& types_list,
    mobile::Function* function);
TORCH_API void parseRegisterSize(size_t rsize, mobile::Function* function);
TORCH_API void applyUpgrader(
    mobile::Function* function,
    uint64_t operator_version);
} // namespace torch::jit::mobile
