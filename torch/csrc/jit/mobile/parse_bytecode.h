#pragma once
#include <torch/csrc/jit/mobile/function.h>
#include "caffe2/serialize/versions.h"

namespace torch {
namespace jit {
namespace mobile {
using c10::IValue;
TORCH_API void parseInstructions(
    const std::string& function_name,
    c10::ivalue::TupleElements&& ins_list,
    c10::ivalue::TupleElements& debug_handles_m_tuple,
    mobile::Function* function,
    bool use_upgrader = false,
    uint64_t operator_version =
        caffe2::serialize::kMaxSupportedFileFormatVersion);
TORCH_API void parseConstants(
    const c10::ivalue::TupleElements& consts_list,
    mobile::Function* function);
TORCH_API void parseTypes(
    const c10::ivalue::TupleElements& types_list,
    mobile::Function* function);
TORCH_API void parseRegisterSize(size_t rsize, mobile::Function* function);
} // namespace mobile
} // namespace jit
} // namespace torch
