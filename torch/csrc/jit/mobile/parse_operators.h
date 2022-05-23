#pragma once
#include <torch/csrc/jit/mobile/function.h>

namespace torch {
namespace jit {
using c10::IValue;

enum MobileModuleLoadOptions {
  OPERATOR_CHECK = 1,
};

const uint64_t _default_mobile_module_load_options =
    MobileModuleLoadOptions::OPERATOR_CHECK;

namespace mobile {

TORCH_API void parseOperators(
    c10::ivalue::TupleElements&& ops_list,
    const uint64_t& module_load_options,
    mobile::Function* function);
} // namespace mobile
} // namespace jit
} // namespace torch
