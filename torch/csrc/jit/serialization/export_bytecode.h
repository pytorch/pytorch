#pragma once

#include <tuple>
#include <unordered_map>

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

namespace torch {
namespace jit {

struct TORCH_API CompilationOptions {
  bool incl_interface_call = false;
  bool enable_default_value_for_unspecified_arg = false;
  bool enable_default_args_before_out_args = true;
  int model_version = caffe2::serialize::kProducedBytecodeVersion;
};

TORCH_API mobile::Module jitModuleToMobile(
    const Module& module,
    const CompilationOptions& options);

} // namespace jit
} // namespace torch
