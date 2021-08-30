#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.h>

namespace torch {
namespace jit {
namespace mobile {

struct TracerResult {
  std::set<std::string> root_ops;
  std::set<std::string> traced_operators;
  KernelDTypeTracer::kernel_tags_type called_kernel_tags;
  std::vector<std::string> enabled_backends;
};

TracerResult trace_run(const std::string& input_module_path);
} // namespace mobile
} // namespace jit
} // namespace torch
