#pragma once

#include <set>
#include <string>
#include <vector>

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/model_tracer/BuildFeatureTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/CustomClassTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.h>

namespace torch {
namespace jit {
namespace mobile {

const std::vector<std::string> always_included_traced_ops = {
    // The following are called from setup sections.
    "aten::resize_",
    "aten::slice.Tensor",
};

struct TracerResult {
  std::set<std::string> root_ops;
  std::set<std::string> traced_operators;
  KernelDTypeTracer::kernel_tags_type called_kernel_tags;
  CustomClassTracer::custom_classes_type loaded_classes;
  BuildFeatureTracer::build_feature_type build_features;
  std::set<std::string> enabled_backends;
};

/**
 * Trace a single model and return the TracerResult.
 */
TracerResult trace_run(const std::string& input_module_path);

/**
 * Trace multiple models and return the TracerResult.
 */
TracerResult trace_run(const std::vector<std::string>& input_module_paths);

} // namespace mobile
} // namespace jit
} // namespace torch
