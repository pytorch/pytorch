#pragma once

#include <string>
#include <unordered_map>
#include <ATen/ATen.h>
#include <ATen/record_function.h>

namespace torch { namespace autograd {
namespace profiler {

std::unordered_map<std::string, c10::IValue> TORCH_API saveExtraArgs(const at::RecordFunction& fn);

uint64_t TORCH_API computeFlops(const std::string &op_name,
                                const std::unordered_map<std::string, c10::IValue> &extra_args);

}}}
