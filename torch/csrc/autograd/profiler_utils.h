#pragma once

#include <string>
#include <unordered_map>
#include <ATen/ATen.h>
#include <ATen/record_function.h>

namespace torch { namespace autograd {
namespace profiler {

void TORCH_API saveExtraArgs(std::unordered_map<std::string, c10::IValue> &extra_args,
                             const at::RecordFunction& fn);

uint64_t TORCH_API computeFlops(const at::StringView &op_name,
                                const std::unordered_map<std::string, c10::IValue> &extra_args);

}}}

