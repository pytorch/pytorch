// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/TracingGuard.hpp>

#include <string>
#include <string_view>

#include <ATen/core/functional.h>
#include <ATen/core/ivalue.h>
#include <ATen/record_function.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>

namespace c10d::nccl2 {

// Creates a ParamCommsDebugInfo object containing metadata about a collective
// operation for integration with PyTorch's debugging and profiling
// infrastructure. The debug info includes communicator details, operation name,
// tensor sizes, data types, and split sizes for variable-length collectives.
// This information is used by PyTorch's PARAM_COMMS tracing to track and
// analyze distributed communication patterns.
std::shared_ptr<torch::ParamCommsDebugInfo> TracingGuard::getDebugInfo(
    const TracingGuardInfo& info,
    std::string_view collective_name,
    int collective_rank,
    const std::vector<at::Tensor>& input_tensor_list,
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<int64_t>& input_split_sizes,
    const std::vector<int64_t>& output_split_sizes) {
  int64_t input_total_numel = 0;
  for (const auto& input_tensor_list_elem : input_tensor_list) {
    input_total_numel += input_tensor_list_elem.numel();
  }
  int64_t output_total_numel = 0;
  for (const auto& output_tensor_list_elem : output_tensor_list) {
    output_total_numel += output_tensor_list_elem.numel();
  }

  // If both input and output tensor lists are empty, use a default data type.
  auto data_type = at::kByte;
  if (input_tensor_list.size() > 0) {
    data_type = input_tensor_list.front().scalar_type();
  } else if (output_tensor_list.size() > 0) {
    data_type = output_tensor_list.front().scalar_type();
  }

  return std::make_shared<torch::ParamCommsDebugInfo>(
      std::make_tuple(std::string(info.pgUid), std::string(info.pgDesc)),
      collective_rank,
      std::string(collective_name).c_str(),
      input_total_numel,
      output_total_numel,
      data_type,
      input_split_sizes,
      output_split_sizes,
      info.globalRankStart,
      info.globalRankStride,
      info.commSize);
}

void TracingGuard::initializeTracingCommon(
    const TracingGuardInfo& info,
    std::string_view collective_name,
    int collective_rank,
    uint64_t sequence_number,
    const std::vector<at::Tensor>& input_tensor_list,
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<int64_t>& in_split_sizes,
    const std::vector<int64_t>& out_split_sizes) {
  auto debug_info = getDebugInfo(
      info,
      collective_name,
      collective_rank,
      input_tensor_list,
      output_tensor_list,
      in_split_sizes,
      out_split_sizes);
  // Where the profiler reads the sequence number from to build its cross-rank
  // "Comms Id" (torch/csrc/profiler/util.cpp); it stays at -1 unless set here.
  // isP2P is false throughout: nccl2 counts p2p and collectives in the one
  // per-PG sequence_number_, so there is no separate p2p sequence space.
  debug_info->setSequenceInfo(static_cast<int64_t>(sequence_number), false);
  debug_info_guard_ = std::make_unique<c10::DebugInfoGuard>(
      c10::DebugInfoKind::PARAM_COMMS_INFO, std::move(debug_info));

  if (record_function_guard_->needsInputs()) {
    std::initializer_list<const c10::IValue> paramList = {
        c10::IValue(input_tensor_list),
        std::make_tuple(static_cast<int64_t>(sequence_number), false),
        std::make_tuple(std::string(info.pgUid), std::string(info.pgDesc)),
        collective_rank,
        std::string(collective_name),
        in_split_sizes,
        out_split_sizes,
        info.globalRankStart,
        info.globalRankStride,
        info.commSize};
    c10::ArrayRef<const c10::IValue> paramInputs(paramList);
    record_function_guard_->before(
        at::kParamCommsCallName, std::move(paramInputs));
  } else {
    record_function_guard_->before(at::kParamCommsCallName);
  }
  if (record_function_guard_->needsOutputs()) {
    record_function_guard_->setOutputs(
        std::vector<c10::IValue>(1, c10::IValue(output_tensor_list)));
  }
}

namespace {

std::vector<int64_t> numels(const std::vector<at::Tensor>& tensors) {
  return c10::fmap(tensors, [](const at::Tensor& t) { return t.numel(); });
}

} // namespace

TracingGuard::TracingGuard(
    const TracingGuardInfo& info,
    std::string_view collective_name,
    int collective_rank,
    uint64_t sequence_number,
    const std::vector<at::Tensor>& input_tensor_list,
    const std::vector<at::Tensor>& output_tensor_list) {
  record_function_guard_.emplace(at::RecordScope::FUNCTION);
  if (!record_function_guard_->isActive()) {
    return;
  }
  initializeTracingCommon(
      info,
      collective_name,
      collective_rank,
      sequence_number,
      input_tensor_list,
      output_tensor_list,
      numels(input_tensor_list),
      numels(output_tensor_list));
}

TracingGuard::TracingGuard(
    const TracingGuardInfo& info,
    std::string_view collective_name,
    int collective_rank,
    uint64_t sequence_number,
    const at::Tensor& input_tensor,
    const at::Tensor& output_tensor) {
  record_function_guard_.emplace(at::RecordScope::FUNCTION);
  if (!record_function_guard_->isActive()) {
    return;
  }
  initializeTracingCommon(
      info,
      collective_name,
      collective_rank,
      sequence_number,
      {input_tensor},
      {output_tensor},
      {input_tensor.numel()},
      {output_tensor.numel()});
}

TracingGuard::TracingGuard(
    const TracingGuardInfo& info,
    std::string_view collective_name,
    int collective_rank,
    uint64_t sequence_number,
    const at::Tensor& input_tensor,
    const at::Tensor& output_tensor,
    const std::vector<int64_t>& input_split_sizes,
    const std::vector<int64_t>& output_split_sizes) {
  record_function_guard_.emplace(at::RecordScope::FUNCTION);
  if (!record_function_guard_->isActive()) {
    return;
  }
  initializeTracingCommon(
      info,
      collective_name,
      collective_rank,
      sequence_number,
      {input_tensor},
      {output_tensor},
      input_split_sizes,
      output_split_sizes);
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
