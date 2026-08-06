// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef USE_C10D_NCCL

#include <ATen/core/ivalue.h>
#include <cstdint>
#include <string_view>
#include <vector>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>

namespace c10d::nccl2 {

// Snapshot of communicator metadata captured at construction time.
// Stored by work handles to avoid calling checked comm accessors that
// may throw after the communicator transitions to UNINITIALIZED or FINALIZED.
struct TracingGuardInfo {
  std::string commName;
  std::string commId;
  int commSize{0};
  int rank{-1};
};

class TracingGuard {
 public:
  // `sequence_number` is the issuing process group's own collective counter
  // (ProcessGroupNCCL::sequence_number_, the value the work handle also
  // carries). Profilers pair collectives across ranks by (process group,
  // sequence number), so it has to be per-PG: a process-wide counter makes a
  // rank's numbering depend on how it interleaved its other groups' work.
  TracingGuard(
      std::string_view comm_name,
      int comm_size,
      std::string_view collective_name,
      int collective_rank,
      uint64_t sequence_number,
      const std::vector<at::Tensor>& input_tensor_list = {},
      const std::vector<at::Tensor>& output_tensor_list = {});

  TracingGuard(
      std::string_view comm_name,
      int comm_size,
      std::string_view collective_name,
      int collective_rank,
      uint64_t sequence_number,
      const at::Tensor& input_tensor,
      const at::Tensor& output_tensor);

  TracingGuard(
      const TracingGuardInfo& info,
      std::string_view collective_name,
      uint64_t sequence_number,
      const std::vector<at::Tensor>& input_tensor_list = {},
      const std::vector<at::Tensor>& output_tensor_list = {});

  TracingGuard(
      const TracingGuardInfo& info,
      std::string_view collective_name,
      uint64_t sequence_number,
      const at::Tensor& input_tensor,
      const at::Tensor& output_tensor);

  void initializeTracingCommon(
      std::string_view comm_name,
      std::string_view comm_id,
      int comm_size,
      std::string_view collective_name,
      int collective_rank,
      uint64_t sequence_number,
      const std::vector<at::Tensor>& input_tensor_list,
      const std::vector<at::Tensor>& output_tensor_list);

  std::shared_ptr<torch::ParamCommsDebugInfo> getDebugInfo(
      std::string_view comm_name,
      std::string_view comm_id,
      int comm_size,
      std::string_view collective_name,
      int collective_rank,
      const std::vector<at::Tensor>& input_tensor_list,
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<int64_t>& input_split_sizes,
      const std::vector<int64_t>& output_split_sizes);

 private:
  std::unique_ptr<c10::DebugInfoGuard> debug_info_guard_;
  std::optional<at::RecordFunction> record_function_guard_;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
