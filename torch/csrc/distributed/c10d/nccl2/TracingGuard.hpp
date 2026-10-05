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

// Process group metadata recorded with every record_param_comms event, matching
// stock ProcessGroupNCCL: the PG name tuple is (uid, desc) and the global rank
// start/stride describe the group's members in world-rank terms (see
// getGlobalRankStartAndStride). Non-owning: views into the issuing process
// group, which outlives the guard.
struct TracingGuardInfo {
  std::string_view pgUid;
  std::string_view pgDesc;
  int commSize{0};
  int globalRankStart{-1};
  int globalRankStride{-1};
};

class TracingGuard {
 public:
  // `sequence_number` is the issuing process group's own collective counter
  // (ProcessGroupNCCL::sequence_number_, the value the work handle also
  // carries). Profilers pair collectives across ranks by (process group,
  // sequence number), so it has to be per-PG: a process-wide counter makes a
  // rank's numbering depend on how it interleaved its other groups' work.
  TracingGuard(
      const TracingGuardInfo& info,
      std::string_view collective_name,
      int collective_rank,
      uint64_t sequence_number,
      const std::vector<at::Tensor>& input_tensor_list = {},
      const std::vector<at::Tensor>& output_tensor_list = {});

  TracingGuard(
      const TracingGuardInfo& info,
      std::string_view collective_name,
      int collective_rank,
      uint64_t sequence_number,
      const at::Tensor& input_tensor,
      const at::Tensor& output_tensor);

  // Records the given split sizes rather than each tensor's numel, like stock
  // all_to_allv: empty for an equal split, else the per-rank splits.
  TracingGuard(
      const TracingGuardInfo& info,
      std::string_view collective_name,
      int collective_rank,
      uint64_t sequence_number,
      const at::Tensor& input_tensor,
      const at::Tensor& output_tensor,
      const std::vector<int64_t>& input_split_sizes,
      const std::vector<int64_t>& output_split_sizes);

  void initializeTracingCommon(
      const TracingGuardInfo& info,
      std::string_view collective_name,
      int collective_rank,
      uint64_t sequence_number,
      const std::vector<at::Tensor>& input_tensor_list,
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<int64_t>& input_split_sizes,
      const std::vector<int64_t>& output_split_sizes);

  std::shared_ptr<torch::ParamCommsDebugInfo> getDebugInfo(
      const TracingGuardInfo& info,
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
