#pragma once

#include <chrono>
#include <cstdint>

#include <torch/csrc/utils/comm.h>

namespace c10d {

using ReduceOp = torch::utils::comm::ReduceOp;

constexpr auto kUnsetTimeout = std::chrono::milliseconds(-1);

struct BroadcastOptions {
  int rootRank = 0;
  int rootTensor = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

struct AllreduceOptions {
  ReduceOp reduceOp = ReduceOp::SUM;
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

struct AllreduceCoalescedOptions : AllreduceOptions {};

struct ReduceOptions {
  ReduceOp reduceOp = ReduceOp::SUM;
  int rootRank = 0;
  int rootTensor = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

struct AllgatherOptions {
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

struct GatherOptions {
  int rootRank = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

struct ScatterOptions {
  int rootRank = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

struct ReduceScatterOptions {
  ReduceOp reduceOp = ReduceOp::SUM;
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

struct AllToAllOptions {
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

struct BarrierOptions {
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

} // namespace c10d
