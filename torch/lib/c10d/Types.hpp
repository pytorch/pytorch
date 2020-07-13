#pragma once

#include <chrono>
#include <cstdint>

namespace c10d {

enum class ReduceOp : std::uint8_t {
  SUM = 0,
  PRODUCT,
  MIN,
  MAX,
  BAND, // Bitwise AND
  BOR, // Bitwise OR
  BXOR, // Bitwise XOR
  UNUSED,
};

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

struct NCCLAllreduceOptions : AllreduceOptions {
  c10::optional<std::vector<at::cuda::CUDAStream>> cudaStreams;
};

struct NCCLAllgatherOptions : AllgatherOptions {
  c10::optional<std::vector<at::cuda::CUDAStream>> cudaStreams;
};

struct NCCLReduceOptions : ReduceOptions {
  c10::optional<std::vector<at::cuda::CUDAStream>> cudaStreams;
};

struct NCCLBroadcastOptions : BroadcastOptions {
  c10::optional<std::vector<at::cuda::CUDAStream>> cudaStreams;
};

struct NCCLReduceScatterOptions : ReduceScatterOptions {
  c10::optional<std::vector<at::cuda::CUDAStream>> cudaStreams;
};

} // namespace c10d
