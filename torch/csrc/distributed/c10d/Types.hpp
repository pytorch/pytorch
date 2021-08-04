#pragma once

#include <chrono>
#include <cstdint>


#if defined(NCCL_MAJOR) && ((NCCL_MAJOR > 2) || \
                            (NCCL_MAJOR == 2) && (NCCL_MINOR >= 10))
#define NCCL_HAS_AVG 1
#endif

namespace c10d {

enum class ReduceOp : std::uint8_t {
  SUM = 0,
#ifdef NCCL_HAS_AVG
  AVG,
#endif
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
  std::vector<int> device_ids;
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

} // namespace c10d
