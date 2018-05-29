#pragma once

#include <cstdint>

namespace c10d {

enum class CollectiveType : std::uint8_t {
  ALLGATHER = 0,
  GATHER,
  SCATTER,
  ALLREDUCE,
  REDUCE,
  BROADCAST,
  SEND,
  BARRIER,
  UNUSED,
};

enum class DeviceType : std::uint8_t {
  CPU = 0,
  CUDA,
  UNUSED,
};

enum class ReduceOp : std::uint8_t {
  SUM = 0,
  PRODUCT,
  MIN,
  MAX,
  UNUSED,
};

struct BroadcastOptions {
  int rootRank = 0;
  int rootTensor = 0;
};

struct AllreduceOptions {
  ReduceOp reduceOp = ReduceOp::SUM;
};

} // namespace c10d
