#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d {

// Identifies the operation a pre/post hook is firing for.
enum class HookOpName : uint8_t {
  SEND = 0,
  RECV,
  BROADCAST,
  ALLREDUCE,
  REDUCE,
  ALLGATHER,
  REDUCE_SCATTER,
  ALLTOALL,
  BARRIER,
  SCATTER,
  GATHER,
  SPLIT,
  NEW_WINDOW,
  UNKNOWN,
};

// Arguments passed to a pre-hook, fired before an operation is issued.
struct PreHookArgs {
  HookOpName name = HookOpName::UNKNOWN;
  bool async_op = false;
  std::vector<at::Tensor> input_tensors;
  std::vector<at::Tensor> output_tensors;
  int64_t root = -1;
  // Correlates a pre-hook call with its matching post-hook call.
  int64_t op_id = 0;
};

using PreHook = std::function<void(const PreHookArgs&)>;

// Arguments passed to a post-hook, fired after an operation is issued.
struct PostHookArgs {
  HookOpName name = HookOpName::UNKNOWN;
  bool async_op = false;
  // Work handle for the issued operation; may be null for synchronous ops.
  c10::intrusive_ptr<Work> work;
  int64_t op_id = 0;
};

using PostHook = std::function<void(const PostHookArgs&)>;

// Abort hook - called before aborting when a collective times out or fails.
// This allows users to capture debug information before the abort.
using AbortHook = std::function<void()>;

} // namespace c10d
