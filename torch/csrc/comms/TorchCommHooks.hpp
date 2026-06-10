// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/comms/TorchCommTypes.hpp>
#include <torch/csrc/comms/TorchWork.hpp>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace torch::comms {

// Forward declarations
class TorchComm;
class TorchCommWindow;

// Enum for collective operation names
enum class OpName {
  send,
  recv,
  broadcast,
  all_reduce,
  reduce,
  all_gather,
  all_gather_v,
  all_gather_single,
  reduce_scatter,
  reduce_scatter_v,
  reduce_scatter_single,
  all_to_all_single,
  all_to_all_v_single,
  all_to_all,
  barrier,
  scatter,
  gather,
  gather_single,
  split,
  new_window,
  batch_op_issue,
  finalize,
};

// Convert OpName enum to string
constexpr std::string_view opToString(OpName name) {
  switch (name) {
    case OpName::send:
      return "send";
    case OpName::recv:
      return "recv";
    case OpName::broadcast:
      return "broadcast";
    case OpName::all_reduce:
      return "all_reduce";
    case OpName::reduce:
      return "reduce";
    case OpName::all_gather:
      return "all_gather";
    case OpName::all_gather_v:
      return "all_gather_v";
    case OpName::all_gather_single:
      return "all_gather_single";
    case OpName::reduce_scatter:
      return "reduce_scatter";
    case OpName::reduce_scatter_v:
      return "reduce_scatter_v";
    case OpName::reduce_scatter_single:
      return "reduce_scatter_single";
    case OpName::all_to_all_single:
      return "all_to_all_single";
    case OpName::all_to_all_v_single:
      return "all_to_all_v_single";
    case OpName::all_to_all:
      return "all_to_all";
    case OpName::barrier:
      return "barrier";
    case OpName::scatter:
      return "scatter";
    case OpName::gather:
      return "gather";
    case OpName::gather_single:
      return "gather_single";
    case OpName::split:
      return "split";
    case OpName::new_window:
      return "new_window";
    case OpName::batch_op_issue:
      return "batch_op_issue";
    case OpName::finalize:
      return "finalize";
  }
  return "unknown";
}

// -- Per-collective pre-hook argument structs --
//
// Each collective operation has its own typed args struct carrying exactly
// the parameters relevant to that operation. All structs use explicit
// constructors so that omitting or reordering parameters is a compile
// error. References are valid for the duration of the hook call (they
// point to the caller's stack).

// Point-to-point
struct SendPreHookArgs {
  SendPreHookArgs(const at::Tensor& tensor, int peer, bool async_op)
      : tensor(tensor), peer(peer), async_op(async_op) {}
  const at::Tensor& tensor;
  int peer;
  bool async_op;
};

struct RecvPreHookArgs {
  RecvPreHookArgs(at::Tensor& tensor, int peer, bool async_op)
      : tensor(tensor), peer(peer), async_op(async_op) {}
  at::Tensor& tensor;
  int peer;
  bool async_op;
};

// In-place collectives (input == output)
struct BroadcastPreHookArgs {
  BroadcastPreHookArgs(at::Tensor& tensor, int root, bool async_op)
      : tensor(tensor), root(root), async_op(async_op) {}
  at::Tensor& tensor;
  int root;
  bool async_op;
};

struct AllReducePreHookArgs {
  AllReducePreHookArgs(at::Tensor& tensor, const ReduceOp& op, bool async_op)
      : tensor(tensor), op(op), async_op(async_op) {}
  at::Tensor& tensor;
  const ReduceOp& op;
  bool async_op;
};

struct ReducePreHookArgs {
  ReducePreHookArgs(
      const at::Tensor& tensor,
      int root,
      const ReduceOp& op,
      bool async_op)
      : tensor(tensor), root(root), op(op), async_op(async_op) {}
  const at::Tensor& tensor;
  int root;
  const ReduceOp& op;
  bool async_op;
};

// Gather-style (input tensor -> output tensor list)
struct AllGatherPreHookArgs {
  AllGatherPreHookArgs(
      const at::Tensor& input,
      const std::vector<at::Tensor>& output,
      bool async_op)
      : input(input), output(output), async_op(async_op) {}
  const at::Tensor& input;
  const std::vector<at::Tensor>& output;
  bool async_op;
};

struct AllGatherVPreHookArgs {
  AllGatherVPreHookArgs(
      const at::Tensor& input,
      const std::vector<at::Tensor>& output,
      bool async_op)
      : input(input), output(output), async_op(async_op) {}
  const at::Tensor& input;
  const std::vector<at::Tensor>& output;
  bool async_op;
};

struct AllGatherSinglePreHookArgs {
  AllGatherSinglePreHookArgs(
      const at::Tensor& input,
      at::Tensor& output,
      bool async_op)
      : input(input), output(output), async_op(async_op) {}
  const at::Tensor& input;
  at::Tensor& output;
  bool async_op;
};

// Scatter-style (input tensor list -> output tensor)
struct ReduceScatterPreHookArgs {
  ReduceScatterPreHookArgs(
      const std::vector<at::Tensor>& input,
      at::Tensor& output,
      const ReduceOp& op,
      bool async_op)
      : input(input), output(output), op(op), async_op(async_op) {}
  const std::vector<at::Tensor>& input;
  at::Tensor& output;
  const ReduceOp& op;
  bool async_op;
};

struct ReduceScatterVPreHookArgs {
  ReduceScatterVPreHookArgs(
      const std::vector<at::Tensor>& input,
      at::Tensor& output,
      const ReduceOp& op,
      bool async_op)
      : input(input), output(output), op(op), async_op(async_op) {}
  const std::vector<at::Tensor>& input;
  at::Tensor& output;
  const ReduceOp& op;
  bool async_op;
};

struct ReduceScatterSinglePreHookArgs {
  ReduceScatterSinglePreHookArgs(
      const at::Tensor& input,
      at::Tensor& output,
      const ReduceOp& op,
      bool async_op)
      : input(input), output(output), op(op), async_op(async_op) {}
  const at::Tensor& input;
  at::Tensor& output;
  const ReduceOp& op;
  bool async_op;
};

// All-to-all
struct AllToAllSinglePreHookArgs {
  AllToAllSinglePreHookArgs(
      const at::Tensor& input,
      at::Tensor& output,
      bool async_op)
      : input(input), output(output), async_op(async_op) {}
  const at::Tensor& input;
  at::Tensor& output;
  bool async_op;
};

struct AllToAllVSinglePreHookArgs {
  AllToAllVSinglePreHookArgs(
      const at::Tensor& input,
      at::Tensor& output,
      const std::vector<uint64_t>& input_split_sizes,
      const std::vector<uint64_t>& output_split_sizes,
      bool async_op)
      : input(input),
        output(output),
        input_split_sizes(input_split_sizes),
        output_split_sizes(output_split_sizes),
        async_op(async_op) {}
  const at::Tensor& input;
  at::Tensor& output;
  const std::vector<uint64_t>& input_split_sizes;
  const std::vector<uint64_t>& output_split_sizes;
  bool async_op;
};

struct AllToAllPreHookArgs {
  AllToAllPreHookArgs(
      const std::vector<at::Tensor>& input,
      const std::vector<at::Tensor>& output,
      bool async_op)
      : input(input), output(output), async_op(async_op) {}
  const std::vector<at::Tensor>& input;
  const std::vector<at::Tensor>& output;
  bool async_op;
};

// Barrier (no tensors)
struct BarrierPreHookArgs {
  explicit BarrierPreHookArgs(bool async_op) : async_op(async_op) {}
  bool async_op;
};

// Scatter/gather with root
struct ScatterPreHookArgs {
  ScatterPreHookArgs(
      at::Tensor& output,
      const std::vector<at::Tensor>& input,
      int root,
      bool async_op)
      : output(output), input(input), root(root), async_op(async_op) {}
  at::Tensor& output;
  const std::vector<at::Tensor>& input;
  int root;
  bool async_op;
};

struct GatherPreHookArgs {
  GatherPreHookArgs(
      const at::Tensor& input,
      const std::vector<at::Tensor>& output,
      int root,
      bool async_op)
      : input(input), output(output), root(root), async_op(async_op) {}
  const at::Tensor& input;
  const std::vector<at::Tensor>& output;
  int root;
  bool async_op;
};

struct GatherSinglePreHookArgs {
  GatherSinglePreHookArgs(
      const at::Tensor& input,
      at::Tensor& output,
      int root,
      bool async_op)
      : input(input), output(output), root(root), async_op(async_op) {}
  const at::Tensor& input;
  at::Tensor& output;
  int root;
  bool async_op;
};

// Batch operations
struct BatchOpIssuePreHookArgs {
  BatchOpIssuePreHookArgs(size_t num_ops, bool async_op)
      : num_ops(num_ops), async_op(async_op) {}
  size_t num_ops;
  bool async_op;
};

// Communicator management
struct SplitPreHookArgs {
  SplitPreHookArgs(const std::vector<int>& ranks, const std::string& name)
      : ranks(ranks), name(name) {}
  const std::vector<int>& ranks;
  const std::string& name;
};

struct NewWindowPreHookArgs {};

struct FinalizePreHookArgs {};

// Variant of all per-collective pre-hook argument types
using PreHookArgs = std::variant<
    SendPreHookArgs,
    RecvPreHookArgs,
    BroadcastPreHookArgs,
    AllReducePreHookArgs,
    ReducePreHookArgs,
    AllGatherPreHookArgs,
    AllGatherVPreHookArgs,
    AllGatherSinglePreHookArgs,
    ReduceScatterPreHookArgs,
    ReduceScatterVPreHookArgs,
    ReduceScatterSinglePreHookArgs,
    AllToAllSinglePreHookArgs,
    AllToAllVSinglePreHookArgs,
    AllToAllPreHookArgs,
    BarrierPreHookArgs,
    ScatterPreHookArgs,
    GatherPreHookArgs,
    GatherSinglePreHookArgs,
    SplitPreHookArgs,
    NewWindowPreHookArgs,
    BatchOpIssuePreHookArgs,
    FinalizePreHookArgs>;

using PreHook = std::function<void(size_t op_id, const PreHookArgs& args)>;

// -- Per-collective post-hook argument structs --
//
// Base struct for collectives that produce a work object.
// Split and new_window have their own result types.

struct CollectivePostHookArgs {
  explicit CollectivePostHookArgs(c10::weak_intrusive_ptr<TorchWork> work)
      : work(std::move(work)) {}
  c10::weak_intrusive_ptr<TorchWork> work;
};

struct SendPostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct RecvPostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct BroadcastPostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct AllReducePostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct ReducePostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct AllGatherPostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct AllGatherVPostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct AllGatherSinglePostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct ReduceScatterPostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct ReduceScatterVPostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct ReduceScatterSinglePostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct AllToAllSinglePostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct AllToAllVSinglePostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct AllToAllPostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct BarrierPostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct ScatterPostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct GatherPostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};
struct GatherSinglePostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};

struct SplitPostHookArgs {
  explicit SplitPostHookArgs(std::weak_ptr<TorchComm> new_comm)
      : new_comm(std::move(new_comm)) {}
  std::weak_ptr<TorchComm> new_comm;
};

struct NewWindowPostHookArgs {
  explicit NewWindowPostHookArgs(std::weak_ptr<TorchCommWindow> new_window)
      : new_window(std::move(new_window)) {}
  std::weak_ptr<TorchCommWindow> new_window;
};

struct BatchOpIssuePostHookArgs : CollectivePostHookArgs {
  using CollectivePostHookArgs::CollectivePostHookArgs;
};

struct FinalizePostHookArgs {};

using PostHookArgs = std::variant<
    SendPostHookArgs,
    RecvPostHookArgs,
    BroadcastPostHookArgs,
    AllReducePostHookArgs,
    ReducePostHookArgs,
    AllGatherPostHookArgs,
    AllGatherVPostHookArgs,
    AllGatherSinglePostHookArgs,
    ReduceScatterPostHookArgs,
    ReduceScatterVPostHookArgs,
    ReduceScatterSinglePostHookArgs,
    AllToAllSinglePostHookArgs,
    AllToAllVSinglePostHookArgs,
    AllToAllPostHookArgs,
    BarrierPostHookArgs,
    ScatterPostHookArgs,
    GatherPostHookArgs,
    GatherSinglePostHookArgs,
    SplitPostHookArgs,
    NewWindowPostHookArgs,
    BatchOpIssuePostHookArgs,
    FinalizePostHookArgs>;

using PostHook = std::function<void(size_t op_id, const PostHookArgs& args)>;

// Abort hook - called before aborting when a collective times out or fails.
// This allows users to capture debug information before the abort.
using AbortHook = std::function<void()>;

// Graph replay hook - called by backends that support CUDA graph capture
// when graph replays are detected. The hook receives the graph ID, replay
// number, an opaque stream handle, a zero-based per-stream collective index,
// and an event name ("S" or "E"). Backends call this from the watchdog thread.
using GraphReplayHook = std::function<void(
    uint64_t graph_id,
    uint64_t replay_id,
    void* stream,
    size_t collective_index,
    std::string_view event)>;

} // namespace torch::comms
