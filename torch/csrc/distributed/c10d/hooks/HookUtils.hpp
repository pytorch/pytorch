// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Helpers shared by the hooks in this directory.

#pragma once

#include <optional>
#include <string_view>

#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/csrc/distributed/c10d/Hooks.hpp>

namespace c10d::detail {

// These must be spelled the way the trace analyzer spells them, i.e. match
// COLLECTIVES / P2P in torch/distributed/flight_recorder/components/types.py
// and the profiling titles the native backends use ("nccl:all_reduce",
// "gloo:all_gather", ...). Anything else makes Op() reject the entry.
inline std::string_view hookOpName(HookOpName name) {
  switch (name) {
    case HookOpName::SEND:
      return "send";
    case HookOpName::RECV:
      return "recv";
    case HookOpName::BROADCAST:
      return "broadcast";
    case HookOpName::ALLREDUCE:
      return "all_reduce";
    case HookOpName::ALLREDUCE_COALESCED:
      return "allreduce_coalesced";
    case HookOpName::REDUCE:
      return "reduce";
    case HookOpName::ALLGATHER:
      return "all_gather";
    case HookOpName::ALLGATHER_BASE:
      return "_all_gather_base";
    case HookOpName::ALLGATHER_COALESCED:
      return "allgather_coalesced";
    case HookOpName::ALLGATHER_INTO_TENSOR_COALESCED:
      return "all_gather_into_tensor_coalesced";
    case HookOpName::REDUCE_SCATTER:
      return "reduce_scatter";
    case HookOpName::REDUCE_SCATTER_BASE:
      return "_reduce_scatter_base";
    case HookOpName::REDUCE_SCATTER_TENSOR_COALESCED:
      return "reduce_scatter_tensor_coalesced";
    case HookOpName::ALLTOALL:
      return "all_to_all";
    case HookOpName::ALLTOALL_BASE:
      return "all_to_all_single";
    case HookOpName::BARRIER:
      return "barrier";
    case HookOpName::SCATTER:
      return "scatter";
    case HookOpName::GATHER:
      return "gather";
    case HookOpName::SPLIT:
      return "split";
    case HookOpName::NEW_WINDOW:
      return "new_window";
    case HookOpName::UNKNOWN:
      break;
  }
  return "unknown";
}

// PreHookArgs carries no device, so take it from the op's tensors.
inline std::optional<c10::Device> opDevice(const PreHookArgs& args) {
  if (!args.input_tensors.empty() && args.input_tensors.front().defined()) {
    return args.input_tensors.front().device();
  }
  if (!args.output_tensors.empty() && args.output_tensors.front().defined()) {
    return args.output_tensors.front().device();
  }
  return std::nullopt;
}

// Whether a CUDA graph capture (or the equivalent on another accelerator) is
// active on the stream an op on this device would be issued on. Asked through
// the device guard impl so the hook stays device agnostic.
//
// Both ways of answering false here are positive facts, not guesses: a device
// type with no guard impl registered has no capture mechanism at all, and a
// device whose current stream cannot even be obtained -- CUDA linked in but no
// visible GPU, say -- has no stream for a capture to be running on. "I do not
// know which device to ask" is a different answer and is not expressed here;
// see captureActive.
inline bool streamIsCapturing(c10::Device device) {
  if (device.is_cpu() || !c10::impl::hasDeviceGuardImpl(device.type())) {
    return false;
  }
  try {
    auto* guard_impl = c10::impl::getDeviceGuardImpl(device.type());
    return guard_impl->getStream(device).is_capturing();
  } catch (const std::exception&) {
    return false;
  }
}

} // namespace c10d::detail
