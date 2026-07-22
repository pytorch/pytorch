// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// FlightRecorderHook: backend-agnostic FlightRecorder integration built on the
// ProcessGroup pre/post collective hooks (Hooks.hpp). Port of torchcomms'
// hooks/fr FlightRecorderHook onto c10d.
//
// The pre-hook records an entry into the generic FlightRecorder<c10::Event>
// ring buffer (null start/end events, but full op/tensor/sequencing metadata).
// The post-hook associates the entry with the returned Work so it can be
// retired when the operation completes. Because the hooks fire from the
// dispatcher kernels in Ops.cpp, this works for any backend routed through
// c10d ops.

#pragma once

#include <memory>

#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace c10d {

class TORCH_API FlightRecorderHook {
 public:
  // Returns a handle for the process group's automatically installed hook, or
  // attaches one when automatic recording is disabled.
  static std::shared_ptr<FlightRecorderHook> attach(
      c10::intrusive_ptr<ProcessGroup> pg);

  static bool isEnabled();

  static std::string getFlightRecorderTraceback(
      const c10::intrusive_ptr<c10::ivalue::Future>& future);

  ~FlightRecorderHook();

  FlightRecorderHook(const FlightRecorderHook&) = delete;
  FlightRecorderHook(FlightRecorderHook&&) = delete;
  FlightRecorderHook& operator=(const FlightRecorderHook&) = delete;
  FlightRecorderHook& operator=(FlightRecorderHook&&) = delete;

  // Detach from the process group. Idempotent.
  void remove();

 private:
  friend class ProcessGroup;

  explicit FlightRecorderHook(c10::intrusive_ptr<ProcessGroup> pg);

  static void install(ProcessGroup* pg);
  static bool isInstalled(ProcessGroup* pg);

  c10::intrusive_ptr<ProcessGroup> pg_;
};

} // namespace c10d
