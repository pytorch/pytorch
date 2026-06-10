// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/comms/TorchCommTypes.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <chrono>
#include <string>
#include <string_view>
#include <unordered_map>

namespace torch::comms {

// Options classes for collective operations
class SendOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  SendOptions() : timeout(kNoTimeout) {}
};

class RecvOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  RecvOptions() : timeout(kNoTimeout) {}
};

class BatchP2POptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  BatchP2POptions() : timeout(kNoTimeout) {}
};

class BroadcastOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  BroadcastOptions() : timeout(kNoTimeout) {}
};

class AllReduceOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllReduceOptions() : timeout(kNoTimeout) {}
};

class ReduceOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  ReduceOptions() : timeout(kNoTimeout) {}
};

class AllGatherOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllGatherOptions() : timeout(kNoTimeout) {}
};

class AllGatherSingleOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllGatherSingleOptions() : timeout(kNoTimeout) {}
};

class ReduceScatterOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  ReduceScatterOptions() : timeout(kNoTimeout) {}
};

class ReduceScatterSingleOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  ReduceScatterSingleOptions() : timeout(kNoTimeout) {}
};

class AllToAllOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllToAllOptions() : timeout(kNoTimeout) {}
};

class AllToAllSingleOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllToAllSingleOptions() : timeout(kNoTimeout) {}
};

class AllToAllvSingleOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllToAllvSingleOptions() : timeout(kNoTimeout) {}
};

class BarrierOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  BarrierOptions() : timeout(kNoTimeout) {}
};

class ScatterOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  ScatterOptions() : timeout(kNoTimeout) {}
};

class GatherOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  GatherOptions() : timeout(kNoTimeout) {}
};

class GatherSingleOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  GatherSingleOptions() : timeout(kNoTimeout) {}
};

class CommOptions {
 public:
  bool abort_process_on_timeout_or_error{true};
  std::chrono::milliseconds timeout{kDefaultTimeout};
  bool high_priority_stream{false};
  c10::intrusive_ptr<c10d::Store> store{nullptr};
  /**
   * If true, enables reconfigure() for fault tolerance.
   * With reconfigure enabled, the communicator is not initialized until
   * reconfigure() is called. Default is false.
   */
  bool enable_reconfigure{false};
  std::unordered_map<std::string, std::string> hints;

 public:
  CommOptions();

  bool operator==(const CommOptions& other) const;

  // Look up a hint by key and convert to the requested type.
  // Returns default_value if the key is not present.
  template <typename T>
  T getHint(std::string_view key, const T& default_value) const;
};

class PutOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  PutOptions() : timeout(kNoTimeout) {}
};

class SignalOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  SignalOptions() : timeout(kNoTimeout) {}
};

class WaitSignalOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  WaitSignalOptions() : timeout(kNoTimeout) {}
};

class AllGatherPInitOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllGatherPInitOptions() : timeout(kNoTimeout) {}
};

class AllGatherPExecOptions {
 public:
  std::unordered_map<std::string, std::string> hints;
  std::chrono::milliseconds timeout;

  AllGatherPExecOptions() : timeout(kNoTimeout) {}
};

} // namespace torch::comms
