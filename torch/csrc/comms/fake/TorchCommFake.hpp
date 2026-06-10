// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <torch/csrc/comms/TorchCommBackend.hpp>
#include <torch/csrc/comms/TorchWork.hpp>
#include <unordered_set>
#include <vector>

namespace torch::comms {

class TorchCommFake : public TorchCommBackend {
 public:
  static constexpr std::string_view kBackendName = "fake";

  TorchCommFake();
  ~TorchCommFake() override = default;

  // Initialize the communication backend
  void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) override;
  void finalize() override;
  int getRank() const override;
  int getSize() const override;
  std::string_view getCommName() const override;
  std::string_view getBackendName() const override;

  // Point-to-Point Operations
  c10::intrusive_ptr<TorchWork> send(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      const SendOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> recv(
      at::Tensor& tensor,
      int src,
      bool async_op,
      const RecvOptions& options = {}) override;

  c10::intrusive_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      const BatchP2POptions& options = {}) override;

  // Collective Operations
  c10::intrusive_ptr<TorchWork> broadcast(
      at::Tensor& tensor,
      int root,
      bool async_op,
      const BroadcastOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      const ReduceOp& op,
      bool async_op,
      const AllReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce(
      const at::Tensor& tensor,
      int root,
      const ReduceOp& op,
      bool async_op,
      const ReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_v(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllGatherSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor& output,
      const at::Tensor& input,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllToAllSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      const AllToAllvSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      const AllToAllOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> barrier(
      bool async_op,
      const BarrierOptions& options = {}) override;

  // Scatter and Gather Operations
  c10::intrusive_ptr<TorchWork> scatter(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      const ScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> gather(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      const GatherOptions& options = {}) override;

  // Window & One-sided Operations
  std::shared_ptr<TorchCommWindow> new_window(
      const std::optional<at::Tensor>& tensor = std::nullopt) override;

  // Fault Tolerance
  bool supportsReconfigure() const override {
    return true;
  }
  c10::intrusive_ptr<TorchWork> reconfigure(
      const ReconfigureOptions& opts) override;

  bool isInitialized() const override {
    return initialized_;
  }

  // Test helpers
  void setSize(int size) {
    size_ = size;
  }

  void setReconfigureFailure(bool fail) {
    shouldFailReconfigure_ = fail;
  }

  // Communicator Management
  std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) override;

  const CommOptions& getOptions() const override;
  const at::Device& getDevice() const override;

  // Test helper to trigger abort hooks (for testing purposes only).
  // Intentionally kept separate from abort() — fires hooks without
  // changing communicator state, used by TorchCommHooksTest.
  void triggerAbort() {
    runAbortHooks();
  }

  // Test helper to enable abort simulation (one-way, mirrors MCCL's
  // immutable Abort::enabled_)
  void enableAbort() {
    abortEnabled_ = true;
  }

  // Full abort path: sets aborted state + fires hooks.
  void abort() override {
    if (!abortEnabled_) {
      return;
    }
    aborted_ = true;
    runAbortHooks();
  }

  bool isAbortSupported() const override {
    return abortEnabled_;
  }

  // Returns false when abortEnabled_ is false, matching the
  // base class no-op contract.
  bool isAborted() const override {
    return abortEnabled_ && aborted_;
  }

  // Memory registration — tracks registered addresses for test verification
  void tensor_register(const at::Tensor& tensor) override {
    registered_addrs_.insert(tensor.data_ptr());
  }

  void tensor_deregister(const at::Tensor& tensor) override {
    registered_addrs_.erase(tensor.data_ptr());
  }

  bool is_tensor_registered(const at::Tensor& tensor) const {
    return registered_addrs_.count(tensor.data_ptr()) > 0;
  }

 private:
  bool initialized_;
  at::Device device_;
  CommOptions options_;
  int rank_;
  int size_;
  std::string name_;
  bool abortEnabled_{false};
  bool aborted_{false};
  bool shouldFailReconfigure_{false};
  std::unordered_set<void*> registered_addrs_;
};

} // namespace torch::comms
