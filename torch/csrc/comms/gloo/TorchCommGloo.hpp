// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <memory>
#include <string_view>
#include <vector>

#include <ATen/ATen.h>
#include <gloo/context.h>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp

#include <torch/csrc/comms/TorchComm.hpp>
#include <torch/csrc/comms/TorchCommBackend.hpp>
#include <torch/csrc/comms/TorchCommBatch.hpp>
#include <torch/csrc/comms/gloo/TorchWorkGloo.hpp>

namespace torch::comms {

constexpr bool kDefaultPersistentStore = false;

class TorchCommGloo : public TorchCommBackend,
                      public std::enable_shared_from_this<TorchCommGloo> {
 public:
  static constexpr std::string_view kBackendName = "gloo";

  TorchCommGloo();
  ~TorchCommGloo() override;

  // Delete copy and move operations
  TorchCommGloo(const TorchCommGloo&) = delete;
  TorchCommGloo(TorchCommGloo&&) = delete;
  TorchCommGloo& operator=(const TorchCommGloo&) = delete;
  TorchCommGloo& operator=(TorchCommGloo&&) = delete;

  void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) override;
  void finalize() override;
  int getRank() const override;
  int getSize() const override;
  std::string_view getBackendName() const override;
  std::string_view getCommName() const override;

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

  // Batch P2P Operations
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

  // Communicator Management
  std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) override;

  // Fault Tolerance API
  bool supportsReconfigure() const override {
    return true;
  }
  InitHandle getInitHandle() const override;
  c10::intrusive_ptr<TorchWork> reconfigure(
      const ReconfigureOptions& opts) override;

  // Friend access for TorchWorkGloo
  friend class TorchWorkGloo;

  const CommOptions& getOptions() const override {
    return options_;
  }

  const at::Device& getDevice() const override {
    return device_;
  }

 protected:
  // Event management for friend classes
  enum class CommState {
    NORMAL,
    ERROR,
    TIMEOUT,
  };

  std::atomic<CommState> comm_state_{
      CommState::NORMAL}; // State of the communicator

 private:
  c10::intrusive_ptr<TorchWork> createWork(
      std::function<void()> fn,
      bool async_op);

  void checkInitialized();
  void checkAndAbortIfTimedOutOrError();
  void connectGlooContext(
      const CommOptions& options,
      const std::string& storePrefix = {});

  uint32_t nextTag() {
    return collectiveCounter_++;
  }

 private:
  // Member variables
  at::Device device_;
  int comm_size_{-1};
  int rank_{-1};
  CommOptions options_;
  enum class InitializationState {
    UNINITIALIZED,
    INITIALIZED,
    FINALIZED,
  } init_state_{InitializationState::UNINITIALIZED};

  std::string name_;

  c10::intrusive_ptr<c10d::Store> store_;
  c10::intrusive_ptr<c10d::Store> reconfigure_store_;
  int64_t uuid_{-1};
  std::shared_ptr<gloo::Context> context_;

  uint32_t collectiveCounter_{0};
  uint32_t splitCounter_{0};
};

// Registers the Gloo backend with the global TorchCommFactory.
void register_gloo_backend();

} // namespace torch::comms
