#pragma once

#ifdef USE_C10D_GLOO

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/gloo/ProcessGroupGlooAsyncWork.hpp>

namespace c10d {

class TORCH_API AsyncBroadcastWork : public ProcessGroupGlooAsyncWork {
 public:
  AsyncBroadcastWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout);
  ~AsyncBroadcastWork() override = default;

  const std::vector<at::Tensor> getInputTensors() override;
  const std::vector<at::Tensor> getOutputTensors() override;
  void run() override;

 protected:
  void broadcast(at::Tensor tensor);
  std::vector<at::Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const uint32_t tag;
};

class TORCH_API AsyncAllreduceWork : public ProcessGroupGlooAsyncWork {
 public:
  AsyncAllreduceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout);
  ~AsyncAllreduceWork() override = default;

  const std::vector<at::Tensor> getInputTensors() override;
  const std::vector<at::Tensor> getOutputTensors() override;
  void run() override;

 protected:
  AsyncAllreduceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      OpType opType,
      std::chrono::milliseconds timeout);
  void allreduce(std::vector<at::Tensor>& tensors);

  std::vector<at::Tensor> inputs;
  const ReduceOp reduceOp;
  const uint32_t tag;
};

class TORCH_API AsyncSparseAllreduceWork : public ProcessGroupGlooAsyncWork {
 public:
  AsyncSparseAllreduceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout);
  ~AsyncSparseAllreduceWork() override = default;

  const std::vector<at::Tensor> getInputTensors() override;
  const std::vector<at::Tensor> getOutputTensors() override;
  void run() override;

 protected:
  at::Tensor allreduce(std::vector<at::Tensor>& tensors);
  std::vector<at::Tensor> inputs;
  const uint32_t tag;

 private:
  class SparseTensorMetadata;
  std::vector<SparseTensorMetadata> allgatherMetadata(const at::Tensor& tensor);
  std::vector<at::Tensor> allgatherIndices(
      const at::Tensor& tensor,
      const std::vector<SparseTensorMetadata>& metadata);
  std::vector<at::Tensor> allgatherValues(
      const at::Tensor& tensor,
      const std::vector<SparseTensorMetadata>& metadata);
};

class TORCH_API AsyncAllreduceCoalescedWork : public AsyncAllreduceWork {
 public:
  AsyncAllreduceCoalescedWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout);
  ~AsyncAllreduceCoalescedWork() override = default;
  void run() override;

 private:
  void allreduceCoalesced(std::vector<at::Tensor>& tensors);
};

class TORCH_API AsyncReduceWork : public ProcessGroupGlooAsyncWork {
 public:
  AsyncReduceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout);
  ~AsyncReduceWork() override = default;

  const std::vector<at::Tensor> getInputTensors() override;
  const std::vector<at::Tensor> getOutputTensors() override;
  void run() override;

 protected:
  void reduce(std::vector<at::Tensor>& tensors);

  std::vector<at::Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const ReduceOp reduceOp;
  const uint32_t tag;
};

class TORCH_API AsyncAllgatherWork : public ProcessGroupGlooAsyncWork {
 public:
  AsyncAllgatherWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout);
  ~AsyncAllgatherWork() override = default;

  const std::vector<at::Tensor> getInputTensors() override;
  const std::vector<at::Tensor> getOutputTensors() override;
  void run() override;

 protected:
  void allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs);
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<at::Tensor> inputs;
  const uint32_t tag;
};

class TORCH_API AsyncAllgatherCoalescedWork : public ProcessGroupGlooAsyncWork {
 public:
  AsyncAllgatherCoalescedWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout);
  ~AsyncAllgatherCoalescedWork() override = default;

  const std::vector<at::Tensor> getInputTensors() override;
  const std::vector<at::Tensor> getOutputTensors() override;
  void run() override;

 protected:
  void allgatherCoalesced();
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<at::Tensor> inputs;
  const uint32_t tag;
};

class TORCH_API AsyncGatherWork : public ProcessGroupGlooAsyncWork {
 public:
  AsyncGatherWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      int root,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout);
  ~AsyncGatherWork() override = default;

  const std::vector<at::Tensor> getInputTensors() override;
  const std::vector<at::Tensor> getOutputTensors() override;
  void run() override;

 protected:
  void gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs);
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<at::Tensor> inputs;
  const int root;
  const uint32_t tag;
};

class TORCH_API AsyncScatterWork : public ProcessGroupGlooAsyncWork {
 public:
  AsyncScatterWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      int root,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout);
  ~AsyncScatterWork() override = default;

  const std::vector<at::Tensor> getInputTensors() override;
  const std::vector<at::Tensor> getOutputTensors() override;
  void run() override;

 protected:
  void scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs);
  std::vector<at::Tensor> outputs;
  std::vector<std::vector<at::Tensor>> inputs;
  const int root;
  const uint32_t tag;
};

class TORCH_API AsyncAlltoallWork : public ProcessGroupGlooAsyncWork {
 public:
  AsyncAlltoallWork(
      std::shared_ptr<gloo::Context> context,
      at::Tensor& output,
      at::Tensor& input,
      std::vector<int64_t>& outputCounts,
      std::vector<int64_t>& inputCounts,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout);
  ~AsyncAlltoallWork() override = default;

  const std::vector<at::Tensor> getInputTensors() override;
  const std::vector<at::Tensor> getOutputTensors() override;
  void run() override;

 protected:
  void alltoall(at::Tensor& output, at::Tensor& input);
  at::Tensor output;
  at::Tensor input;
  std::vector<int64_t> outputCounts;
  std::vector<int64_t> inputCounts;
  const uint32_t tag;
};

class TORCH_API AsyncAlltoallListWork : public ProcessGroupGlooAsyncWork {
 public:
  AsyncAlltoallListWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout);
  ~AsyncAlltoallListWork() override = default;

  const std::vector<at::Tensor> getInputTensors() override;
  const std::vector<at::Tensor> getOutputTensors() override;
  void run() override;

 protected:
  void alltoall(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs);
  std::vector<at::Tensor> outputs;
  std::vector<at::Tensor> inputs;
  const uint32_t tag;
};

} // namespace c10d

#endif // USE_C10D_GLOO
