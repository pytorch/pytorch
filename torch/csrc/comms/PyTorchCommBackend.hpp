// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/comms/TorchCommBackend.hpp>
#include <torch/csrc/comms/TorchWork.hpp>

namespace py = pybind11;

namespace torch::comms {

class PyTorchWork : public TorchWork {
 public:
  explicit PyTorchWork(py::object py_work);
  ~PyTorchWork() override = default;
  void wait() override;
  void publicSetStatus(WorkStatus status) {
    setStatus(status);
  }

 private:
  py::object py_work_;
};

class PyTorchCommBackend : public TorchCommBackend {
 public:
  using TorchCommBackend::TorchCommBackend;

  void setPySelf(py::object self) {
    py_self_ = std::move(self);
  }

  static c10::intrusive_ptr<TorchWork> wrapPyWork(py::object py_result);

  void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options) override {
    device_ = device;
    options_ = options;
    PYBIND11_OVERRIDE_PURE(void, TorchCommBackend, init, device, name, options);
  }

  void finalize() override {
    PYBIND11_OVERRIDE_PURE(void, TorchCommBackend, finalize);
  }

  int getRank() const override {
    PYBIND11_OVERRIDE_PURE_NAME(int, TorchCommBackend, "get_rank", getRank);
  }

  int getSize() const override {
    PYBIND11_OVERRIDE_PURE_NAME(int, TorchCommBackend, "get_size", getSize);
  }

  std::string_view getBackendName() const override;
  std::string_view getCommName() const override;

  const CommOptions& getOptions() const override {
    return options_;
  }
  const at::Device& getDevice() const override {
    return device_;
  }

  c10::intrusive_ptr<TorchWork> send(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      const SendOptions& options) override;

  c10::intrusive_ptr<TorchWork> recv(
      at::Tensor& tensor,
      int src,
      bool async_op,
      const RecvOptions& options) override;

  c10::intrusive_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      const BatchP2POptions& options) override;

  c10::intrusive_ptr<TorchWork> broadcast(
      at::Tensor& tensor,
      int root,
      bool async_op,
      const BroadcastOptions& options) override;

  c10::intrusive_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      const ReduceOp& op,
      bool async_op,
      const AllReduceOptions& options) override;

  c10::intrusive_ptr<TorchWork> reduce(
      const at::Tensor& tensor,
      int root,
      const ReduceOp& op,
      bool async_op,
      const ReduceOptions& options) override;

  c10::intrusive_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options) override;

  c10::intrusive_ptr<TorchWork> all_gather_v(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options) override;

  c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllGatherSingleOptions& options) override;

  c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options) override;

  c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options) override;

  c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor& output,
      const at::Tensor& input,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterSingleOptions& options) override;

  c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllToAllSingleOptions& options) override;

  c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      const AllToAllvSingleOptions& options) override;

  c10::intrusive_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      const AllToAllOptions& options) override;

  c10::intrusive_ptr<TorchWork> barrier(
      bool async_op,
      const BarrierOptions& options) override;

  c10::intrusive_ptr<TorchWork> scatter(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      const ScatterOptions& options) override;

  c10::intrusive_ptr<TorchWork> gather(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      const GatherOptions& options) override;

  std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options) override;

 private:
  py::object py_self_;
  mutable std::string backend_name_;
  mutable std::string comm_name_;
  CommOptions options_;
  at::Device device_{at::kCPU};
};

void initPyBackendBindings(py::module_& m);

} // namespace torch::comms
