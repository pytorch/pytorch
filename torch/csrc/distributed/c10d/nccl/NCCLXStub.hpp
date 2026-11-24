#pragma once

#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

namespace c10d {

// Minimal C++ Backend for testing, modeled after ncclx
// (fbcode/comms/ncclx/pg/).  Wraps a real nccl2 ProcessGroupNCCL so that
// collectives (including the allgather used by symmetric-memory PG
// rendezvous) actually work.  Registers as a custom backend plugin via
// Backend.register_backend() with extended_api=True, which means
// _new_process_group_helper may leave the wrapper ProcessGroup's
// backendType_ as UNDEFINED when used in a multi-backend config like
// "cpu:gloo,cuda:ncclx_stub".
class NCCLXStub : public Backend {
 public:
  NCCLXStub(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<NCCLOptions> options = NCCLOptions::create())
      : Backend(rank, size) {
    nccl_ = c10::make_intrusive<nccl2::ProcessGroupNCCL>(
        store, rank, size, std::move(options));
    // nccl2 creates its communicator when bound to a device; c10d does not
    // bind extended_api plugin backends, so bind the current device.
    nccl_->setBoundDeviceId(at::Device(at::kCUDA, c10::cuda::current_device()));
  }

  const std::string getBackendName() const override {
    return "ncclx_stub";
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    return nccl_->broadcast(tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    return nccl_->allreduce(tensors, opts);
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    return nccl_->allgather(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> all_gather_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    return nccl_->all_gather_single(outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    return nccl_->barrier(opts);
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    nccl_->setTimeout(timeout);
  }

 private:
  c10::intrusive_ptr<nccl2::ProcessGroupNCCL> nccl_;
};

} // namespace c10d
