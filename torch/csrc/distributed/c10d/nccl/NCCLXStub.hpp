#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {

// Minimal C++ Backend for testing, modeled after ncclx
// (fbcode/comms/ncclx/pg/).  Wraps a real ProcessGroupNCCL so that
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
      c10::intrusive_ptr<ProcessGroupNCCL::Options> options =
          ProcessGroupNCCL::Options::create())
      : Backend(rank, size) {
    nccl_ = c10::make_intrusive<ProcessGroupNCCL>(
        store, rank, size, std::move(options));
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

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    return nccl_->_allgather_base(outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    return nccl_->barrier(opts);
  }

 private:
  c10::intrusive_ptr<ProcessGroupNCCL> nccl_;
};

} // namespace c10d
