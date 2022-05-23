#pragma once

#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/utils/pybind.h>

namespace c10d {

// PyProcessGroup is a pybind11 trampoline class to allow a Python
// class to inherit from torch.distributed.ProcessGroup
class PyProcessGroup : public ProcessGroup {
 public:
  // PyWork is a pybind11 trampoline class to allow a Python
  // class to inherit from torch.distributed.Work
  class PyWork : public ProcessGroup::Work {
   public:
    PyWork() = default;

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
      PYBIND11_OVERRIDE(
          bool, /* Return type */
          ProcessGroup::Work, /* Parent class */
          wait, /* Name of function in C++ */
          timeout);
    }
  };

  using ProcessGroup::ProcessGroup;

  const std::string getBackendName() const override {
    PYBIND11_OVERRIDE_PURE(
        std::string, /* Return type */
        ProcessGroup, /* Parent class */
        getBackendName, /* Name of function in C++ */
    );
  }

  c10::intrusive_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<ProcessGroup::Work>, /* Return type */
        ProcessGroup, /* Parent class */
        allgather, /* Name of function in C++ */
        outputTensors,
        inputTensors,
        opts);
  }

  c10::intrusive_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<ProcessGroup::Work>, /* Return type */
        ProcessGroup, /* Parent class */
        allreduce, /* Name of function in C++ */
        tensors,
        opts);
  }

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<ProcessGroup::Work>, /* Return type */
        ProcessGroup, /* Parent class */
        barrier, /* Name of function in C++ */
        opts);
  }

  c10::intrusive_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<ProcessGroup::Work>, /* Return type */
        ProcessGroup, /* Parent class */
        broadcast, /* Name of function in C++ */
        tensors,
        opts);
  }

  c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<ProcessGroup::Work>, /* Return type */
        ProcessGroup, /* Parent class */
        reduce_scatter, /* Name of function in C++ */
        outputTensors,
        inputTensors,
        opts);
  }

  c10::intrusive_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<ProcessGroup::Work>, /* Return type */
        ProcessGroup, /* Parent class */
        send, /* Name of function in C++ */
        tensors,
        dstRank,
        tag);
  }

  c10::intrusive_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<ProcessGroup::Work>, /* Return type */
        ProcessGroup, /* Parent class */
        recv, /* Name of function in C++ */
        tensors,
        srcRank,
        tag);
  }
};

} // namespace c10d
