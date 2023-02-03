#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace c10d {

// PyProcessGroup is a pybind11 trampoline class to allow a Python
// class to inherit from torch.distributed.ProcessGroup
class PyProcessGroup : public ProcessGroup {
 public:
  // PyWork is a pybind11 trampoline class to allow a Python
  // class to inherit from torch.distributed.Work
  class PyWork : public Work {
   public:
    PyWork() = default;

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
      PYBIND11_OVERRIDE(
          bool, /* Return type */
          Work, /* Parent class */
          wait, /* Name of function in C++ */
          timeout);
    }

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
        // We cannot use PYBIND11_OVERRIDE because:
        // 1. We have to >MANUALLY< unwrap the PyFutureWrapper and
        // 2. The python name is get_future
        pybind11::gil_scoped_acquire gil;
        auto override = pybind11::get_override(static_cast<const Work *>(this), "get_future");

        if (override) {
            py::object o = override();
            auto futWrapper = o.cast<std::shared_ptr<torch::jit::PythonFutureWrapper>>();
            return futWrapper->fut;
        }

        return Work::getFuture();
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

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        allgather, /* Name of function in C++ */
        outputTensors,
        inputTensors,
        opts);
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        allreduce, /* Name of function in C++ */
        tensors,
        opts);
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        barrier, /* Name of function in C++ */
        opts);
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        broadcast, /* Name of function in C++ */
        tensors,
        opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        reduce_scatter, /* Name of function in C++ */
        outputTensors,
        inputTensors,
        opts);
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        send, /* Name of function in C++ */
        tensors,
        dstRank,
        tag);
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        recv, /* Name of function in C++ */
        tensors,
        srcRank,
        tag);
  }
};

} // namespace c10d
