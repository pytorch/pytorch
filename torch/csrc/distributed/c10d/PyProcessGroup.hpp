#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace c10d {

// PyProcessGroup is a pybind11 trampoline class to allow a Python
// class to inherit from torch.distributed.ProcessGroup
class PyProcessGroup : public ProcessGroup {
 public:
  // PyWork is a pybind11 trampoline class to allow a Python
  // class to inherit from torch.distributed.Work
  class TORCH_PYTHON_API PyWork : public Work {
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
      auto override =
          pybind11::get_override(static_cast<const Work*>(this), "get_future");

      if (override) {
        py::object o = override();
        auto futWrapper =
            o.cast<std::shared_ptr<torch::jit::PythonFutureWrapper>>();
        return futWrapper->fut;
      }

      return Work::getFuture();
    }

    // Take a reference of the corresponding py::object.
    // With functional collectives, ownership of work objects is generally
    // transferred to C++. For pure C++ work objects, it is sufficient to
    // transfer the ownership of work object. For user-defined work objects in
    // Python, it is necessary to keep the corresponding py::object alive in
    // addition to ensure that the user-defined methods can be executed.
    void ref_py_object() {
      py_obj_ = py::cast(this);
    }

   private:
    py::object py_obj_;
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

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        allgather_into_tensor_coalesced, /* Name of function in C++ */
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

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        allreduce_coalesced, /* Name of function in C++ */
        tensors,
        opts);
  }

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        alltoall_base, /* Name of function in C++ */
        outputBuffer,
        inputBuffer,
        outputSplitSizes,
        inputSplitSizes,
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

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        reduce_scatter_tensor_coalesced, /* Name of function in C++ */
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

class TORCH_PYTHON_API PythonOnCompletionHook {
 public:
  // Wraps a py::object hook and acquires Python GIL in dtor before
  // destructing the hook object.
  PythonOnCompletionHook(py::object hook) : hook_(std::move(hook)) {}

  ~PythonOnCompletionHook() {
    py::gil_scoped_acquire ag;
    hook_.dec_ref();
    // Explicitly set hook_ to nullptr to prevent py::object's dtor
    // to decref on the PyObject again.
    // See Note [Destructing py::object] in python_ivalue.h
    hook_.ptr() = nullptr;
  }

  void operator()(const std::shared_ptr<WorkInfo>& workInfo) const {
    std::exception_ptr eptr;
    {
      py::gil_scoped_acquire acquire;
      try {
        hook_(workInfo);
      } catch (py::error_already_set& e) {
        // py::error_already_set requires GIL to destruct, take
        // special care.
        eptr = std::make_exception_ptr(std::runtime_error(e.what()));
        e.restore();
        PyErr_Clear();
      } catch (std::exception& e) {
        eptr = std::current_exception();
      }
    }
    // No more Python-related stuff at this point, i.e., this
    // exception can be captured and handled by PG backend.
    if (eptr)
      std::rethrow_exception(eptr);
  }

 private:
  py::object hook_;
};

} // namespace c10d
