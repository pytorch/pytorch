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
  };

#define WORK_OVERRIDE(cname, name, ...)                                 \
  do {                                                                  \
    pybind11::gil_scoped_acquire gil;                                   \
    pybind11::function override =                                       \
        pybind11::get_override(static_cast<const cname*>(this), #name); \
    if (override) {                                                     \
      auto o = override(__VA_ARGS__);                                   \
      return c10::make_intrusive<PyWorkHolder>(o);                      \
    }                                                                   \
    return cname::name(__VA_ARGS__);                                    \
  } while (false)

  // This class is used to wrap a PyWork trampoline with it's corresponding
  // Python object to prevent the Python object from being garbage collected.
  class PyWorkHolder : public Work {
   public:
    PyWorkHolder(const c10::intrusive_ptr<Work>& work, py::object pyWork)
        : work_(work), pyWork_(std::move(pyWork)) {}

    PyWorkHolder(py::object pyWork)
        : work_(pyWork.cast<c10::intrusive_ptr<Work>>()),
          pyWork_(std::move(pyWork)) {}

    ~PyWorkHolder() override {
      // GIL must be held when freeing python objects.
      py::gil_scoped_acquire gil;
      pyWork_ = py::object();
    }

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
      return work_->wait(timeout);
    }

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
      return work_->getFuture();
    }

   private:
    c10::intrusive_ptr<Work> work_;
    py::object pyWork_;
  };

  using ProcessGroup::ProcessGroup;

  const std::string getBackendName() const override {
    PYBIND11_OVERRIDE(
        std::string, /* Return type */
        ProcessGroup, /* Parent class */
        getBackendName, /* Name of function in C++ */
    );
  }

  int getRank() const override {
    PYBIND11_OVERRIDE(
        int, /* Return type */
        ProcessGroup, /* Parent class */
        getRank, /* Name of function in C++ */
    );
  }

  int getSize() const override {
    PYBIND11_OVERRIDE(
        int, /* Return type */
        ProcessGroup, /* Parent class */
        getSize, /* Name of function in C++ */
    );
  }

  void abort() override {
    PYBIND11_OVERRIDE(
        void, /* Return type */
        ProcessGroup, /* Parent class */
        abort, /* Name of function in C++ */
    );
  }

  const std::string& getGroupName() const override {
    PYBIND11_OVERRIDE(
        const std::string&, /* Return type */
        ProcessGroup, /* Parent class */
        getGroupName, /* Name of function in C++ */
    );
  }

  void setGroupName(const std::string& group_name) override {
    PYBIND11_OVERRIDE(
        void, /* Return type */
        ProcessGroup, /* Parent class */
        setGroupName, /* Name of function in C++ */
        group_name);
  }

  const std::string& getGroupDesc() const override {
    PYBIND11_OVERRIDE(
        const std::string&, /* Return type */
        ProcessGroup, /* Parent class */
        getGroupDesc, /* Name of function in C++ */
    );
  }

  void setGroupDesc(const std::string& group_desc) override {
    PYBIND11_OVERRIDE(
        void, /* Return type */
        ProcessGroup, /* Parent class */
        setGroupDesc, /* Name of function in C++ */
        group_desc);
  }

  c10::intrusive_ptr<ProcessGroup> splitGroup(
      const std::vector<int>& ranks,
      const std::optional<std::chrono::milliseconds>& timeout,
      const std::optional<c10::intrusive_ptr<Backend::Options>>& opts,
      const std::optional<std::string>& group_name,
      const std::optional<std::string>& group_desc) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<ProcessGroup>, /* Return type */
        ProcessGroup, /* Parent class */
        splitGroup, /* Name of function in C++ */
        ranks,
        timeout,
        opts,
        group_name,
        group_desc);
  }

  c10::intrusive_ptr<ProcessGroup> mergeRemoteGroup(
      const c10::intrusive_ptr<c10d::Store>& store,
      const MergeOptions& opts,
      const int& size) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<ProcessGroup>, /* Return type */
        ProcessGroup, /* Parent class */
        mergeRemoteGroup, /* Name of function in C++ */
        store,
        opts,
        size);
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    WORK_OVERRIDE(
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
    WORK_OVERRIDE(
        ProcessGroup, /* Parent class */
        allgather_into_tensor_coalesced, /* Name of function in C++ */
        outputTensors,
        inputTensors,
        opts);
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    WORK_OVERRIDE(
        // py::object, /* Return type */
        ProcessGroup, /* Parent class */
        allreduce, /* Name of function in C++ */
        tensors,
        opts);
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override {
    WORK_OVERRIDE(
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
    WORK_OVERRIDE(
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
    WORK_OVERRIDE(
        ProcessGroup, /* Parent class */
        barrier, /* Name of function in C++ */
        opts);
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    WORK_OVERRIDE(
        ProcessGroup, /* Parent class */
        broadcast, /* Name of function in C++ */
        tensors,
        opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    WORK_OVERRIDE(
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
    WORK_OVERRIDE(
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
    WORK_OVERRIDE(
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
    WORK_OVERRIDE(
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
  PythonOnCompletionHook(const PythonOnCompletionHook&) = default;

  // NOLINTNEXTLINE(bugprone-exception-escape)
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
