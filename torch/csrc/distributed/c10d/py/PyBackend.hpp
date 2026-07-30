#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/py/PyProcessGroup.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pyobject_preservation.h>

namespace c10d {

// PyBackend is a pybind11 trampoline class to allow a Python
// class to inherit from torch.distributed.Backend
class PyBackend : public Backend {
 public:
  using Backend::Backend;

  // -- Collectives (return intrusive_ptr<Work>) --
  // Use WORK_OVERRIDE for methods where C++ and Python names match.

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    WORK_OVERRIDE(Backend, broadcast, tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    WORK_OVERRIDE(Backend, allreduce, tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    WORK_OVERRIDE(Backend, allreduce_sparse, tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override {
    WORK_OVERRIDE(Backend, allreduce_coalesced, tensors, opts);
  }

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override {
    WORK_OVERRIDE(Backend, reduce, tensors, opts);
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    WORK_OVERRIDE(Backend, allgather, outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> all_gather_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    WORK_OVERRIDE(Backend, all_gather_single, outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    WORK_OVERRIDE(
        Backend, allgather_coalesced, outputTensorLists, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> all_gather_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    WORK_OVERRIDE(Backend, all_gather_single_coalesced, outputs, inputs, opts);
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override {
    WORK_OVERRIDE(Backend, gather, outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override {
    WORK_OVERRIDE(Backend, scatter, outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    WORK_OVERRIDE(Backend, reduce_scatter, outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    WORK_OVERRIDE(
        Backend, reduce_scatter_single, outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    WORK_OVERRIDE(
        Backend, reduce_scatter_single_coalesced, outputs, inputs, opts);
  }

  c10::intrusive_ptr<Work> all_to_all_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    WORK_OVERRIDE(
        Backend,
        all_to_all_single,
        outputBuffer,
        inputBuffer,
        outputSplitSizes,
        inputSplitSizes,
        opts);
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    WORK_OVERRIDE(Backend, alltoall, outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    WORK_OVERRIDE(Backend, send, tensors, dstRank, tag);
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    WORK_OVERRIDE(Backend, recv, tensors, srcRank, tag);
  }

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "recv_anysource");
    if (override) {
      auto o = override(tensors, tag);
      if (o.is_none()) {
        return c10::intrusive_ptr<Work>();
      }
      return c10::make_intrusive<PyProcessGroup::PyWorkHolder>(o);
    }
    return Backend::recvAnysource(tensors, tag);
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    WORK_OVERRIDE(Backend, barrier, opts);
  }

  c10::intrusive_ptr<Work> reconfigure(
      const ReconfigureOptions& opts) override {
    WORK_OVERRIDE(Backend, reconfigure, opts);
  }

  void registerOnCompletionHook(
      std::function<void(std::shared_ptr<WorkInfo>)>&& hook) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "_register_on_completion_hook");
    if (override) {
      // Store the hook on the C++ side so hasHooks() returns true,
      // then forward to the Python override for custom handling.
      onCompletionHook_ = hook;
      override(std::move(hook));
      return;
    }
    return Backend::registerOnCompletionHook(std::move(hook));
  }

  // startCoalescing/endCoalescing: C++ names differ from Python binding
  // names (_start_coalescing/_end_coalescing). Follow PyProcessGroup's
  // pattern of using manual get_override with the Python name.
  void startCoalescing() override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "start_coalescing");
    if (override) {
      override();
      return;
    }
    return Backend::startCoalescing();
  }

  c10::intrusive_ptr<Work> endCoalescing() override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "end_coalescing");
    if (override) {
      auto o = override();
      if (o.is_none()) {
        return c10::intrusive_ptr<Work>();
      }
      return c10::make_intrusive<PyProcessGroup::PyWorkHolder>(o);
    }
    return Backend::endCoalescing();
  }

  // -- Methods returning non-Work intrusive_ptr --

  c10::intrusive_ptr<Backend> shrink(
      const std::vector<int64_t>& ranks_to_exclude,
      int shrink_flags = 0,
      const c10::intrusive_ptr<Options>& opts_override = nullptr) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override =
        pybind11::get_override(static_cast<const Backend*>(this), "shrink");
    if (override) {
      py::object o = override(ranks_to_exclude, shrink_flags, opts_override);
      return initSlotAndCast(o);
    }
    return Backend::shrink(ranks_to_exclude, shrink_flags, opts_override);
  }

  c10::intrusive_ptr<Backend> split(
      const c10::intrusive_ptr<Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<Options>& opts) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override =
        pybind11::get_override(static_cast<const Backend*>(this), "split");
    if (override) {
      py::object o = override(store, ranks, opts);
      return initSlotAndCast(o);
    }
    return Backend::split(store, ranks, opts);
  }

  c10::intrusive_ptr<Backend> merge(
      const c10::intrusive_ptr<Store>& store,
      const c10::intrusive_ptr<Options>& opts,
      const int& rank,
      const int& size) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override =
        pybind11::get_override(static_cast<const Backend*>(this), "merge");
    if (override) {
      py::object o = override(store, opts, rank, size);
      return initSlotAndCast(o);
    }
    return Backend::merge(store, opts, rank, size);
  }

  c10::intrusive_ptr<Window> new_window(
      const std::optional<at::Tensor>& tensor = std::nullopt) override {
    PYBIND11_OVERRIDE(c10::intrusive_ptr<Window>, Backend, new_window, tensor);
  }

  c10::intrusive_ptr<Options> getBackendOptions() override {
    pybind11::gil_scoped_acquire gil;
    auto self = pybind11::cast(this);
    auto cls = pybind11::type::of(self);
    if (pybind11::hasattr(cls, "options")) {
      auto attr = cls.attr("options");
      if (PyObject_IsInstance(
              attr.ptr(), reinterpret_cast<PyObject*>(&PyProperty_Type)) ||
          !pybind11::isinstance<pybind11::cpp_function>(attr)) {
        return pybind11::getattr(self, "options")
            .cast<c10::intrusive_ptr<Options>>();
      }
    }
    return Backend::getBackendOptions();
  }

  // -- Value-returning methods --

  const std::string getBackendName() const override {
    PYBIND11_OVERRIDE(std::string, Backend, getBackendName);
  }

  ReconfigureHandle get_reconfigure_handle() const override {
    PYBIND11_OVERRIDE(ReconfigureHandle, Backend, get_reconfigure_handle);
  }

  uint64_t getSequenceNumberForGroup() override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "_get_sequence_number_for_group");
    if (override) {
      return override().cast<uint64_t>();
    }
    return Backend::getSequenceNumberForGroup();
  }

  ErrorType getError() override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override =
        pybind11::get_override(static_cast<const Backend*>(this), "get_error");
    if (override) {
      return override().cast<ErrorType>();
    }
    return Backend::getError();
  }

  std::shared_ptr<c10::Allocator> getMemAllocator() override {
    PYBIND11_OVERRIDE(
        std::shared_ptr<c10::Allocator>, Backend, getMemAllocator);
  }

  at::Tensor allocateTensor(long size, at::TensorOptions options = {})
      override {
    PYBIND11_OVERRIDE(at::Tensor, Backend, allocateTensor, size, options);
  }

  std::unordered_map<std::string, uint64_t> getMemoryStats() override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "memory_stats");
    if (override) {
      return override().cast<std::unordered_map<std::string, uint64_t>>();
    }
    return Backend::getMemoryStats();
  }

  // -- Bool properties --
  // These are bound as def_property_readonly, so Python subclasses override
  // them with @property. get_override won't find @property descriptors, so
  // we use py::getattr to access them through normal Python attribute
  // resolution, which handles both @property and regular methods.

  bool supportsSplitting() const override {
    return getPropertyOverride(
        "supports_splitting", Backend::supportsSplitting());
  }

  bool supportsCoalescing() const override {
    return getPropertyOverride(
        "supports_coalescing", Backend::supportsCoalescing());
  }

  bool supportsTimeEstimation() const override {
    return getPropertyOverride(
        "supports_time_estimate", Backend::supportsTimeEstimation());
  }

  bool supportsShrinking() const override {
    return getPropertyOverride(
        "supports_shrinking", Backend::supportsShrinking());
  }

  bool supportsReconfigure() const override {
    return getPropertyOverride(
        "supports_reconfigure", Backend::supportsReconfigure());
  }

  bool supportsWindow() const override {
    return getPropertyOverride("supports_window", Backend::supportsWindow());
  }

  bool supportsTensorAlloc(c10::DeviceIndex deviceIdx) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "supports_tensor_alloc");
    if (override) {
      return override(deviceIdx).cast<bool>();
    }
    return Backend::supportsTensorAlloc(deviceIdx);
  }

  // -- Void methods --

  void setTimeout(std::chrono::milliseconds timeout) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "set_timeout");
    if (override) {
      override(timeout);
      return;
    }
    return Backend::setTimeout(timeout);
  }

  void abort() override {
    PYBIND11_OVERRIDE(void, Backend, abort);
  }

  void shutdown() override {
    PYBIND11_OVERRIDE(void, Backend, shutdown);
  }

  void suspend() override {
    PYBIND11_OVERRIDE(void, Backend, suspend);
  }

  void resume() override {
    PYBIND11_OVERRIDE(void, Backend, resume);
  }

  void monitoredBarrier(const BarrierOptions& opts, bool waitAllRanks = false)
      override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "monitored_barrier");
    if (override) {
      override(opts, waitAllRanks);
      return;
    }
    return Backend::monitoredBarrier(opts, waitAllRanks);
  }

  void setSequenceNumberForGroup() override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "_set_sequence_number_for_group");
    if (override) {
      override();
      return;
    }
    return Backend::setSequenceNumberForGroup();
  }

  void waitForPendingWorks() override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "_wait_for_pending_works");
    if (override) {
      override();
      return;
    }
    return Backend::waitForPendingWorks();
  }

  void enableCollectivesTiming() override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "_enable_collectives_timing");
    if (override) {
      override();
      return;
    }
    return Backend::enableCollectivesTiming();
  }

  void eagerConnectSingleDevice(at::Device device) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "eager_connect_single_device");
    if (override) {
      override(device);
      return;
    }
    return Backend::eagerConnectSingleDevice(device);
  }

  void setGroupUid(const std::string& pg_uid) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(
        static_cast<const Backend*>(this), "_set_group_uid");
    if (override) {
      override(pg_uid);
      return;
    }
    return Backend::setGroupUid(pg_uid);
  }

  void registerAbortHook(int64_t hook_id, AbortHook hook) override {
    PYBIND11_OVERRIDE(void, Backend, registerAbortHook, hook_id, hook);
  }

  void unregisterAbortHook(int64_t hook_id) override {
    PYBIND11_OVERRIDE(void, Backend, unregisterAbortHook, hook_id);
  }

 private:
  static c10::intrusive_ptr<Backend> initSlotAndCast(py::object o) {
    auto backend = o.cast<c10::intrusive_ptr<Backend>>();
    auto* pyobj = torch::utils::PyObjectPreservation::get_or_init(
        *backend, [&]() { return Py_NewRef(o.ptr()); });
    Py_DECREF(pyobj);
    return backend;
  }

  bool getPropertyOverride(const char* name, bool defaultValue) const {
    pybind11::gil_scoped_acquire gil;
    auto self = pybind11::cast(this);
    auto cls = pybind11::type::of(self);
    if (pybind11::hasattr(cls, name)) {
      auto attr = cls.attr(name);
      if (!pybind11::isinstance<pybind11::cpp_function>(attr)) {
        return pybind11::getattr(self, name).cast<bool>();
      }
    }
    return defaultValue;
  }
};

} // namespace c10d
