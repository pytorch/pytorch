#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/PyProcessGroup.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace c10d {

class PyBackend : public Backend {
 public:
  using Backend::Backend;

  PyBackend(py::object backend, int rank, int size)
      : Backend(rank, size), pyBackend_(std::move(backend)) {}

  ~PyBackend() override {
    pybind11::gil_scoped_acquire gil;
    if (pyBackend_) {
      pyBackend_.dec_ref();
      pyBackend_.ptr() = nullptr;
    }
  }

  static c10::intrusive_ptr<Backend> wrap(py::object backend) {
    if (backend.is_none()) {
      return nullptr;
    }

    auto base = backend.cast<c10::intrusive_ptr<Backend>>();
    if (dynamic_cast<PyBackend*>(base.get()) == nullptr ||
        !hasPythonBackendClass(backend)) {
      return base;
    }
    return c10::make_intrusive<PyBackend>(
        std::move(backend), base->getRank(), base->getSize());
  }

  py::object getPyBackendAttr(const std::string& name) const {
    if (pyBackend_) {
      return py::getattr(pyBackend_, name.c_str());
    }
    throw py::attribute_error(name);
  }

 private:
  static bool isC10dBackendType(py::handle type) {
    py::object name = py::getattr(type, "__name__", py::none());
    py::object module = py::getattr(type, "__module__", py::none());
    return !name.is_none() && !module.is_none() &&
        name.cast<std::string>() == "Backend" &&
        module.cast<std::string>() == "torch._C._distributed_c10d";
  }

  static bool hasPythonBackendClass(py::handle pySelf) {
    py::tuple mro = py::type::handle_of(pySelf).attr("__mro__");
    for (py::handle typeHandle : mro) {
      if (isC10dBackendType(typeHandle)) {
        return false;
      }
      py::object module = py::getattr(typeHandle, "__module__", py::none());
      if (!module.is_none() &&
          module.cast<std::string>() != "torch._C._distributed_c10d") {
        return true;
      }
    }
    return false;
  }

  py::object pySelf() const {
    if (pyBackend_) {
      return pyBackend_;
    }
    return py::cast(
        const_cast<PyBackend*>(this), py::return_value_policy::reference);
  }

  std::optional<py::object> getAttrOverride(const char* name) const {
    py::object self = pySelf();
    py::tuple mro = py::type::handle_of(self).attr("__mro__");
    for (py::handle typeHandle : mro) {
      if (isC10dBackendType(typeHandle)) {
        return std::nullopt;
      }
      py::object dict = py::getattr(typeHandle, "__dict__", py::none());
      if (!dict.is_none()) {
        int hasKey = PyMapping_HasKeyString(dict.ptr(), name);
        if (hasKey == -1) {
          throw py::error_already_set();
        }
        if (hasKey == 1) {
          return py::getattr(self, name);
        }
      }
    }
    return std::nullopt;
  }

  static c10::intrusive_ptr<Work> wrapWork(py::object work) {
    return c10::make_intrusive<PyProcessGroup::PyWorkHolder>(std::move(work));
  }

  py::object getMethodOverride(const char* name) const {
    if (pyBackend_) {
      if (auto override = getAttrOverride(name)) {
        return *override;
      }
      return py::object();
    }
    return pybind11::get_override(static_cast<const Backend*>(this), name);
  }

  template <typename Return>
  std::optional<Return> getAttrOverrideAs(const char* name) const {
    pybind11::gil_scoped_acquire gil;
    if (auto override = getAttrOverride(name)) {
      return override->template cast<Return>();
    }
    return std::nullopt;
  }

  template <typename Return, typename... Args>
  std::optional<Return> callWrappedOverride(const char* name, Args&&... args)
      const {
    if (!pyBackend_) {
      return std::nullopt;
    }
    pybind11::gil_scoped_acquire gil;
    if (auto override = getAttrOverride(name)) {
      return (*override)(std::forward<Args>(args)...).template cast<Return>();
    }
    return std::nullopt;
  }

  template <typename... Args>
  bool callWrappedVoidOverride(const char* name, Args&&... args) const {
    if (!pyBackend_) {
      return false;
    }
    pybind11::gil_scoped_acquire gil;
    if (auto override = getAttrOverride(name)) {
      (*override)(std::forward<Args>(args)...);
      return true;
    }
    return false;
  }

  template <typename... Args>
  c10::intrusive_ptr<Work> callWorkOverride(const char* name, Args&&... args)
      const {
    pybind11::gil_scoped_acquire gil;
    py::object override = getMethodOverride(name);
    if (override) {
      return wrapWork(override(std::forward<Args>(args)...));
    }
    return nullptr;
  }

  template <typename... Args>
  c10::intrusive_ptr<Backend> callBackendOverride(
      const char* name,
      Args&&... args) const {
    pybind11::gil_scoped_acquire gil;
    py::object override = getMethodOverride(name);
    if (override) {
      return wrap(override(std::forward<Args>(args)...));
    }
    return nullptr;
  }

  py::object pyBackend_;
  using AllocatorPtr = std::shared_ptr<c10::Allocator>;
  using OptionsPtr = c10::intrusive_ptr<Options>;
  using WindowPtr = c10::intrusive_ptr<Window>;
  using MemoryStats = std::unordered_map<std::string, uint64_t>;

 public:
  bool supportsSplitting() const override {
    if (auto override = getAttrOverrideAs<bool>("supports_splitting")) {
      return *override;
    }
    return Backend::supportsSplitting();
  }

  bool supportsCoalescing() const override {
    if (auto override = getAttrOverrideAs<bool>("supports_coalescing")) {
      return *override;
    }
    return Backend::supportsCoalescing();
  }

  bool supportsTimeEstimation() const override {
    if (auto override = getAttrOverrideAs<bool>("supports_time_estimate")) {
      return *override;
    }
    return Backend::supportsTimeEstimation();
  }

  bool supportsShrinking() const override {
    if (auto override = getAttrOverrideAs<bool>("supports_shrinking")) {
      return *override;
    }
    return Backend::supportsShrinking();
  }

  c10::intrusive_ptr<Backend> shrink(
      const std::vector<int64_t>& ranks_to_exclude,
      int shrink_flags = 0,
      const c10::intrusive_ptr<Options>& opts_override = nullptr) override {
    if (auto backend = callBackendOverride(
            "shrink", ranks_to_exclude, shrink_flags, opts_override)) {
      return backend;
    }
    return Backend::shrink(ranks_to_exclude, shrink_flags, opts_override);
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    if (callWrappedVoidOverride("set_timeout", timeout)) {
      return;
    }
    PYBIND11_OVERRIDE_NAME(void, Backend, "set_timeout", setTimeout, timeout);
  }

  bool supportsReconfigure() const override {
    if (auto override = getAttrOverrideAs<bool>("supports_reconfigure")) {
      return *override;
    }
    return Backend::supportsReconfigure();
  }

  ReconfigureHandle get_reconfigure_handle() const override {
    if (auto override =
            callWrappedOverride<ReconfigureHandle>("get_reconfigure_handle")) {
      return *override;
    }
    PYBIND11_OVERRIDE(ReconfigureHandle, Backend, get_reconfigure_handle);
  }

  c10::intrusive_ptr<Work> reconfigure(
      const ReconfigureOptions& opts) override {
    if (auto work = callWorkOverride("reconfigure", opts)) {
      return work;
    }
    return Backend::reconfigure(opts);
  }

  bool supportsWindow() const override {
    if (auto override = getAttrOverrideAs<bool>("supports_window")) {
      return *override;
    }
    return Backend::supportsWindow();
  }

  c10::intrusive_ptr<Window> new_window(
      const std::optional<at::Tensor>& tensor = std::nullopt) override {
    if (auto override = callWrappedOverride<WindowPtr>("new_window", tensor)) {
      return *override;
    }
    PYBIND11_OVERRIDE(c10::intrusive_ptr<Window>, Backend, new_window, tensor);
  }

  void startCoalescing() override {
    if (callWrappedVoidOverride("start_coalescing")) {
      return;
    }
    PYBIND11_OVERRIDE_NAME(void, Backend, "start_coalescing", startCoalescing);
  }

  c10::intrusive_ptr<Work> endCoalescing() override {
    if (auto work = callWorkOverride("end_coalescing")) {
      return work;
    }
    return Backend::endCoalescing();
  }

  const std::string getBackendName() const override {
    if (auto override = callWrappedOverride<std::string>("getBackendName")) {
      return *override;
    }
    PYBIND11_OVERRIDE(std::string, Backend, getBackendName);
  }

  c10::intrusive_ptr<Options> getBackendOptions() override {
    if (auto override = getAttrOverrideAs<OptionsPtr>("options")) {
      return *override;
    }
    return Backend::getBackendOptions();
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    if (auto work = callWorkOverride("broadcast", tensors, opts)) {
      return work;
    }
    return Backend::broadcast(tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    if (auto work = callWorkOverride("allreduce", tensors, opts)) {
      return work;
    }
    return Backend::allreduce(tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    if (auto work = callWorkOverride("allreduce_sparse", tensors, opts)) {
      return work;
    }
    return Backend::allreduce_sparse(tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override {
    if (auto work = callWorkOverride("allreduce_coalesced", tensors, opts)) {
      return work;
    }
    return Backend::allreduce_coalesced(tensors, opts);
  }

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override {
    if (auto work = callWorkOverride("reduce", tensors, opts)) {
      return work;
    }
    return Backend::reduce(tensors, opts);
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    if (auto work =
            callWorkOverride("allgather", outputTensors, inputTensors, opts)) {
      return work;
    }
    return Backend::allgather(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> all_gather_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    if (auto work = callWorkOverride(
            "all_gather_single", outputBuffer, inputBuffer, opts)) {
      return work;
    }
    return Backend::all_gather_single(outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    if (auto work = callWorkOverride(
            "allgather_coalesced", outputTensorLists, inputTensors, opts)) {
      return work;
    }
    return Backend::allgather_coalesced(outputTensorLists, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> all_gather_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    if (auto work = callWorkOverride(
            "all_gather_single_coalesced", outputs, inputs, opts)) {
      return work;
    }
    return Backend::all_gather_single_coalesced(outputs, inputs, opts);
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override {
    if (auto work =
            callWorkOverride("gather", outputTensors, inputTensors, opts)) {
      return work;
    }
    return Backend::gather(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override {
    if (auto work =
            callWorkOverride("scatter", outputTensors, inputTensors, opts)) {
      return work;
    }
    return Backend::scatter(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    if (auto work = callWorkOverride(
            "reduce_scatter", outputTensors, inputTensors, opts)) {
      return work;
    }
    return Backend::reduce_scatter(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    if (auto work = callWorkOverride(
            "reduce_scatter_single", outputBuffer, inputBuffer, opts)) {
      return work;
    }
    return Backend::reduce_scatter_single(outputBuffer, inputBuffer, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    if (auto work = callWorkOverride(
            "reduce_scatter_single_coalesced", outputs, inputs, opts)) {
      return work;
    }
    return Backend::reduce_scatter_single_coalesced(outputs, inputs, opts);
  }

  c10::intrusive_ptr<Work> all_to_all_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    if (auto work = callWorkOverride(
            "all_to_all_single",
            outputBuffer,
            inputBuffer,
            outputSplitSizes,
            inputSplitSizes,
            opts)) {
      return work;
    }
    return Backend::all_to_all_single(
        outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes, opts);
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    if (auto work =
            callWorkOverride("alltoall", outputTensors, inputTensors, opts)) {
      return work;
    }
    return Backend::alltoall(outputTensors, inputTensors, opts);
  }

  void monitoredBarrier(const BarrierOptions& opts, bool waitAllRanks = false)
      override {
    if (callWrappedVoidOverride("monitored_barrier", opts, waitAllRanks)) {
      return;
    }
    PYBIND11_OVERRIDE_NAME(
        void,
        Backend,
        "monitored_barrier",
        monitoredBarrier,
        opts,
        waitAllRanks);
  }

  void setSequenceNumberForGroup() override {
    if (callWrappedVoidOverride("_set_sequence_number_for_group")) {
      return;
    }
    PYBIND11_OVERRIDE_NAME(
        void,
        Backend,
        "_set_sequence_number_for_group",
        setSequenceNumberForGroup);
  }

  uint64_t getSequenceNumberForGroup() override {
    if (auto override =
            callWrappedOverride<uint64_t>("_get_sequence_number_for_group")) {
      return *override;
    }
    PYBIND11_OVERRIDE_NAME(
        uint64_t,
        Backend,
        "_get_sequence_number_for_group",
        getSequenceNumberForGroup);
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    if (auto work = callWorkOverride("send", tensors, dstRank, tag)) {
      return work;
    }
    return Backend::send(tensors, dstRank, tag);
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    if (auto work = callWorkOverride("recv", tensors, srcRank, tag)) {
      return work;
    }
    return Backend::recv(tensors, srcRank, tag);
  }

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override {
    if (auto work = callWorkOverride("recv_anysource", tensors, tag)) {
      return work;
    }
    return Backend::recvAnysource(tensors, tag);
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    if (auto work = callWorkOverride("barrier", opts)) {
      return work;
    }
    return Backend::barrier(opts);
  }

  void registerOnCompletionHook(
      std::function<void(std::shared_ptr<WorkInfo>)>&& hook) override {
    pybind11::gil_scoped_acquire gil;
    auto hookCopy = hook;
    onCompletionHook_ = std::move(hook);
    py::object override = getMethodOverride("_register_on_completion_hook");
    if (override) {
      override(std::move(hookCopy));
    }
  }

  void waitForPendingWorks() override {
    if (callWrappedVoidOverride("_wait_for_pending_works")) {
      return;
    }
    PYBIND11_OVERRIDE_NAME(
        void, Backend, "_wait_for_pending_works", waitForPendingWorks);
  }

  void enableCollectivesTiming() override {
    if (callWrappedVoidOverride("_enable_collectives_timing")) {
      return;
    }
    PYBIND11_OVERRIDE_NAME(
        void, Backend, "_enable_collectives_timing", enableCollectivesTiming);
  }

  c10::intrusive_ptr<Backend> split(
      const c10::intrusive_ptr<Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<Options>& opts) override {
    if (auto backend = callBackendOverride("split", store, ranks, opts)) {
      return backend;
    }
    return Backend::split(store, ranks, opts);
  }

  c10::intrusive_ptr<Backend> merge(
      const c10::intrusive_ptr<Store>& store,
      const c10::intrusive_ptr<Options>& opts,
      const int& rank,
      const int& size) override {
    if (auto backend = callBackendOverride("merge", store, opts, rank, size)) {
      return backend;
    }
    return Backend::merge(store, opts, rank, size);
  }

  void setGroupUid(const std::string& pg_uid) override {
    if (callWrappedVoidOverride("_set_group_name", pg_uid)) {
      return;
    }
    PYBIND11_OVERRIDE_NAME(
        void, Backend, "_set_group_name", setGroupUid, pg_uid);
  }

  void eagerConnectSingleDevice(at::Device device) override {
    if (callWrappedVoidOverride("eager_connect_single_device", device)) {
      return;
    }
    PYBIND11_OVERRIDE_NAME(
        void,
        Backend,
        "eager_connect_single_device",
        eagerConnectSingleDevice,
        device);
  }

  ErrorType getError() override {
    if (auto override = callWrappedOverride<ErrorType>("get_error")) {
      return *override;
    }
    PYBIND11_OVERRIDE_NAME(ErrorType, Backend, "get_error", getError);
  }

  std::shared_ptr<c10::Allocator> getMemAllocator() override {
    if (auto override = getAttrOverrideAs<AllocatorPtr>("mem_allocator")) {
      return *override;
    }
    return Backend::getMemAllocator();
  }

  at::Tensor allocateTensor(long size, at::TensorOptions options = {})
      override {
    pybind11::gil_scoped_acquire gil;
    py::object override = getMethodOverride("allocate_tensor");
    if (override) {
      c10::ScalarType dtype = c10::optTypeMetaToScalarType(options.dtype_opt())
                                  .value_or(at::kFloat);
      c10::Device device =
          options.device_opt().value_or(c10::Device(c10::DeviceType::CPU));
      return override(size, dtype, device).cast<at::Tensor>();
    }
    return Backend::allocateTensor(size, options);
  }

  bool supportsTensorAlloc(c10::DeviceIndex deviceIdx) override {
    if (auto override =
            callWrappedOverride<bool>("supports_tensor_alloc", deviceIdx)) {
      return *override;
    }
    PYBIND11_OVERRIDE_NAME(
        bool, Backend, "supports_tensor_alloc", supportsTensorAlloc, deviceIdx);
  }

  void abort() override {
    if (callWrappedVoidOverride("abort")) {
      return;
    }
    PYBIND11_OVERRIDE(void, Backend, abort);
  }

  void shutdown() override {
    if (callWrappedVoidOverride("shutdown")) {
      return;
    }
    PYBIND11_OVERRIDE(void, Backend, shutdown);
  }

  void suspend() override {
    if (callWrappedVoidOverride("suspend")) {
      return;
    }
    PYBIND11_OVERRIDE(void, Backend, suspend);
  }

  void resume() override {
    if (callWrappedVoidOverride("resume")) {
      return;
    }
    PYBIND11_OVERRIDE(void, Backend, resume);
  }

  std::unordered_map<std::string, uint64_t> getMemoryStats() override {
    if (auto override = callWrappedOverride<MemoryStats>("memory_stats")) {
      return *override;
    }
    PYBIND11_OVERRIDE_NAME(
        MemoryStats, Backend, "memory_stats", getMemoryStats);
  }
};

} // namespace c10d
