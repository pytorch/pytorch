#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <c10/macros/Macros.h>

#include <torch/csrc/distributed/c10d/Hooks.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Window.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/debug.h>

// Feature macro: when defined, c10d::Backend (and ProcessGroup) expose the
// fault-tolerance reconfigure APIs (supportsReconfigure /
// get_reconfigure_handle / reconfigure). Downstream backends can guard their
// overrides with #ifdef so they build against both old and new c10d headers.
#define C10D_BACKEND_HAS_RECONFIGURE 1

// Feature macro: when defined, c10d::Backend (and ProcessGroup) expose the
// one-sided window APIs (supportsWindow / new_window) and the c10d::Window
// interface. Downstream backends can guard their overrides with #ifdef.
#define C10D_BACKEND_HAS_WINDOW 1

// Feature macro: when defined, c10d::Backend exposes abort-hook registration
// and c10d::ProcessGroup additionally exposes pre/post collective hooks (see
// Hooks.hpp). Downstream backends can guard their overrides with #ifdef.
#define C10D_BACKEND_HAS_HOOKS 1

constexpr auto kBackendDefaultTimeout =
    std::chrono::milliseconds(30 * 60 * 1000);

namespace c10d {

enum class ErrorType {
  SUCCESS = 0,
  TIMEOUT = 1,
  // e.g., NCCL error, etc
  COMM_ERROR = 2,
  // TODO, do we need to distinguish between remote timeout or remote COMM
  // errors?
  REMOTE_ERROR = 3
};

namespace {
// RAII helper for C10D_BACKEND_FORWARDING_GUARD (below): sets the re-entry flag
// while a canonical `_single` collective forwards to its deprecated alias, and
// clears it on scope exit (including exceptions).
struct ForwardingGuard {
  bool& flag_;
  explicit ForwardingGuard(bool& flag) : flag_(flag) {
    flag_ = true;
  }
  ~ForwardingGuard() {
    flag_ = false;
  }
};
} // namespace

// Guards a canonical `_single` collective method against infinite recursion.
// Each `_single` method and its deprecated alias forward to each other so that
// a Backend subclass may override (and a caller may call) either name. Placed
// at the top of each canonical method, this macro declares a thread-local
// re-entry flag and -- if neither name is overridden -- reports "Backend <name>
// does not support <method>" (using __func__) instead of looping forever.
#define C10D_BACKEND_FORWARDING_GUARD()                 \
  static thread_local bool forwardingGuardFlag = false; \
  TORCH_CHECK(                                          \
      !forwardingGuardFlag,                             \
      "Backend ",                                       \
      getBackendName(),                                 \
      " does not support ",                             \
      __func__);                                        \
  ForwardingGuard forwardingGuard(forwardingGuardFlag)

class TORCH_API Backend : public torch::CustomClassHolder {
 public:
  // Backend Options is a base struct that defines the basic options
  // when constructing a Backend. Each Backend subclass should
  // extend this struct and define its options if it wants to provide more
  // config options (beyond basic ones defined here) to end user.
  struct TORCH_API Options : torch::CustomClassHolder {
    explicit Options(
        std::string backend,
        std::chrono::milliseconds timeout = kBackendDefaultTimeout)
        : timeout(timeout), backend(std::move(backend)) {}
    ~Options() override = default;
    Options(const Options&) = default;

    std::chrono::milliseconds timeout;

    // backend name
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::string backend;
    std::string group_name;
    std::string group_desc;
    std::vector<uint64_t> global_ranks_in_group;

    // When true, symmetric memory rendezvous exchanges metadata via this
    // PG's allgather instead of TCPStore, which gets overloaded at large
    // rank counts. This will lazily create the backend communicator if it
    // doesn't already exist. If this PG is only used for symmetric memory
    // (no regular collectives), consider calling abort() after rendezvous
    // to release the communicator.
    bool use_pg_for_symm_mem_rendezvous = false;

    // When true, the communicator is created in the reconfigure regime: it is
    // not initialized until reconfigure() is called. Backends that support
    // fault tolerance honor this; others ignore it.
    bool enable_reconfigure = false;
  };

  explicit Backend(int rank, int size);
  ~Backend() override = 0;

  int getRank() const {
    return rank_;
  }

  int getSize() const {
    return size_;
  }

  // Returns a unique opaque ID of this backend that can be used to correlate
  // with its collectives.
  int64_t getID() const {
    return reinterpret_cast<std::intptr_t>(this);
  }

  bool getUsePgForSymmMemRendezvous() const {
    return use_pg_for_symm_mem_rendezvous_;
  }

  void setUsePgForSymmMemRendezvous(bool value) {
    use_pg_for_symm_mem_rendezvous_ = value;
  }

  virtual bool supportsSplitting() const {
    return false;
  }

  virtual bool supportsCoalescing() const {
    return false;
  }

  virtual bool supportsTimeEstimation() const {
    return false;
  }

  virtual bool supportsShrinking() const {
    return false;
  }

  // Shrink the backend by excluding specified ranks. Backends that support
  // communicator shrinking should override this and return a new backend
  // instance representing the shrunken group. Backends may use opts_override
  // to supply backend-specific options for the new group.
  virtual c10::intrusive_ptr<Backend> shrink(
      const std::vector<int64_t>& /*ranks_to_exclude*/,
      int /*shrink_flags*/ = 0,
      const c10::intrusive_ptr<Options>& /*opts_override*/ = nullptr) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support shrink"));
  }

  virtual void setTimeout(std::chrono::milliseconds /*timeout*/) {
    TORCH_WARN(
        "Backend ",
        getBackendName(),
        " does not support setting timeout; the new value is ignored");
  }

  // Fault Tolerance / Reconfigure API
  //
  // Backends that support dynamic membership override these.
  // supportsReconfigure advertises support; get_reconfigure_handle returns an
  // opaque handle that peers exchange out-of-band; reconfigure (re)initializes
  // the communicator with a new set of peers.
  virtual bool supportsReconfigure() const {
    return false;
  }

  virtual ReconfigureHandle get_reconfigure_handle() const {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not support get_reconfigure_handle"));
  }

  virtual c10::intrusive_ptr<Work> reconfigure(
      const ReconfigureOptions& /* opts */) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ", getBackendName(), " does not support reconfigure"));
  }

  // Window & One-sided (RMA) API
  //
  // Backends that support one-sided operations advertise it via supportsWindow
  // and return a concrete c10d::Window from new_window. The optional tensor, if
  // provided, is registered with the new window.
  virtual bool supportsWindow() const {
    return false;
  }

  virtual c10::intrusive_ptr<Window> new_window(
      const std::optional<at::Tensor>& /* tensor */ = std::nullopt) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support new_window"));
  }

  // Abort Hook API
  //
  // Abort hooks are invoked before the backend aborts on a timeout or error,
  // letting users capture debug information. Hooks are keyed by an opaque
  // hook_id so they can be individually unregistered.
  virtual void registerAbortHook(int64_t /* hook_id */, AbortHook /* hook */) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not support registerAbortHook"));
  }

  virtual void unregisterAbortHook(int64_t /* hook_id */) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not support unregisterAbortHook"));
  }

  virtual void startCoalescing() {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not implement startCoalescing"));
  }

  virtual c10::intrusive_ptr<Work> endCoalescing() {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ", getBackendName(), " does not implement endCoalescing"));
  }

  // Subclasses must override this method to return the backend name
  virtual const std::string getBackendName() const {
    TORCH_INTERNAL_ASSERT(false, "getBackendName is not implemented.");
  }

  // Subclasses must override this method to return the backend name
  virtual c10::intrusive_ptr<Options> getBackendOptions() {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not implement getBackendOptions."));
  }

  virtual c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& /* tensors */,
      const BroadcastOptions& /* opts */ = BroadcastOptions()) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support broadcast"));
  }

  virtual c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceOptions& /* opts */ = AllreduceOptions()) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support allreduce"));
  }

  virtual c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceOptions& /* opts */ = AllreduceOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not support allreduce sparse"));
  }

  virtual c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& /* tensors */,
      const AllreduceCoalescedOptions& /* opts */ =
          AllreduceCoalescedOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not support allreduce_coalesced"));
  }

  virtual c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& /* tensors */,
      const ReduceOptions& /* opts */ = ReduceOptions()) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support reduce"));
  }

  virtual c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support allgather"));
  }

  // Gathers a single tensor inputBuffer into a single buffer outputBuffer that
  // is interpreted as a contiguous collection of size inputBuffer * WORLD_SIZE.
  // For implementers of ProcessGroup API and advanced users only.
  // Named after the torchcomms backend naming scheme.
  virtual c10::intrusive_ptr<Work> all_gather_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) {
    C10D_BACKEND_FORWARDING_GUARD();
    return _allgather_base(outputBuffer, inputBuffer, opts);
  }

  // Deprecated: use all_gather_single instead. Kept as an overridable,
  // forwarding alias for backward compatibility with existing backends and
  // callers.
  virtual c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) {
    return all_gather_single(outputBuffer, inputBuffer, opts);
  }

  // This function is deprecated and will be moved out of Backend to comms:
  // * do not add dependencies on this function,
  // * do not implement it in your Backend, implement _allgather_base
  //   instead.
  virtual c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& /* outputTensorLists */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not support allgather_coalesced"));
  }

  // This function is a coalesced version of `all_gather_single`. Each tensor in
  // the vector corresponds to an input/output of one `all_gather_single`
  // operation. Named after the torchcomms backend naming scheme.
  virtual c10::intrusive_ptr<Work> all_gather_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) {
    C10D_BACKEND_FORWARDING_GUARD();
    return allgather_into_tensor_coalesced(outputs, inputs, opts);
  }

  // Deprecated: use all_gather_single_coalesced instead. Kept as an
  // overridable, forwarding alias for backward compatibility with existing
  // backends and callers.
  virtual c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) {
    return all_gather_single_coalesced(outputs, inputs, opts);
  }

  virtual c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const GatherOptions& /* opts */ = GatherOptions()) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support gather"));
  }

  virtual c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ScatterOptions& /* opts */ = ScatterOptions()) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support scatter"));
  }

  virtual c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<std::vector<at::Tensor>>& /* inputTensors */,
      const ReduceScatterOptions& /* opts */ = ReduceScatterOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ", getBackendName(), " does not support reduce_scatter"));
  }

  // Named after the torchcomms backend naming scheme.
  virtual c10::intrusive_ptr<Work> reduce_scatter_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) {
    C10D_BACKEND_FORWARDING_GUARD();
    return _reduce_scatter_base(outputBuffer, inputBuffer, opts);
  }

  // Deprecated: use reduce_scatter_single instead. Kept as an overridable,
  // forwarding alias for backward compatibility with existing backends and
  // callers.
  virtual c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) {
    return reduce_scatter_single(outputBuffer, inputBuffer, opts);
  }

  // This function is a coalesced version of `reduce_scatter_single`. Each
  // tensor in the vector corresponds to an input/output of one
  // `reduce_scatter_single` operation. Named after the torchcomms backend
  // naming scheme.
  virtual c10::intrusive_ptr<Work> reduce_scatter_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) {
    C10D_BACKEND_FORWARDING_GUARD();
    return reduce_scatter_tensor_coalesced(outputs, inputs, opts);
  }

  // Deprecated: use reduce_scatter_single_coalesced instead. Kept as an
  // overridable, forwarding alias for backward compatibility with existing
  // backends and callers.
  virtual c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) {
    return reduce_scatter_single_coalesced(outputs, inputs, opts);
  }

  // Named after the torchcomms backend naming scheme.
  virtual c10::intrusive_ptr<Work> all_to_all_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) {
    C10D_BACKEND_FORWARDING_GUARD();
    return alltoall_base(
        outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes, opts);
  }

  // Deprecated: use all_to_all_single instead. Kept as an overridable,
  // forwarding alias for backward compatibility with existing backends and
  // callers.
  virtual c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) {
    return all_to_all_single(
        outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes, opts);
  }

  virtual c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& /* outputTensors */,
      std::vector<at::Tensor>& /* inputTensors */,
      const AllToAllOptions& opts = AllToAllOptions()) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support alltoall"));
  }

  virtual void monitoredBarrier(
      const BarrierOptions& /* unused */,
      bool /* unused */ = false) {
    auto backendName = getBackendName();
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            backendName,
            " does not support monitoredBarrier, only GLOO supports monitored barrier."));
  }

  // Deprecated no-op: sequence numbers now always start at 0 on every rank, so
  // there is no initial value to agree on. Kept for backward compatibility with
  // existing callers; it warns and does nothing.
  virtual void setSequenceNumberForGroup() {
    TORCH_WARN_ONCE(
        "setSequenceNumberForGroup() is deprecated and is now a no-op; "
        "sequence numbers always start at 0 on every rank. Remove calls to "
        "_set_sequence_number_for_group().");
  }

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  virtual uint64_t getSequenceNumberForGroup() {
    auto backendName = getBackendName();
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            backendName,
            " does not yet support sequence numbers."));
  }

  virtual c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& /* tensors */,
      int /* dstRank */,
      int /* tag */) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support send"));
  }

  virtual c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& /* tensors */,
      int /* srcRank */,
      int /* tag */) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support recv"));
  }

  virtual c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& /* tensors */,
      int /* tag */) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ", getBackendName(), " does not support recvAnysource"));
  }

  virtual c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& /* opts */ = BarrierOptions()) {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support barrier"));
  }

  virtual void registerOnCompletionHook(
      std::function<void(std::shared_ptr<WorkInfo>)>&& hook) {
    TORCH_CHECK(
        false,
        "Only ProcessGrouppNCCL supports onCompletion hook, but got ",
        getBackendName(),
        " backend.");
  }

  virtual void waitForPendingWorks() {
    TORCH_CHECK(
        false,
        "Only ProcessGrouppNCCL supports waitForPendingWorks, but got ",
        getBackendName(),
        " backend.");
  }

  virtual void enableCollectivesTiming() {
    TORCH_CHECK(
        false,
        "Backend ",
        getBackendName(),
        " is missing implementation of enableCollectivesTiming.");
  }

  virtual c10::intrusive_ptr<Backend> split(
      const c10::intrusive_ptr<Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<Options>& opts) {
    TORCH_CHECK(
        false,
        "Backend ",
        getBackendName(),
        " is missing implementation of split.");
  }

  virtual c10::intrusive_ptr<Backend> merge(
      const c10::intrusive_ptr<Store>& store,
      const c10::intrusive_ptr<Options>& opts,
      const int& rank,
      const int& size) {
    TORCH_CHECK(
        false,
        "Backend ",
        getBackendName(),
        " is missing implementation of merge.");
  }

  bool hasHooks() const {
    return onCompletionHook_ != nullptr;
  }

  // Do not call this directly, use ProcessGroup::setGroupName instead.
  virtual void setGroupUid(const std::string& pg_uid) {
    pg_uid_ = pg_uid;
  }

  const std::string& getGroupUid() const {
    return pg_uid_;
  }

  void setGroupDesc(const std::string& desc) {
    pg_desc_ = desc;
  }

  const std::string& getGroupDesc() const {
    return pg_desc_;
  }

  // See similar functions in ProcessGroup.hpp for context.
  std::optional<at::Device> getBoundDeviceId() const {
    return bound_device_id_;
  }

  // Perform an eager connect to the specified device if the backend supports
  // it.
  virtual void eagerConnectSingleDevice(at::Device device) {
    // no-op in the default case; this is an optimization some
    // backends may perform
  }

  void setBoundDeviceId(std::optional<at::Device> device) {
    if (device) {
      TORCH_CHECK(device->has_index(), "setBoundDeviceId must have an index");
    }
    bound_device_id_ = device;
  }

  virtual ErrorType getError() {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support getError"));
  }

  virtual std::shared_ptr<c10::Allocator> getMemAllocator() {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ", getBackendName(), " does not support getMemAllocator"));
  }

  // Allocate tensor (aten::empty) from backend's communication-optimized memory
  // pool
  virtual at::Tensor allocateTensor(long size, at::TensorOptions options = {}) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ", getBackendName(), " does not support allocateTensor"));
  }

  // Returns true if backend supports tensor allocation
  virtual bool supportsTensorAlloc(c10::DeviceIndex deviceIdx) {
    // Change to true in concrete backend if supported
    return false;
  }

  // Aborts all pending operations and connections in the backend if the backend
  // supports it.
  virtual void abort() {}

  // Shutdown the backend if the backend supports it. This should be used for
  // normal shutdown.
  virtual void shutdown() {}

  // APIs related to memory offload
  virtual void suspend() {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support suspend"));
  }

  virtual void resume() {
    TORCH_CHECK(
        false,
        c10::str("Backend ", getBackendName(), " does not support resume"));
  }

  virtual std::unordered_map<std::string, uint64_t> getMemoryStats() {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ", getBackendName(), " does not support getMemoryStats"));
  }

 protected:
  // Implementations of this interface need to call this to setup
  // appropriate logging etc.
  void init();

  int rank_;
  int size_;
  // Debug level setting. It is parsed once when ProcessGroup is constructed and
  // remains the same across use of this process group.
  DebugLevel dist_debug_level_;
  std::string pg_uid_;
  std::string pg_desc_;

  std::function<void(std::shared_ptr<WorkInfo>)> onCompletionHook_;

  std::optional<at::Device> bound_device_id_;

  bool use_pg_for_symm_mem_rendezvous_ = false;
};

} // namespace c10d

#undef C10D_BACKEND_FORWARDING_GUARD
