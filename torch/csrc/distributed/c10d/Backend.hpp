#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/debug.h>

constexpr auto kBackendDefaultTimeout =
    std::chrono::milliseconds(30 * 60 * 1000);

namespace c10d {

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

    std::chrono::milliseconds timeout;

    // backend name
    const std::string backend;
  };

  explicit Backend(int rank, int size);
  ~Backend() override = 0;

  int getRank() const {
    return rank_;
  }

  int getSize() const {
    return size_;
  }

  // Returns an unique opaque ID of this backend that can be used to correlate
  // with its collectives.
  int64_t getID() const {
    return reinterpret_cast<std::intptr_t>(this);
  }

  virtual bool supportsSplitting() const {
    return false;
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
  };

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
  // Note: this function will be deprecated in near future.
  virtual c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ", getBackendName(), " does not support _allgather_base"));
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

  // This function is a coalesced version of `allgather_into_tensor` (currently
  // still named as `_allgather_base`). Each tensor in the vector corresponds to
  // an input/output of one `allgather_into_tensor` operation.
  virtual c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& /* outputs */,
      std::vector<at::Tensor>& /* inputs */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not support allgather_into_tensor_coalesced"));
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

  virtual c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      const ReduceScatterOptions& /* opts */ = ReduceScatterOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not support _reduce_scatter_base"));
  }

  // This function is a coalesced version of `reduce_scatter_tensor` (currently
  // still named as `_reduce_scatter_base`). Each tensor in the vector
  // corresponds to an input/output of one `reduce_scatter_tensor` operation.
  virtual c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& /* outputs */,
      std::vector<at::Tensor>& /* inputs */,
      const ReduceScatterOptions& /* opts */ = ReduceScatterOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            getBackendName(),
            " does not support reduce_scatter_tensor_coalesced"));
  }

  virtual c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& /* outputBuffer */,
      at::Tensor& /* inputBuffer */,
      std::vector<int64_t>& /* outputSplitSizes */,
      std::vector<int64_t>& /* inputSplitSizes */,
      const AllToAllOptions& /* opts */ = AllToAllOptions()) {
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ", getBackendName(), " does not support alltoall_base"));
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

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store. Only implemented
  // for GLOO and NCCL backends currently.
  virtual void setSequenceNumberForGroup() {
    auto backendName = getBackendName();
    TORCH_CHECK(
        false,
        c10::str(
            "Backend ",
            backendName,
            " does not yet support sequence numbers."));
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

  bool hasHooks() const {
    return onCompletionHook_ != nullptr;
  }

  // Do not call this directly, use ProcessGroup::setGroupName instead.
  void setGroupName(const std::string& name) {
    pg_name_ = name;
  }

  const std::string& getGroupName() const {
    return pg_name_;
  }

  void setGroupDesc(const std::string& desc) {
    pg_desc_ = desc;
  }

  const std::string& getGroupDesc() const {
    return pg_desc_;
  }

  // See similar functions in ProcessGroup.hpp for context.
  c10::optional<at::Device> getBoundDeviceId() const {
    return bound_device_id_;
  }

  // Perform an eager connect to the specified device if the backend supports
  // it.
  virtual void eagerConnectSingleDevice(at::Device device) {
    // no-op in the default case; this is an optimization some
    // backends may perform
  }

  void setBoundDeviceId(c10::optional<at::Device> device) {
    if (device) {
      TORCH_CHECK(device->has_index(), "setBoundDeviceId must have an index");
    }
    bound_device_id_ = device;
  }

 protected:
  // Implementations of this interface need to call this to setup
  // appropriate logging etc.
  void init();

  const int rank_;
  const int size_;
  // Debug level setting. It is parsed once when ProcessGroup is constructed and
  // remains the same across use of this process group.
  DebugLevel dist_debug_level_;
  std::string pg_name_;
  std::string pg_desc_;

  std::function<void(std::shared_ptr<WorkInfo>)> onCompletionHook_;

  c10::optional<at::Device> bound_device_id_;
};

} // namespace c10d
