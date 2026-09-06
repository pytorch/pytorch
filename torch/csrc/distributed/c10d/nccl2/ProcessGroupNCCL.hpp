// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// ProcessGroupNCCL: an in-tree c10d::Backend backed by the torchcomms NCCL
// engine. This is a port of torchcomms' TorchCommNCCL collapsed directly onto
// c10d::Backend -- the upstream TorchComm/TorchCommBackend/BackendWrapper
// layers are removed, and the collective methods use the c10d option objects
// (c10d::BroadcastOptions, c10d::ReduceOp, ...) directly rather than the
// torchcomms-specific option/ReduceOp types.
//
// Namespace: the class lives in c10d::nccl2.

#pragma once

#ifdef USE_C10D_NCCL

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <torch/csrc/distributed/c10d/nccl2/Batch.hpp>
#include <torch/csrc/distributed/c10d/nccl2/NcclApi.hpp>
#include <torch/csrc/distributed/c10d/nccl2/WorkNCCL.hpp>

namespace c10d::nccl2 {

// Hint key names for NCCL backend configuration
constexpr std::string_view kHintMaxEventPoolSize = "max_event_pool_size";
constexpr size_t kDefaultMaxEventPoolSize = 1000;

// ncclConfig_t::netName is a strdup'ed const char* (see the NCCLConfig pybind
// setter) and ncclConfig_t tracks no ownership, so plainly copying the struct
// would leave two owners sharing one allocation -- which double frees on the
// NCCL versions that free the caller's netName when the communicator is
// destroyed. Copy the string as well so each config owns its own.
TORCH_API ncclConfig_t cloneNcclConfig(const ncclConfig_t& config);

TORCH_API void waitForNcclCompletion(
    NcclApi& nccl_api,
    ncclComm_t comm,
    ncclResult_t status,
    std::chrono::milliseconds timeout,
    std::string_view operation);

TORCH_API void waitForNcclChildComm(
    NcclApi& nccl_api,
    ncclComm_t parent_comm,
    ncclComm_t* child_comm,
    ncclResult_t status,
    bool expect_child,
    std::chrono::milliseconds timeout,
    std::string_view operation);

// Custom exception class for better error handling
class NCCLException : public std::exception {
 public:
  NCCLException(
      NcclApi& api,
      const std::string& message,
      ncclResult_t result,
      ncclComm_t comm);

  const char* what() const noexcept override;
  [[nodiscard]] ncclResult_t getResult() const noexcept;

 private:
  std::string message_;
  ncclResult_t result_;
};

#define NCCL_CHECK(nccl_api, nccl_comm, call, err_str)            \
  do {                                                            \
    ncclResult_t status = call;                                   \
    if (status != ncclSuccess) {                                  \
      throw NCCLException(*nccl_api, err_str, status, nccl_comm); \
    }                                                             \
  } while (0)

#define NCCL_CHECK_NONBLOCKING(nccl_api, nccl_comm, call, err_str) \
  do {                                                             \
    ncclResult_t status = call;                                    \
    if (status != ncclSuccess && status != ncclInProgress) {       \
      throw NCCLException(*nccl_api, err_str, status, nccl_comm);  \
    }                                                              \
  } while (0)

// Ignore variant for use in destructors - logs errors instead of throwing
#define NCCL_CHECK_IGNORE(nccl_api, call, err_str)                         \
  do {                                                                     \
    ncclResult_t status = call;                                            \
    if (status != ncclSuccess) {                                           \
      LOG(ERROR) << "[TC] " << err_str << ": "                             \
                 << nccl_api->getErrorString(status) << " at " << __FILE__ \
                 << ":" << __LINE__;                                       \
    }                                                                      \
  } while (0)

class TORCH_API ProcessGroupNCCL : public ::c10d::Backend {
 public:
  static constexpr std::string_view kBackendName = "nccl2";

  using Options = ::c10d::ProcessGroupNCCL::Options;

  // c10d-style constructor. Communicator initialization happens when c10d
  // binds the backend to a device via setBoundDeviceId().
  ProcessGroupNCCL(
      c10::intrusive_ptr<::c10d::Store> store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());
  ~ProcessGroupNCCL() override;

  ProcessGroupNCCL(const ProcessGroupNCCL&) = delete;
  ProcessGroupNCCL(ProcessGroupNCCL&&) = delete;
  ProcessGroupNCCL& operator=(const ProcessGroupNCCL&) = delete;
  ProcessGroupNCCL& operator=(ProcessGroupNCCL&&) = delete;

  // ---- c10d::Backend overrides ----
  const std::string getBackendName() const override {
    return std::string(kBackendName);
  }
  c10::intrusive_ptr<::c10d::Backend::Options> getBackendOptions() override;

  c10::intrusive_ptr<::c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const ::c10d::BroadcastOptions& opts =
          ::c10d::BroadcastOptions()) override;
  c10::intrusive_ptr<::c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const ::c10d::AllreduceOptions& opts =
          ::c10d::AllreduceOptions()) override;
  c10::intrusive_ptr<::c10d::Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const ::c10d::AllreduceOptions& opts =
          ::c10d::AllreduceOptions()) override;
  c10::intrusive_ptr<::c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const ::c10d::AllreduceCoalescedOptions& opts =
          ::c10d::AllreduceCoalescedOptions()) override;
  c10::intrusive_ptr<::c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ::c10d::ReduceOptions& opts = ::c10d::ReduceOptions()) override;
  c10::intrusive_ptr<::c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ::c10d::AllgatherOptions& opts =
          ::c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<::c10d::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const ::c10d::AllgatherOptions& opts =
          ::c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<::c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ::c10d::AllgatherOptions& opts =
          ::c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<::c10d::Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ::c10d::AllgatherOptions& opts =
          ::c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<::c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ::c10d::GatherOptions& opts = ::c10d::GatherOptions()) override;
  c10::intrusive_ptr<::c10d::Work> gather_single(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ::c10d::GatherOptions& opts = ::c10d::GatherOptions()) override;
  c10::intrusive_ptr<::c10d::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ::c10d::ScatterOptions& opts = ::c10d::ScatterOptions()) override;
  c10::intrusive_ptr<::c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ::c10d::ReduceScatterOptions& opts =
          ::c10d::ReduceScatterOptions()) override;
  c10::intrusive_ptr<::c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ::c10d::ReduceScatterOptions& opts =
          ::c10d::ReduceScatterOptions()) override;
  c10::intrusive_ptr<::c10d::Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ::c10d::ReduceScatterOptions& opts =
          ::c10d::ReduceScatterOptions()) override;
  c10::intrusive_ptr<::c10d::Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const ::c10d::AllToAllOptions& opts = ::c10d::AllToAllOptions()) override;
  c10::intrusive_ptr<::c10d::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ::c10d::AllToAllOptions& opts = ::c10d::AllToAllOptions()) override;
  c10::intrusive_ptr<::c10d::Work> barrier(
      const ::c10d::BarrierOptions& opts = ::c10d::BarrierOptions()) override;
  c10::intrusive_ptr<::c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;
  c10::intrusive_ptr<::c10d::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  bool supportsCoalescing() const override {
    return true;
  }
  bool supportsTimeEstimation() const override {
#ifdef NCCL_SIM_INFO_INITIALIZER
    return true;
#else
    return false;
#endif
  }
  bool supportsSplitting() const override {
    return true;
  }
  bool isInitialized() override;
  bool supportsShrinking() const override {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
    return true;
#else
    return false;
#endif
  }
  void startCoalescing() override;
  c10::intrusive_ptr<::c10d::Work> endCoalescing() override;
  void startTimeEstimate() override;
  float endTimeEstimate() override;

  // Create a child backend over `ranks` (a subset of this group's ranks) via
  // ncclCommSplit. Collective over the parent communicator: every parent rank
  // must call it (members with their color, non-members with
  // NCCL_SPLIT_NOCOLOR) in the same order. Non-members get an empty pointer.
  c10::intrusive_ptr<::c10d::Backend> split(
      const c10::intrusive_ptr<::c10d::Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<::c10d::Backend::Options>& opts) override;
  void performNocolorSplit(at::Device device);
  c10::intrusive_ptr<::c10d::Backend> shrink(
      const std::vector<int64_t>& ranks_to_exclude,
      int shrink_flags = 0,
      const c10::intrusive_ptr<::c10d::Backend::Options>& opts_override =
          nullptr) override;

  std::shared_ptr<c10::Allocator> getMemAllocator() override;
  at::Tensor allocateTensor(long size, at::TensorOptions options) override;
  bool supportsTensorAlloc(c10::DeviceIndex deviceIdx) override;
  void setTimeout(std::chrono::milliseconds timeout) override;
  void addEphemeralTimeout(const std::chrono::milliseconds& timeout) override;
  void setBoundDeviceId(std::optional<at::Device> device) override;
  void eagerConnectSingleDevice(at::Device device) override;
  uint64_t getSequenceNumberForGroup() override {
    return sequence_number_;
  }
  void enableCollectivesTiming() override;
  void shutdown() override;
  void abort() override;
  ::c10d::ErrorType getError() override;

  // Memory offload API (see Backend.hpp): suspend() releases NCCL's dynamic
  // GPU allocations (offloading contents to CPU backups), resume() restores
  // them, and getMemoryStats() reports the ncclCommMemStats counters. The
  // communicator cannot be used while suspended. Requires NCCL 2.29.7+ at
  // runtime.
  void suspend() override;
  void resume() override;
  std::unordered_map<std::string, uint64_t> getMemoryStats() override;

  // Fault tolerance / reconfigure API (see Backend.hpp). The handle encodes
  // "nccl2:<rank>:<uuid>:<store host:port>"; reconfigure() tears down the
  // current communicator generation (if any) and bootstraps a fresh ncclComm
  // over the surviving/new members. Implemented in
  // ReconfigureNCCL.cpp.
  bool supportsReconfigure() const override {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0) && !defined(USE_ROCM)
    return true;
#else
    return false;
#endif
  }
  ::c10d::ReconfigureHandle get_reconfigure_handle() const override;
  c10::intrusive_ptr<::c10d::Work> reconfigure(
      const ::c10d::ReconfigureOptions& opts) override;

  // Window / one-sided RMA API (see Backend.hpp and WindowNCCL.hpp). Windows
  // are zero-copy: tensors must be allocated from this backend's NCCL mempool
  // (torch.cuda.MemPool(backend.mem_allocator)); the caching-allocator hook
  // registers each mempool segment with the communicator and WindowNCCL
  // lazily upgrades the segment to a collective NCCL_WIN_COLL_SYMMETRIC
  // window on first use. Requires NCCL 2.29+ at runtime.
  bool supportsWindow() const override;
  c10::intrusive_ptr<::c10d::Window> new_window(
      const std::optional<at::Tensor>& tensor = std::nullopt) override;

  // MemPool registration (::c10d::ProcessGroupNCCL::registerMemPool parity).
  // Unlike the stock backend, plain ncclCommRegister of every private-pool
  // segment already happens unconditionally through NCCLCachingAllocatorHook,
  // so this only has to cover the segments the hook could not reach and -- for
  // symm -- perform the collective NCCL_WIN_COLL_SYMMETRIC upgrade that the
  // hook must not do from an allocator thread. Collective when symm is set:
  // every rank must call it with the same pool contents, in the same order.
  void registerMemPool(at::cuda::MemPool* pool, bool symm = false);
  void deregisterMemPool(at::cuda::MemPool* pool);

  // Caching-allocator segment registration (called by
  // NCCLCachingAllocatorHook, potentially from allocator threads).
  void register_address(void* addr, size_t len);
  // from_allocator_hook only controls diagnostics: tearing a symmetric window
  // down from the hook is the uncollective path worth warning about.
  void deregister_address(void* addr, bool from_allocator_hook = false);
  // Returns {window handle, byte offset of ptr within the segment}, or
  // {nullptr, 0} if ptr is not inside a window-registered segment.
  std::pair<ncclWindow_t, size_t> lookupSegmentWindow(const void* ptr);
#if defined(USE_ROCM)
  // Returns true when ptr exactly matches an ncclMemAlloc segment base on this
  // process group's device and len does not exceed the allocation size.
  bool isNcclAllocatorSegment(const void* ptr, size_t len) const;
#endif
  // Registers the segment containing ptr as a NCCL_WIN_COLL_SYMMETRIC window
  // if it is not one already. Collective: all ranks must call it together.
  // On ROCm the segment must be a live ncclMemAlloc/getMemAllocator range:
  // an ineligible segment returns ncclInvalidArgument (caller should throw),
  // while ncclInvalidUsage stays reserved for a missing symmetric transport
  // (caller keeps the plain registration and warns).
  ncclResult_t ensureSegmentWindow(const void* ptr);

  bool supportsAbortHooks() const override {
    return true;
  }
  void registerAbortHook(int64_t hook_id, ::c10d::AbortHook hook) override;
  void unregisterAbortHook(int64_t hook_id) override;

  bool supportsCompletionHooks() const override {
    return true;
  }
  void registerCompletionHook(int64_t hook_id, ::c10d::CompletionHook hook)
      override;
  void unregisterCompletionHook(int64_t hook_id) override;

  // ---- accessors used by friend classes (work) ----
  NcclApi* getNcclApi() const {
    return nccl_api_.get();
  }
  void setNcclApi(std::shared_ptr<NcclApi> api) {
    nccl_api_ = std::move(api);
  }
  const at::Device& getDevice() const {
    return device_;
  }
  std::string_view getCommName() const {
    return name_;
  }
  // Underlying host ncclComm_t as an opaque integer pointer.
  int64_t getCommPtr() const;
  bool collectivesTimingEnabled() const {
    return event_pool_->timingEnabled();
  }
  const std::shared_ptr<NCCLEventPool>& getEventPool() const {
    return event_pool_;
  }

  friend class WorkNCCL;
  friend class WindowNCCL;

 protected:
  void waitForNcclOperation(
      ncclResult_t status,
      std::chrono::milliseconds timeout,
      std::string_view operation);
  // Tears the NCCL communicator down. This NEVER terminates the process --
  // a user-initiated abort()/shutdown() must be survivable, matching
  // ::c10d::ProcessGroupNCCL::abort(). Callers that are handling a
  // watchdog-detected timeout or async error use handleWatchdogFailure().
  void abortNcclComm();
  // Applies the cleanup/process-teardown action selected by
  // TORCH_NCCL_ASYNC_ERROR_HANDLING. Reconfigurable communicators are revoked
  // instead, independent of the selected mode.
  void handleWatchdogFailure(const std::string& reason);
  // Blocking wait has no watchdog, so the waiting thread tears down a failed
  // communicator before surfacing the exception.
  void handleBlockingWaitFailure(
      WorkNCCL::WorkStatus status,
      int64_t reconfigure_uuid);
  // Terminates the process (after running the abort hooks) if requested by the
  // async error mode. `reason` is logged after "Aborting process on rank N due
  // to ", so it must describe the actual trigger.
  void abortProcess(const std::string& reason);
  // Signals the watchdog thread to exit and reaps it. Detaches instead of
  // joining when called from the watchdog thread itself.
  void stopWatchdog();
  void revokeNcclComm();

  enum class CommState {
    NORMAL,
    ERROR,
    TIMEOUT,
  };

  std::atomic<CommState> comm_state_{CommState::NORMAL};
  std::atomic<bool> revoked_{false};

  ncclDataType_t getNcclDataType(const at::Tensor& tensor);
  c10::intrusive_ptr<WorkNCCL> createWork(
      cudaStream_t stream,
      std::chrono::milliseconds timeout,
      const std::vector<at::Tensor>& inputTensors = {});
  c10::intrusive_ptr<WorkNCCL> createWork(
      cudaStream_t stream,
      std::chrono::milliseconds timeout,
      const at::Tensor& inputTensor);

 private:
  // RAII helper that cleans up NCCL premul-sum reduction ops. Built from a
  // c10d::ReduceOp (the premul factor is read from its supplement).
  struct RedOpRAII {
    /* implicit */ RedOpRAII(ncclRedOp_t op);
    explicit RedOpRAII(
        const ::c10d::ReduceOp& op,
        ncclComm_t comm,
        const ncclDataType_t dataType,
        std::shared_ptr<NcclApi> nccl_api);

    RedOpRAII() = delete;
    RedOpRAII(const RedOpRAII&) = delete;
    RedOpRAII& operator=(const RedOpRAII&) = delete;

    RedOpRAII(RedOpRAII&& other) noexcept
        : ncclRedOp_(other.ncclRedOp_),
          comm_(other.comm_),
          nccl_api_(std::move(other.nccl_api_)) {
      other.comm_ = nullptr;
    }
    RedOpRAII& operator=(RedOpRAII&& other) noexcept {
      if (this != &other) {
        if (comm_ && nccl_api_) {
          NCCL_CHECK_IGNORE(
              nccl_api_,
              nccl_api_->redOpDestroy(ncclRedOp_, comm_),
              "failed to destroy NCCL reduction operation");
        }
        ncclRedOp_ = other.ncclRedOp_;
        comm_ = other.comm_;
        nccl_api_ = std::move(other.nccl_api_);
        other.comm_ = nullptr;
      }
      return *this;
    }
    ~RedOpRAII();

    operator ncclRedOp_t() const {
      return ncclRedOp_;
    }

    ncclRedOp_t ncclRedOp_{ncclMaxRedOp};
    ncclComm_t comm_{nullptr};
    std::shared_ptr<NcclApi> nccl_api_;
  };

  // One-time bootstrap of the NCCL communicator on `device`. Subsequent calls
  // validate the same device.
  void ensureInitialized(at::Device device);
  void init(at::Device device);
  void finalize();
  void initNcclResources();
  // Adopt a child communicator and bring this backend to the INITIALIZED state,
  // sharing the parent's NcclApi.
  void initFromComm(
      ncclComm_t comm,
      at::Device device,
      std::shared_ptr<NcclApi> nccl_api);

  // Internal NCCL engine helpers (port of TorchCommNCCL). These take c10d
  // option fields directly (c10d::ReduceOp + resolved timeout/root/async),
  // not torchcomms-specific option objects.
  c10::intrusive_ptr<WorkNCCL> sendImpl(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> recvImpl(
      at::Tensor& tensor,
      int src,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> broadcastImpl(
      at::Tensor& tensor,
      int root,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> all_reduce(
      at::Tensor& tensor,
      const ::c10d::ReduceOp& op,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> reduceImpl(
      const at::Tensor& tensor,
      int root,
      const ::c10d::ReduceOp& op,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> allGatherSingleImpl(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ::c10d::ReduceOp& op,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> reduceScatterSingleImpl(
      at::Tensor& output,
      const at::Tensor& input,
      const ::c10d::ReduceOp& op,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> allToAllSingleImpl(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> barrierImpl(
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> scatterImpl(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkNCCL> gatherImpl(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      std::chrono::milliseconds timeout);

  // Resolve a c10d per-op timeout (kUnsetTimeout -> communicator default).
  std::chrono::milliseconds operationTimeout(
      std::chrono::milliseconds opt_timeout) const;

  size_t wordSize(ncclDataType_t type) const;
  RedOpRAII getNcclReduceOp(
      const ::c10d::ReduceOp& op,
      ncclComm_t comm,
      const at::Tensor& tensor);
  void timeoutWatchdog() noexcept;
  void checkInitialized() const;
  void checkAndAbortIfTimedOutOrError();
  void checkWorkQueue();
  std::pair<std::chrono::milliseconds, std::chrono::milliseconds>
  applyEphemeralTimeout(std::chrono::milliseconds timeout);
  void releaseEphemeralTimeout(std::chrono::milliseconds timeout);
  void enqueueWork(
      const c10::intrusive_ptr<WorkNCCL>& work,
      cudaStream_t stream);
  bool getGraphCaptureMode();
  cudaStream_t getOperationStream(bool async_op);
  void ensureTensorContiguous(const at::Tensor& tensor);
  void checkTensorDevice(const at::Tensor& tensor) const;
  void checkTensorsDevice(const std::vector<at::Tensor>& tensors) const;
  void runAbortHooks();
  // Whether anyone is listening, so a work that completes with no hook
  // registered does not pay for a duration measurement nobody reads.
  bool hasCompletionHooks();
  void runCompletionHooks(
      uint64_t completion_key,
      std::optional<float> duration_ms);

  void attachMemoryHook();
  void detachMemoryHook();

  // Publish/retire nccl_comm_ to the symmetric-memory registry via the NCCL
  // comm-registration hook (NCCLCommRegistrationHook.hpp), so this class does
  // not depend on the symm_mem DevCommManager header. Fired from
  // initNcclResources() and the comm-teardown paths, respectively.
  void publishComm();
  void retireComm();

  // Member variables (port of TorchCommNCCL).
  ncclComm_t nccl_comm_{};
  at::Device device_;
  int comm_size_{};
  // NOTE: the rank is stored in the inherited c10d::Backend::rank_ (set in the
  // ctor and refreshed from NCCL in initNcclResources). The ported engine code
  // reads/writes `rank_` directly, which resolves to that protected member.
  std::optional<at::cuda::CUDAStream> internal_stream_;
  std::optional<at::cuda::CUDAEvent> dependency_event_;
  at::DataPtr barrier_buffer_;
  enum class InitializationState {
    UNINITIALIZED,
    INITIALIZED,
    FINALIZED,
  } init_state_{InitializationState::UNINITIALIZED};
  std::mutex reconfigure_mutex_;

  c10::intrusive_ptr<::c10d::Store> store_;
  uint64_t bootstrap_generation_{0};
  uint64_t sequence_number_{0};

  std::shared_ptr<NcclApi> nccl_api_;
  std::unique_ptr<at::cuda::MemPool> memPool_;

  std::shared_ptr<NCCLEventPool> event_pool_;

  WorkNCCLQueue workq_;

  std::thread timeout_thread_;
  std::atomic<bool> shutdown_{false};
  std::condition_variable timeout_cv_;
  std::mutex timeout_mutex_;

  bool is_high_priority_stream_{false};
  std::string name_;

  const ::c10d::ErrorHandlingMode async_error_handling_;
  const bool blocking_wait_;

  c10::intrusive_ptr<Options> options_c10d_;

  std::mutex ephemeral_timeout_mutex_;
  std::chrono::milliseconds ephemeral_timeout_active_{0};
  std::chrono::milliseconds ephemeral_timeout_inflight_{0};

  // Identifies the current communicator generation in the reconfigure regime;
  // -1 until the first reconfigure(). Baked into the reconfigure handle so
  // peers can detect membership of the same generation.
  int64_t reconfigure_uuid_{-1};

  // Registration handle for a caching-allocator segment (and its symmetric
  // window, once ensureSegmentWindow upgraded it). Sorted by base address so
  // lookupSegmentWindow can find the containing segment via upper_bound.
  struct RegistrationHandle {
    void* regHandle{nullptr};
    ncclWindow_t winHandle{nullptr};
    size_t len{0};
  };
  using RegistrationMap = std::map<void*, RegistrationHandle, std::less<>>;
  RegistrationMap memoryRegistrationHandles_;
  // Caller must hold memory_registration_mutex_. Returns end() when ptr is not
  // inside a registered segment.
  RegistrationMap::iterator findContainingRegistrationLocked(const void* ptr);
  // Guards memoryRegistrationHandles_ and registeredMemPools_:
  // register/deregister_address run on allocator threads while window ops look
  // segments up on the main thread.
  std::mutex memory_registration_mutex_;
  // MemPools passed to registerMemPool, so deregisterMemPool can reject a pool
  // that was never registered (stock does the same via its global
  // ncclCommMemPoolMap). The symm mode is not tracked: a segment carries its
  // own window handle, so teardown does not need to be told which mode to use.
  std::set<c10::cuda::MempoolId_t> registeredMemPools_;
  // Caller must hold memory_registration_mutex_ and have checked that `addr`
  // is not already registered.
  void registerAddressLocked(void* addr, size_t len);

  // Abort hooks (c10d::Backend API; storage was in torchcomms' TorchCommBackend
  // base, folded in here). Guarded because the watchdog thread fires them while
  // the owner of a hook may be un/registering on another thread -- a
  // FlightRecorderHook unregisters from its destructor, i.e. whenever Python
  // drops the handle.
  std::unordered_map<int64_t, ::c10d::AbortHook> abortHooks_;
  std::mutex abort_hooks_mutex_;

  // Completion hooks (c10d::Backend API). Same guarding rationale as the abort
  // hooks: the watchdog fires them from checkStatus() while their owner may be
  // un/registering on another thread.
  std::unordered_map<int64_t, ::c10d::CompletionHook> completionHooks_;
  std::mutex completion_hooks_mutex_;

  // Active coalescing batch (port of BackendWrapper). Engaged between
  // startCoalescing() and endCoalescing(); send()/recv() append into it.
  std::optional<BatchSendRecv> coalescing_batch_;
  c10::intrusive_ptr<WorkNCCL> coalesced_work_;

  std::unordered_map<
      unsigned long long,
      std::vector<std::shared_ptr<WorkNCCL::State>>>
      graph_capture_work_refs_;
  std::mutex graph_capture_work_mutex_;

  struct GraphCleanupData {
    ProcessGroupNCCL* comm;
    unsigned long long graph_id;
    GraphCleanupData(ProcessGroupNCCL* comm_, unsigned long long id)
        : comm(comm_), graph_id(id) {}
  };
  // NOTE: no CUDART_CB here -- it is empty on Linux CUDA and undefined under
  // HIP/ROCm; the plain void(void*) signature matches cuda/hipHostFn_t.
  static void graphCleanupCallback(void* userData);
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
