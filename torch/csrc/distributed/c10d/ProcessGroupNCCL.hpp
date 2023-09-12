#pragma once

#ifdef USE_C10D_NCCL

#include <chrono>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/UCCForNCCL.hpp>

#include <ATen/DynamicLibrary.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/custom_class.h>

namespace c10d {
// Environment variable which controls whether we perform a NCCL healt check
// which ensures communicators are healthy at the beginning of init.
constexpr const char* ENABLE_NCCL_HEALTH_CHECK = "ENABLE_NCCL_HEALTH_CHECK";

// Environment variable which controls whether or not wait() is blocking or
// non-blocking.
constexpr const char* NCCL_BLOCKING_WAIT = "NCCL_BLOCKING_WAIT";

// Environment variable which controls whether or not we perform Async Error
// Handling with NCCL.
constexpr const char* NCCL_ASYNC_ERROR_HANDLING = "NCCL_ASYNC_ERROR_HANDLING";

// Environment Variable to control whether Desync Debug is enabled.
// This variable must be set together with NCCL_ASYNC_ERROR_HANDLING.
constexpr const char* NCCL_DESYNC_DEBUG = "NCCL_DESYNC_DEBUG";

constexpr const char* NCCL_ENABLE_TIMING = "NCCL_ENABLE_TIMING";

constexpr const char* NCCL_BACKEND_NAME = "nccl";

// NoHandling: do not handle asynchronous NCCL errors
// TearDown: tear down process upon error, see `WorkNCCL::handleException`
// CleanUpOnly: just clean up collectives and abort communicators without
// tearing down process SkipCleanUp: (this is a temporary option and can be
// removed in future) tear down process without cleaning up NCCL communicators.
// This should be used as a last resort in case `ncclCommAbort` itself is
// hanging
enum ErrorHandlingMode {
  NoHandling = 0,
  TearDown = 1,
  CleanUpOnly = 2,
  SkipCleanUp = 3
};

#define SHOULD_CLEAN_UP(a) (a != NoHandling && a != SkipCleanUp)

#define SHOULD_TEAR_DOWN(a) (a != NoHandling && a != CleanUpOnly)

// If set, ProcessGroupNCCL doesn't use recordStream calls to ensure
// caching allocator safety for tensors used on both user-facing and
// internal comm streams.
// Instead, it stashes live references to those tensors until after
// user-facing streams are synced with comm streams.
// See stashed_for_allocator_safety_ below.
constexpr const char* TORCH_NCCL_AVOID_RECORD_STREAMS =
    "TORCH_NCCL_AVOID_RECORD_STREAMS";

// ProcessGroupNCCL implements NCCL bindings for c10d.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// All NCCL functions provided by this class are asynchronous functions. More
// specifically, each NCCL call is scheduled on a separate CUDA stream that is
// different from the current CUDA stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibility to make sure that the CUDA stream their
// code works on needs to wait for the NCCL operation from
// this class.
//
// This can be done by calling:
//
// either WorkNCCL::wait() or WorkNCCL::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Also note that WorkNCCL::finishedGPUExecution() is a helper function only
// provided by ProcessGroupNCCL to check if the NCCL operation of WorkNCCL has
// finished execution on the GPU (not just scheduled).
//
// Example on using the NCCL process group
//
//   ProcessGroupNCCL pg(store, rank, size);
//   std::shared_ptr<WorkNCCL> work = pg.allreduce(tensors);
//
//   // At this point, NCCL kernel has already by queued successfully
//   // Now, let current stream wait for the NCCL to finish, this function is
//   // async operation as well
//
//   work->wait()
//
//   // Now continue on other work in the current stream.
class TORCH_API ProcessGroupNCCL : public Backend {
 public:
  class WorkNCCL : public Work, public std::enable_shared_from_this<WorkNCCL> {
   public:
    friend struct WorkInfo;

    // Constructor takes a list of CUDA devices
    WorkNCCL(
        const std::vector<at::Device>& devices,
        int rank,
        OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr,
        const c10::optional<std::vector<at::Tensor>>& inputs = c10::nullopt,
        bool desyncDebug = false,
        bool enableTiming = false);
    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    WorkNCCL(const WorkNCCL& w);

    ~WorkNCCL() override;

    // Checks if the NCCL kernel has started to execute.
    bool isStarted();

    // Checks if request has completed. In this specific case of NCCL, it checks
    // if the NCCL operation has completed on the GPU in its own NCCL stream.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for NCCL work.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    // Let current stream wait on the completing of the NCCL work
    // Throws on exceptions. Blocking operation, which will wait for work
    // completion.
    void synchronize() override;

    // Synchronize streams by blocking each on the NCCL stream
    void synchronizeStreams();

    // Helper function to handle exception (throw if needed).
    void handleException(ErrorHandlingMode asyncErrorHandling);

    // Helper function that checks if the NCCL kernels have finished
    // execution on the GPUs
    bool finishedGPUExecution();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    float getDuration() const override;

    uint64_t getSequencenumber() const override;

    // Helper function that sets an exception_ptr on the WorkNCCL object.
    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkNCCL object has timed out
    // and False otherwise.
    // In case of timeout, set exception on the WorkNCCL object.
    bool checkTimeout(
        c10::optional<std::chrono::milliseconds> timeout = c10::nullopt);

    std::vector<at::Tensor> result() override;

   protected:
    // The cached list of CUDA devices to operate on
    std::vector<at::Device> devices_;

    // The start CUDA events of NCCL operator tracking this work item on
    // multiple CUDA devices. These start CUDA events are needed by desync
    // debugging if enabled.
    std::shared_ptr<std::vector<at::cuda::CUDAEvent>> ncclStartEvents_;

    // The end CUDA events of NCCL operator tracking this work item on
    // multiple CUDA devices.
    std::shared_ptr<std::vector<at::cuda::CUDAEvent>> ncclEndEvents_;

    // The NCCL communicators used for this work item.
    std::vector<std::shared_ptr<NCCLComm>> ncclComms_;

    // Tensors used for barrier op
    std::vector<at::Tensor> barrierTensors_;

    // Clone of blockingWait_ from ProcessGroupNCCL.
    bool blockingWait_ = false;

    // Clone of avoidRecordStreams_ from ProcessGroupNCCL.
    bool avoidRecordStreams_ = false;

    // Clone of opTimeout_ from ProcessGroupNCCL.
    std::chrono::milliseconds opTimeout_;

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    // Record the collective sequential number.
    uint64_t seq_;

    // Indicates if the nccl start event has been updated to the store trace.
    // This will be used by desync debug.
    bool startTraceUpdated_{false};

    // Record collective sizes for debug. We only record the size on the first
    // device as multi-device per process is deprecated
    size_t numelIn_ = -1;
    size_t numelOut_ = -1;

    // Wrapper method for the static checkForNCCLErrors which can be overridden
    // for tests.
    virtual std::exception_ptr checkForNCCLErrors(
        const std::vector<std::shared_ptr<NCCLComm>>& ncclComms) const;

    friend std::ostream& operator<<(
        std::ostream& output,
        const WorkNCCL& workNCCL);

   private:
    // Helper function for synchronize
    void synchronizeInternal(std::chrono::milliseconds timeout);

    // Checks for NCCL errors and sets an appropriate exception_ptr.
    void checkAndSetException();

    // Just checks whether GPU execution has started, without modifying
    // exception_ptr.
    bool startedGPUExecutionInternal() const;

    // Just checks whether GPU execution has completed, without modifying
    // exception_ptr.
    bool finishedGPUExecutionInternal() const;

    // Reference to the store so that we can write aborted communicators
    // to the store.
    c10::intrusive_ptr<Store> store_;

    // Store a reference to NCCL collective's outputs, used by result and to
    // give a more descriptive message when representing the Work as a string.
    std::shared_ptr<std::vector<at::Tensor>> outputs_;

    // TORCH_NCCL_AVOID_RECORD_STREAMS implementation helper.
    // Stores references to participating non-output tensors (ie inputs,
    // flattened intermediates).
    // We'll clear this list in synchronizeStreams, just after user-facing
    // stream(s) are synced with the nccl work stream(s).
    // By keeping these refs (as well as outputs_) alive until after the
    // collective's work rejoins the user-facing streams, we achieve
    // caching allocator safety without any recordStream calls.
    // For in-place collectives, some refs stashed here may alias outputs_,
    // but that doesn't do any harm.
    std::shared_ptr<std::vector<at::Tensor>> stashed_for_allocator_safety_;

    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;

    bool timingEnabled_;
    friend class ProcessGroupNCCL;
  };

  class CoalescedWorkNCCL
      : public Work,
        public std::enable_shared_from_this<CoalescedWorkNCCL> {
   public:
    // Constructor takes a list of WorkNCCL works
    CoalescedWorkNCCL(
        std::vector<ProcessGroupNCCL::WorkNCCL> works,
        int rank,
        OpType opType);

    ~CoalescedWorkNCCL() override;

    // Same as calling synchronize() for NCCL work.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

   protected:
    // The cached list of CUDA devices to operate on
    std::vector<ProcessGroupNCCL::WorkNCCL> works_;

    friend class ProcessGroupNCCL;
  };

  struct Options : Backend::Options {
    // NOTE: timeout in ProcessGroupNCCL::Options denote the timeout for
    // operations. This is only used when blockingWait_ is enabled.
    explicit Options(bool is_high_priority_stream = false);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    // Schedule NCCL operations on high priority CUDA streams
    bool is_high_priority_stream;

#ifdef NCCL_HAS_COMM_NONBLOCKING
    // Configure ranks
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
#endif
  };

  // If you wish to create multiple process groups, each with a potentially
  // different rank and size, you can do so by passing a new store instance
  // to each one. If you have only a single store object, you can
  // use the `c10d::PrefixStore` to derive scoped instances.
  // This is also what the Python API in torch.distributed does.
  //
  // The process group instance keeps a reference to the store because
  // it may be used long after the constructor runs. In fact, the constructor
  // doesn't create any NCCL communicators. A single NCCL communicator can
  // only be used on a specific set of devices, and are therefore created
  // on-demand when a collective runs. If another collective is executed later,
  // against a different set of devices, the process group creates another NCCL
  // communicator. These NCCL communicators are cached and reused if possible.
  //
  ProcessGroupNCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  // This constructor includes the deprecated `groupName` argument.
  // If you have existing code that uses the `groupName`, you can replace
  // it by specifying a `c10d::PrefixStore(groupName, store)` for store.
  C10_DEPRECATED ProcessGroupNCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      const std::string& groupName,
      c10::intrusive_ptr<Options> options = Options::create())
      : ProcessGroupNCCL(store, rank, size, options) {}

  ~ProcessGroupNCCL() override;

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  const std::string getBackendName() const override {
    return std::string(NCCL_BACKEND_NAME);
  }

  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> _broadcast_oop(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const BroadcastOptions& opts = BroadcastOptions());

  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> _reduce_oop(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ReduceOptions& opts = ReduceOptions());

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  void groupStart();

  void groupEnd();

  void groupEndNonblocking(std::vector<std::shared_ptr<NCCLComm>> comms);

  // Unsupported Ops
  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store.
  void setSequenceNumberForGroup() override;

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override;

  void registerOnCompletionHook(
      std::function<void(std::shared_ptr<WorkInfo>)>&& hook) override;
  void waitForPendingWorks() override;

  void enableCollectivesTiming() override;

  // Tests if the UCC fallback path is available
  bool isUCCAvailable() const;

  // Provides an API to abort the ProcessGroup (similar to ncclCommAbort)
  // instead of relying on ProcessGroupNCCL destructor.
  void abort(c10::optional<std::string> abortReason = c10::nullopt);

 protected:
  // Helper that broadcasts nccl unique ID to all ranks through the store
  void broadcastUniqueNCCLID(
      ncclUniqueId* ncclID,
      bool isSingleP2POp,
      const std::string& devicesKey,
      int p2pRank);

  // Helper that either looks up the cached NCCL communicators or creates
  // a new set of NCCL communicators as a cache entry
  std::vector<std::shared_ptr<NCCLComm>>& getNCCLComm(
      const std::string& devicesKey,
      const std::vector<at::Device>& devices,
      OpType opType,
      int p2pRank = 0,
      bool isSendRecvSelf = false);

  // Wrapper method which can be overridden for tests.
  virtual std::exception_ptr checkForNCCLErrors(
      const std::vector<std::shared_ptr<NCCLComm>>& ncclComms);

  virtual c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> initWork(
      std::vector<at::Device> devices,
      int rank,
      OpType opType,
      const char* profilingTitle = nullptr,
      const c10::optional<std::vector<at::Tensor>>& inputs = c10::nullopt);

  virtual c10::intrusive_ptr<ProcessGroupNCCL::CoalescedWorkNCCL>
  initCoalescedWork(
      const std::vector<c10::intrusive_ptr<Work>>& works,
      int rank,
      OpType opType);

 private:
  // Helper that encapsulates work shared across all collective communication
  // primitives.  The callbacks have the following signatures:
  //
  //    ncclResult_t fn(at::Tensor& input, at::Tensor& output,
  //                    ncclComm_t, at::cuda::CUDAStream&);
  //    void {pre,post}(std::vector<at::cuda::CUDAStream&>);
  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      OpType opType,
      const char* profilingTitle = nullptr);
  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      const char* profilingTitle = nullptr);

  // Helper that encapsulates work shared across point-to-point communication
  // primitives. It is the same structure as the helper used for collective
  // communication primitives.
  template <typename Fn>
  c10::intrusive_ptr<Work> pointToPoint(
      std::vector<at::Tensor>& tensor,
      Fn fn,
      int peer,
      OpType opType,
      const char* profilingTitle = nullptr);
  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> pointToPoint(
      std::vector<at::Tensor>& tensor,
      Fn fn,
      int peer,
      OpType opType,
      PreProcess pre,
      PostProcess post,
      const char* profilingTitle);

  c10::intrusive_ptr<Work> allreduce_impl(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions());

  // Checks for NCCL errors on each of the communicators and returns an
  // appropriate exception_ptr (nullptr if no errors).
  static std::exception_ptr checkForNCCLErrorsInternal(
      const std::vector<std::shared_ptr<NCCLComm>>& ncclComms);

  // Function that runs as part of a separate thread and checks for errors on
  // NCCL communicators. We need a separate thread to check for NCCL errors
  // since we can't rely on the user calling certain methods like wait(),
  // isCompleted() etc. to detect and remediate errors. In addition to this, we
  // need a mechanism to safely abort and remove NCCL communicators from our
  // cache. This can be done cleanly by having a thread for the ProcessGroupNCCL
  // class. Attempting to modify the communicator cache from the WorkNCCL class
  // might run into issues with object lifetime since the ProcessGroupNCCL
  // object might get destroyed before the WorkNCCL object.
  void ncclCommWatchdog();

  // Performs a health check by initializing dummy NCCL communicators and then
  // destroying them. This will help indicate and signal any NCCL-related issues
  // prior to the first collective. The actual initialization and subsequent
  // destruction is ran on a separate thread and the main thread is signalled
  // about timeouts/errors to report to the application.
  void runHealthCheck();

  // Destroys initialized NCCL communicators in devNCCLComMap_ given by input
  // key. Throws if there are no communicators to destroy. Also removes
  // communicators from the cache and clears used device indices.
  void destroyNCCLComms(const std::string& devNCCLCommMapKey);

  // Watchdog's inside loop.
  // Takes care of cleaning up completed work, and aborting upon failure or
  // timeout.
  void workCleanupLoop();

  void runHookLoop();

  // Desync debug helper
  void logWorkStart(WorkNCCL& work, bool emitDesyncInfo);

  // Desync debug helper
  void logWorkEnd(WorkNCCL& work, bool emitDesyncInfo);

 protected:
  static const int64_t kWatchdogThreadSleepMillis;

  // The store is used to broadcast the NCCL unique ID of rank 0.
  c10::intrusive_ptr<Store> store_;

  bool storeError_{false};

  const c10::intrusive_ptr<Options> options_;

  // The number of NCCL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t ncclCommCounter_{0};

  // The store keys to trace the last NCCL collective kernel CUDA events - start
  // event and end event respectively. These are used to do desync root cause
  // analysis.
  const std::string traceKeyStart_;
  const std::string traceKeyEnd_;

  // The NCCL communicator that the process group has cached.
  //
  // For collective operations:
  // The key is a list of GPU devices that an operation is operating on
  // The GPU devices are stored in a device sequence and the cache NCCL
  // communicator is associated with this GPU device sequence
  //
  // e.g. If the process group op only uses device 0, then the value of
  // the used device string stored (value of the hashmap) would be "0".
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
  //      then the value of the used device string (key) stored would be
  //      "0,1,2,3,4,5,6,7"
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
  //      then the value of the used device string stored would be
  //      "0,4,5,6,7,1,2,3"
  //
  //      Note that the order of the device for the tensor list matters.
  //
  // For point-to-point operations:
  // The key is a string of my current rank and the peer process rank.
  // e.g. If process 1 and process 2 are involved in a point-to-point
  // communication, the key will be "1:2" on both processes. Note: this is for
  // the scenario where there is only 1 GPU per process. When it comes to
  // multiple GPUs per process, this part may need to redesigned.
  std::unordered_map<std::string, std::vector<std::shared_ptr<NCCLComm>>>
      devNCCLCommMap_;

  // The NCCL communicators currently in process of being initialized.
  std::unordered_map<std::string, std::vector<std::shared_ptr<NCCLComm>>>
      inInitializationCommMap_;

  // Map from ncclUniqueId to appropriate communicator.
  std::unordered_map<std::string, std::vector<std::shared_ptr<NCCLComm>>>
      ncclIdToCommMap_;

  // Mutex to guard maps like devNCCLCommMap_ and ncclIdToCommMap_.
  std::mutex mutex_;

  // Watchdog thread which looks for errors on the cached NCCL communicators.
  std::thread ncclCommWatchdogThread_;

  std::thread onCompletionHookThread_;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminateProcessGroup_;

  // Whether there are hooks pending to be fired
  std::atomic<bool> hasPendingHooks_;

  // Mutex to Guard workMetaList_
  std::mutex workMetaListMutex_;

  // Condition Variable for watchdog thread sleep
  std::condition_variable workMetaListCV_;

  // Vector to Store WorkNCCL pointers
  std::list<ProcessGroupNCCL::WorkNCCL> workMetaList_;

  // Mutex to Guard workMetaList_
  std::mutex completedWorkListMutex_;

  // Condition Variable for watchdog thread sleep
  std::condition_variable completedWorkListCV_;

  std::list<ProcessGroupNCCL::WorkNCCL> completedWorkList_;

  // Add Work Pointer to workVector
  void workEnqueue(c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>);

  // The CUDA streams used by NCCL kernels
  std::unordered_map<std::string, std::vector<at::cuda::CUDAStream>>
      ncclStreams_;

  // The CUDA events used to sync NCCL streams
  std::unordered_map<std::string, std::vector<at::cuda::CUDAEvent>> ncclEvents_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  int coalescing_state_ = 0;

  // Stores device indexes for all collectives run inside a coalescing block
  std::vector<std::vector<at::Device>> coalescedDevices_;

  // Stores communicators for all collectives run inside a coalescing block
  std::vector<std::vector<std::shared_ptr<NCCLComm>>> coalescedComms_;

  // map from the key: "group name + pg counter (ID)" to the
  // unique NCCL ID count. This needs to be group and pg specific
  //
  // For each process group, we need a uniform unique NCCL ID counter to ensure
  // that NCCL operation in this process group can be completed successfully.
  // Since each process group ID belongs to a group name, the key to this map
  // is a combination of group name and ProcessGroupNCCL ID.
  static std::unordered_map<std::string, ssize_t> pgUniqueNCCLIDCnt_;

  // map from group name to the pg counter (ID) within that group
  //
  // For each group with the "group name" (which is the key), we need to
  // keep track of a unique process group ID when creating a new
  // ProcessGroupNCCL for this "group name". Therefore, the value of this
  // map keeps the unique ProcessGroupNCCL's ID for a specific group with
  // the "group name". The reason we need a per-group process group ID counter
  // is that different group can have different ranks and we need ensure that
  // each group has its own uniform process group ID for all its ranks.
  static std::unordered_map<std::string, ssize_t> processGroupCounterMap_;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  // Whether or not the workCleanupThread is used to perform async error
  // handling.
  ErrorHandlingMode asyncErrorHandling_ = NoHandling;

  // Whether or not to enable timeout root cause analysis.
  bool desyncDebug_;

  // Whether or not to create start CUDAEvent and enable timing for start
  // and end events. Note that enableTiming_ is always true if desyncDebug_
  // is set to true.
  std::atomic<bool> enableTiming_;

  // Whether or not TORCH_NCCL_AVOID_RECORD_STREAMS was set
  bool avoidRecordStreams_ = false;

  // Set of communicators that this process group has aborted and their
  // ncclUniqueId has been written to the store. We don't need a lock
  // for this map since only the watchdog thread accesses this set. The
  // set contains the string representation of ncclUniqueId.
  std::unordered_set<std::string> abortedComms_;

  // The number of active ncclGroupStart() calls. This counter will be increased
  // by 1 when ncclGroupStart() is called and decreased by 1 when ncclGroupEnd()
  // is called.
  static thread_local uint64_t ncclActiveGroupCounter_;

  // Counting for the sequential number of NCCL collective call.
  uint64_t seq_{0};

  std::exception_ptr watchDogException_ = nullptr;

#ifdef USE_NCCL_WITH_UCC
  // ProcessGroupUCC shared library handle and ProcessGroup pointer
  static std::shared_ptr<at::DynamicLibrary> uccLib_;
  c10::intrusive_ptr<Backend> uccPG_;
#endif
};

} // namespace c10d

#endif // USE_C10D_NCCL
