#pragma once

#ifdef USE_C10D_NCCL

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/NCCLCommon.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/cuda/CUDAEventCache.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp>

#include <ATen/DynamicLibrary.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/MemPool.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/custom_class.h>

namespace c10d {
#define PRINT_COLLECTIVE_HASH_SIGNATURE(phase, opType, numel, hashValue)      \
  LOG(WARNING) << logPrefix() << "Hash of " << phase << " to NCCL " << opType \
               << " with size " << numel << " is " << hashValue;

// If set, ProcessGroupNCCL registers postAlloc and preFree hooks to cuda cache
// allocator so that whenever a tensor is allocated or freed, ProcessGroupNCCL
// can register/deregister the tensor on all available NCCL communicators.
static std::vector<std::string> TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK =
    {"TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK",
     "NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"};

#if defined(__linux__)
struct DumpPipe {
  DumpPipe(int rank, const std::string& fileStem, int traceBufferSize) {
    if (fileStem.empty() || traceBufferSize <= 0) {
      return;
    }
    std::string filename = c10::str(fileStem, rank, ".pipe");
    TORCH_CHECK(
        unlink(filename.c_str()) != -1 || errno == ENOENT,
        "Error removing existing named pipe ",
        filename,
        ", Error: ",
        std::strerror(errno));
    TORCH_CHECK(
        mkfifo(filename.c_str(), 0666) != -1,
        "Error creating named pipe ",
        filename,
        ", Error: ",
        std::strerror(errno));
    fd_ = open(filename.c_str(), O_RDONLY | O_NONBLOCK);
    LOG(INFO) << "Pipe file " << filename
              << " has been opened, write to it to trigger NCCL Debug Dump.";
    TORCH_CHECK(fd_ != -1, "Error opening named pipe ", filename);
  }
  bool shouldDump() {
    if (fd_ == -1) {
      return false;
    }
    // NOLINTNEXTLINE(*array*)
    char buf[128]{};
    // non-blocking from O_NONBLOCK above.
    // Ignore EINTR because we already will poll this
    // again later.
    ssize_t bytesRead = read(fd_, &buf, 128);
    return bytesRead > 0;
  }
  ~DumpPipe() {
    if (fd_ != -1) {
      close(fd_);
    }
  }

 private:
  int fd_ = -1;
};
#else
struct DumpPipe {
  DumpPipe(int rank) {}
  bool shouldDump() {
    return false;
  }
};
#endif

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
//   // At this point, NCCL kernel has already been queued successfully
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
        std::string pgUID,
        std::string pgDesc,
        at::Device& device,
        int rank,
        OpType opType,
        uint64_t seq,
        bool isP2P = false,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt,
        bool enableTiming = false,
        bool cudaEventCacheEnabled = false,
        DebugLevel distDebugLevel = DebugLevel::Off);
    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    WorkNCCL(const WorkNCCL& w);

    ~WorkNCCL() override = default;

    // Checks if the NCCL kernel has started to execute.
    bool isStarted();

    // Checks if request has completed. In this specific case of NCCL, it checks
    // if the NCCL operation has completed on the GPU in its own NCCL stream.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for NCCL work if timeout is not set.
    // Otherwise, it will block the CPU thread until the NCCL work is completed
    // or timed out. If timeout, exception will be thrown.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void blockCurrentStream() override {
      synchronize();
    }

    void abort() override;

    // Let current stream wait on the completion of the NCCL work
    // Throws on exceptions.
    void synchronize() override;

    // Synchronize streams by blocking each on the NCCL stream
    void synchronizeStream();

    // Helper function to handle exception (throw if needed).
    void handleException(ErrorHandlingMode asyncErrorHandling);

    // Helper function that checks if the NCCL kernels have finished
    // execution on the GPUs
    bool finishedGPUExecution();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    // Get a Future result of each work (e.g. success, different error types).
    // instead of the tensor output.
    c10::intrusive_ptr<c10::ivalue::Future> getFutureResult() override;

    bool hasFutureResult() const override {
      return true;
    }

    float getDuration() const override;

    uint64_t getSequencenumber() const override;

    std::chrono::milliseconds getTimeout() const override {
      return opTimeout_;
    }

    const std::string& logPrefix() const;

    // Helper function that sets an exception_ptr on the WorkNCCL object.
    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkNCCL object has timed out
    // and False otherwise.
    // In case of timeout, set exception on the WorkNCCL object.
    bool checkTimeout(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt);

    // Print the traceback of the collective at call time
    void printTraceback() const;

    std::string getTraceback() const;

    std::vector<at::Tensor> result() override;

   protected:
    // The process group unique id
    std::string pgUID_;

    // The process group description
    std::string pgDesc_;

    // The cached list of CUDA devices to operate on
    at::Device device_;

    // The start CUDA event of NCCL operator tracking this work item. These
    // start CUDA events are needed by desync debugging if enabled.
    std::shared_ptr<at::cuda::CUDAEvent> ncclStartEvent_;

    // The end CUDA event of NCCL operator tracking this work item.
    std::shared_ptr<at::cuda::CUDAEvent> ncclEndEvent_;

    // The NCCL communicator used for this work item.
    std::shared_ptr<NCCLComm> ncclComm_;

    // whether this work is a barrier op
    bool isBarrierOp_{false};

    // Clone of blockingWait_ from ProcessGroupNCCL.
    bool blockingWait_{false};

    // Clone of opTimeout_ from ProcessGroupNCCL.
    std::chrono::milliseconds opTimeout_{};

    // Ephemeral timeouts are owned by exactly one work,
    // and reset after that work completes.
    // There may be more than one ephemeral timeout active at the same time,
    // and this variable is used to track the ownership of ephemeral timeout.
    std::chrono::milliseconds ownedEphermeralTimeout_ =
        std::chrono::milliseconds(0);

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    // Record the sequential number of collective or p2p.
    uint64_t seq_;
    bool isP2P_;

    // Indicates if the nccl start event has been updated to the store trace.
    // This will be used by desync debug.
    bool startTraceUpdated_{false};

    // Record collective sizes for debug. We only record the size on the first
    // device as multi-device per process is deprecated
    size_t numelIn_ = 0;
    size_t numelOut_ = 0;

    // Wrapper method for the static checkForNCCLErrors which can be overridden
    // for tests.
    virtual std::exception_ptr checkForNCCLErrors();

    friend std::ostream& operator<<(
        std::ostream& output,
        const WorkNCCL& workNCCL);

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

    // Stashing implementation helper.
    // Stores references to participating non-output tensors (ie inputs,
    // flattened intermediates).
    // We'll clear this list in synchronizeStream, just after user-facing
    // stream(s) are synced with the nccl work stream(s).
    // By keeping these refs (as well as outputs_) alive until after the
    // collective's work rejoins the user-facing streams, we achieve
    // caching allocator safety without any recordStream calls.
    // For in-place collectives, some refs stashed here may alias outputs_,
    // but that doesn't do any harm.
    std::shared_ptr<TensorShelf> stashed_for_allocator_safety_;

    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;

    // the future result (e.g., success or failure) of the work
    c10::intrusive_ptr<at::ivalue::Future> futureWorkResult_;

    bool timingEnabled_;
    // unique id used to tell the trace buffer that this
    // work has completed
    std::optional<uint64_t> trace_id_;
    std::optional<uint64_t> trace_reset_epoch_;
    DebugLevel distDebugLevel_;
    std::string logPrefix_;
    friend class ProcessGroupNCCL;
  };

  using Options = NCCLOptions;

  // Helper class related to TORCH_NCCL_DESYNC_DEBUG
  class DesyncDebugger {
   public:
    // Initialize and enable DesyncDebugger
    void init(
        int rank,
        int size,
        int globalRank,
        int pgId,
        c10::intrusive_ptr<Store> store);

    // Run desync debug. This function is called by watchdog at time of timeout.
    void run();

    // Buffer a work start for the next flush().
    void logWorkStart(WorkNCCL& work);

    // Buffer a work end for the next flush().
    void logWorkEnd(WorkNCCL& work);

    // Publish buffered start/end updates. Each trace key retains only its
    // latest value, so one write per key per pass replaces one per work. Normal
    // watchdog passes call this after releasing workMetaListMutex_. Timeout and
    // error reporting may call it while holding the mutex so state is published
    // before the report or dump.
    void flush();

   private:
    // Whether desync debug is enabled.
    // If false, all functions are no-op.
    bool enabled_{false};

    // Buffered until flush().
    std::optional<std::pair<uint64_t, std::string>> pendingStart_;
    std::optional<std::pair<uint64_t, std::string>> pendingEnd_;

    // From ProcessGroupNCCL
    int rank_;
    int size_;
    int globalRank_;
    int pgId_;

    // Reference to the store so that we can log start/end event.
    c10::intrusive_ptr<Store> store_;

    // The store keys to trace the last NCCL collective kernel CUDA events -
    // start event and end event respectively. These are used to do desync root
    // cause analysis.
    std::string traceKeyStart_;
    std::string traceKeyEnd_;
  };

  // Class that runs as a separate thread aside from watchdog
  // thread because we need to check the heartbeat from watchdog thread
  // so that when we get stuck in some NCCL/CUDA calls,
  // we can dump the debugging information and abort the process.
  class HeartbeatMonitor {
   public:
    HeartbeatMonitor(ProcessGroupNCCL* pg);
    virtual ~HeartbeatMonitor() = default;

    // Start the heartbeat monitor thread.
    void start();

    // Join the heartbeat monitor thread.
    void join();

    // Run the actual loop to check watchdog heartbeat.
    virtual void runLoop();

    // Set the terminal flag and notify the heartbeat monitor thread to stop.
    void stop();

    // Set the last update time of watchdog thread.
    void setLastWorkListUpdateTime(
        std::chrono::time_point<std::chrono::steady_clock> time);

    int getDumpTimeout() const;

    // Util function to get the timeout error message
    std::string getNCCLWatchdogTimeoutErrorMsg(const std::string& extraMsg);

    // Util function to get the timeout exit message
    std::string getNCCLWatchdogTimeoutExitMsg(const std::string& exitReason);

   protected:
    // We need to keep a reference to the PG instance so that we can access
    // the member functions of the PG instance. We store a raw pointer on
    // purpose because the heartbeat monitor thread now still lives within the
    // lifetime of the PG instance.
    ProcessGroupNCCL* pg_;

   private:
    // Whether or not to print C++ stack traces to logs on unclean shutdown.
    bool logCppStackOnUncleanShutdown_;

    // The time interval used for deciding whether there is no watchdog
    // heartbeat.
    int heartbeatTimeoutInSec_;

    // timeout for the dump to finish.
    int waitTimeoutDumpInMilSec_;

    // Interval of check coordinated signals in ProcessGroupNCCL from other
    // ranks e.g., trigger the dump of the debugging info for timeout when
    // notified.
    int coordCheckIntervalMilSec_;

    // We gate the heartbeat monitor thread so that we can roll it out
    // gradually.
    bool watchdogHeartbeatMonitorEnabled_;

    // Monitor thread which checks the heartbeat of Watchdog thread.
    // If the monitor thread finds there is no heartbeat, it will dump debug
    // info and then kill the watchdog thread to avoid hang.
    std::thread ncclHeartbeatMonitorThread_;

    // Whether or not we should terminate the heartbeat monitoring threads.
    std::atomic<bool> terminateHeartbeatMonitorThread_{false};

    // Condition Variable for monitor thread to wake up early
    std::condition_variable monitorWakeUpCV_;

    // Whether or not to dump debug info on exception including both watchdog
    // timeout and nccl errors.
    bool dumpOnTimeoutOrEx_;

    // Mutex to Guard monitorWakeUpCV_
    std::mutex monitorMutex_;

    // The last update time of WorkList inside watchdog thread.
    std::chrono::time_point<std::chrono::steady_clock> lastWorkListUpdateTime_;
  };

  // Class that runs as a side thread to check whether the NCCL collective
  // is timed out or errors on the cached NCCL communicators.
  class Watchdog {
   public:
    Watchdog(ProcessGroupNCCL* pg);
    virtual ~Watchdog() = default;

    // Start the watchdog thread.
    void start();

    // Join the watchdog thread.
    void join();

    // Function that runs as part of a separate thread and checks for errors on
    // NCCL communicators. We need a separate thread to check for NCCL errors
    // since we can't rely on the user calling certain methods like wait(),
    // isCompleted() etc. to detect and remediate errors. In addition to this,
    // we need a mechanism to safely abort and remove NCCL communicators from
    // our cache. This can be done cleanly by having a thread for the
    // ProcessGroupNCCL class. Attempting to modify the communicator cache from
    // the WorkNCCL class might run into issues with object lifetime since the
    // ProcessGroupNCCL object might get destroyed before the WorkNCCL object.
    void run();

    // Watchdog's inside loop.
    // Takes care of cleaning up completed work, and aborting upon failure or
    // timeout.
    void runLoop();

    // Notify the loop inside watchdog.
    void notify();

    void checkAndSetRemoteError();

    // A helper function to get the src rank of a signal from the Store. This is
    // nonblocking function returning -1 if the signal is not available yet.
    int getSignalSrcRank(
        c10::intrusive_ptr<Store>& store,
        const std::string& signal);

    uint64_t getHeartbt() const;

    void setDesyncDebug(bool desyncDebug);

   private:
    std::thread ncclCommWatchdogThread_;

    // We need to keep a reference to the PG instance so that we can access
    // the member functions of the PG instance. We store a raw pointer on
    // purpose because the watchdog thread now still lives within the
    // lifetime of the PG instance.
    ProcessGroupNCCL* pg_;

    // Whether the NCCL watchdog should rethrow CUDA errors.
    bool rethrowCUDAErrors_ = false;

    // Opt-in for the timeout teardown below. Read once at construction.
    bool tearDownOnTimeout_ = false;

    // Whether a NCCL timeout exception should be rethrown in Watchdog::run()
    // to terminate the process. Written in runLoop() and read in run()'s catch,
    // which runs it, so both touch it from the single watchdog thread.
    bool rethrowTimeoutException_ = false;

    std::exception_ptr watchDogException_ = nullptr;

    // Condition Variable for watchdog thread sleep
    std::condition_variable workMetaListCV_;

    // Heartbeat of watchdog thread.
    std::atomic_uint64_t heartbeat_;

    // Whether or not to propagate detected errors to all ranks in the same PG
    // through TCPStore.
    bool propagatePgError_;

    // Whether or not to enable timeout root cause analysis.
    bool desyncDebug_;

    DesyncDebugger desyncDebugger_;
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
      c10::intrusive_ptr<Store> store,
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
      : ProcessGroupNCCL(store, rank, size, std::move(options)) {}

  ~ProcessGroupNCCL() override;

  // This function returns a local uid for ProcessGroupNCCL.
  uint64_t getUid() {
    return static_cast<uint64_t>(local_id_);
  }

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return c10::static_intrusive_pointer_cast<Backend::Options>(options_);
  }

  const std::string getBackendName() const override {
    return std::string(NCCL_BACKEND_NAME);
  }

  bool supportsSplitting() const override {
    return true;
  }

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

  void setTimeout(std::chrono::milliseconds timeout) override {
    options_->timeout = timeout;
  }

  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  void startTimeEstimate() override;

  float endTimeEstimate() override;

  // For specifying a composite optype, such as ALLGATHER and REDUCE_SCATTER
  c10::intrusive_ptr<Work> endCoalescing(OpType optype);

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> _broadcast_oop(
      at::Tensor& outputTensors,
      at::Tensor& inputTensors,
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
      at::Tensor& outputTensors,
      at::Tensor& inputTensors,
      const ReduceOptions& opts = ReduceOptions());

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> all_gather_single(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> all_gather_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_single(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_single_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> all_to_all_single(
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

  int64_t getCommPtr();

  void groupStart();

  void groupEnd();

  void groupEndNonblocking(const std::shared_ptr<NCCLComm>& comm);

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> gather_single(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  // Unsupported Ops
  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override;

  // Return the total number of splits the communicators held by this process
  // group have performed.  Counts ncclCommCreateFromRanks() for ncclx v2.21.5+
  uint64_t getCommSplitCounter() const;

  void registerOnCompletionHook(
      std::function<void(std::shared_ptr<WorkInfo>)>&& hook) override;
  void waitForPendingWorks() override;

  void enableCollectivesTiming() override;
  bool collectivesTimingEnabled() const override {
    return enableTiming_.load();
  }

  c10::intrusive_ptr<Backend> split(
      const c10::intrusive_ptr<Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<Backend::Options>& opts) override;

  c10::intrusive_ptr<Backend> merge(
      const c10::intrusive_ptr<Store>& store,
      const c10::intrusive_ptr<Backend::Options>& opts,
      const int& rank,
      const int& size) override;

  // Helper function for iteratively aborting communicators in the provided map
  void abortCommsFromMap(
      std::unordered_map<std::string, std::shared_ptr<NCCLComm>>& ncclCommsMap,
      const std::optional<std::string>& abortReason);

  c10::intrusive_ptr<intra_node_comm::IntraNodeComm> initIntraNodeComm();

  // Destroy (shutdown) this backend -- normal exit.
  void shutdown() override;

  // Provides an API to abort the ProcessGroup (similar to ncclCommAbort)
  // instead of relying on ProcessGroupNCCL destructor.
  void abort() override;

  void eagerConnectSingleDevice(at::Device device) override;

  void performNocolorSplit(at::Device device);

  // If all comms on this PG are fully initialized, return true.
  bool isInitialized() override;

  ErrorType getError() override;

  bool supportsShrinking() const override {
#ifdef NCCL_HAS_COMM_SHRINK
    return true;
#else
    return false;
#endif
  }

  // Backend-style shrink override that returns a Backend instance.
  c10::intrusive_ptr<Backend> shrink(
      const std::vector<int64_t>& ranks_to_exclude,
      int shrink_flags = 0,
      const c10::intrusive_ptr<Backend::Options>& opts_override =
          nullptr) override;

  std::shared_ptr<c10::Allocator> getMemAllocator() override;

  // Allocate tensor from communication-optimized memory pool
  at::Tensor allocateTensor(long size, at::TensorOptions options = {}) override;

  // Whether tensor allocation from NCCL memory pool is supported
  bool supportsTensorAlloc(c10::DeviceIndex deviceIdx) override;

  // Performs NCCL user buffer registration for all buffers in
  // the given MemPool
  void registerMemPool(at::cuda::MemPool* pool, bool symm = false);

  // Performs NCCL user buffer de-registration for all buffers in
  // the given MemPool
  void deregisterMemPool(at::cuda::MemPool* pool);

  // This method adds a temporary extension for the timeout period, applying to
  // collectives issued after this call until the first such collective
  // completes on the GPU. Existing work retains its original timeout.
  void addEphemeralTimeout(const std::chrono::milliseconds& timeout) override;

  void setEnableNanCheck(bool enableNanCheck);

  // APIs related to memory offload (require NCCL 2.29.7+ at runtime)
  void suspend() override;

  void resume() override;

  std::unordered_map<std::string, uint64_t> getMemoryStats() override;

 protected:
  uint64_t getWatchdogHeartbt() const;

  // Instance of the heartbeat monitor thread.
  std::unique_ptr<HeartbeatMonitor> heartbeatMonitor_;

  // Instance of the watchdog thread.
  std::unique_ptr<Watchdog> watchdog_;

  // Helper that broadcasts nccl unique ID to all ranks through the store
  void broadcastUniqueNCCLID(
      ncclUniqueId* ncclID,
      bool isSingleP2POp,
      const std::string& devicesKey,
      int p2pRank);

  // Helper that allgathers nccl unique IDs to all ranks through the store
  void allgatherUniqueNCCLIDs(
      int rootIdx,
      ncclUniqueId* ncclID,
      std::vector<ncclUniqueId>& ncclIDs);

  // Helper that looks up the cached NCCL communicators only
  std::shared_ptr<NCCLComm> getNCCLComm(const std::string& deviceKey);

  std::shared_ptr<NCCLComm> initNCCLComm(
      const std::string& deviceKey,
      at::Device& device,
      OpType opType,
      int p2pRank = 0,
      bool isSendRecvSelf = false);

  // Initialize device-specific state (comm, stream, event, bookkeeping) for a
  // given communicator on this process group instance.
  void initializeDeviceStateForComm(
      const at::Device& device,
      std::shared_ptr<NCCLComm> comm);

  // Wrapper method which can be overridden for tests.
  virtual std::exception_ptr checkForNCCLErrors(
      std::shared_ptr<NCCLComm>& ncclComm);

  // Ensure that if record is True, the work obj will be enqueued via
  // workEnqueue
  virtual c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> initWork(
      at::Device& device,
      int rank,
      OpType opType,
      bool isP2P,
      const char* profilingTitle = nullptr,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {},
      bool record = false);

  // In the timeout case and we will dump debug info such as the NCCL flight
  // recorder to storage. Down the road, if we have more complicated or blocking
  // operations, we might need to use a side thread to do it.
  bool dumpDebuggingInfo(
      bool includeStackTrace = true,
      bool onlyActive = false);

  void dumpExtraDebuggingInfo();

  // Abort all communicators on this rank.
  bool abortComms(const std::optional<std::string>& abortReason = std::nullopt);

  // A helper function to check if nonblocking API mode should be used.
  // Use this helper instead of directly checking `useNonblocking_` variable.
  bool useNonblocking();

 protected:
  int globalRankStart_{};
  int globalRankStride_{};

 private:
  bool eagerInit_{false};
  bool showSerializationWarning_{true};

  // Helper that encapsulates work shared across all collective communication
  // primitives.  The callbacks have the following signatures:
  //
  //    ncclResult_t fn(at::Tensor& input, at::Tensor& output,
  //                    ncclComm_t, at::cuda::CUDAStream&);
  //    void {pre,post}(std::vector<at::cuda::CUDAStream&>);
  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr,
      bool nanCheck = true);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr,
      bool nanCheck = true);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr,
      bool nanCheck = true);

  template <typename Fn>
  c10::intrusive_ptr<Work> collectiveCoalesced(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr);

  // Helper that encapsulates work shared across point-to-point communication
  // primitives. It is the same structure as the helper used for collective
  // communication primitives.
  template <typename Fn>
  c10::intrusive_ptr<Work> pointToPoint(
      at::Tensor& tensor,
      Fn fn,
      int peer,
      OpType opType,
      const char* profilingTitle = nullptr);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> pointToPoint(
      at::Tensor& tensor,
      Fn fn,
      int peer,
      OpType opType,
      PreProcess pre,
      PostProcess post,
      const char* profilingTitle);

  c10::intrusive_ptr<Work> allreduce_impl(
      at::Tensor& tensor,
      const char* profilingTitle = "nccl:all_reduce",
      const AllreduceOptions& opts = AllreduceOptions());

  // Checks for NCCL errors on each of the communicators and returns an
  // appropriate exception_ptr (nullptr if no errors).
  static std::exception_ptr checkForNCCLErrorsInternal(
      std::shared_ptr<NCCLComm>& ncclComm);

  // Return the CUDA device most likely associated with this backend.
  // If we aren't bound to a specific device, there is no strict
  // guarantee that this heuristic is the correct assignment of ranks
  // to GPUs that Python layers use, but in practice it tends to be.
  // Fortunately we don't rely on this for correctness of any tensor
  // operations, just for ancillary uses like barriers.
  at::Device guessDeviceForRank() const;

  // Destroys initialized NCCL communicators in devNCCLComMap_ given by input
  // key. Throws if there are no communicators to destroy. Also removes
  // communicators from the cache and clears used device indices.
  void destroyNCCLComms(const std::string& devNCCLCommMapKey);

  void runHookLoop();

  // Generates a prefix that is unique to this process group and rank, for
  // disambiguating logs
  std::string createLogPrefix() const;

  // Returns the unique prefix created in createLogPrefix
  const std::string& logPrefix() const;

  // Returns the global rank of the device. This function assumes that users
  // always create a default global process group(PG) which includes all
  // devices. It is called in the constructor of ProcessGroupNCCL, so it always
  // return the rank_ of the very first PG created, aka, default global PG.
  const int& globalRank() const;

  const c10::intrusive_ptr<Store>& globalStore() const;

  // Returns the global ranks of a PG.
  const std::vector<uint64_t>& groupRanks() const;

  // Util function to assign timeout to each work.
  void assignTimeoutToWork(
      const c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work,
      const c10::intrusive_ptr<Options>& option);

  // Broadcast flight-recorder dump signal
  void broadcastDumpSignal();

  // A helper function to broadcast a signal (key) from a src rank to all other
  // ranks using the specified store.
  void broadcastSignal(
      c10::intrusive_ptr<Store>& store,
      const std::string& signal,
      int srcRank);

 protected:
  // Function that directly trigger std::abort so that the whole process
  // gets terminated.
  virtual void terminateProcess(const std::string& errMsg);

  // A helper function to wait for a future to complete or timeout.
  // Returns true if the future completes before timeout, false otherwise.
  bool waitForFutureOrTimeout(
      std::future<bool>& fut,
      const std::chrono::milliseconds& timeOutMilSec,
      const std::string& futDescription,
      ::c10d::C10dLoggingData& debugLog,
      bool throwException = false);

  // A helper function to guess the device id of the current rank, based on
  // bounded device or used device. Do not use this function if you already know
  // the device id to operate on.
  c10::DeviceIndex guessDeviceId() const;

  static const int64_t kWatchdogThreadSleepMillis;

  // The store is used to broadcast the NCCL unique ID of rank 0. This store
  // comes with prefix and it is different across ProcessGroup NCCL instances
  // (aka, different ProcessGroups).
  c10::intrusive_ptr<Store> store_;

  // Reference to the store without prefix so that keys are same across all
  // ProcessGroup NCCL instances and (key, value) pairs written to the store are
  // global.
  c10::intrusive_ptr<Store> globalStore_;

  // The lock which protects the write/read of
  // ephemeralTimeoutActive_/ephemeralTimeoutInflight_.
  // TODO(fduwjj): We need to have an audit on all mutexes we are adding here.
  // And consolidate them if possible.
  std::mutex mtxTimeoutExtension_;

  // The ephemeral timeout added on top of existing timeout for works issued
  // before first work finishes.
  std::chrono::milliseconds ephemeralTimeoutActive_ =
      std::chrono::milliseconds(0);

  // The ephemeral timeout addition which has been already applied to work.
  std::chrono::milliseconds ephemeralTimeoutInflight_ =
      std::chrono::milliseconds(0);

  const c10::intrusive_ptr<Options> options_;

  // The number of NCCL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t ncclCommCounter_{0};

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
  // TODO: we probably need a separate map for P2P comms
  std::unordered_map<std::string, std::shared_ptr<NCCLComm>> devNCCLCommMap_;

  // The NCCL communicators currently in process of being initialized.
  std::unordered_map<std::string, std::shared_ptr<NCCLComm>>
      inInitializationCommMap_;

  // Mutex to guard maps like devNCCLCommMap_.
  std::mutex mutex_;

  // Size of ring buffer where we store NCCL Traces for debugging.
  int traceBufferSize_;

  // Stores TORCH_NCCL_DEBUG_INFO_PIPE_FILE
  std::string debugInfoPipeFile_;

  // We gate the cudaEventCache so that we can roll it out gradually.
  std::atomic<bool> cudaEventCacheEnabled_;

  std::thread onCompletionHookThread_;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminateProcessGroup_;

  // Whether there are hooks pending to be fired
  std::atomic<bool> hasPendingHooks_;

  // This is the signal from watchdog threads to indicate whether the monitor
  // thread should dump. Making it static so that it is accessible from all the
  // PGs. With this flag, monitor thread would dump debug info under any one of
  // the three conditions:
  //
  // 1: watchdog thread of any PG detects a collective timeout.
  // 2: timeout signal is received from other ranks through tcpstore.
  // 3: current PG's watchdog heartbeat timeout occurs.
  //
  // Note that only the monitor thread from PG0 will dump the debug info for
  // case one and two so that the debug info is only dumped once.
  static std::atomic<bool> shouldDump_;

  // Mutex to Guard workMetaList_
  std::mutex workMetaListMutex_;

  bool writeDebugInfo_ = false;

  // Vector to store WorkNCCL pointers
  std::list<ProcessGroupNCCL::WorkNCCL> workMetaList_;

  // Mutex to Guard workMetaList_
  std::mutex completedWorkListMutex_;

  // Condition Variable for watchdog thread sleep
  std::condition_variable completedWorkListCV_;

  std::list<ProcessGroupNCCL::WorkNCCL> completedWorkList_;

  // Add Work Pointer to workVector
  void workEnqueue(
      const c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& /*work*/);

  // The CUDA streams used by NCCL kernels
  std::unordered_map<std::string, at::cuda::CUDAStream> ncclStreams_;

  // The CUDA events used to sync NCCL streams
  std::unordered_map<std::string, at::cuda::CUDAEvent> ncclEvents_;

  // Device Indexes used for all collectives in this group
  std::set<c10::DeviceIndex> usedDeviceIdxs_;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  int coalescing_state_ = 0;

  // Stores device indexes for all collectives run inside a coalescing block
  at::Device coalescedDevice_ = at::Device("cuda");

  // Stores communicators for all collectives run inside a coalescing block
  std::shared_ptr<NCCLComm> coalescedComm_ = nullptr;

  // Whether the coalesced calls are sync or async.
  bool coalescedAsync_{};

  // keeps track of input and output tensors when coalescing is in flight.  Will
  // hand over these tensors to WorkNCCL's stash when coalescing is ended.
  TensorShelf coalescedTensors_;

  // Some ops may have completed, but user still hasn't called `work.wait()`.
  // When watchdog detects this, it transfers the TensorShelf from `work` to
  // this `shelves` structure. Next time we execute ProcessGroupNCCL's methods
  // on main thread, we clear the `shelves` in one shot. This is mainly because
  // watchdog (a side thread) unstashing the shelf directly seems to cause some
  // problem.
  std::vector<std::shared_ptr<TensorShelf>> shelvesToUnstash_;
  std::mutex shelvesMutex_;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  // Whether or not the workCleanupThread is used to perform async error
  // handling.
  ErrorHandlingMode asyncErrorHandling_ = NoHandling;

  ErrorType error_ = ErrorType::SUCCESS;

  std::mutex errorMutex_;

  // Whether or not to sleep after an exception is thrown in the watchdog.
  bool sleepAfterException_{};

  // Whether or not to enable nan check for input tensors to collectives.
  bool enableNanCheck_;

  // Whether or not to create start CUDAEvent and enable timing for start
  // and end events. Note that enableTiming_ is always true if desyncDebug_
  // is set to true.
  std::atomic<bool> enableTiming_;

  // Flag to enable the print of hash value of input/output of collectives for
  // verification.
  std::atomic<bool> enableCollectiveHashDebug_;

  // The number of active ncclGroupStart() calls. This counter will be increased
  // by 1 when ncclGroupStart() is called and decreased by 1 when ncclGroupEnd()
  // is called.
  static thread_local uint64_t ncclActiveGroupCounter_;

  // Counting for the sequential number of NCCL collective call.
  // (specifically, how many actual kernels we launched, which differs from
  // op_id_ when coalescing is enabled)
  uint64_t seqCollective_{0};

  // Counting for the sequential number of NCCL P2P calls.
  uint64_t seqP2P_{0};

  // Incrementing counter for logical operations (collective or p2p) issued on
  // the ProcessGroup
  uint64_t op_id_{0};

  // The number of ProcessGroupNCCL created on the current rank.
  size_t local_id_;

  // Identity rank mapping [0, size_), used by groupRanks() when
  // options_->global_ranks_in_group is empty. Filled in the constructor.
  std::vector<uint64_t> defaultRanks_;

  std::string logPrefix_;

  c10::intrusive_ptr<intra_node_comm::IntraNodeComm> intraNodeComm_;

  // Number of devices on this node.
  int localDeviceCount_{0};

  std::shared_ptr<ProcessGroupStatus> pgStatus_ =
      std::make_shared<ProcessGroupStatus>();

  // Internal cached value: use NCCL non-blocking API mode or not.
  // Use `useNonblocking()` method instead of accessing this variable directly.
  std::optional<bool> useNonblocking_{std::nullopt};

  // Communication-optimized memory pool associated with this PG
  std::unique_ptr<at::cuda::MemPool> memPool_ = nullptr;
};

} // namespace c10d

#endif // USE_C10D_NCCL
