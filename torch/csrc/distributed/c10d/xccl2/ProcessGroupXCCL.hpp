// ProcessGroupXCCL: an in-tree c10d::Backend backed by the torchcomms XCCL
// engine. This is a port of torchcomms' TorchCommXCCL collapsed directly onto
// c10d::Backend -- the upstream TorchComm/TorchCommBackend/BackendWrapper
// layers are removed, and the collective methods use the c10d option objects
// (c10d::BroadcastOptions, c10d::ReduceOp, ...) directly rather than the
// torchcomms-specific option/ReduceOp types. This mirrors how nccl2 ported
// TorchCommNCCL.
//
// Namespace: the class lives in c10d::xccl2, which is what keeps it distinct
// from the pre-existing c10d::ProcessGroupXCCL in third_party/torch-xpu-ops.

#pragma once

#ifdef USE_C10D_XCCL

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/xpu/XPUEvent.h>
#include <c10/xpu/XPUStream.h>
#include <oneapi/ccl.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <torch/csrc/distributed/c10d/xccl2/Batch.hpp>
#include <torch/csrc/distributed/c10d/xccl2/WorkXCCL.hpp>
#include <torch/csrc/distributed/c10d/xccl2/XcclApi.hpp>

namespace c10d::xccl2 {

// Hint key names for XCCL backend configuration
constexpr std::string_view kHintIsHighPriorityStream =
    "is_high_priority_stream";
constexpr std::string_view kHintMaxEventPoolSize = "max_event_pool_size";

constexpr size_t kDefaultMaxEventPoolSize = 1000;

// Custom exception class for better error handling
class XCCLException : public std::exception {
 public:
  XCCLException(
      XcclApi& api,
      const std::string& message,
      onecclResult_t result);

  const char* what() const noexcept override;
  [[nodiscard]] onecclResult_t getResult() const noexcept;

 private:
  std::string message_;
  onecclResult_t result_;
};

#define XCCL_CHECK(xccl_api, call, err_str)            \
  do {                                                 \
    onecclResult_t status = call;                      \
    if (status != onecclSuccess) {                     \
      throw XCCLException(*xccl_api, err_str, status); \
    }                                                  \
  } while (0)

// Ignore variant for use in destructors - logs errors instead of throwing
#define XCCL_CHECK_IGNORE(xccl_api, call, err_str)                         \
  do {                                                                     \
    onecclResult_t status = call;                                          \
    if (status != onecclSuccess) {                                         \
      LOG(ERROR) << "[TC] " << err_str << ": "                             \
                 << xccl_api->getErrorString(status) << " at " << __FILE__ \
                 << ":" << __LINE__;                                       \
    }                                                                      \
  } while (0)

class TORCH_API ProcessGroupXCCL : public ::c10d::Backend {
 public:
  static constexpr std::string_view kBackendName = "xccl2";

  // Unlike nccl2 -- which reuses ::c10d::ProcessGroupNCCL::Options because that
  // type already exists in core -- there is no core-side XCCL options type to
  // reuse (the torch-xpu-ops backend does not define one), so the backend
  // carries its own. `hints` is the c10d spelling of torchcomms' CommOptions
  // hint map and is consumed by populateXcclConfigFromHints() plus the
  // kHint* keys above.
  struct TORCH_API Options : ::c10d::Backend::Options {
    explicit Options(
        bool is_high_priority_stream = false,
        std::chrono::milliseconds timeout = ::kBackendDefaultTimeout);

    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false,
        std::chrono::milliseconds timeout = ::kBackendDefaultTimeout) {
      return c10::make_intrusive<Options>(is_high_priority_stream, timeout);
    }

    bool is_high_priority_stream;
    std::unordered_map<std::string, std::string> hints;
  };

  // c10d-style constructor: the XCCL communicator is bootstrapped lazily, on
  // the first collective (or via eagerConnectSingleDevice / bound_device_id),
  // matching c10d's device-binding model -- unlike torchcomms which took an
  // eager init(device).
  ProcessGroupXCCL(
      c10::intrusive_ptr<::c10d::Store> store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());
  ~ProcessGroupXCCL() override;

  ProcessGroupXCCL(const ProcessGroupXCCL&) = delete;
  ProcessGroupXCCL(ProcessGroupXCCL&&) = delete;
  ProcessGroupXCCL& operator=(const ProcessGroupXCCL&) = delete;
  ProcessGroupXCCL& operator=(ProcessGroupXCCL&&) = delete;

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
  void startCoalescing() override;
  c10::intrusive_ptr<::c10d::Work> endCoalescing() override;

  void setTimeout(std::chrono::milliseconds timeout) override;
  void eagerConnectSingleDevice(at::Device device) override;
  void shutdown() override;
  void abort() override;
  ::c10d::ErrorType getError() override;

  void registerAbortHook(int64_t hook_id, ::c10d::AbortHook hook) override;
  void unregisterAbortHook(int64_t hook_id) override;

  c10::intrusive_ptr<::c10d::Backend> split(
      const c10::intrusive_ptr<::c10d::Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<::c10d::Backend::Options>& opts) override;

  bool supportsSplitting() const override {
    return true;
  }

  // ---- caching-allocator segment registration ----
  // Called by XCCLCachingAllocatorHook, possibly from an allocator thread, so
  // both take memory_registration_mutex_.
  void register_address(void* addr, size_t len);
  void deregister_address(void* addr);

  // ---- accessors used by friend classes (work) ----
  XcclApi* getXcclApi() const {
    return xccl_api_.get();
  }
  // Overrides the XcclApi implementation; used by tests to inject a fake.
  void setXcclApi(std::shared_ptr<XcclApi> api) {
    xccl_api_ = std::move(api);
  }
  const at::Device& getDevice() const {
    return device_;
  }
  std::string_view getCommName() const {
    return name_;
  }
  // Underlying host onecclComm_t as an opaque integer pointer.
  int64_t getCommPtr() const;

  friend class WorkXCCL;

 protected:
  const std::shared_ptr<XcclEventPool>& getEventPool() const {
    return event_pool_;
  }
  // NOTE: oneCCL has no communicator abort (onecclCommAbort is declared
  // CCL_C_NOT_IMPLEMENTED, i.e. calling it is a compile-time error), so this
  // tears down with onecclCommDestroy instead. See XcclApi.cpp.
  void abortXcclComm();

  enum class CommState {
    NORMAL,
    ERROR,
    TIMEOUT,
  };

  std::atomic<CommState> comm_state_{CommState::NORMAL};

  onecclDataType_t getXcclDataType(const at::Tensor& tensor);
  c10::intrusive_ptr<WorkXCCL> createWork(
      c10::xpu::XPUStream stream,
      std::chrono::milliseconds timeout,
      const std::vector<at::Tensor>& inputTensors = {});
  c10::intrusive_ptr<WorkXCCL> createWork(
      c10::xpu::XPUStream stream,
      std::chrono::milliseconds timeout,
      const at::Tensor& inputTensor);

 private:
  // RAII helper that cleans up oneCCL premul-sum reduction ops. Built from a
  // c10d::ReduceOp (the premul factor is read from its supplement).
  struct RedOpRAII {
    /* implicit */ RedOpRAII(onecclRedOp_t op);
    explicit RedOpRAII(
        const ::c10d::ReduceOp& op,
        onecclComm_t comm,
        const onecclDataType_t dataType,
        std::shared_ptr<XcclApi> xccl_api);

    RedOpRAII() = delete;
    RedOpRAII(const RedOpRAII&) = delete;
    RedOpRAII& operator=(const RedOpRAII&) = delete;

    RedOpRAII(RedOpRAII&& other) noexcept
        : xcclRedOp_(other.xcclRedOp_),
          comm_(other.comm_),
          xccl_api_(std::move(other.xccl_api_)) {
      other.comm_ = nullptr;
    }
    RedOpRAII& operator=(RedOpRAII&& other) noexcept {
      if (this != &other) {
        if (comm_ && xccl_api_) {
          XCCL_CHECK_IGNORE(
              xccl_api_,
              xccl_api_->redOpDestroy(xcclRedOp_, comm_),
              "XCCL redOpDestroy failed");
        }
        xcclRedOp_ = other.xcclRedOp_;
        comm_ = other.comm_;
        xccl_api_ = std::move(other.xccl_api_);
        other.comm_ = nullptr;
      }
      return *this;
    }
    ~RedOpRAII();

    operator onecclRedOp_t() const {
      return xcclRedOp_;
    }

    onecclRedOp_t xcclRedOp_{onecclMaxRedOp};
    onecclComm_t comm_{nullptr};
    std::shared_ptr<XcclApi> xccl_api_;
  };

  // Lazy, one-time bootstrap of the XCCL communicator on `device`. Subsequent
  // calls validate the same device. Replaces torchcomms' eager init(device).
  void ensureInitialized(at::Device device);
  void init(at::Device device);
  // Adopts an already-created communicator (from split()) instead of
  // bootstrapping one through the Store.
  void initFromComm(
      onecclComm_t comm,
      at::Device device,
      std::shared_ptr<XcclApi> xccl_api);
  void finalize();
  void initXcclResources();

  void attachMemoryHook();
  void detachMemoryHook();

  // Internal XCCL engine helpers (port of TorchCommXCCL). These take c10d
  // option fields directly (c10d::ReduceOp + resolved timeout/root/async),
  // not torchcomms-specific option objects.
  c10::intrusive_ptr<WorkXCCL> sendImpl(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> recvImpl(
      at::Tensor& tensor,
      int src,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> broadcastImpl(
      at::Tensor& tensor,
      int root,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> all_reduce(
      at::Tensor& tensor,
      const ::c10d::ReduceOp& op,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> reduceImpl(
      const at::Tensor& tensor,
      int root,
      const ::c10d::ReduceOp& op,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> allGatherSingleImpl(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ::c10d::ReduceOp& op,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> reduceScatterSingleImpl(
      at::Tensor& output,
      const at::Tensor& input,
      const ::c10d::ReduceOp& op,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> allToAllSingleImpl(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> barrierImpl(
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> scatterImpl(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      std::chrono::milliseconds timeout);
  c10::intrusive_ptr<WorkXCCL> gatherImpl(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      std::chrono::milliseconds timeout);

  // Resolve a c10d per-op timeout (kUnsetTimeout -> communicator default).
  std::chrono::milliseconds operationTimeout(
      std::chrono::milliseconds opt_timeout) const;

  size_t wordSize(onecclDataType_t type) const;
  RedOpRAII getXcclReduceOp(
      const ::c10d::ReduceOp& op,
      onecclComm_t comm,
      const onecclDataType_t dataType);
  void timeoutWatchdog() noexcept;
  void checkInitialized() const;
  void checkAndAbortIfTimedOutOrError();
  [[noreturn]] void throwAsyncError(bool abort_comm = true);
  void checkWorkQueue();
  void enqueueWork(
      c10::intrusive_ptr<WorkXCCL> work,
      c10::xpu::XPUStream stream);
  c10::xpu::XPUStream getOperationStream(bool async_op);
  void ensureTensorContiguous(const at::Tensor& tensor);
  void checkTensorDevice(const at::Tensor& tensor) const;
  void checkTensorsDevice(const std::vector<at::Tensor>& tensors) const;
  void runAbortHooks();

  // Member variables (port of TorchCommXCCL).
  onecclComm_t xccl_comm_{};
  at::Device device_;
  int comm_size_{};
  // NOTE: the rank is stored in the inherited c10d::Backend::rank_ (set in the
  // ctor and refreshed from XCCL in initXcclResources). The ported engine code
  // reads/writes `rank_` directly, which resolves to that protected member.
  std::optional<c10::xpu::XPUStream> internal_stream_;
  std::optional<at::xpu::XPUEvent> dependency_event_;
  at::DataPtr barrier_buffer_;
  enum class InitializationState {
    UNINITIALIZED,
    INITIALIZED,
    FINALIZED,
  } init_state_{InitializationState::UNINITIALIZED};

  c10::intrusive_ptr<::c10d::Store> store_;
  uint64_t bootstrap_generation_{0};

  std::shared_ptr<XcclApi> xccl_api_;

  // Shared with every WorkXCCL created here, which may outlive this backend.
  // Sized in the constructor from the kHintMaxEventPoolSize hint.
  const std::shared_ptr<XcclEventPool> event_pool_;

  WorkXCCLQueue workq_;

  std::thread timeout_thread_;
  std::atomic<bool> shutdown_{false};
  std::condition_variable timeout_cv_;
  std::mutex timeout_mutex_;

  bool is_high_priority_stream_{false};
  bool abort_process_on_timeout_or_error_{true};
  std::string name_;

  c10::intrusive_ptr<Options> options_c10d_;

  // Abort hooks (c10d::Backend API; storage was in torchcomms' TorchCommBackend
  // base, folded in here).
  std::unordered_map<int64_t, ::c10d::AbortHook> abortHooks_;

  // Active coalescing batch (port of BackendWrapper). Engaged between
  // startCoalescing() and endCoalescing(); send()/recv() append into it.
  std::optional<BatchSendRecv> coalescing_batch_;
  c10::intrusive_ptr<WorkXCCL> coalesced_work_;

  // Caching-allocator segments currently registered with xccl_comm_, keyed by
  // segment base address. Touched from allocator threads, hence the mutex.
  struct RegistrationHandle {
    void* handle{nullptr};
    size_t len{0};
  };
  std::map<void*, RegistrationHandle, std::less<>> memoryRegistrationHandles_;
  std::mutex memory_registration_mutex_;
};

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
