// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/Logging.h>
#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp

#include <torch/csrc/comms/TorchComm.hpp>
#include <torch/csrc/comms/TorchCommBackend.hpp>
#include <torch/csrc/comms/TorchCommBatch.hpp>
#include <torch/csrc/comms/device/cuda/CudaApi.hpp>
#include <torch/csrc/comms/nccl/NcclApi.hpp>
#include <torch/csrc/comms/nccl/TorchWorkNCCL.hpp>

namespace torch::comms {

// Hint key names for NCCL backend configuration
constexpr std::string_view kHintHighPriorityStream = "high_priority_stream";
constexpr std::string_view kHintMaxEventPoolSize = "max_event_pool_size";

constexpr size_t kDefaultMaxEventPoolSize = 1000;

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

class TorchCommNCCL : public TorchCommBackend,
                      public std::enable_shared_from_this<TorchCommNCCL> {
 public:
  static constexpr std::string_view kBackendName = "nccl";

  TorchCommNCCL();
  ~TorchCommNCCL() override;

  // Delete copy and move operations
  TorchCommNCCL(const TorchCommNCCL&) = delete;
  TorchCommNCCL(TorchCommNCCL&&) = delete;
  TorchCommNCCL& operator=(const TorchCommNCCL&) = delete;
  TorchCommNCCL& operator=(TorchCommNCCL&&) = delete;

  void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) override;
  void finalize() override;
  int getRank() const override;
  int getSize() const override;
  std::string_view getBackendName() const override;
  std::string_view getCommName() const override;

  // Point-to-Point Operations
  c10::intrusive_ptr<TorchWork> send(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      const SendOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> recv(
      at::Tensor& tensor,
      int src,
      bool async_op,
      const RecvOptions& options = {}) override;

  // Batch P2P Operations
  c10::intrusive_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      const BatchP2POptions& options = {}) override;

  // Collective Operations
  c10::intrusive_ptr<TorchWork> broadcast(
      at::Tensor& tensor,
      int root,
      bool async_op,
      const BroadcastOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      const ReduceOp& op,
      bool async_op,
      const AllReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce(
      const at::Tensor& tensor,
      int root,
      const ReduceOp& op,
      bool async_op,
      const ReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_v(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllGatherSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor& output,
      const at::Tensor& input,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllToAllSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      const AllToAllvSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      const AllToAllOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> barrier(
      bool async_op,
      const BarrierOptions& options = {}) override;

  // Scatter and Gather Operations
  c10::intrusive_ptr<TorchWork> scatter(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      const ScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> gather(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      const GatherOptions& options = {}) override;

  // Communicator Management
  std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) override;

  // Window / one-sided RMA. Requires NCCL 2.29+.
  std::shared_ptr<TorchCommWindow> new_window(
      const std::optional<at::Tensor>& tensor = std::nullopt) override;

  // Fault Tolerance API
  bool supportsReconfigure() const override {
    return true;
  }
  InitHandle getInitHandle() const override;
  c10::intrusive_ptr<TorchWork> reconfigure(
      const ReconfigureOptions& opts) override;
  void abort() override;

  // Friend access for TorchCommNCCL
  friend class TorchWorkNCCL;
  friend class NcclCachingAllocatorHookImpl;
  friend class TorchCommWindowNCCL;

  // Getter for CUDA API (for friend classes)
  CudaApi* getCudaApi() const {
    return cuda_api_.get();
  }

  // Getter for NCCL API (for friend classes)
  NcclApi* getNcclApi() const {
    return nccl_api_.get();
  }

  // Method to override the NCCL API implementation for testing
  void setNcclApi(std::shared_ptr<NcclApi> api) {
    nccl_api_ = std::move(api);
  }

  // Method to override the CUDA API implementation for testing
  void setCudaApi(std::shared_ptr<CudaApi> api) {
    cuda_api_ = std::move(api);
  }

  const CommOptions& getOptions() const override {
    return options_;
  }

  const at::Device& getDevice() const override {
    return device_;
  }

 protected:
  // Event management for friend classes
  [[nodiscard]] cudaEvent_t getEvent();
  void returnEvent(cudaEvent_t event);
  void abortNcclComm();
  void revokeNcclComm();

  enum class CommState {
    NORMAL,
    ERROR,
    TIMEOUT,
  };

  struct Address {
    void* addr;
  };

  struct AddressWithLen {
    void* addr;
    size_t len;
  };

  std::atomic<CommState> comm_state_{
      CommState::NORMAL}; // State of the communicator

  void register_address(const AddressWithLen& addr);
  void deregister_address(const Address& addr);
  ncclDataType_t getNcclDataType(const at::Tensor& tensor);
  c10::intrusive_ptr<TorchWorkNCCL> createWork(
      cudaStream_t stream,
      std::chrono::milliseconds timeout,
      const std::vector<at::Tensor>& inputTensors = {});

  c10::intrusive_ptr<TorchWorkNCCL> createWork(
      cudaStream_t stream,
      std::chrono::milliseconds timeout,
      const at::Tensor& inputTensor);

 private:
  // Helper that automatically cleans up premul sums.
  struct RedOpRAII {
    /* implicit */ RedOpRAII(ncclRedOp_t op);

    // Constructor for Premulsum Reduction
    explicit RedOpRAII(
        const ReduceOp& op,
        const ncclComm_t comm,
        const ncclDataType_t dataType,
        std::shared_ptr<NcclApi> nccl_api);

    RedOpRAII() = delete;
    RedOpRAII(const RedOpRAII&) = delete;
    RedOpRAII& operator=(const RedOpRAII&) = delete;

    RedOpRAII(RedOpRAII&& other) noexcept
        : ncclRedOp_(other.ncclRedOp_),
          comm_(other.comm_),
          nccl_api_(std::move(other.nccl_api_)) {
      other.comm_ = nullptr; // Prevent destructor from destroying the op
    }

    RedOpRAII& operator=(RedOpRAII&& other) noexcept {
      if (this != &other) {
        // Destroy current op if we own one
        if (comm_ && nccl_api_) {
          NCCL_CHECK_IGNORE(
              nccl_api_,
              nccl_api_->redOpDestroy(ncclRedOp_, comm_),
              "failed to destroy NCCL reduction operation");
        }
        ncclRedOp_ = other.ncclRedOp_;
        comm_ = other.comm_;
        nccl_api_ = std::move(other.nccl_api_);
        other.comm_ = nullptr; // Prevent destructor from destroying the op
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

  // Struct to hold the registration handle for a buffer (and its symmetric
  // window, if window registration succeeded for this segment).
  struct RegistrationHandle {
    void* regHandle{nullptr};
    ncclWindow_t winHandle{nullptr};
    size_t len{0};

    RegistrationHandle() = default;
    RegistrationHandle(void* regHandle, ncclWindow_t winHandle, size_t len)
        : regHandle{regHandle}, winHandle{winHandle}, len{len} {}

    RegistrationHandle(RegistrationHandle&& other) noexcept
        : regHandle{other.regHandle},
          winHandle{other.winHandle},
          len{other.len} {
      other.regHandle = nullptr;
      other.winHandle = nullptr;
      other.len = 0;
    }

    RegistrationHandle(const RegistrationHandle&) = delete;
    RegistrationHandle& operator=(const RegistrationHandle&) = delete;

    RegistrationHandle& operator=(RegistrationHandle&& other) noexcept {
      if (this != &other) {
        regHandle = other.regHandle;
        winHandle = other.winHandle;
        len = other.len;
        other.regHandle = nullptr;
        other.winHandle = nullptr;
        other.len = 0;
      }
      return *this;
    }

    ~RegistrationHandle() = default;
  };

 public:
  // Look up the symmetric NCCL window covering `ptr` (which must lie inside
  // a segment allocated from the NCCL mempool). Returns {win, offset_in_bytes}
  // on success and {nullptr, 0} if `ptr` is not in any registered segment or
  // window registration was unavailable for that segment.
  std::pair<ncclWindow_t, size_t> lookupSegmentWindow(const void* ptr) const;

  // Ensure the segment containing `ptr` is registered as a
  // NCCL_WIN_COLL_SYMMETRIC window (collective; all ranks must call with the
  // same segment). Returns ncclSuccess on success; ncclInvalidArgument if
  // `ptr` is not in any segment from the NCCL mempool; or the underlying
  // ncclCommWindowRegister error code.
  ncclResult_t ensureSegmentWindow(const void* ptr);

 private:
  // Constructor for split communicators
  explicit TorchCommNCCL(const ncclComm_t nccl_comm);

  // Private utility methods
  size_t wordSize(ncclDataType_t type) const;
  RedOpRAII getNcclReduceOp(
      const ReduceOp& op,
      const ncclComm_t comm,
      const ncclDataType_t dataType);
  void timeoutWatchdog() noexcept;
  void checkInitialized() const;
  void checkAndAbortIfTimedOutOrError();
  void checkWorkQueue();
  void enqueueWork(c10::intrusive_ptr<TorchWorkNCCL> work, cudaStream_t stream);
  bool getGraphCaptureMode();
  cudaStream_t getOperationStream(bool async_op);
  void ensureTensorContiguous(const at::Tensor& tensor);
  void checkTensorDevice(const at::Tensor& tensor) const;
  void checkTensorsDevice(const std::vector<at::Tensor>& tensors) const;

  void attachMemoryHook();
  void detachMemoryHook();
  void initNcclResources();

  // Member variables
  ncclComm_t nccl_comm_{};
  at::Device device_;
  int comm_size_{};
  int rank_{};
  int64_t uuid_{-1};
  CommOptions options_;
  size_t max_event_pool_size_{};
  cudaStream_t internal_stream_{};
  cudaEvent_t
      dependency_event_{}; // Pre-allocated event for stream dependencies
  void* barrier_buffer_{}; // Pre-allocated CUDA buffer for barrier operations
  enum class InitializationState {
    UNINITIALIZED,
    INITIALIZED,
    FINALIZED,
  } init_state_;

  // List of [comm, regHandlesMap] pairs.  Each regHandlesMap is a map from the
  // buffer address to the registration handle
  std::map<void*, RegistrationHandle> memoryRegistrationHandles_;

  // Store held for reconfigure bootstrap (kept alive across reconfigure calls)
  c10::intrusive_ptr<c10d::Store> reconfigure_store_;

  // NCCL API abstraction
  std::shared_ptr<NcclApi> nccl_api_;

  // CUDA API abstraction
  std::shared_ptr<CudaApi> cuda_api_;

  // Event pool management
  std::queue<cudaEvent_t> event_pool_;
  std::mutex event_pool_mutex_;

  // Work tracking per stream
  TorchWorkNCCLQueue workq_;

  // Timeout monitoring
  std::thread timeout_thread_;
  std::atomic<bool> shutdown_;
  std::condition_variable timeout_cv_;
  std::mutex timeout_mutex_;

  bool high_priority_stream_{false};
  std::string name_;

  // Graph capture mode work references
  // Keep references to work objects during graph capture to prevent premature
  // destruction, organized per graph using capture ID
  std::unordered_map<
      unsigned long long,
      std::vector<c10::intrusive_ptr<TorchWorkNCCL>>>
      graph_capture_work_refs_;
  std::mutex graph_capture_work_mutex_;

  // Structure to hold cleanup data for CUDA user objects
  struct GraphCleanupData {
    TorchCommNCCL* comm;
    unsigned long long graph_id;

    GraphCleanupData(TorchCommNCCL* comm_, unsigned long long id)
        : comm(comm_), graph_id(id) {}
  };

  // Static callback function for CUDA user object cleanup
  static void CUDART_CB graphCleanupCallback(void* userData);

  friend class TorchWorkNCCLQueueCommTest;
};

// Registers the NCCL backend and its allocator factory with the global
// TorchCommFactory.
void register_nccl_backend();

} // namespace torch::comms
