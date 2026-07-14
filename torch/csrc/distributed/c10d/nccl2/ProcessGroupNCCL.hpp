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
#include <cuda_runtime.h>
#include <nccl.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <torch/csrc/distributed/c10d/nccl2/Batch.hpp>
#include <torch/csrc/distributed/c10d/nccl2/CudaApi.hpp>
#include <torch/csrc/distributed/c10d/nccl2/NcclApi.hpp>
#include <torch/csrc/distributed/c10d/nccl2/WorkNCCL.hpp>

namespace c10d::nccl2 {

// Hint key names for NCCL backend configuration
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

class TORCH_API ProcessGroupNCCL : public ::c10d::Backend {
 public:
  static constexpr std::string_view kBackendName = "nccl2";

  // c10d Backend options for this backend (a c10d::Backend::Options subclass,
  // like ProcessGroupNCCL::Options); surfaced to Python via the Options pybind.
  struct TORCH_API Options : ::c10d::Backend::Options {
    bool abort_process_on_timeout_or_error{true};
    bool is_high_priority_stream{false};
    std::unordered_map<std::string, std::string> hints;

    explicit Options(bool is_high_priority_stream = false)
        : ::c10d::Backend::Options(std::string(kBackendName)),
          is_high_priority_stream(is_high_priority_stream) {}

    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }
  };

  // c10d-style constructor: the NCCL communicator is bootstrapped lazily, on
  // the first collective (or via eagerConnectSingleDevice / bound_device_id),
  // matching c10d's device-binding model -- unlike torchcomms which took an
  // eager init(device).
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

  std::shared_ptr<c10::Allocator> getMemAllocator() override;
  void setTimeout(std::chrono::milliseconds timeout) override;
  void eagerConnectSingleDevice(at::Device device) override;
  void shutdown() override;
  void abort() override;

  void registerAbortHook(int64_t hook_id, ::c10d::AbortHook hook) override;
  void unregisterAbortHook(int64_t hook_id) override;

  // ---- accessors used by friend classes (work) ----
  CudaApi* getCudaApi() const {
    return cuda_api_.get();
  }
  NcclApi* getNcclApi() const {
    return nccl_api_.get();
  }
  void setNcclApi(std::shared_ptr<NcclApi> api) {
    nccl_api_ = std::move(api);
  }
  void setCudaApi(std::shared_ptr<CudaApi> api) {
    cuda_api_ = std::move(api);
  }
  const at::Device& getDevice() const {
    return device_;
  }
  std::string_view getCommName() const {
    return name_;
  }
  // Underlying host ncclComm_t as an opaque integer pointer.
  int64_t getCommPtr() const;

  friend class WorkNCCL;

 protected:
  [[nodiscard]] cudaEvent_t getEvent();
  void returnEvent(cudaEvent_t event);
  void abortNcclComm();
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

  // Lazy, one-time bootstrap of the NCCL communicator on `device`. Subsequent
  // calls validate the same device. Replaces torchcomms' eager init(device).
  void ensureInitialized(at::Device device);
  void init(at::Device device);
  void finalize();
  void initNcclResources();

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
      const ncclComm_t comm,
      const ncclDataType_t dataType);
  void timeoutWatchdog() noexcept;
  void checkInitialized() const;
  void checkAndAbortIfTimedOutOrError();
  void checkWorkQueue();
  void enqueueWork(c10::intrusive_ptr<WorkNCCL> work, cudaStream_t stream);
  bool getGraphCaptureMode();
  cudaStream_t getOperationStream(bool async_op);
  void ensureTensorContiguous(const at::Tensor& tensor);
  void checkTensorDevice(const at::Tensor& tensor) const;
  void checkTensorsDevice(const std::vector<at::Tensor>& tensors) const;
  void runAbortHooks();

  void attachMemoryHook();
  void detachMemoryHook();

  // Member variables (port of TorchCommNCCL).
  ncclComm_t nccl_comm_{};
  at::Device device_;
  int comm_size_{};
  // NOTE: the rank is stored in the inherited c10d::Backend::rank_ (set in the
  // ctor and refreshed from NCCL in initNcclResources). The ported engine code
  // reads/writes `rank_` directly, which resolves to that protected member.
  size_t max_event_pool_size_{};
  cudaStream_t internal_stream_{};
  cudaEvent_t dependency_event_{};
  void* barrier_buffer_{};
  enum class InitializationState {
    UNINITIALIZED,
    INITIALIZED,
    FINALIZED,
  } init_state_{InitializationState::UNINITIALIZED};

  c10::intrusive_ptr<::c10d::Store> store_;

  std::shared_ptr<NcclApi> nccl_api_;
  std::shared_ptr<CudaApi> cuda_api_;

  std::queue<cudaEvent_t> event_pool_;
  std::mutex event_pool_mutex_;

  WorkNCCLQueue workq_;

  std::thread timeout_thread_;
  std::atomic<bool> shutdown_{false};
  std::condition_variable timeout_cv_;
  std::mutex timeout_mutex_;

  bool is_high_priority_stream_{false};
  std::string name_;

  c10::intrusive_ptr<Options> options_c10d_;

  // Abort hooks (c10d::Backend API; storage was in torchcomms' TorchCommBackend
  // base, folded in here).
  std::unordered_map<int64_t, ::c10d::AbortHook> abortHooks_;

  // Active coalescing batch (port of BackendWrapper). Engaged between
  // startCoalescing() and endCoalescing(); send()/recv() append into it.
  std::optional<BatchSendRecv> coalescing_batch_;

  std::unordered_map<
      unsigned long long,
      std::vector<c10::intrusive_ptr<WorkNCCL>>>
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
