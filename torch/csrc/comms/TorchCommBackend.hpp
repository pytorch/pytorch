// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/comms/TorchCommBatch.hpp>
#include <torch/csrc/comms/TorchCommHooks.hpp>
#include <torch/csrc/comms/TorchCommOptions.hpp>
#include <torch/csrc/comms/TorchCommTypes.hpp>
#include <torch/csrc/comms/TorchCommWindow.hpp>
#include <torch/csrc/comms/TorchWork.hpp>
#include <memory>
#include <vector>

namespace torch::comms {

inline constexpr const char* TORCHCOMM_BACKEND_ABI_VERSION = "1.1";

/**
 * TorchCommBackend - Abstract base class for communication backends.
 *
 * Thread Safety:
 * TorchCommBackend implementations are NOT thread-safe. All operations
 * (collectives, point-to-point, split, finalize, etc.) must be serialized
 * by the caller.
 *
 * Internal threads (e.g., timeout watchdog) are properly synchronized with
 * the main thread using mutexes and condition variables.
 */
class TorchCommBackend {
 public:
  virtual ~TorchCommBackend() = default;

  // Initialize the communication backend
  virtual void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) = 0;
  virtual void finalize() = 0;
  virtual int getRank() const = 0;
  virtual int getSize() const = 0;

  // Name of the backend impl that's the same for all instances of a backend.
  virtual std::string_view getBackendName() const = 0;

  virtual std::string_view getBackendVersion() const {
    throw std::logic_error(
        "[TorchCommBackend]: version not implemented for communicator:" +
        std::string(getCommName()));
  }

  // Unique name for this instance of the communicator.
  virtual std::string_view getCommName() const = 0;

  // Point-to-Point Operations
  virtual c10::intrusive_ptr<TorchWork> send(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      const SendOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> recv(
      at::Tensor& tensor,
      int src,
      bool async_op,
      const RecvOptions& options = {}) = 0;

  virtual c10::intrusive_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      const BatchP2POptions& options = {}) = 0;

  // Collective Operations
  virtual c10::intrusive_ptr<TorchWork> broadcast(
      at::Tensor& tensor,
      int root,
      bool async_op,
      const BroadcastOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      const ReduceOp& op,
      bool async_op,
      const AllReduceOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> reduce(
      const at::Tensor& tensor,
      int root,
      const ReduceOp& op,
      bool async_op,
      const ReduceOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_gather_v(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllGatherSingleOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor& output,
      const at::Tensor& input,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterSingleOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllToAllSingleOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      const AllToAllvSingleOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      const AllToAllOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> barrier(
      bool async_op,
      const BarrierOptions& options = {}) = 0;

  // Scatter and Gather Operations
  virtual c10::intrusive_ptr<TorchWork> scatter(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      const ScatterOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> gather(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      const GatherOptions& options = {}) = 0;

  virtual c10::intrusive_ptr<TorchWork> gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      int root,
      bool async_op,
      const GatherSingleOptions& /*options*/ = {}) {
    // Default: split output along dim-0 into getSize() views and delegate to
    // gather. The views share the same underlying storage so data lands
    // contiguously in the original output tensor.
    std::vector<at::Tensor> output_list;
    if (getRank() == root) {
      output_list = output.chunk(getSize(), /*dim=*/0);
    }
    GatherOptions gather_opts;
    return gather(output_list, input, root, async_op, gather_opts);
  }

  // Communicator Management
  virtual std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) = 0;

  virtual const CommOptions& getOptions() const = 0;

  virtual const at::Device& getDevice() const = 0;
  // Window & One-sided Operations, not required for all backends, so we added
  // default implementation here
  virtual std::shared_ptr<TorchCommWindow> new_window(
      [[maybe_unused]] const std::optional<at::Tensor>& tensor = std::nullopt) {
    throw std::logic_error(
        "[TorchCommBackend]: new_window not implemented for communicator:" +
        std::string(getCommName()));
  }

  // Abort hook support (AbortHook defined in TorchCommHooks.hpp)
  // Called before aborting when a collective times out or fails.
  // Multiple hooks can be registered and will be called in order.

  virtual void registerAbortHook(int64_t hookId, AbortHook hook) {
    abortHooks_.emplace(hookId, std::move(hook));
  }

  virtual void unregisterAbortHook(int64_t hookId) {
    abortHooks_.erase(hookId);
  }

  std::unordered_map<int64_t, AbortHook> abortHooks_;

  // Graph replay hook support (GraphReplayHook defined in TorchCommHooks.hpp)
  // Called by backends when graph replay events are detected.

  virtual void registerGraphReplayHook(int64_t hookId, GraphReplayHook hook) {
    graphReplayHooks_.emplace(hookId, std::move(hook));
  }

  virtual void unregisterGraphReplayHook(int64_t hookId) {
    graphReplayHooks_.erase(hookId);
  }

  std::unordered_map<int64_t, GraphReplayHook> graphReplayHooks_;

  // Persistent AllGather operations
  // Handle type for persistent AllGather (opaque pointer)
  using AllGatherPHandle = void*;

  // Initialize persistent AllGather operation
  // Returns a handle that can be used for multiple executions
  virtual AllGatherPHandle all_gather_p_init(
      at::Tensor& /* output */,
      const AllGatherPInitOptions& /* options */ = {}) {
    throw std::logic_error(
        "[TorchCommBackend]: all_gather_p_init not implemented for "
        "communicator:" +
        std::string(getCommName()));
  }

  // Execute persistent AllGather
  // Can be called multiple times with the same handle
  virtual c10::intrusive_ptr<TorchWork> all_gather_p_exec(
      AllGatherPHandle /* handle */,
      const at::Tensor& /* input */,
      bool /* async_op */,
      const AllGatherPExecOptions& /* options */ = {}) {
    throw std::logic_error(
        "[TorchCommBackend]: all_gather_p_exec not implemented for "
        "communicator:" +
        std::string(getCommName()));
  }

  // Free persistent AllGather handle
  virtual void all_gather_p_free(AllGatherPHandle /* handle */) {
    throw std::logic_error(
        "[TorchCommBackend]: all_gather_p_free not implemented for "
        "communicator:" +
        std::string(getCommName()));
  }

  // Fault Tolerance API

  /**
   * Check if this backend supports reconfigure for fault tolerance.
   * Override this method in backends that support reconfigure.
   *
   * @return True if the backend supports reconfigure, false otherwise.
   */
  virtual bool supportsReconfigure() const {
    return false;
  }

  /**
   * Check if abort/fault-tolerance is supported on this communicator.
   *
   * @return True if abort is supported, false otherwise.
   */
  virtual bool isAbortSupported() const {
    return false;
  }

  /**
   * Check if the communicator is in an aborted state.
   *
   * Useful in CUDA graph mode where per-operation work handles are
   * unavailable and polling the communicator state is the only way to
   * detect failures.
   *
   * @return True if the communicator has been aborted.
   */
  virtual bool isAborted() const {
    return false;
  }

  /**
   * Check if the backend is fully initialized and ready for collective
   * operations. In dynamic regime, the backend transitions to initialized
   * state after a successful reconfigure().
   *
   * This method is non-throwing and safe to call regardless of backend state.
   *
   * @return True if the backend is initialized, false otherwise.
   */
  virtual bool isInitialized() const {
    return true;
  }

  /**
   * Get the initialization handle for this backend.
   * In dynamic regime, this URL encodes information required by the backend
   * to complete the initialization process via reconfigure().
   *
   * @return An InitHandle containing the initialization URL/handle.
   * @throws std::runtime_error if not implemented by the backend.
   */
  virtual InitHandle getInitHandle() const {
    throw std::runtime_error(
        "[TorchCommBackend]: getInitHandle not implemented for communicator:" +
        std::string(getCommName()));
  }

  /**
   * Reconfigure the communicator with a new set of peers.
   * In dynamic regime, this method initializes the communicator with the
   * provided set of peers. After a successful reconfigure call, the
   * communicator is fully initialized and collective operations are permitted.
   *
   * @param opts ReconfigureOptions containing uuid, handles, timeout, and
   * hints.
   * @return A TorchWork handle that can be used to wait for completion.
   * @throws std::runtime_error if not implemented by the backend.
   */
  virtual c10::intrusive_ptr<TorchWork> reconfigure(
      const ReconfigureOptions& /*opts*/) {
    throw std::runtime_error(
        "[TorchCommBackend]: reconfigure not implemented for communicator:" +
        std::string(getCommName()));
  }

  /**
   * Abort the communicator.
   *
   * Must be non-blocking and return immediately. Implementations must ensure
   * all pending operations transition to CANCELED status on their work
   * handles.
   *
   * After abort(), the communicator is in an uninitialized state;
   * reconfigure() must be called before issuing further collectives.
   *
   * Must be thread-safe; may be called from a background watchdog thread.
   */
  virtual void abort() {
    throw std::runtime_error(
        "[TorchCommBackend]: abort not implemented for communicator:" +
        std::string(getCommName()));
  }

  /**
   * Register a tensor's memory with the backend for optimized data transfer.
   *
   * Pre-registers the memory region for zero-copy RDMA or similar transport
   * optimizations. Backends that support registration must override this.
   *
   * The caller is responsible for calling tensor_deregister() before the
   * tensor is freed. Failing to deregister leaks the backend registration
   * handle but does not cause a crash — the transport layer will clean up
   * on communicator finalization.
   *
   * @param tensor The tensor whose memory to register.
   */
  virtual void tensor_register(const at::Tensor& /*tensor*/) {
    throw std::runtime_error(
        "[TorchCommBackend]: tensor_register not implemented for "
        "communicator:" +
        std::string(getCommName()));
  }

  /**
   * Deregister a tensor's previously registered memory.
   *
   * @param tensor The tensor whose memory to deregister.
   */
  virtual void tensor_deregister(const at::Tensor& /*tensor*/) {
    throw std::runtime_error(
        "[TorchCommBackend]: tensor_deregister not implemented for "
        "communicator:" +
        std::string(getCommName()));
  }

 protected:
  void runAbortHooks() {
    for (const auto& [_, hook] : abortHooks_) {
      try {
        hook();
      } catch (const std::exception& e) {
        LOG(ERROR) << "[TorchCommBackend] Abort hook threw exception: "
                   << e.what();
      } catch (...) {
        LOG(ERROR) << "[TorchCommBackend] Abort hook threw unknown exception.";
      }
    }
  }

  void fireGraphReplayHook(
      uint64_t graph_id,
      uint64_t replay_id,
      void* stream,
      size_t collective_index,
      std::string_view event) {
    for (const auto& [_, hook] : graphReplayHooks_) {
      try {
        hook(graph_id, replay_id, stream, collective_index, event);
      } catch (const std::exception& e) {
        LOG(ERROR) << "[TorchCommBackend] Graph replay hook threw exception: "
                   << e.what();
      } catch (...) {
        LOG(ERROR)
            << "[TorchCommBackend] Graph replay hook threw unknown exception.";
      }
    }
  }
};

/**
 * Interface for a dynamic loader to be able to load a backend library
 * from a dynamic library.
 */
struct DynamicLoaderInterface {
  // Function pointers
  TorchCommBackend* (*new_comm)(void);
  void (*destroy_comm)(TorchCommBackend* comm);
  const char* (*get_supported_version)();
};

// Factory function signature (implemented in each .so)
using CreateDynamicLoaderFn = DynamicLoaderInterface (*)();

} // namespace torch::comms
