#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>
#include <memory>
#include <vector>

HIDDEN_NAMESPACE_BEGIN(torch, stable, c10d)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

/** Reduction operations; support depends on the process group's backend. */
enum class ReduceOp : int32_t {
  SUM = 0,
  AVG = 1,
  PRODUCT = 2,
  MIN = 3,
  MAX = 4,
  BAND = 5,
  BOR = 6,
  BXOR = 7
};

/**
 * Owns a collective operation and retains its group and tensor objects.
 * Copies share ownership. Keep a Work until completion; destruction neither
 * waits nor cancels. Release all handles before Python interpreter shutdown.
 * Calls and destruction acquire the GIL as needed.
 *
 * Ownership does not prevent explicit group destruction or mutation/freeing of
 * tensor storage. Retain external buffers and follow backend stream and graph
 * lifetime requirements.
 */
class Work {
 public:
  /** Takes ownership of the handle. */
  explicit Work(TorchWorkHandle work)
      : work_(work, [](TorchWorkHandle value) { torch_delete_work(value); }) {}

  /**
   * Delegates to the backend, returning its boolean result and propagating
   * errors. Zero timeout selects the backend default. On CUDA, establishes
   * dependencies on the calling stream without necessarily blocking the CPU
   * until completion. Other streams need their own dependencies; a successful
   * wait alone does not make it safe to release externally managed resources.
   */
  bool wait(int64_t timeout_ms = 0) const {
    bool result = false;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_work_wait(work_.get(), timeout_ms, &result));
    return result;
  }

  /** Queries backend completion without establishing CUDA stream dependencies.
   */
  bool is_completed() const {
    bool result = false;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_work_is_completed(work_.get(), &result));
    return result;
  }

 private:
  std::shared_ptr<TorchWorkOpaque> work_;
};

/**
 * Owns a reference to an existing process group; copies share ownership.
 * Group creation, registration, and explicit destruction remain with the
 * caller. Release all handles before Python interpreter shutdown. Except for
 * from_pyobject(), calls and destruction acquire the GIL as needed.
 *
 * All ranks must issue matching collectives in matching order. Calls report an
 * error if PyTorch was built without distributed support.
 */
class ProcessGroup {
 public:
  /** Takes ownership of the handle. */
  explicit ProcessGroup(TorchProcessGroupHandle group)
      : group_(group, [](TorchProcessGroupHandle value) {
          torch_delete_process_group(value);
        }) {}

  /**
   * Retains the Python ProcessGroup, preserving Python overrides.
   * The caller must hold the GIL and have libtorch_python loaded at runtime.
   */
  static ProcessGroup from_pyobject(void* obj) {
    TorchProcessGroupHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_process_group_from_pyobject(obj, &result));
    return ProcessGroup(result);
  }

  int64_t rank() const {
    int64_t result = 0;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_process_group_rank(group_.get(), &result));
    return result;
  }

  int64_t size() const {
    int64_t result = 0;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_process_group_size(group_.get(), &result));
    return result;
  }

  std::string backend() const {
    StringHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_process_group_backend(group_.get(), &result));
    return detail::to<std::string>(detail::from(result));
  }

  Work allreduce(
      const std::vector<Tensor>& tensors,
      ReduceOp op = ReduceOp::SUM) const {
    auto handles = tensor_handles(tensors);
    TorchWorkHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_process_group_allreduce(
        group_.get(),
        handles.data(),
        handles.size(),
        static_cast<int32_t>(op),
        &result));
    return Work(result);
  }

  Work allreduce_coalesced(
      const std::vector<Tensor>& tensors,
      ReduceOp op = ReduceOp::SUM) const {
    auto handles = tensor_handles(tensors);
    TorchWorkHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_process_group_allreduce_coalesced(
        group_.get(),
        handles.data(),
        handles.size(),
        static_cast<int32_t>(op),
        &result));
    return Work(result);
  }

  /** The root rank is relative to this group; root_tensor indexes its tensors.
   */
  Work broadcast(
      const std::vector<Tensor>& tensors,
      int64_t root_rank,
      int64_t root_tensor = 0) const {
    auto handles = tensor_handles(tensors);
    TorchWorkHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_process_group_broadcast(
        group_.get(),
        handles.data(),
        handles.size(),
        root_rank,
        root_tensor,
        &result));
    return Work(result);
  }

  /** Gathers one input per rank into outputs, which must contain one per rank.
   */
  Work allgather(const Tensor& input, const std::vector<Tensor>& outputs)
      const {
    auto handles = tensor_handles(outputs);
    TorchWorkHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_process_group_allgather(
        group_.get(), input.get(), handles.data(), handles.size(), &result));
    return Work(result);
  }

  Work barrier(const std::vector<int64_t>& device_ids = {}) const {
    TorchWorkHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_process_group_barrier(
        group_.get(), device_ids.data(), device_ids.size(), &result));
    return Work(result);
  }

 private:
  static std::vector<AtenTensorHandle> tensor_handles(
      const std::vector<Tensor>& tensors) {
    std::vector<AtenTensorHandle> result;
    result.reserve(tensors.size());
    for (const auto& tensor : tensors)
      result.push_back(tensor.get());
    return result;
  }

  std::shared_ptr<TorchProcessGroupOpaque> group_;
};

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable, c10d)
