#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <c10d/ProcessGroup.hpp>

namespace c10d {

// Broadcast many tensors to all processes in the process group.
TORCH_API void broadcast_coalesced(
    c10::intrusive_ptr<c10d::ProcessGroup> process_group,
    at::TensorList tensors,
    size_t buffer_size,
    int rank = 0);

// This class passes bucket contents tensor to DDP communication hook.
class TORCH_API GradBucket {
 public:
  explicit GradBucket(
      size_t index,
      const at::Tensor& tensor,
      const std::vector<size_t>& offsets,
      const std::vector<size_t>& lengths,
      const std::vector<c10::IntArrayRef>& sizes_vec,
      const std::vector<at::Tensor>& model_params_for_bucket)
      : index_(index),
        tensor_(tensor),
        offsets_(offsets),
        lengths_(lengths),
        sizes_vec_(sizes_vec),
        model_params_for_bucket_(model_params_for_bucket) {}

  // Returns the index of the bucket, which is unique across all the buckets.
  size_t getIndex() const {
    return index_;
  }

  const at::Tensor& getTensor() const {
    return tensor_;
  }

  // Returns a mutable tensor compared with the above method.
  at::Tensor& getTensorRef() {
    return tensor_;
  }

  // Overwrites the tensor at a specific index.
  void setTensor(at::Tensor& tensor) {
    tensor_ = tensor;
  }

  // Each tensor in the list that getPerParameterTensors corresponds to a
  // parameter.
  std::vector<at::Tensor> getPerParameterTensors() const;

  // Returns model parameters belonging to this bucket. They are returned in the
  // same order as gradient tensors via getPerParameterTensors(). For example,
  // getModelParamsForBucket[i] will have its gradient stored in
  // getPerParameterTensors[i]
  const std::vector<at::Tensor> getModelParamsForBucket() const {
    return model_params_for_bucket_;
  }

  // Returns whther this bucket is the last bucket to allreduce in an iteration.
  bool isTheLastBucketToAllreduce() const {
    return index_ == 0;
  }

 private:
  size_t index_;
  at::Tensor tensor_;

  // Per-variable info in tensor_.
  std::vector<size_t> offsets_;
  std::vector<size_t> lengths_;
  std::vector<c10::IntArrayRef> sizes_vec_;
  // Model parameters for this bucket.
  const std::vector<at::Tensor> model_params_for_bucket_;
};

// Base class of both `PythonCommHook` and `CppCommHook`.
// Requires implementing 1) `runHook` method that communicates gradients
// asynchronously, and 2) `parseHookResult` method that converts the hook
// result into a tensor.
class TORCH_PYTHON_API CommHookInterface {
 public:
  virtual ~CommHookInterface() = default;

  // Passes the input grad bucket to the registered communication hook.
  // Once the tensor in the bucket are ready, kicks off the hook asynchronously
  // and returns a future that holds the communication results.
  virtual c10::intrusive_ptr<c10::ivalue::Future> runHook(
      GradBucket& bucket) = 0;

  // Returns the resulting tensor once the communication hook result is
  // ready. The resulting tensor will then be copied to the grads of
  // individual parameters.
  virtual at::Tensor parseHookResult(
      const c10::IValue& result) = 0;
};

namespace detail {
// This helper function is called both by CppCommHookInterface below and inside
// reducer.
inline at::Tensor parseCppCommHookResult(
    const c10::IValue& result) {
  TORCH_INTERNAL_ASSERT(
      result.isTensor() || result.isTensorList(),
      "expected the hook result is either a Tensor or a TensorList");

  if (result.isTensor()) {
    return result.toTensor();
  }

  return result.toTensorVector()[0];
}
} // namespace detail

// This CppCommHook interface only requires implementing runHook method that
// potentially uses a state.
// Still need TORCH_PYTHON_API instead of TORCH_API to support Windows platform.
template <typename T>
class TORCH_PYTHON_API CppCommHookInterface : public CommHookInterface {
 public:
  explicit CppCommHookInterface(T& state) : state_(state) {}

  ~CppCommHookInterface() override = default;

  at::Tensor parseHookResult(const c10::IValue& result) override {
    return detail::parseCppCommHookResult(result);
  }

 protected:
  T state_;
};

} // namespace c10d
