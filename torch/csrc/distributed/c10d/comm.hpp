#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <utility>

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
      size_t bucket_count,
      at::Tensor tensor,
      std::vector<size_t> offsets,
      std::vector<size_t> lengths,
      std::vector<c10::IntArrayRef> sizes_vec,
      std::vector<at::Tensor> parameters)
      : index_(index),
        bucket_count_(bucket_count),
        buffer_(std::move(tensor)),
        offsets_(std::move(offsets)),
        lengths_(std::move(lengths)),
        sizes_vec_(std::move(sizes_vec)),
        parameters_(std::move(parameters)) {}

  // Returns the index of the bucket, which is unique across all the buckets.
  size_t getIndex() const {
    return index_;
  }

  const at::Tensor& getBuffer() const {
    return buffer_;
  }

  // Returns a mutable buffer compared with the above method.
  at::Tensor& getBufferRef() {
    return buffer_;
  }

  // Overwrites the buffer at a specific index.
  void setBuffer(at::Tensor& buffer) {
    buffer_ = buffer;
  }

  // Each tensor in the list that getGradients corresponds to a
  // parameter.
  std::vector<at::Tensor> getGradients() const;

  // Returns model parameters belonging to this bucket. They are returned in the
  // same order as gradient tensors via getGradients(). For example,
  // getParameters[i] will have its gradient stored in
  // getGradients[i]
  const std::vector<at::Tensor> getParameters() const {
    return parameters_;
  }

  // Returns whther this bucket is the last bucket to allreduce in an iteration.
  bool isLast() const {
    return index_ == bucket_count_ - 1;
  }

 private:
  size_t index_;
  size_t bucket_count_;
  at::Tensor buffer_;

  // Per-variable info in buffer_.
  std::vector<size_t> offsets_;
  std::vector<size_t> lengths_;
  std::vector<c10::IntArrayRef> sizes_vec_;
  // Model parameters for this bucket.
  const std::vector<at::Tensor> parameters_;
};

// Base class of both `PythonCommHook` and `CppCommHook`.
// Requires implementing 1) `runHook` method that communicates gradients
// asynchronously, and 2) `parseHookResult` method that converts the hook
// result into a tensor.
class TORCH_API CommHookInterface {
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
TORCH_API at::Tensor parseCppCommHookResult(const c10::IValue& result);
} // namespace detail

// This CppCommHook interface only requires implementing runHook method that
// potentially uses a state.
template <typename T>
class CppCommHookInterface : public CommHookInterface {
 public:
  explicit CppCommHookInterface(T state) : state_(std::move(state)) {}

  ~CppCommHookInterface() override = default;

  at::Tensor parseHookResult(const c10::IValue& result) override {
    return detail::parseCppCommHookResult(result);
  }

 protected:
  T state_;
};

} // namespace c10d
