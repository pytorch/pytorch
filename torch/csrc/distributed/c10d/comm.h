#pragma once

#include <memory>

#include <ATen/ATen.h>
#include <c10d/ProcessGroup.hpp>
#include <ATen/core/ivalue_inl.h>

namespace c10d {

// Broadcast many tensors to all processes in the process group.
TORCH_API void broadcast_coalesced(
    std::shared_ptr<c10d::ProcessGroup> process_group,
    at::TensorList tensors,
    size_t buffer_size,
    int rank = 0);

// This class passes bucket contents tensor (for multiple replicas) to
// DDP communication hook.
// Optionally in the future this can be enhanced with parameter to bucket
// mappings as well.
class GradBucket {
 public:
  explicit GradBucket(const std::vector<at::Tensor>& tensors)
      : tensors_(tensors) {}
  // Each tensor in the list that getTensors returns refers to the replica on
  // each device. There will be multiple replicas only in the case of single
  // process multiple device mode. In the single process single device mode,
  // this list would consist of only a single tensor.
  const std::vector<at::Tensor>& getTensors() const {
    return tensors_;
  }

 private:
  std::vector<at::Tensor> tensors_;
};

// DDP's c10d reducer allows communication hooks defined as a sub class
// of CommHookInterface. CommHookInterface is an abstract class and can
// be used to implement both Python and CPP hooks.
struct TORCH_PYTHON_API CommHookInterface {
 public:
  virtual ~CommHookInterface() {}

  // runHook takes a GradBucket type bucket and passes the tensors of
  // this grad bucket to hook's callback. This function is called once
  // the bucket is ready. The hook can perform whatever processing is
  // needed and return a Future that will hold the new value of the grad
  // bucket's tensors once ready.
  virtual c10::intrusive_ptr<c10::ivalue::Future> runHook(
      const GradBucket& bucket) = 0;

  // Once the grad bucket of Future is ready, c10d reducer will call this
  // function to get the resulting tensors of the grad bucket. Then c10d
  // reducer will use these tensors and copy grads to the grads of individual
  // parameters.
  virtual std::vector<at::Tensor> processFuture(c10::IValue future_value) = 0;
};


} // namespace c10d
