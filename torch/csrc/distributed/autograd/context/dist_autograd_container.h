#pragma once

#include <mutex>
#include <unordered_map>

#include <torch/csrc/distributed/autograd/context/dist_autograd_context.h>

namespace torch {
namespace distributed {
namespace autograd {

// Singleton class per worker which is responsible for storing the distributed
// autograd context for each autograd pass and also cleans up data for an
// autograd pass once its done.
//
// Each autograd pass is assinged a unique autograd_context_id and all data for
// that pass (DistAutogradContext) is stored in this container indexed by the
// autograd_context_id. The autograd_context_id itself is a 64 bit globally
// unique id. The first 16 bits is the worker_id and the next 48 bits is an
// auto-incrementing id for each worker.
class DistAutogradContainer {
 public:
  static DistAutogradContainer& init(int64_t worker_id);
  static DistAutogradContainer& getInstance();
  const DistAutogradContext& newContext();
  void releaseContext(int64_t context_id);
  const DistAutogradContext& retrieveContext(int64_t context_id);

 private:
  DistAutogradContainer();
  ~DistAutogradContainer() = default;

  DistAutogradContainer(const DistAutogradContainer&) = delete;
  DistAutogradContainer& operator=(const DistAutogradContainer&) = delete;
  DistAutogradContainer(DistAutogradContainer&&) = delete;
  DistAutogradContainer& operator=(DistAutogradContainer&&) = delete;

  // Auto incrementing context id used to identify unique autograd passes.
  // Initialized with the first 16 bits being the worker_id.
  int64_t current_context_id_;

  // Unique id to identify a worker in the distributed setting.
  int16_t worker_id_;

  // Map from autograd_context_id to DistAutogradContext.
  std::unordered_map<int64_t, DistAutogradContext> autograd_context_;

  // Whether or not the container has been initialized appropriately.
  bool initialized_;

  // Lock to protect current_context_id_ and autograd_context map. initialized_
  // and worker_id_ are immutable.
  mutable std::mutex autograd_context_lock_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
