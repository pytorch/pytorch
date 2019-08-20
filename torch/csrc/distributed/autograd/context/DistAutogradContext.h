#pragma once

#include <cstdint>

namespace torch {
namespace distributed {
namespace autograd {

// DistAutogradContext which stores information for a single distributed
// autograd pass on a worker.
class DistAutogradContext {
 public:
  explicit DistAutogradContext(int64_t context_id);
  int64_t context_id() const;

 private:
  const int64_t context_id_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
