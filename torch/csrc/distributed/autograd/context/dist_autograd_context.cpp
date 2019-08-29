#include <torch/csrc/distributed/autograd/context/dist_autograd_context.h>
#include <c10/util/Exception.h>

namespace torch {
namespace distributed {
namespace autograd {

DistAutogradContext::DistAutogradContext(int64_t context_id)
    : context_id_(context_id) {}

int64_t DistAutogradContext::context_id() const {
  return context_id_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
