#include <torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>

namespace torch {
namespace distributed {
namespace autograd {

AutogradMetadata::AutogradMetadata(
    int64_t autogradContextId_,
    int64_t autogradMessageId_)
    : autogradContextId(autogradContextId_),
      autogradMessageId(autogradMessageId_) {}

} // namespace autograd
} // namespace distributed
} // namespace torch
