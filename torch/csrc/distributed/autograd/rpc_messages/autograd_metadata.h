#pragma once

#include <torch/csrc/Export.h>
#include <cstdint>

namespace torch::distributed::autograd {

// This structure represents autograd metadata that we need to pass across
// different nodes when we call an RPC which needs autograd computation.
struct TORCH_API AutogradMetadata {
  AutogradMetadata(int64_t autogradContextId, int64_t autogradMessageId);

  // autogradContextId_ is a globally unique integer that identifies a
  // particular distributed autograd pass.
  int64_t autogradContextId;
  // autogradMessageId_ is a globally unique integer that identifies a pair
  // of send/recv autograd functions.
  int64_t autogradMessageId;
};

} // namespace torch::distributed::autograd
