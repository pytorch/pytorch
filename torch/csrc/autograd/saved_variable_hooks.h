#pragma once

#include <ATen/core/Tensor.h>

namespace torch::dynamo::autograd {
class CompiledNodeArgs;
struct VariableMetadata;
} // namespace torch::dynamo::autograd

namespace torch::autograd {

class SavedVariable;

struct TORCH_API SavedVariableHooks {
  virtual void call_pack_hook(const at::Tensor& tensor) = 0;
  virtual at::Tensor call_unpack_hook() = 0;
  virtual void compiled_args(
      torch::dynamo::autograd::CompiledNodeArgs& args,
      const SavedVariable& sv,
      const std::shared_ptr<torch::dynamo::autograd::VariableMetadata>&
          meta) = 0;
  virtual ~SavedVariableHooks() = default;
};

} // namespace torch::autograd
