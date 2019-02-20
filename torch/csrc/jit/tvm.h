#include <tvm/ir_pass.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/tvm.h>

#include <ATen/core/stack.h>
#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/fuser/kernel_cache.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace relay {

struct TVMObject {
  tvm::PackedFunc kernel_;
  tvm::PackedFunc set_input_;
  tvm::PackedFunc get_output_;
};

struct TORCH_API TVMCompiler {
  TVMCompiler(const Node* node);
  void run(Stack& stack);
  std::shared_ptr<Graph> subgraph_;
  std::unordered_map<CompleteArgumentSpec, TVMObject> cache_;
};

bool isSupported(Node* node);

} // namespace relay
} // namespace jit
} // namespace torch
