#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/debug_util.h>

namespace torch {
namespace lazy {

TSOpVector TsNode::Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                         TSLoweringContext* loctx) const {
  // TODO(whc) beginning to invert the design here.  Move to provide a Lower()
  // method on each node, starting with codegen.  Once we delete most
  // non-codegen ops, make this pure-virtual and put Lower() on the remaining
  // non-codegen ops.  For now, returning empty list here triggers fallback to
  // old lowering path.
  return {};
}

}  // namespace lazy
}  // namespace torch
