#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void runCleanupPasses(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
