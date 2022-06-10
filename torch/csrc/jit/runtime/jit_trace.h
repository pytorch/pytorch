#include <torch/csrc/jit/ir/ir.h>
#include <memory>

namespace torch {
namespace jit {
TORCH_API std::shared_ptr<Graph> TraceGraph(
    std::shared_ptr<Graph> graph,
    Stack& stack);
}
} // namespace torch
