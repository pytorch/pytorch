#include <torch/csrc/jit/ir.h>

namespace torch { namespace jit {

void PeepholeOptimizeONNX(std::shared_ptr<Graph>& graph);

}}
