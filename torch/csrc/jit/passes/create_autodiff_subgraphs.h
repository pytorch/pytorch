#pragma once
#include <cstddef>

namespace torch { namespace jit {

struct Graph;

// insert GraphExecutor nodes that group together
// subgraphs that are differentiable by the jit's autodiff passes
// threshold - minimum number of nodes that will appear in a block
void CreateAutodiffSubgraphs(Graph & graph, size_t threshold = 2);

}}
