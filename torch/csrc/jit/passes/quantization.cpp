#include <torch/csrc/jit/passes/quantization.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <algorithm>
#include <stack>
