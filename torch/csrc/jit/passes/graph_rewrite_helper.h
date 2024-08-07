#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {
namespace graph_rewrite_helper {

std::string getFuncName(Value* func_value);
Value* getValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap);
std::optional<IValue> getIValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap);
TORCH_API void replaceConvolutionWithAtenConv(std::shared_ptr<Graph>& graph);

bool isClampFusable(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap);

// This struct contains a compiled IR patterns slated for use in the
// findPatternMatches function. The struct encapsulates the common
// information from parseIR that is used in conjunction with the
// pattern matching facility. A const instance of this struct can
// also be stored away to cache the compiled IR pattern and reduce
// runtime cost
struct PatternInfo {
  std::string pattern_string;
  std::unique_ptr<Graph> pattern_graph;
  std::unordered_map<std::string, Value*> vmap;
  std::vector<MatchFilter> filters;

  static PatternInfo parse_from_str(
      std::string pattern_string,
      const std::vector<MatchFilter>& filters = {}) {
    PatternInfo rv{
        std::move(pattern_string),
        std::make_unique<Graph>(),
        decltype(vmap){},
        filters};
    parseIR(rv.pattern_string, rv.pattern_graph.get(), rv.vmap);
    return rv;
  }
};

} // namespace graph_rewrite_helper
} // namespace jit
} // namespace torch
