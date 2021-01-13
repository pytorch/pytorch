/** This file defines API for pattern-based subgraph rewrites.
 *
 * The API can be used for finding concrete patterns in the model and replacing
 * the corresponding subgraphs with another subgraph. A special case of such
 * rewrites is fusion, where the new subgraph consists of just a single node.
 *
 * There is a default set of the most common patterns that everyone could use.
 * Alternatively, an arbitrary pattern can be registered.
 */
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

#include <functional>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {

// Forward declarations.
struct RewritePatternDescr;
struct Match;

using MatchFilter = std::function<
    bool(const Match&, const std::unordered_map<std::string, Value*>&)>;

/** Run pattern-based subgraph rewrites on all methods in the module.
 *
 * This pass will go through all methods in the module and try to replace all
 * recognized patterns (see SubgraphRewriter::RegisterDefaultPatterns for the
 * list of these patterns).
 */
TORCH_API Module PatternBasedRewrite(const Module& module);

/** A class implementing API for pattern-based subgraph rewrites.
 *
 * To perform pattern-based subgraph rewrites on a module using this API, one
 * needs to create an object of such class, register rewrite patterns and run
 * the transformation pass (`runOnModule`).
 *
 * To use standard patterns, one could use `RegisterDefaultPatterns`.
 *
 * To enable rewrites of custom patterns, the custom patterns must be registered
 * with `RegisterRewritePattern`.
 */
class TORCH_API SubgraphRewriter {
 public:
  // Run pattern-based subgraph rewrite pass on the module.
  Module runOnModule(const Module& module);

  // Run pattern-based subgraph rewrite pass on the graph (used in testing).
  // `filter` is a function that does extra filtering on the match. If it
  // returns false for a given Match, we'll skip the Match. The filter
  // function's arguments consist of a Match and a value map from parsing the
  // pattern graph. Both the Match and the value map are necessary because we
  // need to 1) do extra filtering on the matched result as well as 2) refer to
  // the values in the matched result through the values in the pattern graph.
  void runOnGraph(
      std::shared_ptr<Graph>& graph,
      const std::vector<MatchFilter>& filters);

  void runOnGraph(
      std::shared_ptr<Graph>& graph,
      const MatchFilter& filter =
          [](const Match&, const std::unordered_map<std::string, Value*>&) {
            return true;
          }) {
    runOnGraph(graph, std::vector<MatchFilter>({filter}));
  }

  // Register standard rewrite patterns.
  void RegisterDefaultPatterns();

  /** Register a custom rewrite pattern.
   *
   * The method takes two parameters specifying the pattern:
   * \p PATTERN - IR string representing the pattern subgraph.
   * \p REPLACEMENT - IR string representing the replacement subgraph.
   *
   * See examples of pattern registering in `RegisterDefaultPatterns`.
   */
  void RegisterRewritePattern(
      const std::string& pattern,
      const std::string& replacement);

 private:
  std::vector<RewritePatternDescr> patterns_;
  std::unordered_set<Node*> nodes_to_delete_;

  void rewriteSinglePatternOnGraph(
      std::shared_ptr<Graph>& graph,
      const RewritePatternDescr& pattern,
      const std::vector<MatchFilter>& filters);

  bool overlapsWithPreviousMatches(const Match* match);
};

/** Rewrite pattern descriptor.
 *
 * This structure is used in the implementation of `SubgraphRewriter` and
 * is not supposed to be used externally.
 */
struct RewritePatternDescr {
  std::string pattern;
  std::string replacement;
};

} // namespace jit
} // namespace torch
