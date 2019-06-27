/** This file defines API for pattern-based subgraph rewrites.
 *
 * The API can be used for finding concrete patterns in the model and replacing
 * the corresponding subgraphs with another subgraph. A special case of such
 * rewrites is fusion, where the new subgraph consists of just a single node.
 *
 * There is a default set of most-common patterns that everyone could use, or
 * alternatively an arbitrary pattern can be registered.
 */
#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {

// Forward declarations.
struct RewritePatternDescr;
struct Match;

/** Run pattern-based subgraph rewrites on all methods in the module.
 *
 * This pass will go through all methods in the module and try to replace all
 * recognized patterns (see SubgraphRewriter::RegisterDefaultPatterns for the
 * list of these patterns).
 */
TORCH_API script::Module PatternBasedRewrite(const script::Module& module);

/** A class implementing API for pattern-based subgraph rewrites.
 *
 * To perform pattern-based subgraph rewrites on a module using this API, one
 * needs to crete an object of such class, register rewrite patterns and run the
 * transformation pass (`runOnModule`).
 *
 * To use standard patterns, one could use `RegisterDefaultPatterns`.
 *
 * To enable rewrites of custom patterns, they must be registered with
 * `RegisterRewritePattern`.
 */
class TORCH_API SubgraphRewriter {
 public:
  // Run pattern-based subgraph rewrite pass on the module.
  script::Module runOnModule(const script::Module& module);

  // Run pattern-based subgraph rewrite pass on the graph (used in testing).
  void runOnGraph(std::shared_ptr<Graph>& graph);

  // Register standard rewrite patterns.
  void RegisterDefaultPatterns();

  /** Register a custom rewrite pattern.
   *
   * The method takes two parameters specifying the pattern:
   * \p PATTERN - IR string representing the pattern subgraph.
   * \p REPLACEMENT - IR stringn representing the replacement subgraph.
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
      RewritePatternDescr pattern);
  bool overlapsWithPreviousMatches(const Match* match);
};

/** Rewrite pattern descriptor.
 *
 * This structure is used in implementation of `SubgraphRewriter` and not
 * supposed to be used externally.
 */
struct RewritePatternDescr {
  std::string pattern;
  std::string replacement;
};

} // namespace jit
} // namespace torch
