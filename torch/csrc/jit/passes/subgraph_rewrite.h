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
TORCH_API std::shared_ptr<script::Module> PatternBasedRewrite(
    std::shared_ptr<script::Module>& module);

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
  std::shared_ptr<script::Module> runOnModule(
      std::shared_ptr<script::Module> module);

  // Register standard rewrite patterns.
  void RegisterDefaultPatterns();

  /** Register a custom rewrite pattern.
   *
   * The method takes four parameters specifying the pattern:
   * \p PATTERN - IR string with representing the pattern subgraph.
   * \p FUSED_NODE_NAME - name of the node that will be created to replace
   * matches of the pattern. This will later be generalized to support arbitrary
   * replacement subgraphs, not just a single-node graphs.
   * \p INPUTS - a list of names of the values from the pattern IR string that
   * should be used as inputs of the new subgraph nodes.
   * \p OUTPUT - similar list for outputs.
   *
   * See examples of pattern registering in `RegisterDefaultPatterns`.
   */
  void RegisterRewritePattern(
      const std::string& pattern,
      const std::string& fused_node_name,
      std::vector<std::string> inputs,
      std::vector<std::string> outputs);

 private:
  std::vector<RewritePatternDescr> patterns_;
  std::unordered_set<Node*> nodes_to_delete_;

  void runOnGraph(std::shared_ptr<Graph>& graph);
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
  std::string fused_node_name; // TODO: generalize it to handle arbitrary
                               // subgraphs rather than a single node
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

} // namespace jit
} // namespace torch
