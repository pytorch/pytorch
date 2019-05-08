/** \brief This file defines API for pattern-based fusion.
 *
 * The API can be used for finding concrete patterns in the model and replacing
 * the corresponding subgraphs with a single node (think of Conv-Relu fusion for
 * example).
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
struct FusionPatternDescr;
struct Match;

/** \brief Run pattern-based fusion on all methods in the module.
 *
 * This pass will go through all methods in the module and try to fuse all
 * recognized patterns (see PatternFuser::RegisterDefaultPatterns for the list
 * of these patterns).
 */
TORCH_API std::shared_ptr<script::Module> PatternBasedFusion(
    std::shared_ptr<script::Module> module);

/** \brief A class implementing API for pattern-based fusion.
 *
 * To perform pattern-based fusion on a module using this API, one needs to
 * crete an object of such class, register fusion patterns and run the
 * transformation pass (`runOnModule`).
 *
 * To use standard fusion patterns, one could use `RegisterDefaultPatterns`.
 *
 * To enable fusion of custom patterns, they must be registered with
 * `RegisterFusionPattern`.
 */
class TORCH_API PatternFuser {
 public:
  // \brief Run pattern-based fusion pass on the module.
  std::shared_ptr<script::Module> runOnModule(
      std::shared_ptr<script::Module> module);

  // \brief Register standard fusion patterns.
  void RegisterDefaultPatterns();

  /** \brief Register a custom fusion pattern.
   *
   * The method takes four parameters specifying the pattern:
   * \p PATTERN - IR string with representing the pattern subgraph,
   * \p FUSED_NODE_NAME - name of the node that will be created to replace
   * matches of the pattern,
   * \p INPUTS - a list of names of the values from the pattern IR string that
   * should be used as inputs of the fused nodes,
   * \p OUTPUT - similar list for outputs.
   *
   * See examples of pattern registering in `RegisterDefaultPatterns`.
   */
  void RegisterFusionPattern(
      const std::string& pattern,
      const std::string& fused_node_name,
      std::vector<std::string> inputs,
      std::vector<std::string> outputs);

 private:
  std::vector<FusionPatternDescr> patterns_;
  std::unordered_set<Node*> nodes_to_delete_;

  void runOnGraph(std::shared_ptr<Graph>& graph);
  void fuseSinglePatternOnGraph(
      std::shared_ptr<Graph>& graph,
      FusionPatternDescr pattern);
  bool overlapsWithPreviousMatches(const Match* match);
};

/** \brief Fusion pattern descriptor.
 *
 * This structure is used in implementation of `PatternFuser` and not supposed
 * to be used externally.
 */
struct FusionPatternDescr {
  std::string pattern;
  std::string fused_node_name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

} // namespace jit
} // namespace torch
