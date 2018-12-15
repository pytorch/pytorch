#pragma once

#include <torch/csrc/jit/alias_info.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

/**
 * Alias analysis pass.
 *
 * This pass produces an AliasDb that contains aliasing and mutation
 * information about the graph. Callers (right now moveAfterTopologicallyValid)
 * can use this information to determine whether mutations to the graph are
 * safe, in that they don't reorder/change nodes in a way that affects output.
 *
 * Every value with a mutable type (Tensors, Lists, Tuples, etc.) will be
 * associated with one or more "alias sets". If two values share an alias set,
 * that means they may alias, implying that a mutation to one value cannot be
 * reordered past a use of the other. Only reordering two reads of an alias set
 * is considered safe.
 *
 * There is a special alias set called the "wildcard set", which indicates that
 * we're not sure what this value may alias. To be conservative, we consider
 * the wildcard alias set as potentially aliasing any value.
 */
class AliasDb {
 public:
  explicit AliasDb(std::shared_ptr<Graph> graph);

  // Does `n` use or write to any wildcard aliases?
  bool hasWildcard(const Node* n) const;

  const std::unordered_set<const Node*>& getWildcardNodes() const {
    return wildcardNodes_;
  }

  // Does `n` write to any alias sets?
  bool hasWrites(Node* n) const;

  // Does `n` write to a value that may alias one of the graph inputs?
  bool writesToInputAlias(Node* n) const;

  // Get all nodes that write to any alias set inputed/outputed by `n`
  std::unordered_set<Node*> getWritersForNode(const Node* n) const;

  // Do any nodes  write to an alias set inputed/outputed by `n`?
  bool hasWriters(const Node* n) const {
    return getWritersForNode(n).size() != 0;
  }

  // For debugging: print alias db state to stdout
  void dump() const;

 private:
  void analyze(const std::shared_ptr<Graph>& graph);
  void analyze(Block* block);
  void analyze(Node* node);

  void analyzeIf(Node* node);
  void analyzeLoop(Node* node);
  void analyzeSubgraph(Node* node);
  void analyzeCreator(Node* node);
  void analyzeExtractor(Node* node);
  void analyzeChunk(Node* node);
  void analyzeBroadcastingChunk(Node* node);

  Symbol getFreshAlias(bool isGraphInput = false);
  void addAlias(const Value* value, AliasInfo alias);
  void addAlias(const Value* value, Symbol alias);
  void addAlias(const Value* value, const Value* from);
  void mapAliases(at::ArrayRef<Value*> to, at::ArrayRef<Value*> from);
  void giveFreshAlias(const Value* value);

  bool hasUsesAfter(Symbol alias, const Node* n) const;
  void buildWildcardIndex(const Block* b);
  bool hasWildcardImpl(const Node* n) const;
  bool writesTo(Node* n, const Value* v) const;

  std::shared_ptr<Graph> graph_;
  Symbol latestSymbol_ = Symbol::fromQualString("alias::0");
  std::unordered_map<const Value*, AliasInfo> valueToAlias_;
  std::unordered_map<Symbol, std::unordered_set<const Value*>> aliasToValue_;
  std::unordered_map<Symbol, std::unordered_set<Node*>> aliasToWrites_;
  std::unordered_set<const Node*> wildcardNodes_;
  std::unordered_set<Symbol> graphInputAliases_;
};

inline TORCH_API AliasDb AliasAnalysis(std::shared_ptr<Graph> graph) {
  return AliasDb(std::move(graph));
}
} // namespace jit
} // namespace torch
