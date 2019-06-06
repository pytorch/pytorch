#pragma once

#include <c10/util/flat_hash_map.h>
#include <torch/csrc/jit/alias_info.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/passes/utils/memory_dag.h>

namespace torch {
namespace jit {

/**
 * Alias analysis pass.
 *
 * This pass produces an AliasDb that contains aliasing and mutation
 * information about the graph. Users can use this information to determine
 * whether mutations to the graph are safe, i.e. they don't reorder/change
 * nodes in a way that affects output.
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
  TORCH_API explicit AliasDb(std::shared_ptr<Graph> graph);
  TORCH_API ~AliasDb();

  // There are limitations to what effects the alias analysis can track. Two
  // kinds of nodes may have untracked effects:
  // 1. Nodes that write to a value that may alias the graph inputs (since
  //    the inputs can be used outside the graph).
  // 2. Nodes that write to something in the wildcard set.
  //
  // These nodes are considered not safe to eliminate or mutate under any
  // circumstances.
  bool writesToWildcard(Node* n) const;

  // Does `n` write to an alias of one of the values in `vs`?
  // if `recurseBlocks` is true, consider writes on the nodes in `n`s sub-blocks
  TORCH_API bool writesToAlias(
      Node* n,
      const ValueSet& vs,
      bool recurseBlocks = false) const;

  // Does `a` and `b` potentially share a memory location or do either
  // hold in memory any element that exists in the other
  TORCH_API bool mayContainAlias(Value* a, Value* b) const;

  // Do any values in group `a` share a memory location or hold in memory
  // any element that exists in group `b`
  TORCH_API bool mayContainAlias(
      const at::ArrayRef<Value*>& a,
      const at::ArrayRef<Value*>& b) const;

  // Do `a` and `b` potentially share a memory location?
  TORCH_API bool mayAlias(const Value* a, const Value* b) const;
  // Do any values in group `a` potentially share a memory location with any
  // value in group `b`? i.e. may they overlap?
  TORCH_API bool mayAlias(const ValueSet& a, const ValueSet& b) const;

  // Do any nodes write to an alias set inputed/outputed by `n`?
  TORCH_API bool hasWriters(const Node* n) const;

  // Move 'n' (already in the graph) after 'movePoint' in the topological order.
  //
  // Tries to preserve value dependencies, so other nodes might be moved. We
  // make two gurantees about the postcondition of the node list:
  //   - `n` is directly after `movePoint`.
  //   - only nodes between `n` and `movePoint` have been moved.
  //
  // Returns `false` if it's impossible to move `n` after `MovePoint` without
  // violating dependencies, otherwise executes the move and returns `true`
  TORCH_API bool moveAfterTopologicallyValid(Node* n, Node* movePoint);
  TORCH_API bool moveBeforeTopologicallyValid(Node* n, Node* movePoint);

  bool couldMoveAfterTopologically(Node* n, Node* movePoint);
  bool couldMoveBeforeTopologically(Node* n, Node* movePoint);

  // For debugging: print alias db state to stdout
  TORCH_API void dump() const;

 private:
  // Helper for topologically-safe node moves.
  class WorkingSet;
  enum class MoveSide { BEFORE, AFTER };
  bool tryMove(Node* toMove, Node* movePoint, MoveSide moveSide, bool dryRun);
  void move(Node* toMove, Node* movePoint, MoveSide moveSide);
  bool isBeforeOrAfter(const Node* n, MoveSide moveSide) const;

  /**
   * Write and read internal API
   */
  // Get all the values that `n` writes to.
  // NOTE: this only returns values directly written to, not aliases thereof
  //
  // if `recurseBlocks` is true, gather writes on the nodes in `n`s sub-blocks
  MemoryLocations getWrites(Node* n, bool recurseBlocks = false) const;
  void getWritesImpl(Block* b, MemoryLocations& ret, bool recurseBlocks = false)
      const;
  void getWritesImpl(Node* n, MemoryLocations& ret, bool recurseBlocks = false)
      const;
  // Do any nodes write to `v`s memory location?
  TORCH_API bool hasWriters(const Value* v) const;
  // Register the fact that `n` writes to `v`.
  void registerWrite(const Value* v, Node* n);
  void registerWrite(const Element* e, Node* n);
  // Get all the values that `n` reads from.
  // if `recurseBlocks` is true, gather reads on the nodes in `n`s sub-blocks
  MemoryLocations getReads(Node* n, bool recurseBlocks = false) const;
  void getReadsImpl(Node* n, MemoryLocations& ret, bool recurseBlocks = false)
      const;

  /**
   * Wildcard methods
   */
  // Register `v` as a wildcard value.
  void setWildcard(const Value* v);

  // Is the element a wildcard or an unhandled container type,
  // or does the element contain an element for which that's true
  bool cannotCheckAliasContainment(const Value* elem) const;

  /**
   * Special analysis methods
   */
  void analyze(const std::shared_ptr<Graph>& graph);
  void analyze(Block* block);
  void analyze(Node* node);
  void analyzeImpl(Node* node);
  void analyzeIf(Node* node);
  void analyzeLoop(Node* node);
  void analyzeSubgraph(Node* node);
  void analyzeCreator(Node* node);
  void analyzeExtractor(Node* node);
  void analyzeChunk(Node* node);
  void analyzeBroadcastingChunk(Node* node);
  void analyzeFork(Node* node);
  void analyzeWait(Node* node);
  void analyzeGradOf(Node* node);
  void analyzeSetAttr(Node* node);
  void analyzeTupleConstruct(Node* node);
  void analyzeConservative(Node* node);
  void analyzeContainerConstruct(Node* node);
  bool tryRegisteredAnalysis(Node* node);

  /**
   * Alias manipulation methods
   */
  void makeAllAlias(const std::vector<Value*>& values);
  void makePointerTo(const Value* value, const Value* to);
  TORCH_API void addToContainedElements(
      const Value* element,
      const Value* container);
  void mapAliases(at::ArrayRef<Value*> to, at::ArrayRef<Value*> from);
  void giveFreshAlias(const Value* value);
  Element* getOrCreateElement(const Value* value);

  static bool shouldAnnotate(const Value* v);
  static bool shouldAnnotate(const TypePtr& type);
  static c10::optional<TypeKind> getMutableTypeKind(const TypePtr& type);

  static bool isContainerType(const TypePtr& type);

  std::shared_ptr<Graph> graph_;

  // The points-to graph that stores aliasing relationships
  std::unique_ptr<MemoryDAG> memoryDAG_;
  // Mapping of values to MemoryDAG elements
  ska::flat_hash_map<const Value*, Element*> elementMap_;
  // All wildcard elements (one for each unique mutable type).
  std::map<TypeKind, Element*> wildcardIndex_;
  Element* getWildcard(const TypePtr& type) const;
  Element* getOrCreateWildcard(const TypePtr& type);
  bool mayAliasWildcard(const Value* v) const;

  /**
   * State for tracking write info.
   */
  // Map of nodes to the memory locations that they write to
  ska::flat_hash_map<Node*, MemoryLocations> writeIndex_;
  // Set of all memory locations that may have been written to.
  mutable MemoryLocations writeCache_;
  mutable bool isWriteCacheStale_ = true;
  void rebuildWriteCache() const;
};

// Used to assert that unschematized operators have an analysis method written
TORCH_API bool aliasAnalysisHasSpecialCaseFor(c10::Symbol sym);
} // namespace jit
} // namespace torch
