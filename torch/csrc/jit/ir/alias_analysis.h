#pragma once

#include <ATen/core/alias_info.h>
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/utils/memory_dag.h>
#include <torch/csrc/jit/ir/type_hashing.h>

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
 * the wildcard alias set as potentially aliasing any value within the same
 * type class. Whenever a value becomes contained by another value, such as
 * when a Tensor is appended to a List[Tensor], the contained element becomes
 * part of the wildcard set.
 *
 * Values that contain other mutable types, such as List[Tensor], are
 * initialized as containing the Wildcard set for all contained mutable types.
 *
 */
class AliasDb {
 public:
  TORCH_API explicit AliasDb(
      std::shared_ptr<Graph> graphi,
      bool isFrozen = false);
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
  TORCH_API bool writesToAlias(Node* n, const ValueSet& vs) const;

  // Does `a` and `b` potentially share a memory location or do either
  // hold in memory any element that exists in the other
  TORCH_API bool mayContainAlias(Value* a, Value* b) const;

  // Do any values in group `a` share a memory location or hold in memory
  // any element that exists in group `b`
  TORCH_API bool mayContainAlias(
      const at::ArrayRef<Value*> a,
      const at::ArrayRef<Value*> b) const;

  // Do `a` and `b` potentially share a memory location?
  TORCH_API bool mayAlias(const Value* a, const Value* b) const;
  // Do any values in group `a` potentially share a memory location with any
  // value in group `b`? i.e. may they overlap?
  TORCH_API bool mayAlias(const ValueSet& a, const ValueSet& b) const;

  // Do any nodes write to an alias set input to `n`?
  TORCH_API bool hasInputWriters(const Node* n) const;

  // Do any nodes write to an alias set output by `n`?
  TORCH_API bool hasOutputWriters(const Node* n) const;

  // Do any nodes write to an alias set inputed/outputed by `n`?
  TORCH_API bool hasWriters(const Node* n) const;

  // Do any nodes write to `v`s memory location?
  TORCH_API bool hasWriters(const Value* v) const;

  // Is the operation in-place? i.e. doesn't write anywhere but locations it
  // reads from.
  TORCH_API bool isMutable(Node* n) const;

  // Is it safe to change whether `a` and `b` alias each other ?
  TORCH_API bool safeToChangeAliasingRelationship(
      const at::ArrayRef<Value*>& a,
      const at::ArrayRef<Value*>& b) const;

  // Move 'n' (already in the graph) after 'movePoint' in the topological order.
  //
  // Tries to preserve value dependencies, so other nodes might be moved. We
  // make two guarantees about the postcondition of the node list:
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
  TORCH_API std::string toString() const;

  static bool mutableType(const Value* v);
  static bool mutableType(const TypePtr& type);

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
  MemoryLocations getWrites(Node* n) const;
  void getWritesImpl(Node* n, MemoryLocations& ret) const;
  // Register the fact that `n` writes to `v`.
  void registerWrite(const Value* v, Node* n);
  void registerWrite(const Element* e, Node* n);
  // Get all the values that `n` reads from.
  // if `recurseBlocks` is true, gather reads on the nodes in `n`s sub-blocks
  MemoryLocations getReads(Node* n) const;
  void getReadsImpl(Node* n, MemoryLocations& ret) const;

  /**
   * Wildcard methods
   */
  // Register `v` as a wildcard value.
  c10::optional<Element*> setWildcard(const Value* v);

  // Is this a value which will not alias
  bool nonAliasingValue(const Value* elem) const;

  bool escapesScope(const at::ArrayRef<Value*>& vs) const;

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
  void analyzeRpcAsync(Node* node);
  void analyzeGradOf(Node* node);
  void analyzeSetAttr(Node* node);
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

  static c10::optional<TypeKind> getMutableTypeKind(const TypePtr& type);

  static bool isContainerType(const TypePtr& type);

  std::shared_ptr<Graph> graph_;

  // If the Module is frozen then consider attributes as freshly created
  // objects. Freezing API invokes alias analysis to check if they are mutated
  // internally.
  bool isFrozen_;

  // The points-to graph that stores aliasing relationships
  std::unique_ptr<MemoryDAG> memoryDAG_;
  // Mapping of values to MemoryDAG elements
  ska::flat_hash_map<const Value*, Element*> elementMap_;
  // All wildcard elements (one for each unique mutable type).
  std::unordered_map<TypePtr, Element*, HashType, EqualType> wildcardIndex_;
  Element* getWildcard(const TypePtr& type) const;
  c10::optional<Element*> tryGetOrCreateWildcard(const TypePtr& type);
  void addContainedTypesToFreshElement(
      Element* container_elem,
      const TypePtr& mut_type);

  std::vector<Element*> getElements(at::ArrayRef<Value*> vs) const;
  bool mayAliasWildcard(const Value* v) const;
  bool mayAliasWildcard(const at::ArrayRef<Value*> vs) const;
  bool hasWriters(const at::ArrayRef<Value*>& values) const;

  /**
   * State for tracking write info.
   */
  // Map of nodes to the memory locations that they write to
  ska::flat_hash_map<Node*, MemoryLocations> writeIndex_;
  // Set of all memory locations that may have been written to.
  mutable MemoryLocations writeCache_;
  mutable bool isWriteCacheStale_ = true;
  void rebuildWriteCache() const;
  std::string getElementName(const Element* e) const;
};

} // namespace jit
} // namespace torch
