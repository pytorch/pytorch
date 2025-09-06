#pragma once

#include <ATen/core/alias_info.h>
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/passes/create_functional_graphs.h>
#include <torch/csrc/jit/passes/utils/memory_dag.h>

namespace torch::jit {

class ValueAndMemoryLocationSet;

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
 * we're not sure what this value may alias. To be conservative, we consider the
 * wildcard alias set as potentially aliasing any other wildcard value within
 * the same type class. Whenever a value becomes contained by another value,
 * such as when a Tensor is appended to a List[Tensor], the contained element
 * becomes part of the wildcard set.
 *
 * Values that contain other mutable types, such as List[Tensor], are
 * initialized as containing the Wildcard set for all contained mutable types.
 *
 * The AliasDb API references the idea of "mutable" vs "immutable"
 * types. "Mutable" means that the object's value can change, while
 * "immutable" means that the value is fixed. (For example, `List` is
 * mutable, so you can add and delete elements from it. On the other
 * hand, you can't modify a Tuple once you create it, making `Tuple` an
 * immutable container.)
 *
 * `isFrozen` - if the Module is frozen then consider attributes as freshly
 * created objects. Freezing API invokes alias analysis to check if they are
 * mutated internally.
 *
 * `descendFunctionCalls` - recursively analyze function and method calls
 * instead of conservative analysis. Generally analysis should be done after
 * inlining so the implementation for recursive analysis is unoptimized.
 */
class AliasDb {
 public:
  TORCH_API explicit AliasDb(
      std::shared_ptr<Graph> graphi,
      bool isFrozen = false,
      bool descendFunctionCalls = false);
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

  // Does `n` write to any of the values in `vls`?
  TORCH_API bool writesToAlias(Node* n, const ValueAndMemoryLocationSet& vls)
      const;

  TORCH_API ValueAndMemoryLocationSet getValueAndMemoryLocationSet() const;

  // Does `a` and `b` potentially share a memory location or do either
  // hold in memory any element that exists in the other
  TORCH_API bool mayContainAlias(Value* a, Value* b) const;

  TORCH_API bool mayContainAlias(Value* a, const at::ArrayRef<Value*> b) const;

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

  // Do any nodes write to an alias set inputted/outputted by `n`?
  TORCH_API bool hasWriters(const Node* n) const;

  // Do any nodes write to `v`s memory location?
  TORCH_API bool hasWriters(const Value* v) const;

  // Is the operation in-place? i.e. doesn't write anywhere but locations it
  // reads from.
  TORCH_API bool isMutable(Node* n) const;

  TORCH_API bool escapesScope(const at::ArrayRef<Value*>& vs) const;

  // Is it safe to change whether `a` and `b` alias each other ?
  TORCH_API bool safeToChangeAliasingRelationship(
      const at::ArrayRef<Value*>& a,
      const at::ArrayRef<Value*>& b) const;

  // Move `n` (already in the graph) after `movePoint` in the topological order.
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

  // Generates a DOT (www.graphviz.org) graph representation
  //
  // Returns `true` if the output file was successfully generated
  //
  // WARNING: The output dot file path can't include shell specific notations,
  //  for example you can't use "~/temp/aliasdb.dot"
  //  (instead, use "/home/user/temp/aliasdb.dot")
  //
  TORCH_API bool dumpToGraphvizFile(const char* filename) const;
  TORCH_API std::string toGraphviz() const;

  // Returns `true` if the given element is mutable or if it is a
  // container type with an internal mutable element (e.g.
  // `Tuple[int, Tensor]` has an internal mutable type `Tensor`, so
  // it would be considered a "mutable type" in AliasDb)
  static bool isMutableType(const Value* v);
  static bool isMutableType(const TypePtr& type);

  /**
   * Mutation API
   *
   * These methods allow you to update AliasDb in-place if you are performing
   * graph mutation.
   *
   * WARNING: These methods should be considered INTERNAL. They do not perform
   * very many correctness checks, the user is responsible for making sure they
   * are updating AliasDb correctly. `Lint()`ing the AliasDb can help with
   * this.
   */
  // Copy `existing`s aliasing info to `new_value`, and remove `existing`.
  TORCH_API void replaceWithNewValue(Value* existing, Value* new_value);
  // Copy `from`s aliasing info to `to`.
  TORCH_API void copyValue(Value* from, Value* to);
  // Create a new `value` that does not alias anything else.
  TORCH_API void createValue(const Value* value);

  // Enable more precise treatment of prim::TupleConstruct.
  void enablePreciseTupleContainerAnalysis();

  friend struct MutationRemover;
  friend class ValueAndMemoryLocationSet;

 private:
  // Helper for topologically-safe node moves.
  class WorkingSet;
  enum class MoveSide { BEFORE, AFTER };
  bool tryMove(Node* toMove, Node* movePoint, MoveSide moveSide, bool dryRun);
  void move(Node* toMove, Node* movePoint, MoveSide moveSide);
  bool isBeforeOrAfter(const Node* n, MoveSide moveSide) const;

  bool isMutableTypeInternal(const Value* v) const;
  bool isMutableTypeInternal(const TypePtr& type) const;

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
  void registerWrite(const Value* v, Node* n, bool writeToContained = false);
  // Get all the values that `n` reads from.
  // if `recurseBlocks` is true, gather reads on the nodes in `n`s sub-blocks
  MemoryLocations getReads(Node* n) const;
  void getReadsImpl(Node* n, MemoryLocations& ret) const;
  MemoryLocations getMemoryLocations(Value* v) const;

  /**
   * Wildcard methods
   */
  // Register `v` as a wildcard value.
  std::optional<Element*> setWildcard(const Value* v);

  // Is this a value which will not alias?
  bool nonAliasingValue(const Value* elem) const;

  /**
   * Special analysis methods
   */
  void analyze(const std::shared_ptr<Graph>& graph);
  void analyze(Block* block);
  void analyze(Node* node);
  void analyzeImpl(Node* node);
  void analyzeIf(Node* node);
  void analyzeLoop(Node* node);
  void analyzeSubgraph(Node* node, const std::shared_ptr<Graph>& subgraph);
  void analyzeSubgraph(Node* node);
  void analyzeCreator(Node* node);
  void analyzeExtractor(Node* node);
  void analyzeChunk(Node* node);
  void analyzeBroadcastingChunk(Node* node);
  void analyzeFork(Node* node);
  void analyzeWait(Node* node);
  void analyzeAwaitable(Node* node);
  void analyzeAwaitableWait(Node* node);
  void analyzeRpcAsync(Node* node);
  void analyzeBatchNorm(Node* node);
  void analyzeInstanceNorm(Node* node);
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
  void giveFreshAlias(
      const Value* value,
      bool add_wildcard_to_contained_elems = true);
  Element* getOrCreateElement(const Value* value);

  const AliasTypeSet* mapTypeToAliasTypeSetPtr(const TypePtr& type) const;
  bool functionalNonEscapingListUse(const Use& use) const;
  bool functionalNonEscapingTupleUse(const Use& use) const;

  std::shared_ptr<Graph> graph_;

  // If the Module is frozen then consider attributes as freshly created
  // objects. Freezing API invokes alias analysis to check if they are mutated
  // internally.
  bool isFrozen_;

  bool descend_function_calls_;
  std::unordered_map<Graph*, std::vector<std::shared_ptr<Graph>>>
      function_call_copies_;

  // The points-to graph that stores aliasing relationships
  std::unique_ptr<MemoryDAGBuilder> memoryDAGBuilder_;
  std::unique_ptr<MemoryDAG> memoryDAG_;

  // Mapping of values to MemoryDAG elements
  ska::flat_hash_map<const Value*, Element*> elementMap_;
  // All wildcard Elements (one for each unique mutable type)
  ska::flat_hash_map<TypePtr, Element*, HashType, EqualType> wildcardIndex_;
  Element* getWildcard(const TypePtr& type) const;
  std::optional<Element*> tryGetOrCreateWildcard(const TypePtr& type);
  void addContainedTypesToFreshElement(
      Element* container_elem,
      const AliasTypeSet& mut_types);
  void pointUnionTypeElementToAllContainedTypes(
      Element* container_elem,
      const AliasTypeSet& mut_types);

  std::vector<Element*> getElements(at::ArrayRef<Value*> vs) const;
  bool mayAliasWildcard(const Value* v) const;
  bool mayAliasWildcard(const at::ArrayRef<Value*> vs) const;
  bool hasWriters(const at::ArrayRef<Value*>& values) const;

  // Cached mapping of type ptrs to their mutable types
  mutable ska::flat_hash_map<TypePtr, AliasTypeSet> mapped_mutable_types_;

  /**
   * State for tracking write info.
   */
  // Write registry where the analysis can record the writes as it sees them.
  // This information is later denormalized into various caches to improve query
  // efficiency.
  struct WriteRegistry;
  std::unique_ptr<WriteRegistry> writeRegistry_;

  // Map of nodes to the memory locations that they write to
  using TWriteIndex = ska::flat_hash_map<Node*, MemoryLocations>;
  std::optional<TWriteIndex> writeIndex_;
  // Collection of all memory locations that are written to.
  std::optional<MemoryLocations> writtenToLocationsIndex_;
  void buildWrittenToLocationsIndex();

  std::unordered_set<const Value*> wildcards_;

  std::string getElementName(const Element* e) const;

  friend void Lint(const AliasDb* db);
};

// Helper check that invariants over AliasDb are maintained.
// Useful if you are using the AliasDb mutation API and want to check you did
// the right thing.
TORCH_API void Lint(const AliasDb* db);

/**
 * ValueAndMemoryLocationSet
 *
 * A insert-only set of values which also maintains a MemoryLocations bitset
 * of the memory locations that the values alias. It is insert-only. It
 * should be constructed by calling aliasDb.getValueAndMemoryLocationSet().
 *
 * WARNING:
 *  * The AliasDb must not be mutated after construction of a
 *    ValueAndMemoryLocationsSet, or else the MemoryLocations stored in the
 *    ValueAndMemoryLocationSet will no longer be accurate.
 *  * A ValueAndMemoryLocationsSet is tied to an instance of AliasDb but
 *    does not own the AliasDb. It is the user's responsibility to ensure
 *    that the AliasDb outlives the ValuesAndMemoryLocationsSet.
 *
 * The use case for this is to be able to implement writesToAlias
 * more efficiently for a set of values.
 */
class ValueAndMemoryLocationSet {
 public:
  TORCH_API void insert(Value* v);
  TORCH_API ValueSet& getValueSet();

  friend class AliasDb;

 private:
  ValueAndMemoryLocationSet(const AliasDb* db) : aliasDb_(db) {}

  const AliasDb* aliasDb_;
  ValueSet valueSet_;
  MemoryLocations memoryLocations_;
};

} // namespace torch::jit
