#include <torch/csrc/jit/passes/alias_analysis.h>

#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {
namespace {
bool shouldAnnotate(const TypePtr& type) {
  return type->isSubtypeOf(DynamicType::get()) ||
      type->kind() == TypeKind::ListType ||
      type->kind() == TypeKind::TupleType ||
      type->kind() == TypeKind::VarType ||
      (type->kind() == TypeKind::OptionalType &&
       shouldAnnotate(type->cast<OptionalType>()->getElementType()));
}

// We only need to annotate values that either are mutable or could contain
// mutable types.
bool shouldAnnotate(const Value* v) {
  return shouldAnnotate(v->type());
}
} // namespace

// A union find-ish way to track values and the alias sets they belong to.
//
// Being a member of an alias set means that the value *may* alias any value in
// the set.
//
// Values can belong in more than one alias set. This can happen if, e.g. it
// takes on two different aliases depending on branching control flow.
class AliasSetTracker {
 public:
  // Returns true iff `v` is present in the alias set tracker.
  bool contains(const Value* v) const {
    return map_.count(v);
  }

  void erase(Value* v) {
    if (!contains(v)) {
      return;
    }

    for (auto set : map_.at(v)) {
      erase(set);
      // WARNING `set` is now a dangling pointer, so don't do anything with it!
    }
    map_.erase(v);
  }

  void erase(Node* n) {
    // We should only erase unused nodes
    for (const auto output : n->outputs()) {
      JIT_ASSERT(output->uses().size() == 0);
    }

    // Only support schematized nodes for now. If/loop will require special
    // handling for inner blocks
    JIT_ASSERT(n->maybeSchema());

    for (const auto v : n->inputs()) {
      erase(v);
    }
    for (const auto v : n->outputs()) {
      erase(v);
    }
  }

  // Whether `a` *may be* an alias of `b`
  bool isAlias(const Value* a, const Value* b) const {
    if (isWildcard(a) || isWildcard(b)) {
      return true;
    }
    const auto& aSets = map_.at(a);
    const auto& bSets = map_.at(b);
    for (auto aElement : aSets) {
      for (auto bElement : bSets) {
        if (isSameSet(aElement, bElement)) {
          return true;
        }
      }
    }
    return false;
  }

  // Register `v` as a member of the wildcard set.
  void setWildcard(const Value* v) {
    if (isWildcard(v)) {
      return;
    }

    if (wildcardSet_) {
      registerMembership(v, wildcardSet_);
    } else {
      wildcardSet_ = makeSet(v);
      map_.insert({v, {wildcardSet_}});
    }
  }

  // Returns whether `v` is a wildcard.
  bool isWildcard(const Value* v) const {
    if (!contains(v)) {
      // This value is not tracked
      return false;
    }

    if (!wildcardSet_) {
      // Wildcard set is empty
      return false;
    }

    const auto& sets = map_.at(v);
    return std::any_of(sets.begin(), sets.end(), [&](Element* set) {
      return isSameSet(set, wildcardSet_);
    });
  }

  // Give `v` a "fresh" alias set (i.e. it does not alias any other value).
  void makeFreshAlias(const Value* v) {
    auto set = makeSet(v);
    map_.insert({v, {set}});
  }

  // Make `a` and alias of `b`, by making `a` a member of all of `b`'s alias
  // sets.
  //
  // Returns true if a change occured to a's alias set
  bool makeAlias(const Value* a, const Value* b) {
    if (!map_.count(b)) {
      makeFreshAlias(b);
    }

    bool changed = false;
    for (const auto& set : map_.at(b)) {
      const bool aHasSet = map_.count(a) && map_.at(a).count(set);
      if (!aHasSet) {
        changed = true;
        registerMembership(a, set);
      }
    }
    return changed;
  }

  // Get all values that may alias to `v`.
  // NOTE: This does not include wildcards.
  std::unordered_set<const Value*> getAliases(const Value* v) const {
    std::unordered_set<const Value*> aliases;
    // Collect all aliases in all sets that `v` may have membership in.
    for (auto set : map_.at(v)) {
      auto cur = set;
      auto end = cur;

      // Traverse the next-list and find all the values.
      // Insert the first element
      aliases.insert(cur->value);
      cur = cur->next;

      while (cur != end) {
        aliases.insert(cur->value);
        cur = cur->next;
      }
    }

    return aliases;
  }

  void registerWrite(const Value* v, Node* writer) {
    for (auto set : map_.at(v)) {
      const auto root = find(set);
      writes_[root].insert(writer);
    }
  }

  const std::unordered_set<Node*>& getWildcardWriters() const {
    // Static so we can always return without a copy.
    static std::unordered_set<Node*> empty;
    if (!wildcardSet_) {
      return empty;
    }

    if (!writes_.count(find(wildcardSet_))) {
      return empty;
    }

    return writes_.at(find(wildcardSet_));
  }

  // NOTE: this does not include writes to the wildcard.
  std::unordered_set<Node*> getWrites(const Value* v) const {
    std::unordered_set<Node*> ret;
    if (!map_.count(v)) {
      return ret;
    }

    for (auto set : map_.at(v)) {
      const auto root = find(set);
      if (!writes_.count(root)) {
        continue;
      }

      for (auto write : writes_.at(root)) {
        ret.insert(write);
      }
    }
    return ret;
  }

  // Returns whether `n` writes to the value `v`
  bool writesTo(Node* n, const Value* v) const {
    return getWrites(v).count(n);
  }

  // Returns whether there are no writes in the whole graph
  bool hasNoWrites() const {
    return !writes_.empty();
  }

  // Dump the contents of the alias db to stdout in human-readable form
  void dump() const {
    std::unordered_set<Element*> roots;
    for (auto& ptrPair : sets_) {
      roots.insert(find(ptrPair.first));
    }

    size_t setId = 0;
    for (auto set : roots) {
      if (wildcardSet_ && set == find(wildcardSet_)) {
        std::cout << "WILDCARDS: ";
        dump(set);
        continue;
      }

      std::cout << "Set " << setId << ": ";
      dump(set);

      if (writes_.count(set)) {
        std::cout << "  Writes:\n";
        for (auto writer : writes_.at(set)) {
          std::cout << "    " << *writer;
        }
      }

      setId++;
    }

    std::cout << "\n";
  }

 private:
  // Represents a value's membership in an alias set. This is the "element" of
  // a union find, so the root element represents the set itself.
  struct Element {
    const Value* value;
    // Root entries are their own parent.
    Element* parent;
    // Circular linked list contains all elements of the tree rooted at `this`
    Element* next;
    // Size of tree rooted at `this`
    size_t size;
  };

  // Create a new membership element for `v`.
  Element* makeSet(const Value* v) {
    auto el = torch::make_unique<Element>();
    el->value = v;
    el->parent = el.get();
    el->next = el.get();
    el->size = 1;

    auto rawPtr = el.get();
    sets_.emplace(rawPtr, std::move(el));
    return rawPtr;
  }

  Element* find(Element* el) const {
    JIT_ASSERT(el);
    while (el->parent != el) {
      // Path halving to speed up future queries.
      el->parent = el->parent->parent;
      el = el->parent;
    }

    return el;
  }

  void union_(Element* a, Element* b) {
    JIT_ASSERT(a);
    JIT_ASSERT(b);
    auto aRoot = find(a);
    auto bRoot = find(b);
    if (aRoot == bRoot) {
      // Already in the same set.
      return;
    }

    if (aRoot->size < bRoot->size) {
      // Ensure we're always merging the smaller tree into the bigger one.
      std::swap(aRoot, bRoot);
    }

    JIT_ASSERT(bRoot->size <= aRoot->size);

    // Merge the sets and update the size
    bRoot->parent = aRoot;
    aRoot->size = aRoot->size + bRoot->size;

    // Merge the next lists
    std::swap(aRoot->next, bRoot->next);

    // Merge the write lists
    for (auto writer : writes_[bRoot]) {
      writes_[aRoot].insert(writer);
    }
    writes_.erase(bRoot);
  }

  // Remove `toDelete` from the set tracker and free its associated memory
  //
  // NOTE: this is linear in the number of nodes in an alias set. If we are
  // erasing stuff a lot, we may need to use a more complicated approach to
  // make things faster.
  void erase(Element* toDelete) {
    // TODO HANDLE WILDCARD MAP
    const auto root = find(toDelete);

    // If we are trying to delete the root element of the set, we need to
    // select a new root.
    c10::optional<Element*> newRoot;
    if (root == toDelete) {
      newRoot = root->next;
      if (newRoot.value() == root) {
        // This can only happen if there was a single element in the set.
        // Just return in that case.
        return;
      }
    }

    // Iterate through the next-list, erasing references to `el`
    auto cur = root;
    const auto end = root;

    bool first = true; // We have to consider the root once, so the first time
                       // we see it don't terminate the loop
    while (cur != end && !first) {
      first = false;
      if (cur->next == toDelete) {
        cur->next = toDelete->next;
      }
      if (cur->parent == toDelete) {
        cur->parent = newRoot.value_or(root);
      }
    }

    // If we selected a new root, update the write map
    if (newRoot && writes_.count(root)) {
      JIT_ASSERT(root == toDelete);
      writes_[*newRoot] = std::move(writes_.at(root));
      writes_.erase(root);
    }

    sets_.erase(toDelete);
  }

  void dump(Element* el) const {
    auto cur = el;
    const auto end = el;

    // Traverse the next-list and find all the values.
    std::cout << cur->value->uniqueName();
    cur = cur->next;
    while (cur != end) {
      std::cout << ", " << cur->value->uniqueName();
      cur = cur->next;
    }

    std::cout << "\n";
  }

  // Register membership of value `a` in the given alias set.
  void registerMembership(const Value* a, Element* set) {
    const auto aSet = makeSet(a);
    union_(aSet, set);
    map_[a].insert(set);
  }

  // Test whether elements a and b are in the same set.
  bool isSameSet(Element* a, Element* b) const {
    return find(a) == find(b);
  }

  // Owning stoarge for elements.
  // It's a map of element raw ptrs => the owning ptr for easy retrieval
  std::unordered_map<Element*, std::unique_ptr<Element>> sets_;
  // Mapping values to the alias sets they are in.
  std::unordered_map<const Value*, std::unordered_set<Element*>> map_;
  // Mapping of alias sets to the nodes that write to that alias set.
  std::unordered_map<const Element*, std::unordered_set<Node*>> writes_;
  // Special set representing wildcard values.
  // Nullptr means empty (i.e. there are no wildcards in the graph)
  Element* wildcardSet_ = nullptr;
};

AliasDb::~AliasDb() = default;

AliasDb::AliasDb(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
  setTracker_ = torch::make_unique<AliasSetTracker>();
  analyze(graph_);
}

// Does `n` use or write to any wildcard aliases?
bool AliasDb::hasWildcard(const Node* n) const {
  for (const auto input : n->inputs()) {
    if (setTracker_->isWildcard(input)) {
      return true;
    }
  }

  for (const auto output : n->outputs()) {
    if (setTracker_->isWildcard(output)) {
      return true;
    }
  }
  return false;
}

bool AliasDb::writesTo(Node* n, const Value* v) const {
  if (!shouldAnnotate(v)) {
    // This is a primitive type
    return false;
  }
  return setTracker_->writesTo(n, v);
}

bool AliasDb::hasWriters(const Node* n) const {
  if (hasWildcard(n)) {
    // If `n` has a wildcard, any write in the graph may write to it.
    // So the only way we know there are no writers is if there are no writes
    // at all.
    // TODO this needs to get replaced
    return setTracker_->hasNoWrites();
  }
  return getWriters(n).size() != 0;
}

bool AliasDb::hasWritersBefore(const Node* n) const {
  if (hasWildcard(n)) {
    return true;
  }
  const auto writers = getWriters(n);
  return std::any_of(writers.cbegin(), writers.cend(), [&](const Node* writer) {
    return isBeforeSameGraph(writer, n);
  });
}

bool AliasDb::hasWrites(Node* n) const {
  for (const auto input : n->inputs()) {
    if (writesTo(n, input)) {
      return true;
    }
  }
  for (const auto output : n->outputs()) {
    if (writesTo(n, output)) {
      return true;
    }
  }
  return false;
}

bool AliasDb::writesToInputAlias(Node* n) const {
  std::vector<const Value*> writes;
  for (const auto input : n->inputs()) {
    if (writesTo(n, input)) {
      writes.push_back(input);
    }
  }
  for (const auto output : n->outputs()) {
    if (writesTo(n, output)) {
      writes.push_back(output);
    }
  }

  // For all writes, check if the written value may alias a graph input
  return std::any_of(writes.cbegin(), writes.cend(), [&](const Value* v) {
    return std::any_of(
        graph_->inputs().cbegin(),
        graph_->inputs().cend(),
        [&](const Value* graphInput) {
          return shouldAnnotate(graphInput) &&
              setTracker_->isAlias(graphInput, v);
        });
  });
}

std::unordered_set<Node*> AliasDb::getWriters(const Node* n) const {
  std::unordered_set<Node*> writers;

  for (const auto input : n->inputs()) {
    for (auto writer : setTracker_->getWrites(input)) {
      writers.insert(writer);
    }
  }

  for (const auto output : n->outputs()) {
    for (auto writer : setTracker_->getWrites(output)) {
      writers.insert(writer);
    }
  }

  // A write to the wildcard set should be considered a write to `n`
  const auto& wildcardWriters = setTracker_->getWildcardWriters();
  for (auto writer : wildcardWriters) {
    writers.insert(writer);
  }

  return writers;
}

std::unordered_set<const Value*> AliasDb::getAliases(const Value* v) const {
  std::unordered_set<const Value*> ret;
  if (!setTracker_->contains(v)) {
    return ret;
  }

  return setTracker_->getAliases(v);
}

std::unordered_set<const Value*> AliasDb::getWrites(Node* n) const {
  std::unordered_set<const Value*> writes;
  for (const auto input : n->inputs()) {
    if (writesTo(n, input)) {
      writes.insert(input);
    }
  }
  for (const auto output : n->outputs()) {
    if (writesTo(n, output)) {
      writes.insert(output);
    }
  }
  return writes;
}

void AliasDb::dump() const {
  std::cout << "\n===1. GRAPH===\n";
  graph_->dump();

  std::cout << "\n===2. ALIAS DB===\n";
  setTracker_->dump();
}

void AliasDb::analyze(const std::shared_ptr<Graph>& graph) {
  // Assign aliases to the graph's inputs, assuming that all inputs of a given
  // type may alias to each other.

  // 1. Partition inputs by their type
  std::map<TypeKind, std::vector<Value*>> listTypes;
  std::unordered_map<TupleTypePtr, std::vector<Value*>> tupleTypes;
  std::vector<Value*> tensors;

  for (auto input : graph->inputs()) {
    auto inputType = input->type();
    // unwrap optional types
    if (inputType->kind() == TypeKind::OptionalType) {
      inputType = inputType->cast<OptionalType>()->getElementType();
    }

    if (inputType->isSubtypeOf(DynamicType::get())) {
      tensors.push_back(input);
    } else if (inputType->kind() == TypeKind::ListType) {
      auto containedType = inputType->containedTypes().at(0);
      // All tensor subtypes may alias to each other, so we should consider all
      // lists of them to alias to each other.
      if (containedType->isSubtypeOf(DynamicType::get())) {
        containedType = DynamicType::get();
      }
      listTypes[containedType->kind()].push_back(input);
    } else if (inputType->kind() == TypeKind::TupleType) {
      auto tupleType = inputType->cast<TupleType>();
      tupleTypes[tupleType].push_back(input);
    } else {
      JIT_ASSERT(!shouldAnnotate(input));
    }
  }

  // 2. Make all partitions alias each other
  for (const auto& pr : listTypes) {
    auto& values = pr.second;
    for (const auto value : values) {
      setTracker_->makeAlias(value, values[0]);
    }
  }
  for (const auto& pr : tupleTypes) {
    auto& values = pr.second;
    for (const auto value : values) {
      setTracker_->makeAlias(value, values[0]);
    }
  }
  for (const auto value : tensors) {
    setTracker_->makeAlias(value, tensors[0]);
  }
  analyze(graph->block());
}

void AliasDb::analyze(Block* block) {
  for (auto node : block->nodes()) {
    analyze(node);
  }
}

void AliasDb::analyze(Node* node) {
  analyzeImpl(node);

  // After analyzing, update the wildcard index
  if (hasWildcard(node)) {
    wildcardNodes_.insert(node);
  }
}

// The basic strategy is:
//   1. Retrieve alias information for every input.
//   2. Use the node's schema's alias annotations to propgagate alias/write
//      information to the outputs. For unschematized nodes, a special analyzer
//      will have to be handwritten.
void AliasDb::analyzeImpl(Node* node) {
  // These nodes are not schematized, so we need to handle them specially
  // TODO do the thing that python_printer does to force operator writers to
  // register aliasing information
  switch (node->kind()) {
    case prim::If:
      return analyzeIf(node);
    case prim::Loop:
      return analyzeLoop(node);
    case prim::FusionGroup:
    case prim::DifferentiableGraph:
      return analyzeSubgraph(node);
    case prim::Constant:
    case prim::ListConstruct:
    case prim::TupleConstruct:
    case prim::Undefined:
    case prim::FusedConcat:
    case prim::MMTreeReduce:
    case prim::MMBatchSide:
    case prim::None:
    case prim::BroadcastSizes:
    case prim::ChunkSizes:
    case prim::Function:
      return analyzeCreator(node);
    case prim::TupleUnpack:
    case prim::TupleIndex:
    case prim::TupleSlice:
    case prim::ListUnpack:
    case prim::PythonOp:
      return analyzeExtractor(node);
    case prim::ConstantChunk:
      return analyzeChunk(node);
    case prim::BroadcastingChunk:
      return analyzeBroadcastingChunk(node);
    case aten::add:
    case aten::sub:
    case aten::mul:
    case aten::div: {
      // This is necessary because we sometimes get unschematized combinations
      // of Tensor/primitive.
      auto maybeSchema = node->maybeSchema();
      if (!maybeSchema) {
        return analyzeCreator(node);
      }
      // If the node has a schema, fall through and analyze it normally
      break;
    }
  }

  const auto& schema = node->schema();
  if (schema.is_vararg() || schema.is_varret()) {
    const auto hasMutableOutputs = std::any_of(
        node->outputs().cbegin(),
        node->outputs().cend(),
        [](const Value* output) { return shouldAnnotate(output); });

    // We don't have alias info for this node. Either schematize it, or
    // add it an analyze* method for it.
    if (hasMutableOutputs) {
      throw script::ErrorReport(node->getSourceLocation())
          << "Alias information not found for node. File a bug report.\n"
          << "Node: " << *node << "\n";
    }
  }

  // Bind formal alias annotation to actual alias sets
  std::unordered_map<Symbol, Value*> formalToActual;
  for (size_t i = 0; i < schema.arguments().size(); i++) {
    const auto& formal = schema.arguments()[i].alias_info();
    const auto& actualValue = node->inputs().at(i);
    // Skip if there's no alias annotation
    if (!formal) {
      continue;
    }

    // If this type cannot alias, continue. Can occur with a VarType schema
    if (!shouldAnnotate(actualValue)) {
      continue;
    }

    // We don't support composite types for alias analysis yet.
    JIT_ASSERT(formal->containedTypes().size() == 0);
    // TODO neither unions nor wildcards make sense on an input. We should
    // disallow them in function schema
    JIT_ASSERT(!formal->isWildcard())
    const auto& formalAlias = formal->set();

    // skip if we've already bound this alias
    if (formalToActual.count(formalAlias) != 0) {
      continue;
    }

    // Bind the formal to the actual
    formalToActual[formalAlias] = actualValue;

    // Record writes
    if (formal->isWrite()) {
      setTracker_->registerWrite(actualValue, node);
    }
  }

  // Use the formal-actual mapping to give aliases to the outputs
  for (size_t i = 0; i < schema.returns().size(); i++) {
    const auto actual = node->outputs().at(i);
    const auto& formal = schema.returns()[i].alias_info();
    if (!formal) {
      // This is a fresh tensor
      giveFreshAlias(actual);
      continue;
    }

    // If this type cannot alias, continue. Can occur with a VarType schema
    if (!shouldAnnotate(actual)) {
      continue;
    }

    // We don't support composite types for alias analysis yet.
    JIT_ASSERT(formal->containedTypes().size() == 0);

    const auto& formalAlias = formal->set();
    if (formal->isWildcard()) {
      setTracker_->setWildcard(actual);
    } else {
      auto toAlias = formalToActual.at(formalAlias);
      addAlias(actual, toAlias);
    }

    if (formal->isWrite()) {
      setTracker_->registerWrite(actual, node);
    }
  }
}

void AliasDb::analyzeIf(Node* node) {
  // For if statements, the alias set of an output is the union of the
  // alias sets generated by the if and else block
  const auto trueBlock = node->blocks().at(0);
  const auto falseBlock = node->blocks().at(1);
  analyze(trueBlock);
  analyze(falseBlock);

  for (size_t i = 0; i < node->outputs().size(); i++) {
    const auto nodeOutput = node->outputs()[i];

    const auto trueOutput = trueBlock->outputs().at(i);
    const auto falseOutput = falseBlock->outputs().at(i);

    addAlias(nodeOutput, trueOutput);
    addAlias(nodeOutput, falseOutput);
  }
}

void AliasDb::analyzeLoop(Node* node) {
  const auto bodyBlock = node->blocks().at(0);
  const auto loopCarriedInputs = node->inputs().slice(2); // skip max, cond
  const auto blockInputs = bodyBlock->inputs().slice(1); // skip trip
  const auto blockOutputs = bodyBlock->outputs().slice(1); // skip trip
  JIT_ASSERT(loopCarriedInputs.size() == blockInputs.size());
  JIT_ASSERT(blockOutputs.size() == node->outputs().size());

  // Run alias analysis on the loop body, iterating until the block output
  // alias info converges.
  auto notConverged = true;
  while (notConverged) {
    // Copy node input aliases to block input
    mapAliases(blockInputs, loopCarriedInputs);

    // Populate block output alias info by analyzing the body
    analyze(bodyBlock);

    // Copy the alias info from the block output to the node output
    mapAliases(node->outputs(), blockOutputs);

    // Merge alias info from block outputs to the node inputs.
    notConverged = false;
    for (size_t i = 0; i < blockOutputs.size(); i++) {
      const auto input = loopCarriedInputs[i];
      const auto output = blockOutputs[i];

      // Check whether or not this would change anything
      notConverged = addAlias(input, output);
    }
  }
}

void AliasDb::analyzeSubgraph(Node* node) {
  const auto subgraph = node->g(attr::Subgraph).get();

  subgraphToOwner_.insert({subgraph, node});

  const auto subgraphBlock = subgraph->block();
  mapAliases(subgraphBlock->inputs(), node->inputs());

  analyze(subgraphBlock);

  // TODO(suo): the subgraph outputs and node outputs are NOT NECESSARILY the
  // same length. Autodifferentiation maybe capture additional outputs in the
  // subgraph block.
  JIT_ASSERT(subgraphBlock->outputs().size() >= node->outputs().size());
  for (size_t i = 0; i < node->outputs().size(); i++) {
    addAlias(node->outputs()[i], subgraphBlock->outputs()[i]);
  }
}

// For nodes that generate a fresh value from nothing
void AliasDb::analyzeCreator(Node* node) {
  for (Value* output : node->outputs()) {
    giveFreshAlias(output);
  }
}

// For nodes that extract values from a composite type. Right now, this just
// gives up and creates wildcards for everything.
void AliasDb::analyzeExtractor(Node* node) {
  for (const auto output : node->outputs()) {
    setTracker_->setWildcard(output);
  }
}

// For torch.chunk(), all returned tensors may alias the input tensor
void AliasDb::analyzeChunk(Node* node) {
  for (auto output : node->outputs()) {
    addAlias(output, node->input());
  }
}

// BroadcastingChunk: all inputs are broadcasted, and then individually chunked.
// This is an intermediate node used only in the graph fuser.
void AliasDb::analyzeBroadcastingChunk(Node* node) {
  auto inputs = node->inputs();
  auto outputs = node->outputs();
  auto nchunks = node->i(attr::chunks);
  for (size_t index = 0; index < inputs.size(); ++index) {
    // Each inputs[i] is aliased by exactly `nchunks` distinct output tensors:
    // inputs[i] produces chunks outputs[i * nchunks + k] for k in [0..nchunks)
    auto output_begin = outputs.begin() + index * nchunks;
    for (auto it = output_begin; it != output_begin + nchunks; ++it) {
      addAlias(*it, inputs.at(index));
    }
  }
}

// Union the alias info of `value` with `from`
// Returns true if a change occured to `value`s alias set.
bool AliasDb::addAlias(const Value* value, const Value* from) {
  if (!shouldAnnotate(value)) {
    JIT_ASSERT(!shouldAnnotate(from));
    return false;
  }
  return setTracker_->makeAlias(value, from);
}

void AliasDb::mapAliases(at::ArrayRef<Value*> to, at::ArrayRef<Value*> from) {
  JIT_ASSERT(to.size() == from.size());
  for (size_t i = 0; i < to.size(); i++) {
    addAlias(to[i], from[i]);
  }
}

void AliasDb::giveFreshAlias(const Value* value) {
  if (!shouldAnnotate(value)) {
    return;
  }

  if (setTracker_->contains(value)) {
    // Inside a loop, we may have given a fresh alias to this value already, so
    // skip
    return;
  }

  setTracker_->makeFreshAlias(value);
}

bool AliasDb::moveAfterTopologicallyValid(Node* n, Node* movePoint) {
  return tryMove(n, movePoint, MoveSide::AFTER, /*dryRun=*/false);
}

bool AliasDb::couldMoveAfterTopologically(Node* n, Node* movePoint) {
  return tryMove(n, movePoint, MoveSide::AFTER, /*dryRun=*/true);
}

bool AliasDb::moveBeforeTopologicallyValid(Node* n, Node* movePoint) {
  // We have to distinguish the move side (instead of just moving after
  // n->prev()). Consider the following example:
  // If the dependency graph looks like
  //   n -> movePoint -> o
  // then moveBefore(o) will end up with
  //   n, o, movePoint
  // but moveAfter(n) will return false.
  return tryMove(n, movePoint, MoveSide::BEFORE, /*dryRun=*/false);
}

bool AliasDb::couldMoveBeforeTopologically(Node* n, Node* movePoint) {
  return tryMove(n, movePoint, MoveSide::BEFORE, /*dryRun=*/true);
}

// Helper for topologically-safe node moves. See `tryMove()` for details.
class AliasDb::WorkingSet {
 public:
  explicit WorkingSet(Node* mover, const AliasDb& aliasDb) : aliasDb_(aliasDb) {
    add(mover);
  }

  // Add `n` to the working set
  void add(Node* n) {
    nodes_.push_back(n);
    for (const auto user : getUsersSameBlock(n)) {
      users_[user]++;
    }

    for (const auto& writer : getWritersSameBlock(n)) {
      writers_[writer]++;
    }
    if (aliasDb_.hasWildcard(n)) {
      numWildcards_++;
    }
    if (aliasDb_.hasWrites(n)) {
      numWriterNodes_++;
    }
  }

  void eraseMover() {
    auto mover = nodes_.front();
    for (const auto user : getUsersSameBlock(mover)) {
      // If this user node only uses the mover, we can remove it
      if (users_[user] == 1) {
        users_.erase(user);
      }
    }

    for (const auto& writer : getWritersSameBlock(mover)) {
      if (writers_[writer] == 1) {
        writers_.erase(writer);
      }
    }
    if (aliasDb_.hasWildcard(mover)) {
      numWildcards_--;
    }
    if (aliasDb_.hasWrites(mover)) {
      numWriterNodes_--;
    }
    nodes_.pop_front();
  }

  const std::list<Node*>& nodes() {
    return nodes_;
  }

  // Does the working set depend on `n`?
  bool dependsOn(Node* n) const {
    if (nodes_.empty()) {
      return false;
    }

    return hasDataDependency(n) || hasMutabilityDependency(n);
  }

 private:
  bool hasDataDependency(Node* n) const {
    if (n->isAfter(nodes_.front())) {
      return producesFor(n);
    } else {
      return consumesFrom(n);
    }
  }

  bool hasMutabilityDependency(Node* n) const {
    // 1. Handle wildcard dependencies:
    // If the working set has a wildcard, `n` can't write to anything.
    if (numWildcards_ > 0 && aliasDb_.hasWrites(n)) {
      return true;
    }

    // If `n` has a wildcard, the working set can't write to anything.
    if (aliasDb_.hasWildcard(n) && numWriterNodes_ > 0) {
      return true;
    }

    // 2. Handle regular mutable dependencies
    // Check that this node does not write to anything used by the working set
    if (writers_.count(n) != 0) {
      return true;
    }

    // Check that the working set does not write to anything used by this node
    const auto writersToNode = getWritersSameBlock(n);
    return std::any_of(nodes_.begin(), nodes_.end(), [&](Node* node) {
      return writersToNode.count(node) != 0;
    });
  }

  // Does the working set produce any values consumed by `n`?
  bool producesFor(Node* n) const {
    // This equivalent to asking: does the total use-set of all the nodes in the
    // working set include `n`?
    return users_.count(n) != 0;
  }

  // Does the working set consume any values produced by `n`?
  bool consumesFrom(Node* n) const {
    const auto users = getUsersSameBlock(n);
    return std::any_of(nodes_.begin(), nodes_.end(), [&](Node* node) {
      return users.count(node) != 0;
    });
  }

  // Get all users of outputs of `n`, in the same block as `n`.
  // This means if there is an `if` node that uses an output of `n` in some
  // inner sub-block, we will consider the whole `if` node a user of `n`.
  std::unordered_set<Node*> getUsersSameBlock(Node* n) const {
    std::unordered_set<Node*> users;
    for (const auto output : n->outputs()) {
      for (const auto& use : output->uses()) {
        if (auto sameBlock = findSameBlock(use.user, n)) {
          users.insert(sameBlock);
        }
      }
    }
    return users;
  }

  std::unordered_set<Node*> getWritersSameBlock(Node* n) const {
    std::unordered_set<Node*> writers;
    for (const auto writer : aliasDb_.getWriters(n)) {
      if (auto sameBlock = findSameBlock(writer, n)) {
        writers.insert(sameBlock);
      }
    }
    return writers;
  }

  // Traverse `target`'s blockchain upward until we find a node that shares a
  // block with `n`.
  //
  // If one can't be found (say, because `n` is an inner block and target is
  // outside), then return nullptr. Since we can only reorder nodes within a
  // block, `target` would be irrelevant.
  static Node* findSameBlock(Node* target, Node* n) {
    JIT_ASSERT(target->owningGraph() == n->owningGraph());
    if (target->owningBlock() == n->owningBlock()) {
      return target;
    } else {
      // This user is in a sub-block. Traverse the blockchain upward until
      // we arrive at a node that shares a block with `this`
      auto curNode = target;
      while (curNode->owningBlock() != n->owningBlock()) {
        curNode = curNode->owningBlock()->owningNode();
        if (curNode == nullptr) {
          return curNode;
        }
      }
      return curNode;
    }
  }

  const AliasDb& aliasDb_;
  std::list<Node*> nodes_;
  // users => # of working set nodes it uses
  std::unordered_map<Node*, size_t> users_;
  std::unordered_map<Node*, size_t> writers_;
  size_t numWildcards_ = 0;
  size_t numWriterNodes_ = 0;
};

// Try to move `toMove` before/after `movePoint` while preserving value
// dependencies. Returns false iff such a move could not be made.
//
// If `dryRun` is set, don't actually execute the move, just check if the move
// is possible
//
// The basic approach is: have a "working set" that we are moving forward, one
// node at a time. When we can't move past a node (because it depends on the
// working set), then add it to the working set and keep moving until we hit
// `moveAfter`.
bool AliasDb::tryMove(
    Node* toMove,
    Node* movePoint,
    MoveSide moveSide,
    bool dryRun) {
  JIT_ASSERT(toMove->owningBlock() == movePoint->owningBlock());
  if (toMove == movePoint) {
    return true;
  }

  // 1. Move from `this` toward movePoint, building up the working set of
  // dependencies
  WorkingSet workingSet(toMove, *this);

  int direction;
  if (toMove->isAfter(movePoint)) {
    direction = kPrevDirection;
  } else {
    direction = kNextDirection;
  }

  auto curNode = toMove->next_in_graph[direction];
  // Move forward one node at a time
  while (curNode != movePoint) {
    if (workingSet.dependsOn(curNode)) {
      // If we can't move past this node, add it to the working set
      workingSet.add(curNode);
    }
    curNode = curNode->next_in_graph[direction];
  }

  // 2. Decide whether we can move it all to `movePoint`.

  // Say we are moving directly before movePoint and `toMove` starts before
  // movePoint in the graph. The move looks like
  //
  //  `toMove`            `toMove`         |
  //  <dependencies>  ->  `movePoint`      | `toMove` and deps are split
  //  `movePoint`         <dependencies>   |
  //
  // Contrast with the case where `toMove` starts AFTER movePoint:
  //
  //  `movePoint`           <dependencies>   |
  //  <dependencies>  ->    `toMove`         | `toMove` and deps are together
  //  `toMove`              `movePoint`      |
  //
  // In the first case, we need to split `this` off from its dependencies, so we
  // can move the dependencies below `movePoint` and keep `toMove` above.
  const bool splitToMoveAndDeps =
      (moveSide == MoveSide::BEFORE && toMove->isBefore(movePoint)) ||
      (moveSide == MoveSide::AFTER && toMove->isAfter(movePoint));

  if (splitToMoveAndDeps) {
    // remove `this` from dependencies to be moved past `movePoint`
    workingSet.eraseMover();
  }

  // Check if we can move the working set past the move point
  if (workingSet.dependsOn(movePoint)) {
    // if we can't, then there are intermediate dependencies between the
    // `this` and `movePoint`, so we can't do the move
    return false;
  }

  if (dryRun) {
    return true;
  }

  // 3. Execute the move
  JIT_ASSERT(curNode == movePoint);
  if (splitToMoveAndDeps) {
    // Move `toMove`
    move(toMove, movePoint, moveSide);

    // Then move all of its dependencies on the other side of `movePoint`
    const auto reversed =
        moveSide == MoveSide::BEFORE ? MoveSide::AFTER : MoveSide::BEFORE;
    for (auto n : workingSet.nodes()) {
      move(n, curNode, reversed);
      curNode = n;
    }
  } else {
    // Just append/prepend everything to `movePoint`
    for (auto n : workingSet.nodes()) {
      move(n, curNode, moveSide);
      curNode = n;
    }
  }
  return true;
}

// Helper function so we can generalize `tryMove`
void AliasDb::move(Node* toMove, Node* movePoint, MoveSide moveSide) {
  switch (moveSide) {
    case MoveSide::BEFORE:
      toMove->moveBefore(movePoint);
      break;
    case MoveSide::AFTER:
      toMove->moveAfter(movePoint);
      break;
  }
}

bool AliasDb::hasUntrackedEffects(Node* node) const {
  bool touchesWildcard = false;
  if (const auto lastWildcard = getLastWildcard()) {
    touchesWildcard = hasWrites(node) &&
        (isBeforeSameGraph(node, *lastWildcard) || node == *lastWildcard);
  }

  return writesToInputAlias(node) || touchesWildcard;
}

// Nodes must be in the same graph in order to do `isBefore` or `isAfter`. This
// traverses the subgraph "chain" upward until we find two nodes that share an
// owning graph.
//
// NOTE: this is n^2 in subgraph depth. Right now the maximum depth is like 2,
// but if we ever do huge nested subgraphs we'll need to reconsider this.
bool AliasDb::isBeforeSameGraph(const Node* a, const Node* b) const {
  auto lhs = a;
  while (true) {
    auto rhs = b;
    while (true) {
      if (lhs->owningGraph() == rhs->owningGraph()) {
        return lhs->isBefore(rhs);
      }
      if (!subgraphToOwner_.count(rhs->owningGraph())) {
        break;
      }
      rhs = subgraphToOwner_.at(rhs->owningGraph());
    }
    if (!subgraphToOwner_.count(lhs->owningGraph())) {
      break;
    }
    lhs = subgraphToOwner_.at(lhs->owningGraph());
  }
  JIT_ASSERT(false);
}

c10::optional<const Node*> AliasDb::getLastWildcard() const {
  auto it = std::max_element(
      wildcardNodes_.cbegin(),
      wildcardNodes_.cend(),
      [this](const Node* a, const Node* b) { return isBeforeSameGraph(a, b); });
  if (it != wildcardNodes_.end()) {
    return *it;
  } else {
    return c10::nullopt;
  }
}

bool AliasDb::canDeinplace(Node* node) const {
  auto name = std::string(node->kind().toQualString());
  if (!hasWrites(node) || name.at(name.size() - 1) != '_') {
    // Not an in-place op.
    return false;
  }

  // Check if the node is safe to be de-inplaced:
  // 1. No wildcard node after `node`.
  const auto lastWildcard = getLastWildcard();
  if (lastWildcard && (*lastWildcard)->isAfter(node)) {
    return false;
  }
  // 2. No aliases of the output used after `node`.
  Value* output = node->output();
  for (const Value* v : getAliases(output)) {
    if (v == output) {
      continue;
    }
    for (const Use& u : v->uses()) {
      if (u.user->isAfter(node)) {
        return false;
      }
    }
  }

  return true;
}

Node* AliasDb::deinplace(Node* node) {
  JIT_ASSERT(canDeinplace(node));

  // Replace the in-place node with the out-of-place equivalent.
  auto name = std::string(node->kind().toQualString());
  WithInsertPoint insert_guard{node};
  name.pop_back(); // Remove the underscore
  Graph* graph = node->owningGraph();
  Node* deinplaced = graph->insertNode(
      graph->create(Symbol::fromQualString(name), node->inputs()));
  Value* deinplacedOutput = deinplaced->output();
  node->output()->replaceAllUsesWith(deinplacedOutput);

  // Remove from the alias db.
  erase(node);
  node->destroy();

  // Add the new node to the alias db.
  // insert(deinplaced);
  return deinplaced;
}

void AliasDb::erase(Node* node) {
  wildcardNodes_.erase(node);
  if (node->hasAttribute(attr::Subgraph)) {
    const auto subgraph = node->g(attr::Subgraph).get();
    subgraphToOwner_.erase(subgraph);
  }
  setTracker_->erase(node);
}

// This assumes that all nodes before `node` have been analyzed already.
void AliasDb::insert(Node* node) {
  analyze(node);
}
} // namespace jit
} // namespace torch
