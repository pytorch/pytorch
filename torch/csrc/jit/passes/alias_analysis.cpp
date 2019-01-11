#include <torch/csrc/jit/passes/alias_analysis.h>

#include <torch/csrc/jit/script/error_report.h>

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

AliasDb::AliasDb(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
  analyze(graph_);

  // Build helper indices
  // NOTE: that these assume that AliasDb is immutable once constructed.
  // - Alias set -> value mapping
  for (const auto& pr : valueToAlias_) {
    const auto value = pr.first;
    const auto& aliasInfo = pr.second;
    // We don't support composite types yet
    JIT_ASSERT(aliasInfo.containedTypes().size() == 0);
    for (const auto aliasSet : aliasInfo.sets()) {
      aliasToValue_[aliasSet].insert(value);
    }
  }
  // - Set of all nodes with a wildcard
  buildWildcardIndex(graph_->block());
}

void AliasDb::buildWildcardIndex(const Block* b) {
  for (const auto node : b->nodes()) {
    for (const auto block : node->blocks()) {
      buildWildcardIndex(block);
    }

    if (hasWildcardImpl(node)) {
      wildcardNodes_.insert(node);
    }
  }
}

bool AliasDb::hasWildcard(const Node* n) const {
  return wildcardNodes_.count(n) != 0;
}

// Does `n` use or write to any wildcard aliases?
bool AliasDb::hasWildcardImpl(const Node* n) const {
  for (const auto input : n->inputs()) {
    if (valueToAlias_.count(input) != 0 &&
        valueToAlias_.at(input).isWildcard()) {
      return true;
    }
  }

  for (const auto output : n->outputs()) {
    if (valueToAlias_.count(output) != 0 &&
        valueToAlias_.at(output).isWildcard()) {
      return true;
    }
  }
  return false;
}

bool AliasDb::writesTo(Node* n, const Value* v) const {
  if (valueToAlias_.count(v) == 0) {
    // This is a primitive type
    return false;
  }

  const auto& aliasInfo = valueToAlias_.at(v);
  JIT_ASSERT(aliasInfo.sets().size() > 0);
  // We only need to check one alias set, since if this value belongs to
  // multiple alias sets they are all written to
  const auto& aliasSet = *aliasInfo.sets().begin();

  if (aliasToWrites_.count(aliasSet) == 0) {
    // no writes to this alias set
    return false;
  }

  const auto& writers = aliasToWrites_.at(aliasSet);
  return writers.count(n) != 0;
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
    const auto& aliasInfo = valueToAlias_.at(v);
    const auto& aliasSets = aliasInfo.sets();

    // Check every distinct alias set this value belongs to
    return std::any_of(
        aliasSets.cbegin(), aliasSets.cend(), [&](const Symbol aliasSet) {
          return graphInputAliases_.count(aliasSet) != 0;
        });
  });
}

std::unordered_set<Node*> AliasDb::getWriters(const Node* n) const {
  // Get all alias sets of this node
  // ... check the inputs
  std::unordered_set<Symbol> aliasSets;
  for (const auto& input : n->inputs()) {
    if (valueToAlias_.count(input) != 0) {
      for (const auto& aliasSet : valueToAlias_.at(input).sets()) {
        aliasSets.insert(aliasSet);
      }
    }
  }

  // ... and the outputs
  for (const auto& output : n->outputs()) {
    if (valueToAlias_.count(output) != 0) {
      for (const auto& aliasSet : valueToAlias_.at(output).sets()) {
        aliasSets.insert(aliasSet);
      }
    }
  }

  // Then get the union of all writers to all those alias sets
  std::unordered_set<Node*> writers;
  for (const auto& alias : aliasSets) {
    if (aliasToWrites_.count(alias) != 0) {
      for (const auto writer : aliasToWrites_.at(alias)) {
        writers.insert(writer);
      }
    }
  }
  return writers;
}

std::unordered_set<const Value*> AliasDb::getAliases(const Value* v) const {
  std::unordered_set<const Value*> ret;
  if (!valueToAlias_.count(v)) {
    return ret;
  }

  const auto& aliasSets = valueToAlias_.at(v).sets();
  for (const auto& aliasSet : aliasSets) {
    const auto& aliases = aliasToValue_.at(aliasSet);
    for (auto alias : aliases) {
      ret.insert(alias);
    }
  }
  return ret;
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
  std::cout << "===2. ALIAS SETS===\n";
  for (const auto& pr : valueToAlias_) {
    std::cout << "%" << pr.first->uniqueName() << " : "
              << "(";

    bool first = true;
    for (const auto& alias : pr.second.sets()) {
      if (first) {
        first = false;
      } else {
        std::cout << ", ";
      }
      std::cout << alias.toUnqualString();
    }
    std::cout << ")\n";
  }

  std::cout << "\n===3. WRITES===\n";
  for (const auto& pr : aliasToWrites_) {
    std::cout << "Alias set " << pr.first.toUnqualString() << ":\n";
    for (const auto node : pr.second) {
      std::cout << "  " << *node;
    }
    std::cout << "\n";
  }

  std::cout << "\n===3. WILDCARD INDEX===\n";
  for (const auto node : wildcardNodes_) {
    node->dump();
  }
}

void AliasDb::analyze(const std::shared_ptr<Graph>& graph) {
  // Assign aliases to the graph's inputs, assuming that all inputs of a given
  // type may alias to each other.
  const auto tensorAlias = getFreshAlias(/*isGraphInput=*/true);
  // Create a separate alias set for each list type
  std::map<TypeKind, Symbol> listTypeAliases;
  // Create a separate alias set for each tuple type
  std::map<TupleTypePtr, Symbol> tupleTypeAliases;
  std::map<TypeKind, Symbol> optionalTypeAliases;

  for (auto input : graph->inputs()) {
    auto inputType = input->type();
    // unwrap optional types
    if (inputType->kind() == TypeKind::OptionalType) {
      inputType = inputType->cast<OptionalType>()->getElementType();
    }

    if (inputType->isSubtypeOf(DynamicType::get())) {
      addAlias(input, tensorAlias);
    } else if (inputType->kind() == TypeKind::ListType) {
      auto containedType = inputType->containedTypes().at(0);
      // All tensor subtypes may alias to each other, so we should consider all
      // lists of them to alias to each other.
      if (containedType->isSubtypeOf(DynamicType::get())) {
        containedType = DynamicType::get();
      }
      if (listTypeAliases.count(containedType->kind()) == 0) {
        listTypeAliases[containedType->kind()] =
            getFreshAlias(/*isGraphInput=*/true);
      }

      addAlias(input, listTypeAliases.at(containedType->kind()));
    } else if (inputType->kind() == TypeKind::TupleType) {
      auto tupleType = inputType->cast<TupleType>();
      if (tupleTypeAliases.count(tupleType) == 0) {
        tupleTypeAliases[tupleType] = getFreshAlias(/*isGraphInput=*/true);
      }
      addAlias(input, tupleTypeAliases.at(tupleType));
    } else {
      JIT_ASSERT(!shouldAnnotate(input));
    }
  }

  analyze(graph->block());
}

void AliasDb::analyze(Block* block) {
  for (auto node : block->nodes()) {
    analyze(node);
  }
}

// The basic strategy is:
//   1. Retrieve alias information for every input.
//   2. Use the node's schema's alias annotations to propgagate alias/write
//      information to the outputs. For unschematized nodes, a special analyzer
//      will have to be handwritten.
void AliasDb::analyze(Node* node) {
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
  std::unordered_map<Symbol, AliasInfo> formalToActual;
  formalToActual[AliasInfo::wildcardSet()] = AliasInfo::createWildcard();
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

    const auto& actualAlias = valueToAlias_.at(actualValue);

    // Bind the formal to the actual
    formalToActual[formalAlias] = actualAlias;

    // Record all writes
    for (const auto& alias : actualAlias.sets()) {
      if (formal->isWrite()) {
        aliasToWrites_[alias].insert(node);
      }
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
    auto outputAlias = formalToActual.at(formalAlias);

    // Record writes
    for (const auto& alias : outputAlias.sets()) {
      if (formal->isWrite()) {
        aliasToWrites_[alias].insert(node);
      }
    }

    addAlias(actual, outputAlias);
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
      if (valueToAlias_.count(input) != 0) {
        JIT_ASSERT(valueToAlias_.count(output) != 0)
        if (!valueToAlias_[output].isSubsetOf(valueToAlias_[input])) {
          notConverged = true;
        }
      }
      addAlias(input, output);
    }
  }
}

void AliasDb::analyzeSubgraph(Node* node) {
  const auto subgraphBlock = node->g(attr::Subgraph)->block();
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
    addAlias(output, AliasInfo::createWildcard());
  }
}

// For torch.chunk(), all returned tensors may alias the input tensor
void AliasDb::analyzeChunk(Node* node) {
  auto alias = valueToAlias_.at(node->input());
  for (auto output : node->outputs()) {
    addAlias(output, alias);
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
    auto alias = valueToAlias_.at(inputs.at(index));
    auto output_begin = outputs.begin() + index * nchunks;
    for (auto it = output_begin; it != output_begin + nchunks; ++it) {
      addAlias(*it, alias);
    }
  }
}

Symbol AliasDb::getFreshAlias(bool isGraphInput) {
  auto num = std::stoll(latestSymbol_.toUnqualString());
  latestSymbol_ = Symbol::fromQualString("alias::" + std::to_string(++num));
  if (isGraphInput) {
    graphInputAliases_.insert(latestSymbol_);
  }
  return latestSymbol_;
}

// Give this alias to the value. If the value already has alias info, union
// with this alias
void AliasDb::addAlias(const Value* value, AliasInfo alias) {
  if (!shouldAnnotate(value)) {
    return;
  }
  if (valueToAlias_.count(value) != 0) {
    valueToAlias_[value].unionWith(alias);
  } else {
    valueToAlias_.insert({value, std::move(alias)});
  }
}

// Give this alias to the value. If the value already has alias info, union
// with this alias
void AliasDb::addAlias(const Value* value, Symbol alias) {
  if (!shouldAnnotate(value)) {
    return;
  }
  if (valueToAlias_.count(value) != 0) {
    valueToAlias_[value].addSet(alias);
  } else {
    AliasInfo aliasInfo;
    aliasInfo.addSet(alias);
    valueToAlias_.insert({value, std::move(aliasInfo)});
  }
}

// Union the alias info of `value` with `from`
void AliasDb::addAlias(const Value* value, const Value* from) {
  if (!shouldAnnotate(value)) {
    JIT_ASSERT(!shouldAnnotate(from));
    return;
  }
  addAlias(value, valueToAlias_.at(from));
}

void AliasDb::mapAliases(at::ArrayRef<Value*> to, at::ArrayRef<Value*> from) {
  JIT_ASSERT(to.size() == from.size());
  for (size_t i = 0; i < to.size(); i++) {
    addAlias(to[i], from[i]);
  }
}

void AliasDb::giveFreshAlias(const Value* value) {
  if (valueToAlias_.count(value) != 0) {
    // Inside a loop, we may have given a fresh alias to this value already, so
    // skip
    return;
  }
  addAlias(value, getFreshAlias());
}
} // namespace jit
} // namespace torch
