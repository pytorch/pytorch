#include "alias_analysis.h"

namespace torch {
namespace jit {
bool AliasDb::hasWildcard(const Node* n) const {
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

bool AliasDb::hasWrites(const Node* n) const {
  for (const auto input : n->inputs()) {
    if (valueToAlias_.count(input) != 0 && valueToAlias_.at(input).isWrite()) {
      return true;
    }
  }
  return false;
}

std::unordered_set<Node*> AliasDb::getWritersForNode(const Node* n) const {
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
}

void AliasDb::analyze(std::shared_ptr<Graph> graph) {
  // Assign aliases to the graph's inputs, assuming that all inputs of a given
  // type may alias to each other.
  const auto tensorAlias = getFreshAlias();
  // Create a separate alias set for each list type
  std::map<TypeKind, Symbol> listTypeAliases;
  // Create a separate alias set for each tuple type
  std::map<TupleTypePtr, Symbol> tupleTypeAliases;

  for (auto input : graph->inputs()) {
    if (input->type()->isSubtypeOf(DynamicType::get())) {
      giveAlias(input, tensorAlias);
    } else if (input->type()->kind() == TypeKind::ListType) {
      const auto containedType = input->type()->containedTypes().at(0);
      if (listTypeAliases.count(containedType->kind()) == 0) {
        listTypeAliases[containedType->kind()] = getFreshAlias();
      }

      giveAlias(input, listTypeAliases.at(containedType->kind()));
    } else if (input->type()->kind() == TypeKind::TupleType) {
      auto tupleType = input->type()->cast<TupleType>();
      if (tupleTypeAliases.count(tupleType) == 0) {
        tupleTypeAliases[tupleType] = getFreshAlias();
      }
      giveAlias(input, tupleTypeAliases.at(tupleType));

    } else {
      JIT_ASSERT(!isMutableType(input));
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
      return analyzeCreator(node);
    case prim::TupleUnpack:
    case prim::TupleIndex:
    case prim::TupleSlice:
    case prim::ListUnpack:
    case prim::PythonOp:
      return analyzeExtractor(node);
    case prim::ConstantChunk:
      return analyzeChunk(node);
  }

  const auto& schema = node->schema();
  if (schema.is_vararg() || schema.is_varret()) {
    const auto hasMutableOutputs = std::any_of(
        node->outputs().cbegin(),
        node->outputs().cend(),
        [this](const Value* output) { return isMutableType(output); });

    // We don't have alias info for this node. Either schematize it, or
    // add it an analyze* method for it.
    JIT_ASSERT(!hasMutableOutputs);
  }

  // Bind formal alias annotation to actual alias sets
  std::unordered_map<Symbol, AliasInfo> formalToActual;
  for (size_t i = 0; i < schema.arguments().size(); i++) {
    const auto& formal = schema.arguments()[i].alias_info();
    const auto& actualValue = node->inputs().at(i);
    // Skip if there's no alias annotation
    if (!formal) {
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
    if (!isMutableType(actual)) {
      continue;
    }

    const auto& formal = schema.returns()[i].alias_info();
    if (!formal) {
      // This is a fresh tensor
      giveAlias(actual, getFreshAlias());
      continue;
    }

    // We don't support composite types for alias analysis yet.
    JIT_ASSERT(formal->containedTypes().size() == 0);

    if (formal->isWildcard()) {
      giveAlias(actual, AliasInfo::createWildcard());
      continue;
    }

    const auto& formalAlias = formal->set();
    auto outputAlias = formalToActual.at(formalAlias);

    giveAlias(actual, outputAlias);
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
    if (!isMutableType(nodeOutput)) {
      continue;
    }

    const auto trueOutput = trueBlock->outputs().at(i);
    const auto falseOutput = falseBlock->outputs().at(i);

    // If a value is only used in one of the branches, then other branch
    // won't have alias information for it. Just assign it an empty set in
    // this case.
    if (valueToAlias_.count(trueOutput) == 0) {
      JIT_ASSERT(valueToAlias_.count(falseOutput) != 0);
      giveAlias(trueOutput, AliasInfo());
    } else if (valueToAlias_.count(falseOutput) == 0) {
      JIT_ASSERT(valueToAlias_.count(trueOutput) != 0);
      giveAlias(falseOutput, AliasInfo());
    }

    AliasInfo aliasInfo;
    aliasInfo.unionWith(valueToAlias_.at(trueOutput));
    aliasInfo.unionWith(valueToAlias_.at(falseOutput));
    valueToAlias_[nodeOutput] = aliasInfo;
  }
}

void AliasDb::analyzeLoop(Node* node) {
  // For loops, we need to copy alias information from the node
  // input/outputs to the body block input/outputs
  const auto bodyBlock = node->blocks().at(0);

  // 1. Copy alias info from node input to block input
  const auto loopCarriedInputs = node->inputs().slice(2); // skip max, cond
  const auto blockInputs = bodyBlock->inputs().slice(1); // skip trip

  JIT_ASSERT(loopCarriedInputs.size() == blockInputs.size());
  for (size_t i = 0; i < blockInputs.size(); i++) {
    const auto blockInput = blockInputs[i];
    if (!isMutableType(blockInput)) {
      continue;
    }
    const auto loopInput = loopCarriedInputs[i];
    valueToAlias_[blockInput] = valueToAlias_.at(loopInput);
  }

  // 2. Populate block output alias info by analyzing the body
  analyze(bodyBlock);

  // 3. Copy the alias info from the block output to the node output
  const auto blockOutputs = bodyBlock->outputs().slice(1); // skip trip
  for (size_t i = 0; i < node->outputs().size(); i++) {
    const auto nodeOutput = node->outputs()[i];
    if (!isMutableType(nodeOutput)) {
      continue;
    }
    const auto blockOutput = blockOutputs.at(i);
    valueToAlias_[nodeOutput] = valueToAlias_.at(blockOutput);
  }
}

void AliasDb::analyzeSubgraph(Node* node) {
  const auto subgraphBlock = node->g(attr::Subgraph)->block();
  for (size_t i = 0; i < node->inputs().size(); i++) {
    const auto nodeInput = node->inputs()[i];
    if (!isMutableType(nodeInput)) {
      continue;
    }
    const auto blockInput = subgraphBlock->inputs().at(i);
    valueToAlias_[blockInput] = valueToAlias_.at(nodeInput);
  }

  analyze(subgraphBlock);

  for (size_t i = 0; i < node->outputs().size(); i++) {
    const auto nodeOutput = node->outputs()[i];
    if (!isMutableType(nodeOutput)) {
      continue;
    }
    const auto blockOutput = subgraphBlock->outputs().at(i);
    valueToAlias_[nodeOutput] = valueToAlias_.at(blockOutput);
  }
}

// For nodes that generate a fresh value from nothing
void AliasDb::analyzeCreator(Node* node) {
  if (isMutableType(node->output())) {
    giveAlias(node->output(), getFreshAlias());
  }
}

// For nodes that extract values from a composite type. Right now, this just
// gives up and creates wildcards for everything.
void AliasDb::analyzeExtractor(Node* node) {
  for (const auto output : node->outputs()) {
    if (isMutableType(output)) {
      giveAlias(output, AliasInfo::createWildcard());
    }
  }
}

// For torch.chunk(), all returned tensors may alias the input tensor
void AliasDb::analyzeChunk(Node* node) {
  auto alias = valueToAlias_.at(node->input());
  for (auto output : node->outputs()) {
    giveAlias(output, alias);
  }
}

bool AliasDb::isMutableType(const Value* v) {
  return v->type()->isSubtypeOf(DynamicType::get()) ||
      v->type()->kind() == TypeKind::ListType ||
      v->type()->kind() == TypeKind::TupleType;
}

Symbol AliasDb::getFreshAlias() {
  auto num = std::stoll(latestSymbol_.toUnqualString());
  latestSymbol_ = Symbol::fromDomainAndUnqualString(
      "org.pytorch.alias", std::to_string(++num));
  return latestSymbol_;
}

void AliasDb::giveAlias(const Value* value, AliasInfo alias) {
  valueToAlias_.insert({value, std::move(alias)});
}

void AliasDb::giveAlias(const Value* value, Symbol alias) {
  AliasInfo aliasInfo;
  aliasInfo.addSet(std::move(alias));
  valueToAlias_.insert({value, std::move(aliasInfo)});
}
} // namespace jit
} // namespace torch
