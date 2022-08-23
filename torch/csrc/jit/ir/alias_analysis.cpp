#include <torch/csrc/jit/ir/alias_analysis.h>

#include <ATen/core/interned_strings.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/memory.h>
#include <fstream>

namespace torch {
namespace jit {

namespace {

c10::MaybeOwned<TypePtr> toSingleType(const AliasTypeSet& mut_types) {
  return mut_types.size() == 1
      ? c10::MaybeOwned<TypePtr>::borrowed(mut_types[0])
      : c10::MaybeOwned<TypePtr>::owned(c10::UnionType::create(mut_types));
}

// This class determines whether a type is mutable, and, if so, it maps
// the type to its "mutable equivalent" (see definition in
// `mapTypeToAliasTypeSet`). It uses a cache of TypePtrs to speed up these
// type lookups
class MutableTypePtrHelper {
 public:
  explicit MutableTypePtrHelper(
      ska::flat_hash_map<TypePtr, AliasTypeSet>* mutable_type_cache)
      : mutable_type_cache_(mutable_type_cache) {}

  // Map any mutable type to a type such that all other types which the
  // mutable type can alias will be mapped to the same type. For
  // example, calling this method on `Optional[List[int]]` should be
  // the same as calling this method on `List[int]`.
  //
  // Rules:
  //   - If the type is not mutable, return `nullopt`
  //   - If the type is a `Tuple`, that means that it's an immutable
  //     object that can itself contain mutable objects. We want to make
  //     sure that the mutable objects are correctly aliased, so we
  //     remove the immutable objects. (For example,
  //     `Tuple[int, Tensor]` would become `Tuple[Tensor]`, while
  //     `Tuple[int, str]` would be returned as `nullopt`.) This is a
  //     convenience that makes it easy to check if the `Tuple`
  //     contains only immutable objects, though it's not technically
  //     necessary
  //   - For any Tensor type (including Tensor types that are part of
  //     a larger container, e.g. `List[Tensor]`), return the
  //     "unshaped" version of that Tensor. An "unshaped" Tensor is a
  //     Tensor with shape information removed. For example, a Tensor
  //     of dimension 4 would map to the same type as a Tensor of
  //     dimension 1. This allows us to treat all subclasses of Tensor
  //     as a single, homogenous "Tensor" type.
  c10::optional<AliasTypeSet> mapTypeToAliasTypeSet(const TypePtr& type) {
    if (mutable_type_cache_) {
      const AliasTypeSet* result = mapTypeToBorrowedAliasTypeSet(type);
      if (result) {
        return *result;
      }
    }
    return mapTypeToAliasTypeSetImpl(type);
  }

  const AliasTypeSet* mapTypeToBorrowedAliasTypeSet(const TypePtr& type) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mutable_type_cache_ != nullptr);
    auto maybe_type_mapping = mutable_type_cache_->find(type);
    if (maybe_type_mapping != mutable_type_cache_->end()) {
      return &maybe_type_mapping->second;
    }

    auto mutable_types = mapTypeToAliasTypeSetImpl(type);
    if (mutable_types) {
      auto it =
          mutable_type_cache_->emplace(type, std::move(*mutable_types)).first;
      return &it->second;
    } else {
      return nullptr;
    }
  }

 private:
  c10::optional<AliasTypeSet> mapTypeToAliasTypeSetImpl(const TypePtr& type) {
    switch (type->kind()) {
      case TypeKind::ListType:
      case TypeKind::DictType:
      case TypeKind::ClassType:
      case TypeKind::TensorType:
        // TODO: Look up cached contained types. this is kind of tricky
        // because a `List[Optional[T]]` should still be
        // `List[Optional[Unshaped(T)]]`, but
        // `mapTypeToAliasTypeSet(Optional[T])` should be `T`
        return AliasTypeSet{unshapedType(type)};
      case TypeKind::UnionType: {
        AliasTypeSet mutable_types;
        for (const TypePtr& inner :
             type->expectRef<UnionType>().containedTypes()) {
          if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
            mutable_types.insert(
                mutable_types.end(),
                (*maybe_inner_types).begin(),
                (*maybe_inner_types).end());
          }
        }
        if (mutable_types.size() == 0) {
          return c10::nullopt;
        }
        return mutable_types;
      }
      case TypeKind::OptionalType: {
        auto inner = type->castRaw<OptionalType>()->getElementType();
        return mapTypeToAliasTypeSet(inner);
      }
      case TypeKind::AnyType:
        return {AliasTypeSet{type}};
      case TypeKind::FutureType: {
        if (auto maybe_mut_types = mapTypeToAliasTypeSet(
                type->castRaw<FutureType>()->getElementType())) {
          return {AliasTypeSet{
              FutureType::create(*toSingleType(*maybe_mut_types))}};
        }
        return c10::nullopt;
      }
      case TypeKind::TupleType: {
        std::vector<TypePtr> mutable_types;
        for (const TypePtr& inner : type->expectRef<TupleType>().elements()) {
          if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
            mutable_types.insert(
                mutable_types.end(),
                (*maybe_inner_types).begin(),
                (*maybe_inner_types).end());
          }
        }
        if (mutable_types.size() == 0) {
          return c10::nullopt;
        }
        return {AliasTypeSet{TupleType::create(mutable_types)}};
      }
      default:
        return c10::nullopt;
    }
  }
  ska::flat_hash_map<TypePtr, AliasTypeSet>* mutable_type_cache_;
};

bool isMutableTypeImpl(
    const TypePtr& type,
    ska::flat_hash_map<TypePtr, AliasTypeSet>* mutable_type_cache) {
  // Check common cases to avoid recursively constructing type in
  // `mapTypeToAliasTypeSetPtrImpl`
  auto kind = type->kind();
  if (kind == TypeKind::TensorType || kind == TypeKind::ListType ||
      kind == TypeKind::ClassType || kind == TypeKind::DictType) {
    return true;
  }
  MutableTypePtrHelper helper(mutable_type_cache);
  if (mutable_type_cache) {
    return helper.mapTypeToBorrowedAliasTypeSet(type) != nullptr;
  } else {
    return helper.mapTypeToAliasTypeSet(type).has_value();
  }
}

} // namespace

// Static `isMutableType` does not use cache of type -> mutable type equivalent
bool AliasDb::isMutableType(const TypePtr& type) {
  return isMutableTypeImpl(type, nullptr);
}

bool AliasDb::isMutableType(const Value* v) {
  return isMutableType(v->type());
}

// Make use of type -> mutable cache
bool AliasDb::isMutableTypeInternal(const TypePtr& type) const {
  return isMutableTypeImpl(type, &mapped_mutable_types_);
}

bool AliasDb::isMutableTypeInternal(const Value* v) const {
  return isMutableTypeInternal(v->type());
}

const AliasTypeSet* AliasDb::mapTypeToAliasTypeSetPtr(
    const TypePtr& type) const {
  MutableTypePtrHelper helper(&mapped_mutable_types_);
  return helper.mapTypeToBorrowedAliasTypeSet(type);
}

AliasDb::~AliasDb() = default;

// Structure used during analysis to keep track of all writes at a high
// level. When the analysis is completed, this will be used to construct
// a more efficient WriteIndex
struct AliasDb::WriteRegistry {
  void registerWrite(const Value* v, Node* n) {
    writes_[n].emplace_back(v);
  }
  void registerWriteToAllContained(const Value* v, Node* n) {
    containedWrites_[n].emplace_back(v);
  }
  void registerWriteToAllWildcards(Node* n) {
    writesToAllWildcards_.insert(n);
  }
  std::unordered_map<Node*, std::vector<const Value*>> writes_;
  std::unordered_map<Node*, std::vector<const Value*>> containedWrites_;
  std::unordered_set<Node*> writesToAllWildcards_;
};

AliasDb::AliasDb(
    std::shared_ptr<Graph> graph,
    bool isFrozen,
    bool descendFunctionCalls)
    : graph_(std::move(graph)),
      isFrozen_(isFrozen),
      descend_function_calls_(descendFunctionCalls),
      memoryDAGBuilder_(std::make_unique<MemoryDAGBuilder>()),
      writeRegistry_(std::make_unique<AliasDb::WriteRegistry>()) {
  analyze(graph_);

  memoryDAG_ = std::make_unique<MemoryDAG>(std::move(memoryDAGBuilder_));
  memoryDAGBuilder_ = nullptr; // to make further access a hard error

  memoryDAG_->setWildcards(
      wildcards_, elementMap_, [&](const Value* v) -> Element* {
        return getWildcard(v->type());
      });

  // Now we build up the various write indices based on information in the write
  // registry that we populated during analysis

  // Initialize the write index
  writeIndex_ = TWriteIndex();
  auto& writeIndex = *writeIndex_; // to make operator[] less ugly

  // Build the write index
  for (const auto& write : writeRegistry_->writes_) {
    Node* node = write.first;
    const std::vector<const Value*> writtenValues = write.second;
    for (const Value* writtenValue : writtenValues) {
      auto it = elementMap_.find(writtenValue);
      TORCH_INTERNAL_ASSERT(
          it != elementMap_.end(), "Tried to write to value not in MemoryDAG");
      const auto& writtenMemoryLocations =
          memoryDAG_->getMemoryLocations(it->second);
      writeIndex[node] |= writtenMemoryLocations;
    }
  }

  for (const auto& write : writeRegistry_->containedWrites_) {
    Node* node = write.first;
    const std::vector<const Value*>& writtenValues = write.second;
    for (const Value* writtenValue : writtenValues) {
      auto elem = elementMap_.at(writtenValue);
      MemoryLocations writtenMemoryLocations;
      memoryDAG_->collectAllContainedMemoryLocations(
          elem, writtenMemoryLocations);
      writeIndex[node] |= writtenMemoryLocations;
    }
  }

  for (const auto& write : writeRegistry_->writesToAllWildcards_) {
    for (const auto& pr : wildcardIndex_) {
      writeIndex[write].set(pr.second->index);
    }
  }

  // Now that we've built the write index, we can null out the WriteRegistry to
  // make future access an error. In this way we prevent the index from getting
  // out of sync (since we have no way of registering new writes)
  writeRegistry_ = nullptr;

  // Initialize the write cache
  buildWrittenToLocationsIndex();
  GRAPH_DEBUG(toString());
}

bool AliasDb::isMutable(Node* n) const {
  ValueSet vs;
  for (const auto input : n->inputs()) {
    vs.insert(input);
  }
  return writesToAlias(n, vs);
}

bool AliasDb::hasInputWriters(const Node* n) const {
  for (const auto input : n->inputs()) {
    if (hasWriters(input)) {
      return true;
    }
  }
  return false;
}

bool AliasDb::hasOutputWriters(const Node* n) const {
  for (const auto output : n->outputs()) {
    if (hasWriters(output)) {
      return true;
    }
  }
  return false;
}

bool AliasDb::hasWriters(const Node* n) const {
  return hasInputWriters(n) || hasOutputWriters(n);
}

bool AliasDb::hasWriters(const Value* v) const {
  if (v->mustBeNone()) {
    return false;
  }

  auto it = elementMap_.find(v);
  if (it == elementMap_.end()) {
    return false;
  }

  const auto& el = it->second;
  return writtenToLocationsIndex_->intersects(
      memoryDAG_->getMemoryLocations(el));
}

void AliasDb::getWritesImpl(Node* n, MemoryLocations& ret) const {
  if (writeIndex_->count(n)) {
    const auto& writes = writeIndex_->at(n);
    ret |= writes;
  }

  for (auto block : n->blocks()) {
    for (auto node : block->nodes()) {
      getWritesImpl(node, ret);
    }
  }
}

// Does `n` write to an alias of one of the values in `vs`?
bool AliasDb::writesToAlias(Node* n, const ValueSet& vs) const {
  const auto writtenTo = getWrites(n);
  if (writtenTo.empty()) {
    return false;
  }

  MemoryLocations locs;
  for (const auto v : vs) {
    auto it = elementMap_.find(v);
    if (it != elementMap_.end()) {
      const auto& vlocs = memoryDAG_->getMemoryLocations(it->second);
      if (writtenTo.intersects(vlocs)) {
        return true;
      }
    }
  }

  return false;
}

MemoryLocations AliasDb::getWrites(Node* n) const {
  MemoryLocations writes;
  getWritesImpl(n, writes);
  return writes;
}

void AliasDb::getReadsImpl(Node* n, MemoryLocations& ret) const {
  for (const auto input : n->inputs()) {
    auto it = elementMap_.find(input);
    if (it != elementMap_.end()) {
      auto el = it->second;

      // Add all memory locations this element may alias and their contained
      // elements
      memoryDAG_->collectAllContainedMemoryLocations(el, ret);
    }
  }

  for (auto block : n->blocks()) {
    for (auto node : block->nodes()) {
      getReadsImpl(node, ret);
    }
  }
}

MemoryLocations AliasDb::getReads(Node* n) const {
  MemoryLocations reads;
  getReadsImpl(n, reads);
  return reads;
}

std::string AliasDb::getElementName(const Element* e) const {
  if (e->values.empty()) {
    // Not the most efficient way, but given the fact there are
    // not too many types and even fewer of them will end up in
    // `wildcardIndex_`, we should be fine with a linear search
    // each time we hit a Wildcard leaf
    for (const auto& ent : wildcardIndex_) {
      if (ent.second == e) {
        return std::string("WILDCARD for type ") + ent.first->str();
      }
    }
    return "WILDCARD";
  } else {
    std::ostringstream ss;
    if (e->values.size() == 1) {
      ss << "%" << (*e->values.begin())->debugName();
      return ss.str();
    }
    ss << "(";
    for (const Value* v : e->values) {
      ss << "%" << v->debugName() << ", ";
    }
    ss << ")";
    return ss.str();
  }
}

void AliasDb::dump() const {
  std::cout << toString();
}

std::string AliasDb::toString() const {
  std::stringstream ss{};

  ss << "\n===1. GRAPH===\n";
  ss << graph_->toString();

  ss << "\n===2. ALIAS DB===\n";
  for (const auto& ptrPair : elementMap_) {
    const auto element = ptrPair.second;
    int ct = 0;
    if (!element->pointsTo.empty()) {
      ss << getElementName(element) << " points to: ";
      for (const auto pointedTo : element->pointsTo) {
        if (ct > 0) {
          ss << ", ";
        }
        ++ct;
        ss << getElementName(memoryDAG_->fromIndex(pointedTo));
      }
      ss << "\n";
    }
    ct = 0;
    if (!element->containedElements.empty()) {
      ss << getElementName(element) << " contains: ";
      for (const auto contained : element->containedElements) {
        ss << getElementName(memoryDAG_->fromIndex(contained));
        if (ct > 0) {
          ss << ", ";
        }
        ++ct;
      }
      ss << "\n";
    }
  }

  ss << "\n===3. Writes===\n";
  for (const auto& pr : *writeIndex_) {
    const auto node = pr.first;
    const auto& values = pr.second;
    ss << *node;
    ss << "  ";
    for (const auto value : values) {
      ss << getElementName(memoryDAG_->fromIndex(value)) << ", ";
    }
    ss << "\n";
  }
  ss << "\n";
  return ss.str();
}

bool AliasDb::dumpToGraphvizFile(const char* filename) const {
  std::ofstream dot_file(filename);
  if (!dot_file.good()) {
    std::cout << "Failed to create Graphviz file: '" << filename << "'\n";
    return false;
  }
  dot_file << toGraphviz();
  return true;
}

std::string AliasDb::toGraphviz() const {
  std::stringstream dot;

  // Local helper to generate a graphviz-friendly name encoding
  // See also AliasDb::getElementName()
  const auto name = [this](const Element* e) -> std::string {
    if (e->values.empty()) {
      for (const auto& ent : wildcardIndex_) {
        if (ent.second == e) {
          return std::string("\"WILDCARD for ") + ent.first->str() + "\"";
        }
      }
      return "\"WILDCARD\"";
    } else {
      std::ostringstream ss;
      if (e->values.size() == 1) {
        ss << "\"\\%" << (*e->values.begin())->debugName() << "\"";
        return ss.str();
      }
      ss << "\"(";
      for (const Value* v : e->values) {
        ss << "\\%" << v->debugName() << ", ";
      }
      ss << ")\"";
      return ss.str();
    }
  };

  // Include the textual representation for reference
  dot << "/*\n";
  dot << toString();
  dot << "*/\n";

  dot << "digraph alias_db {\n"
      << "  rankdir=LR\n"
      << "  node [shape=rect, color=gray];\n"
      << "  edge [color=black];\n";

  for (const auto& ptrPair : elementMap_) {
    const auto element = ptrPair.second;
    if (!element->pointsTo.empty()) {
      for (const auto pointedTo : element->pointsTo) {
        dot << "  " << name(element) << " -> "
            << name(memoryDAG_->fromIndex(pointedTo)) << "\n";
      }
    }
    if (!element->containedElements.empty()) {
      for (const auto contained : element->containedElements) {
        dot << "  " << name(element) << " -> "
            << name(memoryDAG_->fromIndex(contained))
            << " [style=dashed, color=blue]\n";
      }
    }
  }

  dot << "}\n";
  return dot.str();
}

void AliasDb::analyze(const std::shared_ptr<Graph>& graph) {
  for (auto input : graph->inputs()) {
    setWildcard(input);
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
}

// Returns true if analysis was run using
// the registered analyzer.
bool AliasDb::tryRegisteredAnalysis(Node* node) {
  const Operator& op = node->getOperator();
  auto analysis = op.aliasAnalysisKind();
  if (AliasAnalysisKind::PURE_FUNCTION == analysis) {
    analyzeCreator(node);
    return true;
  }
  return false;
}

// The basic strategy is:
//   1. Retrieve alias information for every input.
//   2. Use the node's schema's alias annotations to propgagate alias/write
//      information to the outputs. For unschematized nodes, a special analyzer
//      will have to be handwritten.
void AliasDb::analyzeImpl(Node* node) {
  auto op = node->maybeOperator();
  const bool hasSpecialCase = aliasAnalysisHasSpecialCaseFor(node->kind());
  if (op) {
    const auto analysis = op->aliasAnalysisKind();

    const bool registeredAsSpecialCase =
        analysis == AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
    if (C10_UNLIKELY(registeredAsSpecialCase && !hasSpecialCase)) {
      TORCH_INTERNAL_ASSERT(
          false,
          "Op ",
          node->kind().toDisplayString(),
          " is registered with AliasAnalysisKind::INTERNAL_SPECIAL_CASE but doesn't have a special case.");
    } else if (C10_UNLIKELY(!registeredAsSpecialCase && hasSpecialCase)) {
      TORCH_INTERNAL_ASSERT(
          false,
          "Op ",
          node->kind().toDisplayString(),
          " has a special case and should be registered with AliasAnalysisKind::INTERNAL_SPECIAL_CASE but is registered with ",
          c10::toString(analysis));
    }
  } else {
    if (!hasSpecialCase) {
      std::ostringstream oss;
      for (const auto input : node->inputs()) {
        oss << input->type()->str() << ", ";
      }
      oss << "\n\nCandidates:";
      const auto& candidates = getAllOperatorsFor(node->kind());
      for (const auto& candidate : candidates) {
        oss << "\n\t" << candidate->schema();
      }
      TORCH_INTERNAL_ASSERT(
          0,
          "We don't have an op for ",
          node->kind().toDisplayString(),
          " but it isn't a special case.  ",
          "Argument types: ",
          oss.str());
    }
  }

  // These nodes are not schematized, so we need to handle them specially
  switch (node->kind()) {
    case prim::If:
      return analyzeIf(node);
    case prim::Loop:
      return analyzeLoop(node);
    case prim::FusionGroup:
    case prim::CudaFusionGroup:
    case prim::oneDNNFusionGroup:
    case prim::FunctionalGraph:
    case prim::DifferentiableGraph:
    case prim::FallbackGraph:
      return analyzeSubgraph(node);
    case prim::fork:
      return analyzeFork(node);
    case aten::wait:
      return analyzeWait(node);
    case prim::rpc_async:
    case prim::rpc_sync:
    case prim::rpc_remote:
      return analyzeRpcAsync(node);
    case aten::batch_norm:
      return analyzeBatchNorm(node);
    case aten::instance_norm:
      return analyzeInstanceNorm(node);
    case prim::GradOf:
      return analyzeGradOf(node);
    case prim::BroadcastMKLDNNTensors: {
      makePointerTo(node->outputs().at(0), node->inputs().at(0));
      makePointerTo(node->outputs().at(1), node->inputs().at(1));
      return;
    }
    // TODO: think more about TensorExpr alias correctness
    case prim::TensorExprGroup:
    case prim::TensorExprDynamicGroup:
    case prim::MKLDNNGroup:
    case prim::ConstantMKLDNNTensor:
    case prim::StaticSubgraph:
    case prim::Constant:
    case prim::AutogradZero:
    case prim::AutogradAdd:
    case prim::FusedConcat:
    case prim::MMTreeReduce:
    case prim::MMBatchSide:
    case prim::BroadcastSizes:
    case prim::ChunkSizes:
    // this should never be seen outside of initial compilation
    // but because of some dependencies with closure invoking alias
    // db needs to be handled here
    case prim::EmptyListLiteral:
    case prim::Closure:
    case prim::CreateObject:
    case prim::tolist:
    case prim::Uninitialized:
      return analyzeCreator(node);
    case prim::TupleConstruct:
    case prim::DictConstruct:
    case prim::ListConstruct:
      return analyzeContainerConstruct(node);
    case prim::TupleUnpack:
    case prim::TupleIndex:
    case prim::TupleSlice:
    case prim::ListUnpack:
    case prim::PythonOp:
    case prim::GetAttr:
      if (isFrozen_ && node->kind() == prim::GetAttr) {
        auto& ty = node->input()->type();
        if (ty->expectRef<ClassType>().is_module()) {
          return analyzeCreator(node);
        }
      }
      return analyzeExtractor(node);
    case prim::unchecked_cast:
      return makePointerTo(node->output(), node->input());
    case prim::ConstantChunk:
      return analyzeChunk(node);
    case prim::BroadcastingChunk:
      return analyzeBroadcastingChunk(node);
    case prim::SetAttr:
      return analyzeSetAttr(node);
    case prim::profile_ivalue:
    case prim::profile:
      makePointerTo(node->output(), node->inputs().at(0));
      return;
    case prim::TypeCheck:
    case prim::RequiresGradCheck: {
      auto num_inputs = node->inputs().size();
      for (const auto i : c10::irange(num_inputs)) {
        makePointerTo(node->outputs().at(i), node->inputs().at(i));
      }
      return;
    }
    case prim::BailOut:
      TORCH_INTERNAL_ASSERT(
          node->inputs().at(0)->node()->kind() == prim::BailoutTemplate);
      makePointerTo(node->output(), node->inputs().at(1));
      return;
    case prim::Guard:
      makePointerTo(node->output(), node->inputs().at(0));
      return;
    case prim::CallFunction:
    case prim::CallMethod: {
      // TODO: this can be improved with summarizes of what the function does
      // for now we assume the worst
      if (!descend_function_calls_) {
        return analyzeConservative(node);
      }
      auto g = tryToGraphFunction(node);
      if (!g) {
        return analyzeConservative(node);
      }
      // this is an unoptimized path - we copy the subgraph for each function
      // call past the first - so we do not generally enable the recursive
      // analysis. use cases for fine-grained alias analysis without inlining
      // are very uncommon
      auto graph = g->optimized_graph();
      // alias analysis will use Value* as mappings for information,
      // so for each analysis of a particular function call we need a new graph
      // for all copies made, store them for duration of analysis so we do not
      // run into lifetime issues with the graph
      std::vector<std::shared_ptr<Graph>>& graphs =
          function_call_copies_[graph.get()];
      if (graphs.size() == 0) {
        graphs.push_back(graph);
        analyzeSubgraph(node, graph);
      } else {
        auto copied_graph = graph->copy();
        graphs.push_back(copied_graph);
        analyzeSubgraph(node, copied_graph);
      }
      return;
    }
    case prim::Enter:
    case prim::Exit:
      // TODO: this can be improved with summarizes of what the function does
      // for now we assume the worst
      // NB: update safeToChangeAliasingRelationship if changed
      return analyzeConservative(node);
    case prim::Print:
    case prim::isinstance:
      // These ops do nothing
      return;
    default:
      if (tryRegisteredAnalysis(node)) {
        return;
      }
  }

  TORCH_INTERNAL_ASSERT(op, "We should have an op schema if we get to here");
  const AliasAnalysisKind analysis = op->aliasAnalysisKind();
  TORCH_INTERNAL_ASSERT(
      analysis != AliasAnalysisKind::INTERNAL_SPECIAL_CASE &&
          !aliasAnalysisHasSpecialCaseFor(node->kind()),
      "Special cases should be handled already if we're here.");

  if (node->kind().is_aten() || node->kind().is_prim() ||
      node->kind().is_cuda()) {
    // TODO There is nothing in the system that relies on aten:: and prim::
    // ops using AliasAnalysisKind::FROM_SCHEMA or
    // AliasAnalysisKind::INTERNAL_SPECIAL_CASE, but this is the intended
    // behavior for all current ops and a good error check. We can consider
    // lifting this constraint later if we have a use case for it.
    TORCH_INTERNAL_ASSERT(
        analysis == AliasAnalysisKind::FROM_SCHEMA ||
            analysis == AliasAnalysisKind::CONSERVATIVE,
        "aten:: and prim:: operators should use AliasAnalysisKind::FROM_SCHEMA or "
        "AliasAnalysisKind::CONSERVATIVE(if really necessary), but ",
        node->kind().toDisplayString(),
        " doesn't. Note: Ideally, prim:: operators actually shouldn't have a schema ",
        "and then use AliasAnalysisKind::INTERNAL_SPECIAL_CASE instead.");
  }

  if (analysis == AliasAnalysisKind::CONSERVATIVE) {
    // TODO A previous implementation of alias analysis always accessed
    // node->schema , which cause the schema caches in the Node class to be
    // filled for the full graph. Unfortunately, our JIT passes started relying
    // on that, so we need to keep doing this. Details: in
    // caffe2/torch/onnx/utils.py, _jit_pass_onnx is called on an invalid JIT
    // graph because we called _jit_pass_erase_number_types right before and
    // ints are now Tensors instead. So if _jit_pass_onnx tries to look up
    // operator schemas, it will crash. However, _jit_pass_constant_propagation,
    // which is called before it, runs alias analysis and prefills the schema
    // cache in the all Node instances so that _jit_pass_onnx doesn't look up
    // operators to get the schemas anymore. We should fix this.
    node->schema(); // fill the schema cache in the Node class

    return analyzeConservative(node);
  }

  TORCH_INTERNAL_ASSERT(
      analysis == AliasAnalysisKind::FROM_SCHEMA,
      "AliasAnalysisKind::CONSERVATIVE/PURE_FUNCTION/INTERNAL_SPECIAL_CASE should already have been handled above");
  const auto& schema = node->schema();

  // Bind the schema's "formal" alias annotation to the actual values those
  // schema arguments represent
  std::unordered_map<Symbol, Value*> formalToActual;
  for (const auto i : c10::irange(schema.arguments().size())) {
    const at::AliasInfo* formal = schema.arguments()[i].alias_info();
    const auto& actualValue = node->inputs().at(i);

    // Skip if there's no alias annotation
    if (!formal) {
      continue;
    }

    // If this type cannot alias, continue. Can occur with a VarType schema
    if (!isMutableTypeInternal(actualValue)) {
      continue;
    }

    // Do sanity checks on the alias annotation
    TORCH_INTERNAL_ASSERT(
        formal->containedTypes().size() <= 1,
        "Composite types for alias analysis not yet supported");
    TORCH_INTERNAL_ASSERT(
        !formal->isWildcardBefore(),
        "Doesn't make sense for a input value to begin as a wildcard");
    if (formal->containedTypes().size() == 1 && formal->beforeSets().empty()) {
      // Use the first containedType in alias info.
      formal = &(formal->containedTypes()[0]);
    }

    const auto& formalAlias = formal->beforeSet();

    // skip if we've already bound this alias
    if (formalToActual.count(formalAlias) != 0) {
      continue;
    }

    // Bind the formal to the actual
    formalToActual[formalAlias] = actualValue;

    // Record writes
    if (formal->isWrite()) {
      registerWrite(actualValue, node);
    }

    // Now deal with sets after the '->'
    if (formal->isWildcardAfter()) {
      TORCH_INTERNAL_ASSERT(
          formal->afterSets().size() == 1,
          "If the after set contains a wildcard, "
          "there should be no other alias sets specified.");
      setWildcard(actualValue);
    } else {
      // We don't understand anything else in the after yet, so assert there's
      // been no change.
      TORCH_INTERNAL_ASSERT(formal->beforeSets() == formal->afterSets());
    }
  }

  // Use the formal-actual mapping to give aliases to the outputs
  for (const auto i : c10::irange(schema.returns().size())) {
    const auto actual = node->outputs().at(i);
    const at::AliasInfo* formal = schema.returns()[i].alias_info();
    if (!formal) {
      // This is a fresh tensor
      giveFreshAlias(actual);
      continue;
    }

    // If this type cannot alias, continue. Can occur with a VarType schema
    if (!isMutableType(actual)) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        formal->containedTypes().size() <= 1,
        "Composite types for alias analysis not yet supported");
    TORCH_INTERNAL_ASSERT(formal->beforeSets() == formal->afterSets());
    if (formal->containedTypes().size() == 1 && formal->beforeSets().empty()) {
      // Use the first containedType in alias info.
      formal = &(formal->containedTypes()[0]);
    }
    if (formal->isWildcardBefore()) {
      TORCH_INTERNAL_ASSERT(
          formal->beforeSets().size() == 1,
          "If an output is a wildcard, "
          "there should be no other alias sets specified.");
      setWildcard(actual);
      continue;
    }

    bool inputs_has_alias = false;
    for (const auto& formalAlias : formal->beforeSets()) {
      if (formalToActual.count(formalAlias)) {
        inputs_has_alias = true;
        auto toAlias = formalToActual.at(formalAlias);
        makePointerTo(actual, toAlias);
      }
    }
    // If all the alias annotation that we encounter weren't in the inputs:
    //   e.g. foo(Tensor(a) self) -> Tensor(b)
    //   or foo(Tensor(a) self) -> Tensor(b|c)
    // Otherwise it is the form of a|fresh, which we can ignore, taking the
    // conservative assumption that the output must alias `a`, e.g
    //   aten::cuda(Tensor(a) self) -> Tensor(a|fresh)
    if (!inputs_has_alias && formal->beforeSets().size()) {
      giveFreshAlias(actual);
    }

    // Record writes
    if (formal->isWrite()) {
      registerWrite(actual, node);
    }
  }
}

// Register the fact that `n` writes to `v`.
void AliasDb::registerWrite(const Value* v, Node* n, bool writeToContained) {
  if (!isMutableTypeInternal(v)) {
    // don't need to register a write if the value isn't mutable
    return;
  }
  if (writeToContained) {
    writeRegistry_->registerWriteToAllContained(v, n);
  } else {
    writeRegistry_->registerWrite(v, n);
  }
}

void AliasDb::analyzeIf(Node* node) {
  // For if statements, the alias set of an output is the union of the
  // alias sets generated by the if and else block
  const auto trueBlock = node->blocks().at(0);
  const auto falseBlock = node->blocks().at(1);
  analyze(trueBlock);
  analyze(falseBlock);

  for (const auto i : c10::irange(node->outputs().size())) {
    const auto nodeOutput = node->outputs()[i];

    const auto trueOutput = trueBlock->outputs().at(i);
    const auto falseOutput = falseBlock->outputs().at(i);

    makePointerTo(nodeOutput, trueOutput);
    makePointerTo(nodeOutput, falseOutput);
  }
}

void AliasDb::analyzeLoop(Node* node) {
  const auto bodyBlock = node->blocks().at(0);
  const auto loopCarriedInputs = node->inputs().slice(2); // skip max, cond
  const auto blockInputs = bodyBlock->inputs().slice(1); // skip trip
  const auto blockOutputs = bodyBlock->outputs().slice(1); // skip trip
  TORCH_INTERNAL_ASSERT(loopCarriedInputs.size() == blockInputs.size());
  TORCH_INTERNAL_ASSERT(blockOutputs.size() == node->outputs().size());

  // Run alias analysis on the loop body, iterating until the block output
  // alias info converges. Copy node input aliases to block input
  mapAliases(blockInputs, loopCarriedInputs);

  // Populate block output alias info by analyzing the body
  analyze(bodyBlock);

  // Copy the alias info from the block output to the node output
  mapAliases(node->outputs(), blockOutputs);
}

void AliasDb::analyzeGradOf(Node* node) {
  const auto grad_of_block = node->blocks().at(0);
  analyze(grad_of_block);
  mapAliases(node->outputs(), grad_of_block->outputs());
}

void AliasDb::analyzeSubgraph(Node* node, std::shared_ptr<Graph> subgraph) {
  const auto subgraphBlock = subgraph->block();
  // CallFunction nodes have an extra first parameter
  if (node->kind() == prim::CallFunction) {
    mapAliases(subgraphBlock->inputs(), node->inputs().slice(1));
  } else {
    mapAliases(subgraphBlock->inputs(), node->inputs());
  }

  analyze(subgraphBlock);

  // Note: the subgraph outputs and node outputs are NOT NECESSARILY the
  // same length. Autodifferentiation maybe capture additional outputs in the
  // subgraph block.
  TORCH_INTERNAL_ASSERT(
      subgraphBlock->outputs().size() >= node->outputs().size());
  for (size_t i = 0; i < node->outputs().size(); i++) {
    makePointerTo(node->outputs()[i], subgraphBlock->outputs()[i]);
  }
}

void AliasDb::analyzeSubgraph(Node* node) {
  const auto subgraph = node->g(attr::Subgraph);
  return analyzeSubgraph(node, subgraph);
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
    setWildcard(output);
  }
}

// For torch.chunk(), all returned tensors may alias the input tensor
void AliasDb::analyzeChunk(Node* node) {
  for (auto output : node->outputs()) {
    makePointerTo(output, node->input());
  }
}

void AliasDb::analyzeFork(Node* node) {
  for (const auto input : node->inputs()) {
    setWildcard(input);
  }

  // Give the future that the fork emits a fresh value
  for (const auto output : node->outputs()) {
    giveFreshAlias(output);
  }
}

void AliasDb::analyzeWait(Node* node) {
  TORCH_INTERNAL_ASSERT(node->kind() == aten::wait);
  for (const auto output : node->outputs()) {
    setWildcard(output);
  }
  // the forked subgraph that `wait` is waiting on may write to any of its
  // inputs. We don't have a reliable way of recovering the fork inputs, so
  // for safety we just register a write to every wildcard.
  writeRegistry_->registerWriteToAllWildcards(node);
}

void AliasDb::analyzeRpcAsync(Node* node) {
  for (const auto input : node->inputs()) {
    setWildcard(input);
  }

  // Give the future that the rpc_async emits a fresh value
  for (const auto output : node->outputs()) {
    giveFreshAlias(output);
  }
}

namespace {
c10::optional<bool> getConstantBooleanInput(
    Node* node,
    const std::string& inputName) {
  TORCH_INTERNAL_ASSERT(
      node->hasNamedInput(inputName), inputName + " input is expected");
  auto value = node->namedInput(inputName);
  TORCH_INTERNAL_ASSERT(
      value->type() == BoolType::get(),
      inputName + "training input is expected to be a bool");
  return constant_as<bool>(value);
}
} // namespace

// custom behavior for batch_norm because (a!)? annotations currently
// aren't supported, and because behavior differs depending on the value of
// training
void AliasDb::analyzeBatchNorm(Node* node) {
  // we invoking freezing for inference, so we assume training will be folded to
  // a constant false to avoid needing to invoke freezing multiple times in
  // order to make batch norm weights constant
  for (Value* output : node->outputs()) {
    giveFreshAlias(output);
  }

  if (isFrozen_) {
    return;
  }

  auto isTraining = getConstantBooleanInput(node, "training");

  if (!isTraining.has_value() || *isTraining) {
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_mean"), "running_mean input is expected");
    auto runningMean = node->namedInput("running_mean");
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_var"), "running_var input is expected");
    auto runningVar = node->namedInput("running_var");

    registerWrite(runningMean, node);
    registerWrite(runningVar, node);
  }
}

// custom behavior for instance_norm, because (a!)? annotations currently
// aren't supported, and because behavior differs depending on the value of
// use_input_stats
void AliasDb::analyzeInstanceNorm(Node* node) {
  for (Value* output : node->outputs()) {
    giveFreshAlias(output);
  }

  auto useInputStats = getConstantBooleanInput(node, "use_input_stats");

  if (!useInputStats.has_value() || *useInputStats) {
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_mean"), "running_mean input is expected");
    auto runningMean = node->namedInput("running_mean");
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_var"), "running_var input is expected");
    auto runningVar = node->namedInput("running_var");

    registerWrite(runningMean, node);
    registerWrite(runningVar, node);
  }
}

// SetAttr: writes to the `self` field
void AliasDb::analyzeSetAttr(Node* node) {
  const auto self = node->inputs().at(0);
  TORCH_INTERNAL_ASSERT(self->type()->kind() == TypeKind::ClassType);
  registerWrite(self, node);
  // Also the value being set must become a wildcard.
  const auto newValue = node->inputs().at(1);
  setWildcard(newValue);
}

// Used for anything where we do not have accurate alias summaries
// may write to any input and produce wildcards
void AliasDb::analyzeConservative(Node* node) {
  for (const auto input : node->inputs()) {
    if (!isMutableTypeInternal(input)) {
      continue;
    }
    registerWrite(input, node, /*writeToContained=*/true);
    setWildcard(input);
  }

  for (const auto output : node->outputs()) {
    setWildcard(output);
  }
}

bool AliasDb::functionalNonEscapingListUse(const Use& use) const {
  Node* n = use.user;
  size_t offset = use.offset;
  Value* container = n->inputs().at(offset);

  // only consider aten op uses of lists
  if (!container->type()->cast<ListType>()) {
    return false;
  }

  /*
  in the general case, we consider any Value that enters another container as
  entering the heap, and thus aliasing all other heap values of the same type.
  the advantage of this approach are:
  - there are many composite list/container ops that would be tricky to
  schematize if we did something more complicated
  - limits the size of the AliasDb, because a container of size 10 only contains
  1 memory dag element instead of 10
  - we do not need to worry about adding contained elements to the wildcard set
  when a container escapes the graph.
  The downside of this approach is we are unable to handle the common case of a
  list constructed and passed into an aten op. Here, optimize for a set of
  common ops where the output does not alias the list or the list elements
  */

  // only used in output of graph - no further uses,
  // so there will be no use of it where the contained element leaks
  if (use.user->kind() == prim::Return) {
    return use.user->owningBlock() == graph_->block();
  }

  switch (use.user->kind()) {
    case aten::cat:
    case aten::broadcast_tensors:
    case aten::stack:
    case aten::vstack:
    case aten::hstack:
    case aten::dstack:
      return true;
  }
  auto op = use.user->maybeOperator();
  if (op && op->aliasAnalysisKind() == AliasAnalysisKind::PURE_FUNCTION) {
    return true;
  }
  return false;
}

bool AliasDb::functionalNonEscapingTupleUse(const Use& use) const {
  Node* n = use.user;
  size_t offset = use.offset;
  Value* container = n->inputs().at(offset);
  if (!container->type()->cast<TupleType>()) {
    return false;
  }
  // TODO(T97387453): Cover more ops that do not let escape tuples' elements.
  bool in_return_outputs = use.user->kind() == prim::Return;
  bool not_in_nested_subgraph = use.user->owningBlock() == graph_->block();
  return in_return_outputs && not_in_nested_subgraph;
}

// List or dict or tuple construct: create an aliasing element for the actual
// container, then mark all inputs as wildcards, since they've gone inside the
// container. Then, add the wildcard sets of appropriate type to the contained
// elements of the container.
void AliasDb::analyzeContainerConstruct(Node* node) {
  TORCH_INTERNAL_ASSERT(
      node->kind() == prim::ListConstruct ||
      node->kind() == prim::DictConstruct ||
      node->kind() == prim::TupleConstruct);

  // tuples which contain immutable types are immutable
  if (!isMutableTypeInternal(node->output())) {
    return;
  }

  TORCH_INTERNAL_ASSERT(node->outputs().size() == 1);
  auto container = node->output();

  // optimization:
  // if a list is only used once in an aten op, and the op output
  // doesn't alias the input, then we can add all inputs to the list's
  // contained elements instead of the wildcard set.
  if (container->uses().size() == 1 &&
      (functionalNonEscapingListUse(container->uses().at(0)) ||
       functionalNonEscapingTupleUse(container->uses().at(0)))) {
    giveFreshAlias(container, false);
    for (Value* v : node->inputs()) {
      addToContainedElements(v, container);
    }
    return;
  }

  giveFreshAlias(container);
  auto container_elem = elementMap_.at(container);
  for (auto input : node->inputs()) {
    auto maybe_wildcard_elem = setWildcard(input);
    if (maybe_wildcard_elem) {
      memoryDAGBuilder_->addToContainedElements(
          *maybe_wildcard_elem, container_elem);
    }
  }
}

// BroadcastingChunk: all inputs are broadcasted, and then individually chunked.
// This is an intermediate node used only in the graph fuser.
void AliasDb::analyzeBroadcastingChunk(Node* node) {
  auto inputs = node->inputs();
  auto outputs = node->outputs();
  auto nchunks = node->i(attr::chunks);
  for (const auto index : c10::irange(inputs.size())) {
    // Each inputs[i] is aliased by exactly `nchunks` distinct output tensors:
    // inputs[i] produces chunks outputs[i * nchunks + k] for k in [0..nchunks)
    auto output_begin = outputs.begin() + index * nchunks;
    for (auto it = output_begin; it != output_begin + nchunks; ++it) {
      makePointerTo(*it, inputs.at(index));
    }
  }
}

bool AliasDb::nonAliasingValue(const Value* elem) const {
  // these are values which can point to aliasing types in the graph,
  // as with a None value pointing to an optional if node output,
  // but will never alias themselves
  return elem->mustBeNone() || elem->node()->kind() == prim::Uninitialized;
}

// Register the fact that `from` is a pointer to `to`
void AliasDb::makePointerTo(const Value* from, const Value* to) {
  if (nonAliasingValue(from) || nonAliasingValue(to)) {
    // if either value is guaranteed to be non-aliasing, we do not need to
    // connect the two elements. however, it is invariant that aliasing types
    // that are not wildcards have a memory dag element, so we create one if
    // needed
    giveFreshAlias(from);
    giveFreshAlias(to);
    return;
  }

  // The contained types of immutable type containers (`Optional`,
  // `Tuple`, `Future`, and `Union`) are unified, so these types can be
  // mutable or immutable and point to a type which is mutable or
  // immutable. `Any` is mutable but can point to an immutable type
  // through refinement
  if (isMutableTypeInternal(from) != isMutableTypeInternal(to)) {
    return;
  }
  // both immutable
  if (!isMutableTypeInternal(from)) {
    return;
  }
  if (from == to) {
    return;
  }

  // At this point, we are dealing with two mutable types
  auto from_el = getOrCreateElement(from);
  auto to_el = getOrCreateElement(to);

  memoryDAGBuilder_->makePointerTo(from_el, to_el);
}

void AliasDb::addToContainedElements(
    const Value* inner,
    const Value* container) {
  if (!isMutableTypeInternal(inner)) {
    return;
  }

  auto inner_el = getOrCreateElement(inner);
  auto cont_el = getOrCreateElement(container);

  memoryDAGBuilder_->addToContainedElements(inner_el, cont_el);
}

bool AliasDb::mayAlias(const Value* a, const Value* b) const {
  if (!isMutableTypeInternal(a) || !isMutableTypeInternal(b)) {
    return false;
  }

  return memoryDAG_->mayAlias(elementMap_.at(a), elementMap_.at(b));
}

bool AliasDb::mayAlias(const ValueSet& a, const ValueSet& b) const {
  if (a.empty() || b.empty()) {
    return false;
  }

  // Record all memory locations from group `a`
  MemoryLocations aMemLocs;
  for (const auto value : a) {
    auto it = elementMap_.find(value);
    if (it != elementMap_.end()) {
      aMemLocs |= memoryDAG_->getMemoryLocations(it->second);
    }
  }

  // If any of group `b`s memory locations overlap, return true.
  for (const auto value : b) {
    auto it = elementMap_.find(value);
    if (it != elementMap_.end()) {
      if (aMemLocs.intersects(memoryDAG_->getMemoryLocations(it->second))) {
        return true;
      }
    }
  }
  // No overlap, so group `a` and `b` do not share a memory location
  return false;
}

bool AliasDb::mayContainAlias(Value* a, Value* b) const {
  if (!isMutableTypeInternal(a) || !isMutableTypeInternal(b)) {
    return false;
  }
  return memoryDAG_->mayContainAlias(elementMap_.at(a), elementMap_.at(b));
}

std::vector<Element*> AliasDb::getElements(at::ArrayRef<Value*> vs) const {
  std::vector<Element*> elements;
  for (const auto& val : vs) {
    if (isMutableTypeInternal(val)) {
      elements.push_back(elementMap_.at(val));
    }
  }
  return elements;
}

bool AliasDb::mayContainAlias(
    const at::ArrayRef<Value*> a,
    const at::ArrayRef<Value*> b) const {
  auto a_elems = getElements(a);
  return a_elems.size() == 0
      ? false
      : memoryDAG_->mayContainAlias(a_elems, getElements(b));
}

bool AliasDb::mayContainAlias(Value* a, const at::ArrayRef<Value*> b) const {
  if (!isMutableTypeInternal(a)) {
    return false;
  }
  auto b_elems = getElements(b);
  return b_elems.size() == 0
      ? false
      : memoryDAG_->mayContainAlias(elementMap_.at(a), b_elems);
}

// Make each value in the `from` list point to its partner in the `to` list
void AliasDb::mapAliases(at::ArrayRef<Value*> from, at::ArrayRef<Value*> to) {
  TORCH_INTERNAL_ASSERT(to.size() == from.size());
  for (const auto i : c10::irange(to.size())) {
    makePointerTo(from[i], to[i]);
  }
}

// Should only be called from create_functional_graphs.
// The asserts are to guard against unintentional use.
// FIXME refactor aliasdb construction to be more robust to mutation so this
// hack isn't necessary.
void AliasDb::createValue(const Value* value) {
  TORCH_INTERNAL_ASSERT(isMutableTypeInternal(value->type()));
  auto new_elem = memoryDAG_->unsafeMakeFreshValue(value);
  elementMap_[value] = new_elem;
}

void AliasDb::giveFreshAlias(
    const Value* value,
    bool add_wildcard_to_contained_elems) {
  auto maybe_mut_types = mapTypeToAliasTypeSetPtr(value->type());
  if (!maybe_mut_types) {
    return;
  }

  if (elementMap_.count(value)) {
    // Inside a loop, we may have given a fresh alias to this value already, so
    // skip
    return;
  }

  auto new_elem = memoryDAGBuilder_->makeFreshValue(value);
  elementMap_[value] = new_elem;
  if (add_wildcard_to_contained_elems) {
    if (maybe_mut_types->size() > 1) {
      pointUnionTypeElementToAllContainedTypes(new_elem, *maybe_mut_types);
    } else {
      addContainedTypesToFreshElement(new_elem, *maybe_mut_types);
    }
  }
}

Element* AliasDb::getOrCreateElement(const Value* value) {
  if (!elementMap_.count(value)) {
    giveFreshAlias(value);
  }
  return elementMap_.at(value);
}

void AliasDb::replaceWithNewValue(Value* existing, Value* new_value) {
  TORCH_INTERNAL_ASSERT(
      *unshapedType(existing->type()) == *unshapedType(new_value->type()),
      "Types must be strictly equal if you are replacing aliasing information. ",
      "Got existing: '",
      existing->type()->repr_str(),
      "', new_value: '",
      new_value->type()->repr_str(),
      "'");
  if (!isMutableTypeInternal(existing)) {
    return;
  }
  auto existing_elem = elementMap_.at(existing);
  elementMap_[new_value] = existing_elem;
  elementMap_.erase(existing);
  existing_elem->values = {new_value};
}

void AliasDb::copyValue(Value* from, Value* to) {
  TORCH_INTERNAL_ASSERT(
      *unshapedType(from->type()) == *unshapedType(to->type()),
      "Types must be strictly equal if you are copying aliasing information. ",
      "Got from: '",
      from->type()->repr_str(),
      "', to: '",
      to->type()->repr_str(),
      "'");
  if (!isMutableTypeInternal(to)) {
    return;
  }
  auto origElem = elementMap_.at(from);
  elementMap_[to] = origElem;
  origElem->values.insert(to);
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

bool AliasDb::hasWriters(const at::ArrayRef<Value*>& values) const {
  return std::any_of(values.begin(), values.end(), [&](Value* value) {
    return hasWriters(value);
  });
}

bool AliasDb::escapesScope(const at::ArrayRef<Value*>& vs) const {
  return mayContainAlias(graph_->inputs(), vs) ||
      mayContainAlias(graph_->outputs(), vs) || mayAliasWildcard(vs);
}

// Correctness conditions:
// no values in either set can have writers, and values in both sets
// cannot escape the current graph scope. Values can escape the current scope
// by aliasing a graph output or input, or by aliasing the wildcard set.
bool AliasDb::safeToChangeAliasingRelationship(
    const at::ArrayRef<Value*>& a,
    const at::ArrayRef<Value*>& b) const {
  if (hasWriters(a) || hasWriters(b)) {
    return false;
  }

  return !(escapesScope(a) && escapesScope(b));
}

// Helper for topologically-safe node moves. See `tryMove()` for details.
class AliasDb::WorkingSet {
 public:
  explicit WorkingSet(Node* mover, const AliasDb& aliasDb) : aliasDb_(aliasDb) {
    mover_ = mover;
    for (const auto user : getUsersSameBlock(mover_)) {
      moverUsers_.insert(user);
    }
    moverWrites_ |= aliasDb_.getWrites(mover_);
    moverReads_ |= aliasDb_.getReads(mover_);
  }

  // Add `n` to the working set
  void add(Node* n) {
    nodes_.push_back(n);
    node_to_index_[n] = nodes_.size() - 1;
    for (const auto user : getUsersSameBlock(n)) {
      users_.insert(user);
    }

    writes_ |= aliasDb_.getWrites(n);
    reads_ |= aliasDb_.getReads(n);
  }

  void eraseMover() {
    mover_ = nullptr;
    moverWrites_.clear();
    moverReads_.clear();
    moverUsers_.clear();
  }

  const std::vector<Node*>& dependentNodes() {
    return nodes_;
  }

  // Does the working set depend on `n`?
  bool dependsOn(Node* n) const {
    if (!mover_ && nodes_.empty()) {
      return false;
    }

    return hasDataDependency(n) || hasMutabilityDependency(n);
  }

 private:
  bool hasDataDependency(Node* n) const {
    if (!mover_ && nodes_.empty()) {
      return false;
    }
    const Node* pivot = mover_ ? mover_ : nodes_.front();
    if (n->isAfter(pivot)) {
      return producesFor(n);
    } else {
      return consumesFrom(n);
    }
  }

  bool hasMutabilityDependency(Node* n) const {
    // Check that `n` does not write to anything used by the working set
    const auto& nWrites = aliasDb_.getWrites(n);
    if (reads_.intersects(nWrites)) {
      return true;
    }
    if (mover_ && moverReads_.intersects(nWrites)) {
      return true;
    }

    // Check that the working set doesn't write to anything that `n` uses.
    const auto& nReads = aliasDb_.getReads(n);
    if (writes_.intersects(nReads)) {
      return true;
    }
    if (mover_ && moverWrites_.intersects(nReads)) {
      return true;
    }
    return false;
  }

  // Does the working set produce any values consumed by `n`?
  bool producesFor(Node* n) const {
    // This equivalent to asking: does the total use-set of all the nodes in the
    // working set include `n`?
    if (mover_ && moverUsers_.count(n)) {
      return true;
    }
    return users_.count(n) != 0;
  }

  // Does the working set consume any values produced by `n`?
  bool consumesFrom(Node* n) const {
    const auto users = getUsersSameBlock(n);

    if (mover_ && users.count(mover_)) {
      return true;
    }
    return std::any_of(users.begin(), users.end(), [&](Node* user) {
      return node_to_index_.find(user) != node_to_index_.end();
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

  // Traverse `target`'s blockchain upward until we find a node that shares a
  // block with `n`.
  //
  // If one can't be found (say, because `n` is an inner block and target is
  // outside), then return nullptr. Since we can only reorder nodes within a
  // block, `target` would be irrelevant.
  static Node* findSameBlock(Node* target, Node* n) {
    TORCH_INTERNAL_ASSERT(target->owningGraph() == n->owningGraph());
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
  std::vector<Node*> nodes_;
  // Extra data structure for nodes for faster look up
  // Since the tryMove method is used a lot, we want to
  // make it as fast as possible.
  std::unordered_map<Node*, int64_t> node_to_index_;

  // Mover dependencies. We track these separately since we may erase the mover
  // from the working set.
  Node* mover_;
  MemoryLocations moverWrites_;
  MemoryLocations moverReads_;
  std::unordered_set<Node*> moverUsers_;

  // users => # of working set nodes it uses
  std::unordered_set<Node*> users_;
  // Values written to by the working set => number of nodes writing to value
  MemoryLocations writes_;
  MemoryLocations reads_;
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
  if (toMove->owningBlock() != movePoint->owningBlock()) {
    return false;
  }
  if (toMove == movePoint) {
    return true;
  }

  // 1. Move from `this` toward movePoint, building up the working set of
  // dependencies
  WorkingSet workingSet(toMove, *this);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int direction;
  if (toMove->isAfter(movePoint)) {
    direction = kPrevDirection;
  } else {
    direction = kNextDirection;
  }

  auto curNode = toMove->next_in_graph[direction];

  bool toMoveIsOnMoveSide =
      (moveSide == MoveSide::BEFORE && toMove->isBefore(movePoint)) ||
      (moveSide == MoveSide::AFTER && toMove->isAfter(movePoint));

  if (toMoveIsOnMoveSide && curNode == movePoint) {
    return true;
  }

  // it is never valid to move reorder a node with side effects
  if (toMove->hasSideEffects() ||
      (!toMoveIsOnMoveSide && movePoint->hasSideEffects())) {
    return false;
  }

  // Move forward one node at a time
  while (curNode != movePoint) {
    // never valid to reorder around a node with side effects
    if (curNode->hasSideEffects()) {
      return false;
    }

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
  TORCH_INTERNAL_ASSERT(curNode == movePoint);
  if (splitToMoveAndDeps) {
    // Move `toMove`
    move(toMove, movePoint, moveSide);

    // Then move all of its dependencies on the other side of `movePoint`
    const auto reversed =
        moveSide == MoveSide::BEFORE ? MoveSide::AFTER : MoveSide::BEFORE;
    for (auto n : workingSet.dependentNodes()) {
      move(n, curNode, reversed);
      curNode = n;
    }
  } else {
    // Just append/prepend everything to `movePoint`
    move(toMove, curNode, moveSide);
    curNode = toMove;
    for (auto n : workingSet.dependentNodes()) {
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

bool AliasDb::writesToWildcard(Node* n) const {
  if (!writeIndex_->count(n)) {
    return false;
  }
  const auto& writes = writeIndex_->at(n);

  // Are any of these memoryLocs a wildcard element?
  for (const auto& pr : wildcardIndex_) {
    const auto wildcardElement = pr.second;
    if (writes.test(wildcardElement->index)) {
      return true;
    }
  }
  return false;
}

bool AliasDb::mayAliasWildcard(const Value* v) const {
  if (auto e = getWildcard(v->type())) {
    return memoryDAG_->mayAlias(elementMap_.at(v), e);
  }
  // There were no wildcards of this type, so return false.
  return false;
}

bool AliasDb::mayAliasWildcard(const at::ArrayRef<Value*> vs) const {
  return std::any_of(
      vs.begin(), vs.end(), [&](Value* v) { return mayAliasWildcard(v); });
}

c10::optional<Element*> AliasDb::tryGetOrCreateWildcard(const TypePtr& type) {
  auto maybe_mut_types = mapTypeToAliasTypeSetPtr(type);
  if (!maybe_mut_types) {
    return c10::nullopt;
  }
  auto mut_type = toSingleType(*maybe_mut_types);
  auto existing_wildcard = wildcardIndex_.find(*mut_type);
  if (existing_wildcard != wildcardIndex_.end()) {
    return existing_wildcard->second;
  }

  auto wildcard_elem = memoryDAGBuilder_->makeFreshValue(nullptr);
  wildcardIndex_.emplace(*std::move(mut_type), wildcard_elem);
  if (maybe_mut_types->size() > 1) {
    pointUnionTypeElementToAllContainedTypes(wildcard_elem, *maybe_mut_types);
  } else {
    addContainedTypesToFreshElement(wildcard_elem, *maybe_mut_types);
  }
  return wildcard_elem;
}

void AliasDb::pointUnionTypeElementToAllContainedTypes(
    Element* container_elem,
    const AliasTypeSet& mut_types) {
  for (const auto& mut_type : mut_types) {
    auto maybe_elem = tryGetOrCreateWildcard(mut_type);
    if (maybe_elem) {
      TORCH_INTERNAL_ASSERT(*maybe_elem != container_elem);
      memoryDAGBuilder_->makePointerTo(container_elem, *maybe_elem);
    }
  }
}

void AliasDb::addContainedTypesToFreshElement(
    Element* container_elem,
    const AliasTypeSet& mut_types) {
  for (const auto& mut_type : mut_types) {
    for (const auto& contained : mut_type->containedTypes()) {
      auto maybe_elem = tryGetOrCreateWildcard(contained);
      if (maybe_elem) {
        memoryDAGBuilder_->addToContainedElements(*maybe_elem, container_elem);
      }
    }
  }
}

// Search the wildcard index for an element that corresponds to the given type.
// Const version returns nullptr
Element* AliasDb::getWildcard(const TypePtr& type) const {
  auto maybe_mut_types = mapTypeToAliasTypeSetPtr(type);
  if (!maybe_mut_types) {
    return {};
  }
  if (maybe_mut_types->size() > 1) {
    auto union_type = UnionType::create(*maybe_mut_types);
    // Get a <TypePtr, Element*> pair where the TypePtr is this Union
    // type and the Element is the corresponding Wildcard
    auto maybe_union_pair = wildcardIndex_.find(union_type);
    if (maybe_union_pair != wildcardIndex_.end()) {
      return (*maybe_union_pair).second;
    }
  } else {
    // Get a <TypePtr, Element*> pair where the TypePtr is the given
    // type and the Element is the corresponding Wildcard
    auto type_pair = wildcardIndex_.find((*maybe_mut_types)[0]);
    if (type_pair != wildcardIndex_.end()) {
      return type_pair->second;
    }
  }
  return {};
}

// Register `v` as a wildcard value.
c10::optional<Element*> AliasDb::setWildcard(const Value* v) {
  c10::optional<Element*> maybe_wildcardElement =
      tryGetOrCreateWildcard(v->type());
  if (!maybe_wildcardElement) {
    return c10::nullopt;
  }
  // Ensure that we create a corresponding Element for `v` still, as it is an
  // invariant that all mutable values have an Element
  getOrCreateElement(v);
  wildcards_.insert(v);
  return *maybe_wildcardElement;
}

void AliasDb::buildWrittenToLocationsIndex() {
  MemoryLocations ret;
  for (const auto& pr : *writeIndex_) {
    const auto& writtenLocs = pr.second;
    ret |= writtenLocs;
  }
  writtenToLocationsIndex_ = ret;
}

void Lint(const AliasDb* db) {
  bool failed = false;

  std::stringstream ss;
  // Every mutable value in the system has a corresponding element.
  for (const auto& v : db->graph_->all_values) {
    if (!db->isMutableTypeInternal(v)) {
      continue;
    }
    auto it = db->elementMap_.find(v);
    if (it == db->elementMap_.end()) {
      failed = true;
      ss << "Value %" << v->debugName() << " of type " << v->type()->repr_str()
         << " wasn't found in the element map.\n"
         << "It was defined in " << *v->node();
    }
  }
  TORCH_INTERNAL_ASSERT(!failed, ss.str());

  // Two checks that we want to add but can't until the mutation API is more
  // fully developed.
  // - Every mutable value in the aliasdb belongs to the graph
  // - All container values have contained elements
}

} // namespace jit
} // namespace torch
