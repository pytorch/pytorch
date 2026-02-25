#include <torch/nativert/graph/Graph.h>

#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <limits>

#include <c10/util/Enumerate.h>
#include <c10/util/FbcodeMaps.h>
#include <c10/util/StringUtil.h>
#include <torch/nativert/executor/Placement.h>
#include <torch/nativert/graph/TensorMeta.h>

namespace torch::nativert {

namespace {

// Workaround for MSVC bug: "std" ambiguous symbol.
template <typename T, typename U>
constexpr bool is_same_v = std::is_same_v<T, U>;

bool isBlank(char n) {
  return std::isspace(n);
}

size_t consumeWhitespaceImpl(std::string_view source, size_t curPos) {
  while (isBlank(source.at(curPos))) {
    curPos++;
  }
  return curPos;
}

size_t expectImpl(
    std::string_view source,
    std::string_view expected,
    size_t curPos) {
  curPos = consumeWhitespaceImpl(source, curPos);
  const auto actual = source.substr(curPos, expected.size());
  TORCH_CHECK(
      expected == actual,
      fmt::format(
          "Parser error: expected '{}' at position {}, but found '{}'.",
          expected,
          curPos,
          actual));
  curPos += expected.size();
  return curPos;
}

size_t expectImpl(std::string_view source, char expected, size_t curPos) {
  curPos = consumeWhitespaceImpl(source, curPos);
  while (isBlank(source.at(curPos))) {
    curPos++;
  }
  TORCH_CHECK(
      expected == source[curPos],
      "Parser error: expected '{}' at position {}, but found '{}'.",
      expected,
      curPos,
      source[curPos]);
  curPos++;
  return curPos;
}
} // namespace

bool operator==(const Type& left, const Type& right) {
  if (left.kind() != right.kind()) {
    return false;
  }
  if (std::holds_alternative<Type::CustomObjData>(left.kind_) &&
      std::holds_alternative<Type::CustomObjData>(right.kind_)) {
    return std::get<Type::CustomObjData>(left.kind_).classFqn ==
        std::get<Type::CustomObjData>(right.kind_).classFqn;
  }
  return true;
}

Graph::Graph()
    : insertBefore_(nodes_.end()),
      inputNode_(insertNode("prim.Input", {})),
      outputNode_(insertNode("prim.Output", {})) {
  // Set the insertion point to append to the graph
  insertBefore_ = nodes_.iterator_to(*outputNode_);
}

std::string Graph::getUniqueValueName() {
  auto name = fmt::format("v{}", uniqueValueName_);
  while (values_.find(name) != values_.end()) {
    name = fmt::format("v{}", uniqueValueName_++);
  }
  return name;
}

// If `name` is null, create a unique value name
Value* Graph::addValue(
    const std::optional<std::string>& name,
    const Type& type,
    Node* node) {
  const auto valueName = name.value_or(getUniqueValueName());
  ValueId valueId = getNextValueId();
  const auto [it, success] = values_.insert(
      {valueName, std::make_unique<Value>(valueId, valueName, type, node)});
  TORCH_CHECK(
      success,
      fmt::format(
          "Tried to create Value with name: '{}', but it already existed",
          valueName));
  return it->second.get();
}

Value* Graph::addInput(std::string_view name, const Type& type) {
  return inputNode_->addOutput(name, type);
}

void Graph::addInput() {
  inputNode_->addOutput();
}

Value* Graph::addOutput(Value* v) {
  outputNode_->addInput({std::string(v->name()), v});
  return v;
}

void Graph::addConstantOutput(Constant c) {
  constantOutputs_.push_back(std::move(c));
}

// Create a node without inserting it into the execution graph.
Node* Graph::createNode(
    std::string target,
    std::vector<NamedArgument> inputs,
    std::unordered_map<std::string, std::string> metadata) {
  auto& node = nodesOwner_.emplace_back(std::make_unique<Node>(
      this, std::move(target), std::move(inputs), std::move(metadata)));
  return node.get();
}

Node* Graph::insertBefore(Node* toInsert, Node* insertionPoint) {
  TORCH_CHECK(insertionPoint != inputNode_, "can't insert before prim.Input");
  TORCH_CHECK(
      !toInsert->is_linked(), "expected node to be unlinked: ", *toInsert);
  TORCH_CHECK(
      insertionPoint->is_linked(),
      "expected node to be linked: ",
      *insertionPoint);
  auto it = nodes_.insert(nodes_.iterator_to(*insertionPoint), *toInsert);
  return &*it;
}

Node* Graph::insert(Node* toInsert) {
  TORCH_CHECK(
      !toInsert->is_linked(), "expected node to be unlinked: ", *toInsert);
  nodes_.insert(insertBefore_, *toInsert);
  return toInsert;
}

Node* Graph::insertAfter(Node* toInsert, Node* insertionPoint) {
  TORCH_CHECK(insertionPoint != outputNode_, "can't insert after prim.Output");
  TORCH_CHECK(
      !toInsert->is_linked(), "expected node to be unlinked: ", *toInsert);
  TORCH_CHECK(
      insertionPoint->is_linked(),
      "expected node to be linked: ",
      *insertionPoint);

  auto insertIt = nodes_.iterator_to(*insertionPoint);
  // Increment once because we want to insert after the insertion point
  ++insertIt;
  auto it = nodes_.insert(insertIt, *toInsert);
  return &*it;
}

Node* Graph::insertNode(
    std::string target,
    std::vector<NamedArgument> inputs,
    std::unordered_map<std::string, std::string> metadata) {
  auto node =
      createNode(std::move(target), std::move(inputs), std::move(metadata));
  nodes_.insert(insertBefore_, *node);
  return node;
}

std::ostream& operator<<(std::ostream& out, const Type& ty) {
  std::visit(
      [&out](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (is_same_v<T, Type::Kind>) {
          switch (arg) {
            case Type::Kind::None:
              out << "None";
              break;
            case Type::Kind::Tensor:
              out << "Tensor";
              break;
            case Type::Kind::TensorList:
              out << "TensorList";
              break;
            case Type::Kind::NestedTensorList:
              out << "NestedTensorList";
              break;
            case Type::Kind::OptionalTensorList:
              out << "OptionalTensorList";
              break;
            case Type::Kind::SymInt:
              out << "SymInt";
              break;
            case Type::Kind::SymFloat:
              out << "SymFloat";
              break;
            case Type::Kind::SymIntList:
              out << "SymIntList";
              break;
            case Type::Kind::CustomObj:
              out << "CustomObj";
              break;
            default:
              TORCH_CHECK(false, "Unhandled type");
          }
        } else if constexpr (is_same_v<T, Type::CustomObjData>) {
          out << "CustomObj: " << arg.classFqn;
        }
      },
      ty.kind_);
  return out;
}

const NamedArgument* Node::tryGetInput(std::string_view name) const {
  // Just do a scan over the inputs. We expect there to always be a very small
  // number of elements, so it shouldn't be slow. This allows us to avoid a
  // second datastructure for lookups.
  // Drop a debug check here, just to make sure :)
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inputs_.size() < 1000);
  for (const auto& input : inputs_) {
    if (input.name == name) {
      return &input;
    }
  }
  return nullptr;
}

const NamedArgument& Node::getInput(std::string_view name) const {
  const auto ret = tryGetInput(name);
  if (ret == nullptr) {
    TORCH_CHECK(
        false,
        fmt::format(
            "Expected input '{}' on node: '{}' to exist, but it does not.",
            name,
            fmt::streamed(*this)));
  }
  return *ret;
}

const Attribute* Node::tryGetAttribute(std::string_view name) const {
  // Just do a scan over the inputs. We expect there to always be a very small
  // number of elements, so it shouldn't be slow. This allows us to avoid a
  // second datastructure for lookups.
  // Drop a debug check here, just to make sure :)
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(attributes_.size() < 1000);
  for (const auto& attribute : attributes_) {
    if (attribute.name == name) {
      return &attribute;
    }
  }
  return nullptr;
}

const Attribute& Node::getAttribute(std::string_view name) const {
  const auto ret = tryGetAttribute(name);
  if (ret == nullptr) {
    TORCH_CHECK(
        false,
        fmt::format(
            "Expected attribute '{}' on node: '{}' to exist, but it does not.",
            name,
            fmt::streamed(*this)));
  }
  return *ret;
}

void Node::applyDevicePlacement(const Placement& placement) {
  for (auto& attribute : attributes_) {
    if (std::holds_alternative<c10::Device>(attribute.value)) {
      auto device = std::get<c10::Device>(attribute.value);
      auto targetDevice =
          placement.getMappedDevice(std::get<c10::Device>(attribute.value));
      if (!isSameDevice(targetDevice, device)) {
        LOG(INFO) << "Overriding " << device.str() << " to "
                  << targetDevice.str() << " for node " << *this;
        attribute.value = targetDevice;
      }
    }
  }
}

Node* Node::next() {
  return owningGraph()->nodeAfter(this);
}

const Node* Node::next() const {
  return owningGraph()->nodeAfter(this);
}

Node* Node::prev() {
  return owningGraph()->nodeBefore(this);
}

const Node* Node::prev() const {
  return owningGraph()->nodeBefore(this);
}

bool Node::isBefore(const Node* n) const {
  if (this == n) {
    return false;
  }

  for (const Node* cursor = this->next(); cursor != nullptr;
       cursor = cursor->next()) {
    if (cursor == n) {
      return true;
    }
  }
  // Reached the end without finding n
  return false;
}

std::vector<Node*> Node::producers() const {
  std::vector<Node*> ret;

  if (this->prev() == nullptr /* prim.Input */) {
    return ret;
  }

  if (this->next() == nullptr /* prim.Output */) {
    for (auto& node : owningGraph_->nodes()) {
      if (node.next() == nullptr /* prim.Output */ ||
          node.prev() == nullptr /* prim.Input */) {
        continue;
      }
      for (auto* dep : node.users()) {
        if (dep == this /* prim.Output */) {
          ret.push_back(&node);
        }
      }
    }
  } else {
    std::unordered_set<const Node*> seen;

    for (const auto& input : inputs()) {
      auto* n = input.value->producer();
      if (n == nullptr) {
        continue;
      }
      if (const auto [_, inserted] = seen.insert(n); inserted) {
        ret.push_back(n);
      }
    }

    if (ret.empty()) {
      ret.push_back(owningGraph_->inputNode());
    }
  }

  return ret;
}

std::vector<Node*> Node::users() const {
  std::vector<Node*> ret;

  if (this->next() == nullptr /* prim.Output */) {
    return ret;
  }

  if (this->prev() == nullptr /* prim.Input */) {
    for (auto& node : owningGraph_->nodes()) {
      if (node.prev() == nullptr /* prim.Input */ ||
          node.next() == nullptr /* prim.Output */) {
        continue;
      }
      for (auto* dep : node.producers()) {
        if (dep == this /* prim.Input */) {
          ret.push_back(&node);
        }
      }
    }
  } else {
    std::unordered_set<const Node*> seen;

    for (const auto* output : outputs()) {
      for (auto* n : output->users()) {
        if (const auto [_, inserted] = seen.insert(n); inserted) {
          ret.push_back(n);
        }
      }
    }

    if (ret.empty()) {
      ret.push_back(owningGraph_->outputNode());
    }
  }

  return ret;
}

Node* Graph::createListPack(std::vector<Value*> inputs, const Type& inputType) {
  std::vector<NamedArgument> nodeInputs;
  nodeInputs.reserve(inputs.size());
  for (auto [i, input] : c10::enumerate(inputs)) {
    nodeInputs.push_back({fmt::format("l{}", i), input});
  }
  // Create a new named value for this
  auto name = getUniqueValueName();
  auto node = createNode("prim.ListPack", std::move(nodeInputs));

  // Make sure all inputs are the same type
  for (auto& input : inputs) {
    TORCH_CHECK(input->type() == inputType);
  }

  if (inputType == Type::Kind::Tensor) {
    node->addOutput(name, Type::Kind::TensorList);
  } else if (inputType == Type::Kind::SymInt) {
    node->addOutput(name, Type::Kind::SymIntList);
  } else if (inputType == Type::Kind::TensorList) {
    // For nested tensor lists (List[List[Tensor]]), the inner lists are
    // TensorList type. We output a NestedTensorList type.
    node->addOutput(name, Type::Kind::NestedTensorList);
  }

  return node;
}

Node* Graph::createOptionalListPack(std::vector<Value*> inputs) {
  std::vector<NamedArgument> nodeInputs;
  nodeInputs.reserve(inputs.size());
  for (auto [i, input] : c10::enumerate(inputs)) {
    nodeInputs.push_back({fmt::format("l{}", i), input});
  }
  // Create a new named value for this
  auto name = getUniqueValueName();
  auto node = createNode("prim.ListPack", std::move(nodeInputs));
  // Make sure all inputs are either None or Tensor
  for (auto& input : inputs) {
    TORCH_CHECK(
        input->type() == Type::Kind::None ||
        input->type() == Type::Kind::Tensor);
  }
  node->addOutput(name, Type::Kind::OptionalTensorList);

  return node;
}

Value* Graph::createConstantSymIntValue(int value) {
  auto valueName = getUniqueValueName();
  ValueId valueId = getNextValueId();
  const auto [it, success] = values_.insert(
      {valueName,
       std::make_unique<Value>(
           valueId, valueName, Type::Kind::SymInt, nullptr)});
  TORCH_CHECK(
      success,
      fmt::format(
          "Tried to create constant SymInt Value with name: '{}', but it already existed",
          valueName));
  constantSymIntValues_[valueId] = value;
  return it->second.get();
}

Value* Graph::getValue(std::string_view name) const {
  // TODO: can eliminate this string copy by enabling heterogeneous lookup for
  // the container
  return values_.at(std::string(name)).get();
}

Value* Graph::tryGetValue(std::string_view name) const {
  // TODO: can eliminate this string copy by enabling heterogeneous lookup for
  // the container
  const auto key = std::string(name);
  if (values_.find(key) != values_.end()) {
    return values_.at(key).get();
  }
  return nullptr;
}

void Graph::renumberValues() {
  std::vector<Value*> currentValues;
  currentValues.reserve(values_.size());
  for (auto& kv : values_) {
    currentValues.push_back(kv.second.get());
  }

  // Sort values in creation order (by value ids)
  std::sort(currentValues.begin(), currentValues.end(), [](Value* a, Value* b) {
    return a->id() < b->id();
  });

  // Build a new id map with all ids < values_.size()
  std::unordered_map<ValueId, ValueId> oldToNew;
  oldToNew.reserve(currentValues.size());
  ValueId newId = 0;
  for (Value* v : currentValues) {
    oldToNew[v->id()] = newId;
    v->setId(newId);
    newId++;
  }

  std::unordered_map<ValueId, int> newSymIntMap;
  for (auto& [oldId, symIntVal] : constantSymIntValues_) {
    auto it = oldToNew.find(oldId);
    if (it != oldToNew.end()) {
      ValueId updatedId = it->second;
      newSymIntMap[updatedId] = symIntVal;
    }
  }
  constantSymIntValues_ = std::move(newSymIntMap);
  uniqueValueId_ = newId;
}

bool Graph::cleanupDeadNodes() {
  std::unordered_set<const Node*> visited;
  std::vector<const Node*> visitStack;

  // Mark reachable nodes from output
  visitStack.push_back(outputNode_);
  visited.insert(outputNode_);

  while (!visitStack.empty()) {
    const Node* current = visitStack.back();
    visitStack.pop_back();

    for (auto& namedArg : current->inputs()) {
      Value* val = namedArg.value;
      Node* producer = val->producer();

      if (!producer) {
        continue;
      }
      if (!visited.count(producer)) {
        visited.insert(producer);
        visitStack.push_back(producer);
      }
    }
  }

  // Remove all nodes not in visited (other than input/outputs)
  std::vector<Node*> toRemove;
  for (auto& n : nodes()) {
    if (n.target() == "prim.Input" || n.target() == "prim.Output" ||
        visited.count(&n)) {
      continue;
    }
    toRemove.push_back(&n);
  }

  const bool mutated = !toRemove.empty();

  // Remove nodes in reverse order to handle input/output dependencies
  for (auto it = toRemove.rbegin(); it != toRemove.rend(); ++it) {
    removeNode(*it);
  }

  renumberValues();
  lint();

  return mutated;
}

void Graph::lint() const {
  // Check that every value has a producer marked.
  for (const auto& [name, value] : values_) {
    // Some constant symint and None don't have producer nodes
    if (value->type().kind() != Type::Kind::SymInt &&
        value->type().kind() != Type::Kind::None) {
      TORCH_CHECK(value->isFolded() || value->producer() != nullptr);
    }
  }
  for (const auto& node : nodes()) {
    TORCH_CHECK(node.owningGraph() == this);
  }
  // Check that every list type is either produced by a prim.ListPack or
  // immediately consumed by a prim.ListUnpack. We make use of this invariant
  // to retrieve list elements in `getListElements`.
  for (const auto& [_, value] : values_) {
    if (value->type().kind() != Type::Kind::TensorList) {
      continue;
    }
    const bool producedByListPack =
        value->producer(/* resolve_folded = */ true)->target() ==
        "prim.ListPack";
    const bool consumedByListUnpack = value->users().size() == 1 &&
        value->users()[0]->target() == "prim.ListUnpack";
    TORCH_CHECK(producedByListPack || consumedByListUnpack);
  }

  auto getNames = [](const auto& values) {
    c10::FastSet<std::string> names;
    for (const auto* value : values) {
      if (value) {
        names.emplace(value->name());
      }
    }
    return names;
  };
  signature_.lint(getNames(inputs()), getNames(outputs()));
}

void Graph::finalize() {
  // build userOutputs_ view
  userOutputs_.clear();
  size_t constantIndex = 0;
  for (auto& outputName : signature_.userOutputs()) {
    if (outputName.has_value()) {
      userOutputs_.emplace_back(getValue(*outputName));
    } else {
      if (constantIndex < constantOutputs_.size()) {
        // Copy the constant rather than moving it, because finalize() may be
        // called multiple times (e.g. after constant folding). Moving would
        // leave constantOutputs_ entries in a moved-from state, causing
        // subsequent calls to produce empty strings/vectors.
        // Constant is non-copyable due to unique_ptr<Graph>, so we use
        // std::visit to copy each alternative individually.
        userOutputs_.emplace_back(std::visit(
            [](const auto& val) -> Constant {
              using T = std::decay_t<decltype(val)>;
              if constexpr (is_same_v<T, std::unique_ptr<Graph>>) {
                TORCH_CHECK(false, "Graph constant outputs cannot be copied");
                return Constant(None{});
              } else {
                return Constant(val);
              }
            },
            constantOutputs_[constantIndex]));
        constantIndex++;
      } else {
        TORCH_CHECK(false, "No more constant outputs available");
      }
    }
  }
}

namespace {
// Scan through a node's inputs, replacing ALL instances of `old` with
// `replacement`.  Returns true if a replacement occurred, otherwise false.
bool replace(Node* node, Value* old, Value* replacement) {
  bool replacementOccurred = false;
  for (auto& input : node->inputs()) {
    if (input.value == old) {
      input.value = replacement;
      replacementOccurred = true;
    }
  }
  return replacementOccurred;
}
} // namespace

void Graph::replaceAllUses(Value* old, Value* replacement) {
  for (auto user : old->users()) {
    // Find this use in the input list and replace it
    auto replaced = replace(user, old, replacement);
    TORCH_CHECK(replaced);
    replacement->addUser(user);
  }
  old->eraseAllUsers();
  signature_.replaceAllUses(old->name(), replacement->name());
}

void Graph::replaceAllUsesAfterNode(
    Value* old,
    Value* replacement,
    Node* afterThis) {
  auto it = nodes_.iterator_to(*afterThis);
  // Don't search `afterThis`
  ++it;
  // Scan through all node inputs linearly and replace uses
  for (; it != nodes_.end(); ++it) {
    Node* node = &*it;
    const bool replaced = replace(node, old, replacement);
    if (replaced) {
      old->eraseUser(node);
      replacement->addUser(node);
    }
  }
  signature_.replaceAllUses(old->name(), replacement->name());
}

void Graph::applyDevicePlacement(const Placement& placement) {
  TORCH_CHECK(
      !placementApplied_,
      "placement has been applied to the graph! placement must be applied once and once only.");

  placementApplied_ = true;

  // inplace override node's device-typed attributes according to placement
  for (auto& node : nodes_) {
    node.applyDevicePlacement(placement);
  }

  // inplace override weightMeta_'s device according to placement
  for (auto& [_, weightMeta] : weightsMeta_) {
    weightMeta.applyDevicePlacement(placement);
  }

  // inplace override tensorValuesMeta_'s device according to placement
  for (auto& [_, tensorMeta] : tensorValuesMeta_) {
    tensorMeta.applyDevicePlacement(placement);
  }
}

void Graph::overrideWeightsDevice(
    const std::unordered_map<std::string, std::optional<c10::Device>>&
        submodNameToDevice) {
  for (auto& [weightName, weightMeta] : weightsMeta_) {
    for (auto& [name, device] : submodNameToDevice) {
      if (device.has_value() && weightMeta.device() != device &&
          c10::starts_with(weightName, name) &&
          (weightName == name || weightName[name.length()] == '.')) {
        LOG(INFO) << "Overriding " << weightName << " from "
                  << weightMeta.device() << " to device " << device.value();
        weightMeta.setDevice(device.value());
        break;
      }
    }
  }

  for (auto& [tensorName, tensorMeta] : tensorValuesMeta_) {
    for (auto& [name, device] : submodNameToDevice) {
      if (device.has_value() && tensorMeta.device() != device &&
          c10::starts_with(tensorName, name) &&
          (tensorName == name || tensorName[name.length()] == '.')) {
        LOG(INFO) << "Overriding " << tensorName << " from "
                  << tensorMeta.device() << " to device " << device.value();
        tensorMeta.setDevice(device.value());
        break;
      }
    }
  }
}

Node* Graph::nodeAfter(Node* n) {
  TORCH_CHECK(n->owningGraph() == this);
  if (n == outputNode_) {
    return nullptr;
  }
  auto it = nodes_.iterator_to(*n);
  return &*(++it);
}

const Node* Graph::nodeAfter(const Node* n) const {
  TORCH_CHECK(n->owningGraph() == this);
  if (n == outputNode_) {
    return nullptr;
  }
  auto it = nodes_.iterator_to(*n);
  return &*(++it);
}

Node* Graph::nodeBefore(Node* n) {
  TORCH_CHECK(n->owningGraph() == this);
  if (n == inputNode_) {
    return nullptr;
  }
  auto it = nodes_.iterator_to(*n);
  return &*(--it);
}

const Node* Graph::nodeBefore(const Node* n) const {
  TORCH_CHECK(n->owningGraph() == this);
  if (n == inputNode_) {
    return nullptr;
  }
  auto it = nodes_.iterator_to(*n);
  return &*(--it);
}

void Graph::removeNode(Node* n) {
  TORCH_CHECK(n->owningGraph() == this, "Node does not belong to this graph!");

  for (auto* outputVal : n->outputs()) {
    TORCH_CHECK(
        outputVal->users().empty(),
        "Trying to erase a node that still has users: ",
        outputVal->name());
    outputVal->eraseAllUsers();
    removeValue(outputVal);
  }

  for (const auto& input : n->inputs()) {
    input.value->eraseUser(n);
  }

  TORCH_CHECK(n->is_linked(), "Node is not linked to the graph!");
  n->unlink();

  auto it = std::find_if(
      nodesOwner_.begin(),
      nodesOwner_.end(),
      [n](const std::unique_ptr<Node>& ptr) { return ptr.get() == n; });

  TORCH_CHECK(it != nodesOwner_.end(), "Node not found in nodesOwner_!");
  nodesOwner_.erase(it);
}

void Graph::removeValue(Value* value) {
  // TODO: assuming not removing from constantSymIntValues_
  TORCH_CHECK(value->users().empty(), "Cannot erase a value with users.");
  auto it = values_.find(std::string(value->name()));
  TORCH_CHECK(
      it != values_.end(),
      "Attempted to erase a value not in graph ",
      value->name());
  values_.erase(it);
}

std::vector<Value*> Graph::insertGraph(
    const Graph& subgraph,
    std::vector<Value*> inputs,
    std::unordered_map<const Value*, Value*>& valueMap) {
  TORCH_CHECK(subgraph.inputs().size() == inputs.size(), "Input size mismatch");
  for (auto i : c10::irange(subgraph.inputs().size())) {
    valueMap[subgraph.inputs()[i]] = inputs[i];
  }

  // Clone each node from subgraph
  for (const auto& n : subgraph.nodes()) {
    if (n.target() == "prim.Input" || n.target() == "prim.Output") {
      continue;
    }

    std::vector<NamedArgument> clonedInputs;
    auto inputs = n.inputs();
    clonedInputs.reserve(inputs.size());
    for (auto& inp : inputs) {
      auto it = valueMap.find(inp.value);
      TORCH_CHECK(it != valueMap.end(), "Missing input value in subgraph");
      clonedInputs.push_back({inp.name, it->second});
    }

    Node* newNode = insertNode(
        std::string(n.target()), std::move(clonedInputs), n.metadata());

    for (const auto& attr : n.attributes()) {
      Attribute newAttr;
      newAttr.name = attr.name;

      std::visit(
          [&](auto&& val) -> void {
            // Workaround for MSVC bug: "std" ambiguous symbol.
            using std::unique_ptr;
            using std::move;
            using T = std::decay_t<decltype(val)>;
            if constexpr (is_same_v<T, unique_ptr<Graph>>) {
              LOG(ERROR)
                  << "Graph attributes are not supported yet. Skipping attribute: "
                  << attr.name;
            } else {
              newAttr.value = val;
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wunqualified-std-cast-call"
#endif
              newNode->addAttribute(move(newAttr));
#ifdef __clang__
#pragma GCC diagnostic pop
#endif
            }
          },
          attr.value);
    }

    for (const auto* outVal : n.outputs()) {
      const auto& uniqueName = getUniqueValueName();
      Value* newOut = newNode->addOutput(uniqueName, outVal->type());
      valueMap[outVal] = newOut;
    }
  }

  auto subgraphOutputs = subgraph.outputs();
  std::vector<Value*> outputValues;
  outputValues.reserve(subgraphOutputs.size());
  for (auto* outputValue : subgraphOutputs) {
    outputValues.emplace_back(valueMap[outputValue]);
  }
  lint();
  return outputValues;
}

Node::Node(
    Graph* owningGraph,
    std::string target,
    std::vector<NamedArgument> inputs,
    std::unordered_map<std::string, std::string> metadata)
    : owningGraph_(owningGraph),
      target_(std::move(target)),
      inputs_(std::move(inputs)),
      metadata_(std::move(metadata)) {
  for (const auto& input : inputs_) {
    input.value->addUser(this);
  }
}

Value* Node::addInput(NamedArgument input) {
  inputs_.push_back(std::move(input));
  auto val = inputs_.back().value;
  val->addUser(this);
  return val;
}

void Node::addInputs(const std::vector<NamedArgument>& inputs) {
  for (const auto& input : inputs) {
    addInput(input);
  }
}

void Node::addAttribute(Attribute attr) {
  attributes_.push_back(std::move(attr));
}

void Node::addOutput() {
  outputs_.push_back(nullptr);
}

Value* Node::addOutput(const Type& type) {
  TORCH_CHECK(type == Type::Kind::None);
  Value* v = owningGraph_->addValue(std::nullopt, type, this);
  outputs_.push_back(v);
  return v;
}

Value* Node::addOutput(std::string_view name, const Type& type) {
  Value* v = owningGraph_->addValue(std::string(name), type, this);
  outputs_.push_back(v);
  return v;
}

void Node::destroy() {
  owningGraph_->removeNode(this);
}

void Value::addUser(Node* node) {
  for (const auto* user : users_) {
    if (user == node) {
      return;
    }
  }
  users_.push_back(node);
}

void Value::eraseUser(Node* node) {
  users_.erase(
      std::remove_if(
          users_.begin(), users_.end(), [&](Node* el) { return el == node; }),
      users_.end());
}

std::vector<const Value*> Value::getListElements() const {
  std::vector<const Value*> ret;
  if (auto p = producer(); p && p->target() == "prim.ListPack") {
    for (const auto& tv : p->inputs()) {
      ret.push_back(tv.value);
    }
  } else {
    TORCH_CHECK(users().size() == 1);
    const auto listUnpack = users()[0];
    TORCH_CHECK(listUnpack->target() == "prim.ListUnpack");
    for (const auto v : listUnpack->outputs()) {
      ret.push_back(v);
    }
  }
  return ret;
}

template <class>
[[maybe_unused]] inline constexpr bool AlwaysFalse = false;

c10::IValue constantToIValue(const Constant& constant) {
  // Workaround for MSVC bug: "std" ambiguous symbol.
  using std::string;
  using std::unique_ptr;
  using std::vector;
  return std::visit(
      [](auto&& arg) -> c10::IValue {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (is_same_v<T, None>) {
          return c10::IValue();
        } else if constexpr (std::is_convertible_v<T, c10::IValue>) {
          return arg;
        } else if constexpr (is_same_v<T, unique_ptr<Graph>>) {
          TORCH_CHECK(
              false, "subgraph arguments cannot be turned into ivalues!");
        } else {
          static_assert(AlwaysFalse<T>, "non-exhaustive visitor!");
        }
      },
      constant);
}

namespace {

template <class>
[[maybe_unused]] inline constexpr bool always_false_v = false;

void printDouble(std::ostream& out, double arg) {
  fmt::print(out, "{}", arg);
}

template <typename T, typename F>
std::ostream& printList(
    std::ostream& out,
    bool encloseInSquareBrackets,
    const T& list,
    F formatter) {
  if (encloseInSquareBrackets) {
    out << '[';
  }
  for (const auto& [idx, el] : c10::enumerate(list)) {
    if (idx > 0) {
      out << ", ";
    }
    formatter(out, el);
  }
  if (encloseInSquareBrackets) {
    out << ']';
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Constant& constant) {
  // Workaround for MSVC bug: "std" ambiguous symbol.
  using std::quoted;
  using std::string;
  using std::unique_ptr;
  using std::vector;
  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (is_same_v<T, None>) {
          out << "None";
        } else if constexpr (is_same_v<T, int64_t> || is_same_v<T, bool>) {
          out << arg;
        } else if constexpr (
            is_same_v<T, vector<int64_t>> || is_same_v<T, vector<bool>>) {
          out << fmt::format("{}", fmt::streamed(arg));
        } else if constexpr (is_same_v<T, double>) {
          printDouble(out, arg);
        } else if constexpr (is_same_v<T, vector<double>>) {
          printList(out, true, arg, printDouble);
        } else if constexpr (is_same_v<T, string>) {
          out << quoted(arg);
        } else if constexpr (is_same_v<T, c10::ScalarType>) {
          out << kScalarTypePrefix << arg;
        } else if constexpr (is_same_v<T, c10::MemoryFormat>) {
          out << kMemoryFormatPrefix << arg;
        } else if constexpr (is_same_v<T, c10::Layout>) {
          out << kLayoutPrefix << arg;
        } else if constexpr (is_same_v<T, c10::Device>) {
          out << kDevicePrefix << '{' << arg << '}';
        } else if constexpr (is_same_v<T, vector<string>>) {
          out << fmt::format("[{}]", fmt::join(arg, ","));
        } else if constexpr (is_same_v<T, vector<vector<int64_t>>>) {
          out << '[';
          for (const auto& [idx, inner_list] : c10::enumerate(arg)) {
            if (idx > 0) {
              out << ", ";
            }
            out << fmt::format("{}", fmt::streamed(inner_list));
          }
          out << ']';
        } else if constexpr (is_same_v<T, unique_ptr<Graph>>) {
          out << fmt::format("<subgraph>");
          VLOG(0) << "Subgraph pretty print is not implemented";
        } else {
          static_assert(always_false_v<T>, "non-exhaustive visitor!");
        }
      },
      constant);
  return out;
}

void printValue(std::ostream& out, const Value* v) {
  if (!v) {
    out << "<Constant>";
    return;
  }
  out << *v;
}

void printNamedArgument(std::ostream& out, const NamedArgument& nv) {
  out << nv.name << '=' << *nv.value;
}

void printAttribute(std::ostream& out, const Attribute& nv) {
  out << nv.name << '=' << nv.value;
}
} // namespace

std::ostream& operator<<(std::ostream& out, const Value& v) {
  out << '%' << v.name();
  // If a list, distinguish it by adding a []
  // Looks like %my_list[]
  if (v.type() == Type::Kind::TensorList) {
    out << "[]";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Node& node) {
  // special casing for inputs and outputs
  if (node.target() == "prim.Input") {
    out << "graph(";
    printList(out, false, node.outputs(), printValue);
    out << "):";
    return out;
  }
  if (node.target() == "prim.Output") {
    out << "return(";
    printList(out, false, node.inputs(), [](std::ostream& out, const auto& nv) {
      out << *nv.value;
    });
    out << ')';
    return out;
  }

  printList(out, false, node.outputs_, printValue);

  out << " = ";
  out << node.target_ << '(';
  printList(out, false, node.inputs_, printNamedArgument);
  if (!node.inputs_.empty() && !node.attributes_.empty()) {
    // Emit a connective ',' between inputs and attributes.
    out << ", ";
  }

  printList(out, false, node.attributes_, printAttribute);
  out << ')';
  return out;
}

std::ostream& operator<<(std::ostream& out, const Graph& graph) {
  for (const auto& node : graph.nodes_) {
    out << node << '\n';
  }
  return out;
}

c10::Device convertDevice(std::string_view symbol) {
  // Symbol looks like `Device{cuda:1}`
  const auto typeStart = symbol.find('{') + 1;
  TORCH_CHECK(typeStart < symbol.size());

  const auto typeEnd = symbol.find(':');
  TORCH_CHECK(typeEnd != std::string_view::npos);

  const auto type = symbol.substr(typeStart, typeEnd - typeStart);
  const auto indexStart = typeEnd + 1;
  TORCH_CHECK(indexStart < symbol.size());

  const auto indexEnd = symbol.find('}');
  TORCH_CHECK(indexEnd != std::string_view::npos);

  const auto index = symbol.substr(indexStart, indexEnd - indexStart);

  c10::Device device((std::string(type)));
  auto indexValue = c10::tryToNumber<int64_t>(std::string{index});
  TORCH_CHECK(indexValue.has_value(), "Invalid device index format");
  int64_t deviceIndex = indexValue.value();
  TORCH_CHECK(
      deviceIndex >= std::numeric_limits<c10::DeviceIndex>::min() &&
          deviceIndex <= std::numeric_limits<c10::DeviceIndex>::max(),
      "Device index out of range for int8_t");
  device.set_index(static_cast<c10::DeviceIndex>(deviceIndex));
  return device;
}

Constant convertAtomicConstant(std::string_view symbol) {
  if (c10::starts_with(symbol, "\"")) {
    // chop off the outer quotes and return the string
    TORCH_CHECK(symbol.size() >= 2);
    symbol.remove_prefix(1);
    symbol.remove_suffix(1);
    return std::string(symbol);
  } else if (symbol == "None") {
    return None();
  } else if (symbol == "true") {
    return true;
  } else if (symbol == "false") {
    return false;
  } else if (c10::starts_with(symbol, kMemoryFormatPrefix)) {
    torch::_export::MemoryFormat value = torch::_export::MemoryFormat::Unknown;
    symbol.remove_prefix(kMemoryFormatPrefix.length());
    torch::_export::parseEnum(symbol, value);
    return convertJsonMemoryFormat(value);
  } else if (c10::starts_with(symbol, kLayoutPrefix)) {
    torch::_export::Layout value = torch::_export::Layout::Unknown;
    symbol.remove_prefix(kLayoutPrefix.length());
    torch::_export::parseEnum(symbol, value);
    return convertJsonLayout(value);
  } else if (c10::starts_with(symbol, kDevicePrefix)) {
    return convertDevice(symbol);
  } else if (c10::starts_with(symbol, kScalarTypePrefix)) {
    torch::_export::ScalarType value = torch::_export::ScalarType::UNKNOWN;
    symbol.remove_prefix(kScalarTypePrefix.length());
    torch::_export::parseEnum(symbol, value);
    return convertJsonScalarType(value);
  }

  // match number
  // We need to disambiguate between int and float constants
  const auto maybeInt = c10::tryToNumber<int64_t>(std::string{symbol});

  // Libraries may happily convert "5.0" to an int 5, but we want that to
  // become a float. So add an extra check for whether a '.' is in the string
  // to guard against that.
  bool hasDecimalSeparator = symbol.find('.') != std::string_view::npos;
  if (maybeInt.has_value() && !hasDecimalSeparator) {
    return maybeInt.value();
  }

  const auto maybeDouble = c10::tryToNumber<double>(std::string{symbol});
  if (maybeDouble.has_value()) {
    return maybeDouble.value();
  }

  TORCH_CHECK(false, "unhandled symbol: ", symbol);
}

Constant convertListConstant(std::string_view source) {
  std::vector<Constant> values;
  size_t curPos = 0;
  Constant type = None();

  // This basically the same as parseValueList, it's probably better to refactor
  curPos = expectImpl(source, '[', curPos);
  while (true) {
    curPos = consumeWhitespaceImpl(source, curPos);

    size_t start = curPos;
    while (source.at(curPos) != ',' && source.at(curPos) != ']') {
      curPos++;
    }
    auto symbol = source.substr(start, curPos - start);
    auto val = convertAtomicConstant(symbol);
    if (std::holds_alternative<None>(type)) {
      // First time around; initialize our type sentinel with the first value.
      // We will use this on subsequent iterations to check that all types are
      // the same.
      if (auto intPtr = std::get_if<int64_t>(&val)) {
        type = *intPtr;
      } else if (auto doublePtr = std::get_if<double>(&val)) {
        type = *doublePtr;
      } else if (auto boolPtr = std::get_if<bool>(&val)) {
        type = *boolPtr;
      } else {
        TORCH_CHECK(false, "constant lists only support int, float, bool");
      }
    } else {
      TORCH_CHECK(
          type.index() == val.index(), "lists must have all the same type");
    }
    values.push_back(std::move(val));
    if (source.at(curPos) == ']') {
      break;
    }
    curPos = expectImpl(source, ',', curPos);
  }
  expectImpl(source, ']', curPos);

  // Some annoying unwrapping
  //   std::vector<Constant<T>> -->
  //   Constant<std::vector<T>>
  // Do it the dumb way.
  if (std::holds_alternative<int64_t>(type)) {
    std::vector<int64_t> inner;
    inner.reserve(values.size());
    for (const auto& el : values) {
      inner.push_back(std::get<int64_t>(el));
    }
    return inner;
  } else if (std::holds_alternative<double>(type)) {
    std::vector<double> inner;
    inner.reserve(values.size());
    for (const auto& el : values) {
      inner.push_back(std::get<double>(el));
    }
    return inner;
  } else if (std::holds_alternative<bool>(type)) {
    std::vector<bool> inner;
    inner.reserve(values.size());
    for (const auto& el : values) {
      inner.push_back(std::get<bool>(el));
    }
    return inner;
  }
  TORCH_CHECK(false, "constant lists only support int, float, bool");
}

namespace {

/**
 * Deserialization for graphs: parse the output produced by operator<<(Graph).
 * This parser really only expects the exact output generated by well-formed
 * Graph objects, so it is not very permissive and does not give good error
 * messages.
 */
class Parser {
 public:
  explicit Parser(std::string_view source)
      : source_(source), graph_(Graph::createGraph()) {}
  std::unique_ptr<Graph> parse();

 private:
  template <typename T>
  std::vector<T> parseList(
      char open,
      char close,
      const std::function<T()>& parseFn);

  std::string_view parseUntil(
      const std::function<bool()>& fn,
      bool includeEnd = false);

  void expect(std::string_view expected);
  void expect(char expected);
  bool nextEquals(std::string_view expected) const;
  bool nextIf(std::string_view expected);
  bool nextIf(char expected);
  void consumeWhitespace();
  bool validIdent(char n);
  char cur();

  void parseReturn();
  void parseNode();
  std::pair<std::string_view, Type> parseOutput();
  void parseGraphInputs();
  std::string_view parseString();
  std::variant<Value*, Constant> parseArgument();
  std::variant<NamedArgument, Attribute> parseNamedArgument();
  Value* parseSymbolicArgument();
  // Symbols look like %v109, with the same valid ident rules as Python
  // This returns the symbol *without* the % at the front.
  std::string_view parseAtomicSymbol();

  size_t curPos_ = 0;
  std::string_view source_;
  std::unique_ptr<Graph> graph_;
  torch::_export::GraphSignature signature_;
};

std::unique_ptr<Graph> Parser::parse() {
  parseGraphInputs();
  while (true) {
    consumeWhitespace();
    if (nextEquals("return")) {
      parseReturn();
      break;
    }
    parseNode();
  }
  // For graph textual format, it should be safe to assume all
  // inputs/outputs are from users.
  graph_->setSignature(GraphSignature{signature_});
  graph_->finalize();
  graph_->lint();
  // TODO: Might have some source left over, should check it if so.
  return std::move(graph_);
}

bool Parser::nextIf(std::string_view expected) {
  if (nextEquals(expected)) {
    curPos_ += expected.size();
    return true;
  }
  return false;
}

bool Parser::nextIf(char expected) {
  if (cur() == expected) {
    curPos_++;
    return true;
  }
  return false;
}

void Parser::parseGraphInputs() {
  TORCH_CHECK(curPos_ == 0);
  expect("graph");
  const auto inputs = parseList<std::string_view>(
      '(', ')', [&]() { return parseAtomicSymbol(); });
  std::vector<torch::_export::InputSpec> inputSpecs;
  inputSpecs.reserve(inputs.size());
  for (const auto& input : inputs) {
    graph_->addInput(input, Type::Kind::Tensor);

    torch::_export::TensorArgument inputTensorArg;
    inputTensorArg.set_name(std::string{input});
    torch::_export::Argument inputArg;
    inputArg.set_as_tensor(std::move(inputTensorArg));
    torch::_export::UserInputSpec userInput;
    userInput.set_arg(std::move(inputArg));
    torch::_export::InputSpec inputSpec;
    inputSpec.set_user_input(std::move(userInput));
    inputSpecs.push_back(std::move(inputSpec));
  }
  signature_.set_input_specs(std::move(inputSpecs));
  // TODO populate graphinputs
  expect(":");
}

template <typename T>
std::vector<T> Parser::parseList(
    char open,
    char close,
    const std::function<T()>& parseFn) {
  std::vector<T> ret;
  expect(open);

  // Handle empty list
  if (nextIf(close)) {
    return ret;
  }
  while (true) {
    ret.push_back(parseFn());
    if (cur() == close) {
      break;
    }
    expect(',');
  }
  expect(close);
  return ret;
}

// Parse until `fn` returns true, returning the segment of the source that was
// consumed. If `includeEnd` is true, the returned segment will also include
// final character, which caused `fn` to return true.
std::string_view Parser::parseUntil(
    const std::function<bool()>& fn,
    bool includeEnd) {
  size_t start = curPos_;
  while (!fn()) {
    curPos_++;
  }
  if (includeEnd) {
    curPos_++;
  }
  return source_.substr(start, curPos_ - start);
}

// Parse a string, including the outer quotes
std::string_view Parser::parseString() {
  size_t start = curPos_;
  expect('"');
  while (cur() != '"') {
    // Handle escaped characters by skipping the next char when we see a
    // backslash
    if (cur() == '\\') {
      curPos_++;
    }
    curPos_++;
  }

  // Consume final quote
  curPos_++;
  auto ret = source_.substr(start, curPos_ - start);
  return ret;
}

bool Parser::validIdent(char n) {
  return std::isalpha(n) || n == '_' || std::isdigit(n);
}

// Symbols look like %v109, with the same valid ident rules as Python
// This returns the symbol *without* the % at the front.
std::string_view Parser::parseAtomicSymbol() {
  expect("%");
  return parseUntil([&]() { return !validIdent(cur()); });
}

char Parser::cur() {
  return source_.at(curPos_);
}

void Parser::consumeWhitespace() {
  while (isBlank(cur())) {
    curPos_++;
  }
}

void Parser::expect(std::string_view expected) {
  curPos_ = expectImpl(source_, expected, curPos_);
}

void Parser::expect(char expected) {
  curPos_ = expectImpl(source_, expected, curPos_);
}

bool Parser::nextEquals(std::string_view expected) const {
  const auto actual = source_.substr(curPos_, expected.size());
  return expected == actual;
}

// %a, %b = aten.foo.default(input=%foo, foo=[7616], blah=%lol)
void Parser::parseNode() {
  std::vector<std::pair<std::string_view, Type>> outputs;

  outputs.push_back(parseOutput());
  while (nextIf(",")) {
    outputs.push_back(parseOutput());
  }
  expect("=");
  consumeWhitespace();

  // parse target name
  const auto target = parseUntil([&]() { return cur() == '('; });

  Node* node = graph_->insertNode(std::string(target));
  for (auto& [name, var] : outputs) {
    node->addOutput(name, var);
  }

  auto arguments = parseList<std::variant<NamedArgument, Attribute>>(
      '(', ')', [&]() { return parseNamedArgument(); });

  // Split the arguments into symbolic inputs and constant attributes
  for (auto& arg : arguments) {
    if (std::holds_alternative<NamedArgument>(arg)) {
      node->addInput(std::get<NamedArgument>(arg));
    } else {
      node->addAttribute(std::get<Attribute>(std::move(arg)));
    }
  }
}

void Parser::parseReturn() {
  expect("return");
  const auto returns =
      parseList<Value*>('(', ')', [&]() { return parseSymbolicArgument(); });
  std::vector<torch::_export::OutputSpec> outputSpecs;
  outputSpecs.reserve(returns.size());
  for (const auto ret : returns) {
    graph_->addOutput(ret);

    torch::_export::TensorArgument retTensorArg;
    retTensorArg.set_name(std::string{ret->name()});
    torch::_export::Argument retArg;
    retArg.set_as_tensor(std::move(retTensorArg));
    torch::_export::UserOutputSpec userOutput;
    userOutput.set_arg(std::move(retArg));
    torch::_export::OutputSpec outputSpec;
    outputSpec.set_user_output(std::move(userOutput));
    outputSpecs.push_back(std::move(outputSpec));
  }
  signature_.set_output_specs(std::move(outputSpecs));
}

std::variant<NamedArgument, Attribute> Parser::parseNamedArgument() {
  consumeWhitespace();
  // Parse name
  const auto symbol = parseUntil([&]() { return cur() == '='; });
  expect('=');

  // Parse value
  auto value = parseArgument();
  if (std::holds_alternative<Value*>(value)) {
    return NamedArgument{std::string(symbol), std::get<Value*>(value)};
  } else {
    return Attribute{std::string(symbol), std::get<Constant>(std::move(value))};
  }
}

std::pair<std::string_view, Type> Parser::parseOutput() {
  consumeWhitespace();
  TORCH_CHECK(cur() == '%', fmt::format("expected % but got {}", cur()));

  auto symbol = parseAtomicSymbol();
  if (nextIf('[')) {
    expect(']');
    return {symbol, Type::Kind::TensorList};
  } else {
    return {symbol, Type::Kind::Tensor};
  }
}

Value* Parser::parseSymbolicArgument() {
  consumeWhitespace();
  TORCH_CHECK(cur() == '%', fmt::format("expected % but got {}", cur()));

  auto symbol = parseAtomicSymbol();
  std::vector<Value*> listElements;
  if (cur() == '[') {
    listElements = parseList<Value*>(
        '[', ']', [&]() { return graph_->getValue(parseAtomicSymbol()); });
  }
  return graph_->getValue(symbol);
}

std::variant<Value*, Constant> Parser::parseArgument() {
  consumeWhitespace();

  // match symbol
  if (cur() == '%') {
    return parseSymbolicArgument();
  }

  // match list
  if (cur() == '[') {
    const auto symbol =
        parseUntil([&]() { return cur() == ']'; }, /*includeEnd=*/true);
    return convertListConstant(symbol);
  }

  // match string
  if (cur() == '"') {
    return convertAtomicConstant(parseString());
  }

  // otherwise parse this as a value
  const auto symbol =
      parseUntil([&]() { return cur() == ',' || cur() == ')'; });
  return convertAtomicConstant(symbol);
}

} // namespace

std::unique_ptr<Graph> stringToGraph(std::string_view source) {
  return Parser(source).parse();
}

std::string graphToString(const Graph& g, bool include_signature) {
  std::stringstream ss;
  ss << g;

  if (include_signature) {
    ss << "\nGraphSignature\n";
    ss << g.signature();
  }

  return ss.str();
}

} // namespace torch::nativert
