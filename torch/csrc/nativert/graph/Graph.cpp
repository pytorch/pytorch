#include "torch/csrc/nativert/graph/Graph.h"

#include <queue>

#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include "torch/csrc/nativert/common/Conv.h"
#include "torch/csrc/nativert/common/Enumerate.h"
#include "torch/csrc/nativert/common/String.h"
#include "torch/csrc/nativert/graph/TensorMeta.h"

namespace torch::nativert {

namespace {
bool isBlank(char n) {
  return isspace(n) || n == '\n';
}

size_t consumeWhitespaceImpl(std::string_view source, size_t curPos) {
  while (isBlank(source.at(curPos))) {
    curPos++;
  }
  return curPos;
}

size_t
expectImpl(std::string_view source, std::string_view expected, size_t curPos) {
  curPos = consumeWhitespaceImpl(source, curPos);
  const auto actual = source.substr(curPos, expected.size());
  TORCH_CHECK_EQ(expected, actual);
  curPos += expected.size();
  return curPos;
}

size_t expectImpl(std::string_view source, char expected, size_t curPos) {
  curPos = consumeWhitespaceImpl(source, curPos);
  while (isBlank(source.at(curPos))) {
    curPos++;
  }
  TORCH_CHECK_EQ(expected, source[curPos]);
  curPos++;
  return curPos;
}
} // namespace

bool operator==(const Type& left, const Type& right) {
  return left.kind() == right.kind();
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
Value*
Graph::addValue(const std::optional<std::string>& name, Type type, Node* node) {
  const auto valueName = name.value_or(getUniqueValueName());
  ValueId valueId = getNextValueId();
  const auto [it, success] = values_.insert(
      {valueName, std::make_unique<Value>(valueId, valueName, type, node)});
  CHECK(success) << fmt::format(
      "Tried to create Value with name: '{}', but it already existed",
      valueName);
  return it->second.get();
}

Value* Graph::addInput(std::string_view name, Type type) {
  return inputNode_->addOutput(name, type);
}

void Graph::addInput(void) {
  inputNode_->addOutput();
}

Value* Graph::addOutput(Value* v) {
  outputNode_->addInput({std::string(v->name()), v});
  return v;
}

void Graph::addConstantOutput(Constant& c) {
  constantOutputs_.push_back(std::move(c));
}

// Create a node without inserting it into the execution graph.
Node* Graph::createNode(
    std::string target,
    std::vector<NamedArgument> inputs,
    std::unordered_map<std::string, std::string> metadata) {
  auto node = std::make_unique<Node>(
      this, std::move(target), std::move(inputs), std::move(metadata));
  auto ret = node.get();
  nodesOwner_.push_back(std::move(node));
  return ret;
}

Node* Graph::insertBefore(Node* toInsert, Node* insertionPoint) {
  CHECK(insertionPoint != inputNode_) << "can't insert before prim.Input";
  CHECK(!toInsert->is_linked())
      << "expected node to be unlinked: " << *toInsert;
  CHECK(insertionPoint->is_linked())
      << "expected node to be linked: " << *insertionPoint;

  auto it = nodes_.insert(nodes_.iterator_to(*insertionPoint), *toInsert);
  return &*it;
}

Node* Graph::insert(Node* toInsert) {
  CHECK(!toInsert->is_linked())
      << "expected node to be unlinked: " << *toInsert;
  nodes_.insert(insertBefore_, *toInsert);
  return toInsert;
}

Node* Graph::insertAfter(Node* toInsert, Node* insertionPoint) {
  CHECK(insertionPoint != outputNode_) << "can't insert after prim.Output";
  CHECK(!toInsert->is_linked())
      << "expected node to be unlinked: " << *toInsert;
  CHECK(insertionPoint->is_linked())
      << "expected node to be linked: " << *insertionPoint;

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
  switch (ty.kind_) {
    case Type::None:
      out << "None";
      break;
    case Type::Tensor:
      out << "Tensor";
      break;
    case Type::TensorList:
      out << "TensorList";
      break;
    case Type::OptionalTensorList:
      out << "OptionalTensorList";
      break;
    case Type::SymInt:
      out << "SymInt";
      break;
    case Type::SymFloat:
      out << "SymFloat";
      break;
    case Type::SymIntList:
      out << "SymIntList";
      break;
    case Type::CustomObj:
      out << "CustomObj " << ty.classFqn_;
      break;
    default:
      CHECK(false) << "Unhandled type: " << ty.kind_;
  }
  return out;
}

const NamedArgument* Node::tryGetInput(std::string_view name) const {
  // Just do a scan over the inputs. We expect there to always be a very small
  // number of elements, so it shouldn't be slow. This allows us to avoid a
  // second datastructure for lookups.
  // Drop a debug check here, just to make sure :)
  TORCH_CHECK_LT(inputs_.size(), 1000);
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
  TORCH_CHECK_LT(attributes_.size(), 1000);
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

Node* Graph::createListPack(std::vector<Value*> inputs, Type inputType) {
  std::vector<NamedArgument> nodeInputs;
  for (auto [i, input] : enumerate(inputs)) {
    nodeInputs.push_back({fmt::format("l{}", i), input});
  }
  // Create a new named value for this
  auto name = getUniqueValueName();
  auto node = createNode("prim.ListPack", std::move(nodeInputs));

  // Make sure all inputs are the same type
  for (auto& input : inputs) {
    CHECK(input->type() == inputType);
  }

  if (inputType == Type::Tensor) {
    node->addOutput(name, Type::TensorList);
  } else if (inputType == Type::SymInt) {
    node->addOutput(name, Type::SymIntList);
  }

  return node;
}

Node* Graph::createOptionalListPack(std::vector<Value*> inputs) {
  std::vector<NamedArgument> nodeInputs;
  for (auto [i, input] : enumerate(inputs)) {
    nodeInputs.push_back({fmt::format("l{}", i), input});
  }
  // Create a new named value for this
  auto name = getUniqueValueName();
  auto node = createNode("prim.ListPack", std::move(nodeInputs));
  // Make sure all inputs are either None or Tensor
  for (auto& input : inputs) {
    CHECK(input->type() == Type::None || input->type() == Type::Tensor);
  }
  node->addOutput(name, Type::OptionalTensorList);

  return node;
}

Value* Graph::createConstantSymIntValue(int value) {
  auto valueName = getUniqueValueName();
  ValueId valueId = getNextValueId();
  const auto [it, success] = values_.insert(
      {valueName,
       std::make_unique<Value>(valueId, valueName, Type::SymInt, nullptr)});
  CHECK(success) << fmt::format(
      "Tried to create constant SymInt Value with name: '{}', but it already existed",
      valueName);
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

void Graph::cleanupDeadNodes() {
  std::unordered_set<const Node*> visited;
  std::queue<const Node*> visitQueue;

  // Mark reachable nodes from output
  visitQueue.push(outputNode_);
  visited.insert(outputNode_);

  while (!visitQueue.empty()) {
    const Node* current = visitQueue.front();
    visitQueue.pop();

    for (auto& namedArg : current->inputs()) {
      Value* val = namedArg.value;
      Node* producer = val->producer();

      if (!producer) {
        continue;
      }
      if (!visited.count(producer)) {
        visited.insert(producer);
        visitQueue.push(producer);
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

  // Remove nodes in reverse order to handle input/output dependencies
  for (auto it = toRemove.rbegin(); it != toRemove.rend(); ++it) {
    removeNode(*it);
  }

  renumberValues();
  lint();
}

void Graph::lint() const {
  // Check that every value has a producer marked.
  for (const auto& [name, value] : values_) {
    // Some constant symint and None don't have producer nodes
    if (value->type().kind() != Type::SymInt &&
        value->type().kind() != Type::None) {
      CHECK(value->isFolded() || value->producer() != nullptr);
    }
  }
  for (const auto& node : nodes()) {
    TORCH_CHECK_EQ(node.owningGraph(), this);
  }
  // Check that every list type is either produced by a prim.ListPack or
  // immediately consumed by a prim.ListUnpack. We make use of this invariant
  // to retrieve list elements in `getListElements`.
  for (const auto& [_, value] : values_) {
    if (value->type().kind() != Type::TensorList) {
      continue;
    }
    const bool producedByListPack =
        value->producer(/* resolve_folded = */ true)->target() ==
        "prim.ListPack";
    const bool consumedByListUnpack = value->users().size() == 1 &&
        value->users()[0]->target() == "prim.ListUnpack";
    CHECK(producedByListPack || consumedByListUnpack);
  }

  auto getNames = [](const auto& values) {
    std::set<std::string> names;
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
        userOutputs_.emplace_back(std::move(constantOutputs_[constantIndex]));
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
    CHECK(replaced);
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
  // TODO: consolidate device info in weight loading here as well.
  for (auto& node : nodes_) {
    node.applyDevicePlacement(placement);
  }
}

Node* Graph::nodeAfter(Node* n) {
  return const_cast<Node*>(std::as_const(*this).nodeAfter(n));
}

const Node* Graph::nodeAfter(const Node* n) const {
  TORCH_CHECK_EQ(n->owningGraph(), this);
  if (n == outputNode_) {
    return nullptr;
  }
  auto it = nodes_.iterator_to(*n);
  return &*(++it);
}

Node* Graph::nodeBefore(Node* n) {
  return const_cast<Node*>(std::as_const(*this).nodeBefore(n));
}

const Node* Graph::nodeBefore(const Node* n) const {
  TORCH_CHECK_EQ(n->owningGraph(), this);
  if (n == inputNode_) {
    return nullptr;
  }
  auto it = nodes_.iterator_to(*n);
  return &*(--it);
}

void Graph::removeNode(Node* n) {
  TORCH_CHECK_EQ(n->owningGraph(), this)
      << "Node does not belong to this graph!";

  for (auto* outputVal : n->outputs()) {
    TORCH_CHECK_EQ(outputVal->users().size(), 0)
        << "Trying to erase a node that still has users: " << outputVal->name();
    outputVal->eraseAllUsers();
    removeValue(outputVal);
  }

  for (const auto& input : n->inputs()) {
    input.value->eraseUser(n);
  }

  CHECK(n->is_linked()) << "Node is not linked to the graph!";
  n->unlink();

  auto it = std::find_if(
      nodesOwner_.begin(),
      nodesOwner_.end(),
      [n](const std::unique_ptr<Node>& ptr) { return ptr.get() == n; });

  CHECK(it != nodesOwner_.end()) << "Node not found in nodesOwner_!";
  nodesOwner_.erase(it);
}

void Graph::removeValue(Value* value) {
  // TODO: assuming not removing from constantSymIntValues_
  CHECK(value->users().empty()) << "Cannot erase a value with users.";
  auto it = values_.find(std::string(value->name()));
  CHECK(it != values_.end())
      << "Attempted to erase a value not in graph" << value->name();
  values_.erase(it);
}

std::vector<Value*> Graph::insertGraph(
    const Graph& subgraph,
    std::vector<Value*> inputs,
    std::unordered_map<const Value*, Value*>& valueMap) {
  TORCH_CHECK_EQ(subgraph.inputs().size(), inputs.size())
      << "Input size mismatch";
  for (auto i : c10::irange(subgraph.inputs().size())) {
    valueMap[subgraph.inputs()[i]] = inputs[i];
  }

  // Clone each node from subgraph
  for (const auto& n : subgraph.nodes()) {
    if (n.target() == "prim.Input" || n.target() == "prim.Output") {
      continue;
    }

    std::vector<NamedArgument> clonedInputs;
    for (auto& inp : n.inputs()) {
      auto it = valueMap.find(inp.value);
      CHECK(it != valueMap.end()) << "Missing input value in subgraph";
      clonedInputs.push_back({inp.name, it->second});
    }

    Node* newNode =
        insertNode(std::string(n.target()), clonedInputs, n.metadata());

    for (const auto& attr : n.attributes()) {
      Attribute newAttr;
      newAttr.name = attr.name;

      std::visit(
          [&](auto&& val) -> void {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, std::unique_ptr<Graph>>) {
              LOG(ERROR)
                  << "Graph attributes are not supported yet. Skipping attribute: "
                  << attr.name;
            } else {
              newAttr.value = val;
              newNode->addAttribute(std::move(newAttr));
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

  std::vector<Value*> outputValues;
  for (auto* outputValue : subgraph.outputs()) {
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

void Node::addOutput(void) {
  outputs_.push_back(nullptr);
}

Value* Node::addOutput(Type type) {
  TORCH_CHECK_EQ(type, Type::None);
  Value* v = owningGraph_->addValue(std::nullopt, type, this);
  outputs_.push_back(v);
  return v;
}

Value* Node::addOutput(std::string_view name, Type type) {
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
  if (producer()->target() == "prim.ListPack") {
    for (const auto& tv : producer()->inputs()) {
      ret.push_back(tv.value);
    }
  } else {
    TORCH_CHECK_EQ(users().size(), 1);
    const auto listUnpack = users()[0];
    TORCH_CHECK_EQ(listUnpack->target(), "prim.ListUnpack");
    for (const auto v : listUnpack->outputs()) {
      ret.push_back(v);
    }
  }
  return ret;
}

template <class>
[[maybe_unused]] inline constexpr bool AlwaysFalse = false;

c10::IValue constantToIValue(const Constant& constant) {
  return std::visit(
      [](auto&& arg) -> c10::IValue {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, None>) {
          return c10::IValue();
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return arg;
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          return arg;
        } else if constexpr (std::is_same_v<T, double>) {
          return arg;
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          return arg;
        } else if constexpr (std::is_same_v<T, std::string>) {
          return arg;
        } else if constexpr (std::is_same_v<T, bool>) {
          return arg;
        } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
          return arg;
        } else if constexpr (std::is_same_v<T, c10::ScalarType>) {
          return arg;
        } else if constexpr (std::is_same_v<T, c10::MemoryFormat>) {
          return arg;
        } else if constexpr (std::is_same_v<T, c10::Layout>) {
          return arg;
        } else if constexpr (std::is_same_v<T, c10::Device>) {
          return arg;
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          return arg;
        } else if constexpr (std::is_same_v<T, std::unique_ptr<Graph>>) {
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
  out << std::to_string(arg);
}

template <typename T, typename F>
std::ostream& printList(
    std::ostream& out,
    bool encloseInSquareBrackets,
    const T& list,
    F formatter) {
  if (encloseInSquareBrackets) {
    out << "[";
  }
  for (const auto& [idx, el] : enumerate(list)) {
    if (idx > 0) {
      out << ", ";
    }
    formatter(out, el);
  }
  if (encloseInSquareBrackets) {
    out << "]";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Constant& constant) {
  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, None>) {
          out << "None";
        } else if constexpr (std::is_same_v<T, int64_t>) {
          out << arg;
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          out << fmt::format("{}", arg);
        } else if constexpr (std::is_same_v<T, double>) {
          printDouble(out, arg);
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          printList(out, true, arg, printDouble);
        } else if constexpr (std::is_same_v<T, std::string>) {
          out << std::quoted(arg);
        } else if constexpr (std::is_same_v<T, c10::ScalarType>) {
          out << kScalarTypePrefix << arg;
        } else if constexpr (std::is_same_v<T, c10::MemoryFormat>) {
          out << kMemoryFormatPrefix << arg;
        } else if constexpr (std::is_same_v<T, c10::Layout>) {
          out << kLayoutPrefix << arg;
        } else if constexpr (std::is_same_v<T, c10::Device>) {
          out << kDevicePrefix << "{" << arg << "}";
        } else if constexpr (std::is_same_v<T, bool>) {
          out << arg;
        } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
          out << fmt::format("{}", fmt::streamed(arg));
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          out << fmt::format("[{}]", fmt::join(arg, ","));
        } else if constexpr (std::is_same_v<T, std::unique_ptr<Graph>>) {
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
  out << *v;
}

void printNamedArgument(std::ostream& out, const NamedArgument& nv) {
  out << nv.name << "=" << *nv.value;
}

void printAttribute(std::ostream& out, const Attribute& nv) {
  out << nv.name << "=" << nv.value;
}
} // namespace

std::ostream& operator<<(std::ostream& out, const Value& v) {
  out << "%" << v.name();
  // If a list, distinguish it by adding a []
  // Looks like %my_list[]
  if (v.type() == Type::TensorList) {
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
    out << ")";
    return out;
  }

  printList(out, false, node.outputs_, printValue);

  out << " = ";
  out << node.target_ << "(";
  printList(out, false, node.inputs_, printNamedArgument);
  if (!node.inputs_.empty() && !node.attributes_.empty()) {
    // Emit a connective ',' between inputs and attributes.
    out << ", ";
  }

  printList(out, false, node.attributes_, printAttribute);
  out << ")";
  return out;
}

std::ostream& operator<<(std::ostream& out, const Graph& graph) {
  for (const auto& node : graph.nodes_) {
    out << node << "\n";
  }
  return out;
}

c10::Device convertDevice(std::string_view symbol) {
  // Symbol looks like `Device{cuda:1}`
  const auto typeStart = symbol.find('{') + 1;
  TORCH_CHECK_LT(typeStart, symbol.size());

  const auto typeEnd = symbol.find(':');
  TORCH_CHECK_NE(typeEnd, std::string_view::npos);

  const auto type = symbol.substr(typeStart, typeEnd - typeStart);
  const auto indexStart = typeEnd + 1;
  TORCH_CHECK_LT(indexStart, symbol.size());

  const auto indexEnd = symbol.find('}');
  TORCH_CHECK_NE(indexEnd, std::string_view::npos);

  const auto index = symbol.substr(indexStart, indexEnd - indexStart);

  c10::Device device((std::string(type)));
  device.set_index(tryTo<int64_t>(index).value());
  return device;
}

Constant convertAtomicConstant(std::string_view symbol) {
  if (starts_with(symbol, "\"")) {
    // chop off the outer quotes and return the string
    TORCH_CHECK_GE(symbol.size(), 2);
    symbol.remove_prefix(1);
    symbol.remove_suffix(1);
    return std::string(symbol);
  } else if (symbol == "None") {
    return None();
  } else if (symbol == "true") {
    return true;
  } else if (symbol == "false") {
    return false;
  } else if (starts_with(symbol, kMemoryFormatPrefix)) {
    torch::_export::MemoryFormat value;
    symbol.remove_prefix(kMemoryFormatPrefix.length());
    torch::_export::parseEnum(symbol, value);
    return convertJsonMemoryFormat(value);
  } else if (starts_with(symbol, kLayoutPrefix)) {
    torch::_export::Layout value;
    symbol.remove_prefix(kLayoutPrefix.length());
    torch::_export::parseEnum(symbol, value);
    return convertJsonLayout(value);
  } else if (starts_with(symbol, kDevicePrefix)) {
    return convertDevice(symbol);
  } else if (starts_with(symbol, kScalarTypePrefix)) {
    torch::_export::ScalarType value;
    symbol.remove_prefix(kScalarTypePrefix.length());
    torch::_export::parseEnum(symbol, value);
    return convertJsonScalarType(value);
  }

  // match number
  // We need to disambiguate between int and float constants
  const auto maybeInt = tryTo<int64_t>(symbol);

  // Libraries may happily convert "5.0" to an int 5, but we want that to
  // become a float. So add an extra check for whether a '.' is in the string
  // to guard against that.
  bool hasDecimalSeparator = symbol.find('.') != std::string_view::npos;
  if (maybeInt.has_value() && !hasDecimalSeparator) {
    return maybeInt.value();
  }

  const auto maybeDouble = tryTo<double>(symbol);
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
      if (std::holds_alternative<int64_t>(val)) {
        type = std::get<int64_t>(val);
      } else if (std::holds_alternative<double>(val)) {
        type = std::get<double>(val);
      } else if (std::holds_alternative<bool>(val)) {
        type = std::get<bool>(val);
      } else {
        throw std::runtime_error(
            "constant lists only support int, float, bool");
      }
    } else {
      TORCH_CHECK_EQ(type.index(), val.index())
          << "lists must have all the same type";
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
    for (const auto& el : values) {
      inner.push_back(std::get<int64_t>(el));
    }
    return inner;
  } else if (std::holds_alternative<double>(type)) {
    std::vector<double> inner;
    for (const auto& el : values) {
      inner.push_back(std::get<double>(el));
    }
    return inner;
  } else if (std::holds_alternative<bool>(type)) {
    std::vector<bool> inner;
    for (const auto& el : values) {
      inner.push_back(std::get<bool>(el));
    }
    return inner;
  }
  TORCH_CHECK(false, "shouldn't reach here");
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
  std::vector<T>
  parseList(char open, char close, const std::function<T()>& parseFn);

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
  TORCH_CHECK_EQ(curPos_, 0);
  expect("graph");
  const auto inputs = parseList<std::string_view>(
      '(', ')', [&]() { return parseAtomicSymbol(); });
  std::vector<torch::_export::InputSpec> inputSpecs;
  for (const auto& input : inputs) {
    graph_->addInput(input, Type::Tensor);

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
std::vector<T>
Parser::parseList(char open, char close, const std::function<T()>& parseFn) {
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

// Parse a strng, including the outer quotes
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
  return isalpha(n) || n == '_' || isdigit(n);
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
    node->addOutput(name, std::move(var));
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
  TORCH_CHECK_EQ(cur(), '%');

  auto symbol = parseAtomicSymbol();
  if (nextIf('[')) {
    expect(']');
    return {symbol, Type::TensorList};
  } else {
    return {symbol, Type::Tensor};
  }
}

Value* Parser::parseSymbolicArgument() {
  consumeWhitespace();
  TORCH_CHECK_EQ(cur(), '%');

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
