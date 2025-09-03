#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <ATen/core/ivalue.h>
#include <c10/util/IntrusiveList.h>
#include <c10/util/Logging.h>

#include <torch/csrc/utils/generated_serialization_types.h>
#include <torch/nativert/executor/Placement.h>
#include <torch/nativert/graph/GraphSignature.h>
#include <torch/nativert/graph/TensorMeta.h>

namespace torch::nativert {

using NodeIndex = size_t;

class Value;

class Type {
 public:
  enum class Kind {
    None,
    Tensor,
    TensorList,
    OptionalTensorList,
    SymInt,
    SymIntList,
    SymBool,
    SymFloat,
    CustomObj,
  };

  // For simple kinds without classFqn
  /*implicit*/ Type(Kind kind) : kind_(kind) {}

  // For CustomObj kind with classFqn
  explicit Type(Kind kind, const std::string& classFqn)
      : kind_(CustomObjData{classFqn}) {
    TORCH_CHECK(kind == Kind::CustomObj);
    TORCH_CHECK(!classFqn.empty());
  }

  Kind kind() const {
    if (std::holds_alternative<CustomObjData>(kind_)) {
      return Kind::CustomObj;
    }
    return std::get<Kind>(kind_);
  }

  friend std::ostream& operator<<(std::ostream& out, const Type& ty);
  friend bool operator==(const Type& left, const Type& right);

  std::string classFqn() const {
    TORCH_CHECK(
        kind() == Kind::CustomObj, "Only CustomObj type can have classFqn");
    return std::get<CustomObjData>(kind_).classFqn;
  }

 private:
  struct CustomObjData {
    std::string classFqn;
  };
  std::variant<Kind, CustomObjData> kind_;
};

// These are all the constant types that are allowed as attributes on Nodes.
struct None {};
// None always equals itself
inline bool operator==(const None&, const None&) {
  return true;
}

class Graph;

/**
 * We distinguish between a symbolic value (Tensor, TensorList, SymInt, SymInts,
 * etc) and a constant value (int, bool, string, etc). Here Constant is the type
 * for all possible constant values. Along with a name, they are represented as
 * Attributes on a Node.
 */
using Constant = std::variant<
    None,
    int64_t,
    std::vector<int64_t>,
    double,
    std::vector<double>,
    std::string,
    c10::ScalarType,
    c10::MemoryFormat,
    c10::Layout,
    c10::Device,
    bool,
    std::vector<bool>,
    std::vector<std::string>,
    std::unique_ptr<Graph>>;

c10::IValue constantToIValue(const Constant& constant);

class Node;

/**
 * Represents a single symbolic value (tensor/symint/list of them). Values are
 * inputs and outputs of Nodes.
 */
using ValueId = int;
class Value {
 public:
  explicit Value(ValueId id, std::string name, Type t, Node* producer)
      : name_(std::move(name)), id_(id), type_(t), producer_(producer) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(name_ == this->name());
  }

  // Each Value should be uniquely created and managed by a Graph. It's not
  // allowed to copy/move Value instances.
  Value(Value&&) = delete;
  Value& operator=(Value&&) = delete;
  Value(const Value&) = delete;
  Value& operator=(Value&) = delete;

  Type type() const {
    return type_;
  }

  ValueId id() const {
    return id_;
  }

  std::string_view name() const {
    return name_;
  }

  const Node* producer(bool resolve_folded = false) const {
    return (!resolve_folded && isFolded()) ? nullptr : producer_;
  }

  Node* producer() {
    return producer_;
  }

  void addUser(Node* node);
  void eraseUser(Node* node);
  void eraseAllUsers() {
    users_.clear();
  }

  // Throws an exception if the value is not a TensorList
  std::vector<const Value*> getListElements() const;

  const auto& users() const {
    return users_;
  }

  auto& users() {
    return users_;
  }

  void setId(ValueId newId) {
    // This should only be used inside the renumberValues pass
    id_ = newId;
  }

  void setIsFolded() {
    isFolded_ = true;
  }

  bool isFolded() const {
    return isFolded_;
  }

 private:
  friend std::ostream& operator<<(std::ostream& out, const Value& v);
  std::string name_;
  bool isFolded_{false};
  ValueId id_;
  Type type_;
  Node* producer_;
  // All nodes which have this value as input.
  // Note that this is a vector to avoid nondeterminism in iteration, but
  // probably should be an unordered set given usage patterns. If this becomes a
  // perf problem we should revise.
  std::vector<Node*> users_;
};

struct NamedArgument {
  std::string name;
  Value* value;
};

struct Attribute {
  std::string name;
  Constant value;
};

/**
 * Node represents a single unit of execution, typically a PyTorch operator.
 * Using an intrusive list allows us to allocate all the memory at once for a
 * node. This also allows us to track nodes safely without passing around the
 * list object, as an intrusive list maintains a stronger invariant that
 * expiration will always cause unlinking.
 */
class Node : public c10::IntrusiveListHook {
 public:
  Node(
      Graph* owningGraph,
      std::string target,
      std::vector<NamedArgument> inputs,
      std::unordered_map<std::string, std::string> metadata);

  std::string_view target() const {
    return target_;
  }

  void setTarget(std::string_view target) {
    target_ = target;
  }

  const auto& inputs() const {
    return inputs_;
  }

  auto& inputs() {
    return inputs_;
  }

  // NOTE: this invalidates spans given out by inputs()
  Value* addInput(NamedArgument input);
  void addInputs(const std::vector<NamedArgument>& inputs);

  // NOTE: this invalidates spans given out by attributes()
  void addAttribute(Attribute attr);

  // NOTE: this is ONLY for graph's constant inputs and NOT the common case
  void addOutput();

  Value* addOutput(const Type& type);

  // NOTE: this invalidates spans given out by outputs()
  Value* addOutput(std::string_view name, const Type& type);

  size_t numInputs() const {
    return inputs_.size();
  }

  size_t numOutputs() const {
    return outputs_.size();
  }

  // Return the next node in the Graph's node ordering.
  // NOTE: Calling next on the last node (prim.Output) returns nullptr.
  Node* next();
  const Node* next() const;

  // Return the previous node in the Graph's node ordering.
  // NOTE: Calling prev on the first node (prim.Input) returns nullptr.
  Node* prev();
  const Node* prev() const;

  bool isBefore(const Node* n) const;

  std::vector<Node*> producers() const;
  std::vector<Node*> users() const;

  // Returns nullptr if `name` is not an input
  const NamedArgument* tryGetInput(std::string_view name) const;
  // Throws an exception if `name` is not an input
  const NamedArgument& getInput(std::string_view name) const;

  const auto& attributes() const {
    return attributes_;
  }

  // Returns nullptr if `name` is not an attribute
  const Attribute* tryGetAttribute(std::string_view name) const;
  // Throws an exception if `name` is not an attribute
  const Attribute& getAttribute(std::string_view name) const;

  const auto& outputs() const {
    return outputs_;
  }

  void applyDevicePlacement(const Placement& placement);

  std::optional<std::string_view> getMetadata(std::string_view key) const {
    return metadata_.find(std::string{key}) != metadata_.end()
        ? std::optional(std::string_view{metadata_.at(std::string{key})})
        : std::nullopt;
  }

  Graph* owningGraph() {
    return owningGraph_;
  }

  const Graph* owningGraph() const {
    return owningGraph_;
  }

  void destroy();

  const std::unordered_map<std::string, std::string>& metadata() const {
    return metadata_;
  }

  std::string toString() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  void updateInputName(std::string_view oldName, std::string_view newName) {
    for (auto& input : inputs_) {
      if (input.name == oldName) {
        input.name = newName;
        break;
      }
    }
  }

  void updateAttributeName(std::string_view oldName, std::string_view newName) {
    for (auto& attr : attributes_) {
      if (attr.name == oldName) {
        attr.name = newName;
        break;
      }
    }
  }

 private:
  friend std::ostream& operator<<(std::ostream& out, const Node& n);
  Graph* owningGraph_;

  // Target used to retrieve the actual thing to execute.
  // If an aten operator, we expect this to be fully qualified, including an
  // overload name, e.g. "aten.unsqueeze.default"
  std::string target_;
  // *Symbolic* inputs to this node. NOTE: this does not match the ATen operator
  // schema inputs directly. It only represents things that actually participate
  // in dataflow, like tensors/symints and lists thereof.
  //
  // The "name" of the NamedArgument refers to the name of the parameter.
  std::vector<NamedArgument> inputs_;
  // Constant inputs to the node. The "name" of the Attribute refers to the
  // name of the parameter.
  std::vector<Attribute> attributes_;
  std::vector<Value*> outputs_;

  // Extra bits of info added to the node. Contents that are guaranteed will be
  // eventually moved to a first-class field on the json struct of schema.
  std::unordered_map<std::string, std::string> metadata_;
};

/**
 * Graph represents a model's computation graph, which is designed to
 * facilitate transformation and analysis.
 *
 * Ownership semantics:
 *  - Graph owns Nodes and Values
 *  - Nodes own their constant attributes (which we treat as value types)
 *  - Nodes have non-owning pointers back to the graph.
 *
 * NOTE: this class is marked noncopyable/nonmovable and only can be
 * heap-allocated via `createGraph()`. This is to ensure stability of
 * back-pointers held by Nodes/Values.
 */
class Graph {
 public:
  static std::unique_ptr<Graph> createGraph() {
    return std::unique_ptr<Graph>(new Graph());
  }

  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) = delete;
  Graph& operator=(Graph&&) = delete;
  ~Graph() = default;

  // NOTE: this invalidates spans given out by inputs()
  Value* addInput(std::string_view name, const Type& type);

  // NOTE: this is ONLY for graph's constant inputs and NOT the common case
  void addInput();

  // NOTE: this invalidates spans given out by outputs()
  Value* addOutput(Value* v);

  void addConstantOutput(Constant c);

  // Create and insert a node at insertionPoint_
  Node* insertNode(
      std::string target,
      std::vector<NamedArgument> inputs = {},
      std::unordered_map<std::string, std::string> metadata = {});

  // Returns the inserted node.
  Node* insertBefore(Node* toInsert, Node* insertionPoint);
  // Returns the inserted node.
  Node* insertAfter(Node* toInsert, Node* insertionPoint);
  // Insert at the insertionPoint. Returns the inserted node.
  Node* insert(Node* toInsert);

  // Create a node without inserting it into the execution graph.
  // A raw pointer to the node is created when `createNode()` on the
  // owner Graph object is called. It is guranateed that to be valid
  // until the Graph object is destructed.
  Node* createNode(
      std::string target,
      std::vector<NamedArgument> inputs = {},
      std::unordered_map<std::string, std::string> metadata = {});

  Value* createConstantSymIntValue(int value);

  Node* createListPack(std::vector<Value*> inputs, const Type& inputType);

  Node* createOptionalListPack(std::vector<Value*> inputs);

  size_t numValues() const {
    return values_.size();
  }

  // throws on missing name
  Value* getValue(std::string_view name) const;
  // returns nullptr on missing name
  Value* tryGetValue(std::string_view name) const;

  const std::unordered_map<ValueId, int> getConstantSymIntValues() const {
    return constantSymIntValues_;
  }

  Value* addValue(
      const std::optional<std::string>& name,
      const Type& type,
      Node* producer);
  void removeValue(Value* value);

  void replaceAllUses(Value* old, Value* replacement);
  void replaceAllUsesAfterNode(Value* old, Value* replacement, Node* afterThis);
  void removeNode(Node* node);

  void applyDevicePlacement(const Placement& placement);

  std::string getUniqueValueName();

  ValueId getNextValueId() {
    return uniqueValueId_++;
  }

  // NOTE: this range can be invalidated by mutations to the graph.
  const auto& inputs() const {
    return inputNode_->outputs();
  }

  c10::ArrayRef<const Value*> userInputs() const {
    size_t offset = signature().inputsToWeights().size() +
        signature().inputsToCustomObjs().size();
    return {inputs().data() + offset, inputs().data() + inputs().size()};
  }

  c10::ArrayRef<const Value*> weightValues() const {
    return {
        inputs().data(),
        inputs().data() + signature().inputsToWeights().size()};
  }

  // Return a bidirectional range over `const Value*`
  // NOTE: this range can be invalidated by mutations to the graph.
  auto outputs() const {
    std::vector<const Value*> ret;
    ret.reserve(outputNode_->inputs().size());
    for (const auto& namedArg : outputNode_->inputs()) {
      ret.push_back(namedArg.value);
    }
    return ret;
  }

  // Return a bidirectional range over `Value*`
  // NOTE: this range can be invalidated by mutations to the graph.
  auto outputs() {
    std::vector<Value*> ret;
    ret.reserve(outputNode_->inputs().size());
    for (const auto& namedArg : outputNode_->inputs()) {
      ret.push_back(namedArg.value);
    }
    return ret;
  }

  const auto& userOutputs() const {
    return userOutputs_;
  }

  // Return a list over `const Node&`.
  // NOTE: this can be invalidated by mutations to the graph.
  const auto& nodes() const {
    return nodes_;
  }

  auto& nodes() {
    return nodes_;
  }

  // Return a forward range over `const Value*`.
  // NOTE: this range can be invalidated by mutations to the graph.
  auto values() const {
    std::vector<const Value*> ret;
    ret.reserve(values_.size());
    for (const auto& [_, value] : values_) {
      ret.push_back(value.get());
    }
    return ret;
  }

  Node* inputNode() {
    return inputNode_;
  }

  Node* outputNode() {
    return outputNode_;
  }

  const Node* outputNode() const {
    return outputNode_;
  }

  // Assert various graph invariants
  void lint() const;

  bool /* removed > 0? */ cleanupDeadNodes();

  void finalize();

  Node* insertionPoint() {
    // This should never happen, since the last-most insertion point is the
    // prim.Outputs node, not end().
    TORCH_CHECK(insertBefore_ != nodes_.end());
    auto& node = *insertBefore_;
    return &node;
  }

  void setInsertionPoint(Node* n) {
    TORCH_CHECK(n != inputNode_, "can't insert before prim.Input");
    insertBefore_ = nodes_.iterator_to(*n);
  }

  void setInsertionPointAfter(Node* n) {
    TORCH_CHECK(n != outputNode_, "can't insert after prim.Output");
    auto it = nodes_.iterator_to(*n);
    ++it;
    insertBefore_ = it;
  }

  // Return the next node in the Graph's node ordering.
  // NOTE: Calling on the last node (prim.Output) returns nullptr.
  Node* nodeAfter(Node* n);
  const Node* nodeAfter(const Node* n) const;

  // Return the previous node in the Graph's node ordering.
  // NOTE: Calling on the first node (prim.Input) returns nullptr.
  Node* nodeBefore(Node* n);
  const Node* nodeBefore(const Node* n) const;

  // Clone each node from subgraph (except prim.Input/prim.Output) into current
  // graph.
  // @param subgraph: the subgraph to be cloned
  // @param inputs: values from the target graph that will serve as the
  // subgraph's inputs
  // @param valueMap: a map from the cloned subgraph's values to the target
  // graph's values
  std::vector<Value*> insertGraph(
      const Graph& subgraph,
      std::vector<Value*> inputs,
      std::unordered_map<const Value*, Value*>& valueMap);

  const GraphSignature& signature() const {
    return signature_;
  }

  void setSignature(GraphSignature signature) {
    signature_ = std::move(signature);
  }

  void setWeightsMeta(
      const std::unordered_map<std::string, torch::_export::TensorMeta>&
          tensorsMeta) {
    TORCH_CHECK(!placementApplied_);

    for (auto [name, tensorMeta] : tensorsMeta) {
      weightsMeta_.emplace(name, TensorMeta{tensorMeta});
    }
  }

  const std::unordered_map<std::string, TensorMeta>& weightsMeta() const {
    return weightsMeta_;
  }

  std::vector<TensorMeta> userInputsMeta() const {
    std::vector<TensorMeta> userInputsMeta;
    userInputsMeta.reserve(signature_.userInputs().size());
    for (auto inputName : signature_.userInputs()) {
      userInputsMeta.push_back(tensorValuesMeta_.at(inputName));
    }
    return userInputsMeta;
  }

  void setTensorValuesMeta(
      const std::unordered_map<std::string, torch::_export::TensorMeta>&
          tensorsMeta) {
    TORCH_CHECK(!placementApplied_);

    for (auto [name, tensorMeta] : tensorsMeta) {
      tensorValuesMeta_.emplace(name, TensorMeta{tensorMeta});
    }
  }

  const std::unordered_map<std::string, TensorMeta>& tensorValuesMeta() const {
    return tensorValuesMeta_;
  }

  std::string toString() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  /* Reassigns IDs to every Value in this Graph so that they are contiguous from
   * 0..(numValues()-1). Should be used after values are removed
   */
  void renumberValues();

 private:
  Graph();
  friend std::ostream& operator<<(std::ostream& out, const Graph& g);
  GraphSignature signature_;

  bool placementApplied_ = false;

  // keys are parameters, buffers, tensor_constants' names
  std::unordered_map<std::string, TensorMeta> weightsMeta_;

  // keys are tensor_values' names
  std::unordered_map<std::string, TensorMeta> tensorValuesMeta_;

  // Node lifetime is managed by nodesOwner_, but the actual ordering is
  // maintained intrusively using nodes_.
  // This is to facilitate quick insertion before/after a given Node*.
  std::vector<std::unique_ptr<Node>> nodesOwner_;
  c10::IntrusiveList<Node> nodes_;
  // The current insertion point. New nodes are inserted before this node.
  // Defaults to prim.Output.
  c10::IntrusiveList<Node>::iterator insertBefore_;

  // Graphs always start with an input and output node.
  // "prim.input() -> Value[]" take no input, and produces some outputs. AKA
  // "source“ of a graph.
  Node* inputNode_; // target: prim.Input
  // "prim.output(Value[]) -> None", take some inputs, but produce no output.
  // AKA "sink" of a graph.
  Node* outputNode_; // target: prim.Output

  std::unordered_map<std::string, std::unique_ptr<Value>> values_;
  // constantSymIntValues_ is a subset of values_
  std::unordered_map<ValueId, int> constantSymIntValues_;
  // Output values of the graph, which is a subset of values_.
  std::vector<std::variant<Value*, Constant>> userOutputs_;
  // Output constant values of the graph
  std::vector<Constant> constantOutputs_;

  size_t uniqueValueName_ = 0;

  ValueId uniqueValueId_ = 0;
};

/**
 * Scoped utility class for setting temporary insertion points.
 *
 * Use like:
 *   {
 *       InsertingAfter guard(node)
 *       graph.insertNode(...)  // this will be inserted after `node`.
 *   }
 */
class InsertingAfter {
 public:
  explicit InsertingAfter(Node* n)
      : insertAfter_(n), prev_(n->owningGraph()->insertionPoint()) {
    insertAfter_->owningGraph()->setInsertionPointAfter(insertAfter_);
  }
  ~InsertingAfter() {
    insertAfter_->owningGraph()->setInsertionPoint(prev_);
  }

 private:
  Node* insertAfter_;
  Node* prev_;
};

inline constexpr std::string_view kMemoryFormatPrefix = "MemoryFormat::";
inline constexpr std::string_view kLayoutPrefix = "Layout::";
inline constexpr std::string_view kDevicePrefix = "Device";
inline constexpr std::string_view kScalarTypePrefix = "ScalarType::";

/**
 * Debug format serialization. The format here is intended to be human readable
 * and easy to work with, and is intended for debugging and testing only.
 * If you want stable serialization, use the json conversion utils.
 *
 * NOTE: node metadata currently not serialized
 */
std::string graphToString(const Graph& g, bool include_signature = false);
std::unique_ptr<Graph> stringToGraph(std::string_view source);

// Standalone functions to parse common constructs
// Parse something that looks like `Device{cuda:1}` to a device in json format.
c10::Device convertDevice(std::string_view symbol);
// We have separate functions for parsing atomic and list constants because
// there are restrictive rules about which constants can go in lists (i.e.
// it's not recursive).
Constant convertAtomicConstant(std::string_view symbol);
Constant convertListConstant(std::string_view symbol);

} // namespace torch::nativert
