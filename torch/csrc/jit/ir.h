#pragma once

#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/generic_if.h"
#include "torch/csrc/jit/graph_node_list.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/resource_guard.h"
#include "torch/csrc/jit/scope.h"
#include "torch/csrc/jit/source_location.h"
#include "torch/csrc/jit/source_range.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/function_schema.h"
#include "torch/csrc/jit/ivalue.h"
#include "torch/csrc/jit/type.h"
#include "torch/csrc/jit/named_value.h"

#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_stub.h"
#include "torch/csrc/WindowsTorchApiMacro.h"

#include <ATen/ATen.h>
#include "ATen/core/ArrayRef.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

namespace torch { namespace autograd {

struct Function;

}} // namespace torch::autograd

namespace torch { namespace jit {

// Graph represents one "function" of computation.
// It uses a simple ownership model where the graph owns all the nodes inside it.
// All references inside the graph are raw pointers.
// Destroying the Graph will invalidate any pointers to nodes in the graph.
struct Graph;

// Node is the base class of the IR graph. It represents one computation
// and dependencies on a list of Values. The "prim-ops", so to speak.
struct Node;

// A Value represents an input or output to node that is either a
// Tensor or an opaque Handle object, as determined by type().
struct Value;

TORCH_API std::ostream& operator<<(std::ostream & out, const Graph & g);
TORCH_API std::ostream& operator<<(std::ostream & out, const Node & n);

// A list of nodes, with inputs and outputs
struct Block;

// Each use is represented by this type, see Node::uses()
// 'user' is the consumer of the value, offset is the index into
// 'user's input this where the produces will be found.
struct Use {
  Use(Node * user, size_t offset)
  : user(user), offset(offset) {}
  Node * user;
  size_t offset;

  bool operator==(const Use & b) {
    return user == b.user && offset == b.offset;
  }
};

// Note [User node does not uniquely identify use]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A while back, we wrote some code manipulating uses that looked like this:
//
//    for (auto& use : used_val->uses_) {
//      if (use.user == this_node) {
//        use.offset += 1;
//        break;
//      }
//    }
//
// This code is trying to find a particular use (our node's use) to update it.
// However, it's wrong: there may be *multiple* uses of a value %x in a node,
// as might be the case in this IR:
//
//    %y = Add %x %x
//
// In this case, there are two uses of %x whose user is the node 'Add %x %x'.
// So, "use induced by this node" is not a well-formed concept.
//
// If you are looking for "use induced by an input", it's best to use
// findUseForInput() to get it.

// the list types are intentionally simple, but we type-def
// them here so if we need to change them, refactoring will be easier
using node_list = std::vector<Node*>;
using value_list = std::vector<Value*>;
using use_list = std::vector<Use>;
using pyobj_list = std::vector<THPObjectPtr>;
template<typename T>
using ArrayRef = at::ArrayRef<T>;
using NodeKind = Symbol;
using topo_position_t = int64_t;

struct Value {
  TH_DISALLOW_COPY_AND_ASSIGN(Value);
  Value(Node * node_, size_t offset_);
private:
  friend struct Node;
  friend struct Graph;
  Node * node_;
  size_t offset_;
  size_t unique_ = 0;          // unique id
  use_list uses_;
  std::string unique_name_;
  TypePtr type_;
public:
  Value* setType(TypePtr type);
  void inferTypeFrom(const at::Tensor& output) {
    setType(CompleteTensorType::create(output));
  }
  const TypePtr & type() const {
    JIT_ASSERT(type_ != nullptr);
    return type_;
  }
  bool requires_grad() const {
    return type()->requires_grad();
  }
  bool isTensor() const {
    return type()->kind() == TypeKind::CompleteTensorType;
  }
  bool isNone() const {
    return type()->kind() == TypeKind::NoneType;

  }
  size_t unique() const {
    return unique_;
  }
  bool hasUniqueName() const {
    return !unique_name_.empty();
  }
  TORCH_API Value* setUniqueName(const std::string & name);
  std::string uniqueName() const {
    if (hasUniqueName())
      return unique_name_;
    return std::to_string(unique());
  }
  TORCH_API std::string uniqueNameBase() const;
  Node* node() {
    return node_;
  }
  size_t offset() const {
    return offset_;
  }
  void setOffset(size_t offset) {
    offset_ = offset;
  }
  const Node * node() const {
    return node_;
  }
  Graph * owningGraph();
  const Graph * owningGraph() const;
  // TODO: make this more const correct
  const use_list & uses() const {
    return uses_;
  }

  TORCH_API void replaceFirstUseWith(Value * newValue);

  // Replaces all uses of this value with 'newValue'.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%3, %3)
  // Execute: %3.replaceAllUsesWith(%6)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%6)
  //          %5 = h(%6, %6)
  TORCH_API void replaceAllUsesWith(Value * newValue);

  TORCH_API Value* copyMetadata(Value * from);
};


struct Node : public Attributes<Node> {
  TH_DISALLOW_COPY_AND_ASSIGN(Node);
  friend struct Graph;
  friend struct Block;
  friend struct Value;
  friend graph_node_list;
  friend const_graph_node_list;
  friend graph_node_list_iterator;
  friend const_graph_node_list_iterator;
private:
  // each node but Return/Param
  // is associated with exactly one place in the node list...
  // of the graph_
  // this circular is a doubly-linked list, the Return node is used as the sentinel for the beginning and end of the list
  // such that the list never has null pointers
  // next_in_graph[0] is next pointer
  // next_in_graph[1] is prev pointer
  // using an array to allow the same iterator class for forward and reverse node lists
  // This list represents a topological sort

  Node* next_in_graph[2] = { nullptr, nullptr };

  const NodeKind kind_;
  std::vector<Value*> inputs_;
  std::vector<Value*> outputs_;
  // subblocks
  std::vector<Block*> blocks_;
  Graph* graph_;
  Block* owning_block_;
  std::shared_ptr<SourceLocation> source_location_;
  ScopePtr scope_;
  // Assumes FunctionSchemas are persistent, so we don't manage their lifetime.
  // This field is effective a cache that's populated on attribute lookups and
  // invalidated every time we perform an operation that could potentially change
  // the schema.
  // note: mutable because schema_ is effectively a cache
  mutable const FunctionSchema* schema_;
  topo_position_t topo_position_ = 0;
protected:
  TORCH_API Node(Graph * graph_, NodeKind kind_); //defined after graph
public:
  Node* & next() { return next_in_graph[kNextDirection]; }
  Node* & prev() { return next_in_graph[kPrevDirection]; }
  Node* const & next() const { return next_in_graph[kNextDirection]; }
  Node* const & prev() const { return next_in_graph[kPrevDirection]; }

  NodeKind kind() const {
    return kind_;
  }
  Node* setSourceLocation(std::shared_ptr<SourceLocation> sl) {
    source_location_ = std::move(sl);
    return this;
  }
  std::shared_ptr<SourceLocation> getSourceLocation() const {
    return source_location_;
  }
  Graph * owningGraph() {
    return graph_;
  }
  const Graph * owningGraph() const {
    return graph_;
  }
  Block * owningBlock() {
    return owning_block_;
  }
  const Block * owningBlock() const {
    return owning_block_;
  }
  ScopePtr scope() {
    return scope_;
  }
  void setScope(ScopePtr scope) {
    scope_ = scope;
  }
  std::string scopeName() const {
    if (!scope_) {
      return "";
    }
    return scope_->namesFromRoot();
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  at::ArrayRef<Value*> inputs() {
    return inputs_;
  }
  at::ArrayRef<const Value*> inputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {inputs_.data(), inputs_.size()};
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  at::ArrayRef<Value*> outputs() {
    return outputs_;
  }
  at::ArrayRef<const Value*> outputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {outputs_.data(), outputs_.size()};
  }
  Value * output(size_t i) const {
    return outputs_.at(i);
  }
  bool hasUses() const {
    for(auto o : outputs()) {
      if(!o->uses().empty())
        return true;
    }
    return false;
  }

  TORCH_API void replaceAllUsesWith(Node * n);

  // lots of things like chunk have a single input or single output, so we have a
  // helper to make accessing it easier
  Value * input() {
    JIT_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  Value * output() {
    JIT_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  const Value* output() const {
    JIT_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  const  Value * input() const {
    JIT_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  // Access a particular input.  This is a checked index.
  Value * input(size_t i) const {
    return inputs_.at(i);
  }

  Value* namedInput(Symbol name) const;

  c10::optional<IValue> get(Symbol name) const;

  template <typename T>
  c10::optional<T> get(Symbol name) const {
    if(auto v = get(name))
      return v->template to<T>();
    return c10::nullopt;
  }

  // Returns true if the value of input name is statically known
  bool is_constant(Symbol name) const {
    return static_cast<bool>(get(name));
  }

  TORCH_API bool isNondeterministic() const;

  // Graphs

  // Note [Topological invariant]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // We always maintain an up-to-date topological ordering of all nodes via
  // the next()/prev() links.  All transformations to graphs must preserve
  // this topological ordering: for example, it is only valid to 'addInput'
  // with an input which is topologically before the current node.
  //
  // Usually, it is obvious whether or not topological order is maintained;
  // for example, if you are adding nodes to the end of the topsort, it's
  // impossible for them to refer to inputs that are not in the topsort.
  // If it is not obvious, please comment accordingly.

  // Add 'node' as an input to 'this' at the end of existing
  // arguments.  Returns the added node for ease of chaining.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.addInput(%4)
  // Result:  %3 = f(%1, %2, %4)
  TORCH_API Value* addInput(Value * value);

  // Add 'value' as an input to 'this' at the specified position in the
  // arguments. Returns the added value for ease of chaining.
  TORCH_API Value* insertInput(size_t i, Value* value);

  // Replace the input of 'this' at position 'i' with
  // 'newValue', returning the old node.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.replaceInput(1, %4)
  // Result:  %3 = f(%1, %4)
  TORCH_API Value * replaceInput(size_t i, Value * newValue);

  // Replace all occurrences of 'from' in the inputs of this
  // node with 'to'. Corresponds to llvm's replaceUsesOfWith.
  //
  // Given:   %3 = f(%1, %2, %1)
  // Execute: %3.replaceInputWith(%1, %4)
  // Result:  %3 = f(%4, %2, %4)
  TORCH_API void replaceInputWith(Value * from, Value * to);

  TORCH_API Value* addOutput();

  TORCH_API Value* insertOutput(size_t i);

  TORCH_API void eraseOutput(size_t i);

  TORCH_API Block * addBlock();
  TORCH_API void eraseBlock(size_t i);

  // Each Node can have a list of subblocks. These are used to define structured
  // nested control flow operators such as If and Loop.
  // The meaning of a block is specific to the kind of node it is in, but
  // all blocks share these semantics:
  // * Nested lexical scoping: If a node 'Parent' has a subblock which contains a
  //   node 'Child', Child can use any value that was in scope for the Parent
  //   node in addition to any values defined before 'Child' in the subblock.
  // * The list of inputs to the block are in scope for the duration of the block
  // * the outputs of the Parent node are not in scope for the subblocks
  // Typically the inputs to a block that represents control flow act as
  // as the equivalents phi-nodes in standard SSA form,
  // defining a new Value to represent any term that has multiple
  // definitions depending on how control flowed. Outputs of the node containing
  // control flow serve a similiar purpose defining new values for variables
  // that would have different defintions depending on which way control flowed.

  at::ArrayRef<Block*> blocks() {
    return blocks_;
  }
  at::ArrayRef<const Block*> blocks() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {blocks_.data(), blocks_.size()};
  }

  // Is 'this' before 'n' in the topological order?
  TORCH_API bool isBefore(const Node * n) const;

  // Is 'this' after 'n' in the topological order?
  TORCH_API bool isAfter(const Node * n) const;

  // Insert unattached 'this' node before 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertBefore(%4)
  // Result:  %3 = f(%1, %2)
  //          %5 = h(%1)
  //          %4 = g(%3)
  TORCH_API Node* insertBefore(Node * n);

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given: %3 = f(%1, %2)
  //        %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertAfter(%4)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%1)
  TORCH_API Node* insertAfter(Node * n);

  // Move 'this' (already in the graph) after 'n' in the topological order.
  //
  // NOTE: Does not check that value dependencies are preserved, see
  //   moveAfterTopologicallyValid
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.moveAfter(%3)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  //
  TORCH_API void moveAfter(Node * n);

  // Move 'this' (already in the graph) after 'n' in the topological order.
  //
  // Tries to preserve value dependencies, so other nodes might be moved. We
  // make two gurantees about the postcondition of the node list:
  //   - `this` is directly after `n`.
  //   - only nodes between `this` and `n` have been moved
  //
  // Returns `false` if it's impossible to move `this` after `n` without
  // violating dependencies, otherwise executes the move and returns `true`
  TORCH_API bool moveAfterTopologicallyValid(Node* n);

  // Move a node 'n' (already in the graph) before 'this' in the topological
  // order.
  //
  // NOTE: Does not check that value dependencies are preserved, see
  //   moveBeforeTopologicallyValid
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %3.moveBefore(%2)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  TORCH_API void moveBefore(Node * n);

  // Move 'this' (already in the graph) before 'n' in the topological order.
  //
  // Tries to preserve value dependencies, so other nodes might be moved. We
  // make two gurantees about the postcondition of the node list:
  //   - `this` is directly before `n`.
  //   - only nodes between `this` and `n` have been moved
  //
  // Returns `false` if it's impossible to move `this` after `n` without
  // violating dependencies, otherwise executes the move and returns `true`
  TORCH_API bool moveBeforeTopologicallyValid(Node* n);

  // Remove the input at 'i' from this node.
  //
  // WARNING: This is O(n) in the number of inputs, so avoid repeatedly calling
  // removeInput.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeInput(1)
  // Result: %3 = f(%1)
  TORCH_API void removeInput(size_t i);

  // Remove all inputs from a node.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeAllInputs()
  // Result: %3 = f()
  TORCH_API void removeAllInputs();

  // iterators of the node list starting at this node
  // useful for resuming a search starting at this node
  inline graph_node_list_iterator iterator() {
    return {this, 0};
  }
  inline graph_node_list_iterator reverseIterator() {
    return iterator().reverse();
  }
  inline const_graph_node_list_iterator iterator() const {
    return {this, 0};
  }
  inline const_graph_node_list_iterator reverseIterator() const {
    return iterator().reverse();
  }

  // Remove 'this' from the instruction list and deallocate it.
  //
  // Invariant: no outputs of 'this' may have any uses.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.destroy()
  // Result: %3 = g(%1)
  TORCH_API void destroy();

  // Dynamically cast this node to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid..
  //
  // Example usage: if(auto s = n.cast<Select>()) { ... }
  //
  // TODO: Make this const correct
  template<typename T>
  T* cast() {
    if(T::Kind == kind())
      return static_cast<T*>(this);
    return nullptr;
  }
  template<typename T>
  T* expect() {
    JIT_ASSERTM(
        T::Kind == kind(),
        "expected a ", T::Kind.toDisplayString(),
        " but found a ", kind().toDisplayString());
    return static_cast<T*>(this);
  }

  // XXX: this function is meant to be used with string literals only!
  TORCH_API bool matches(const char *signature_literal, at::ArrayRef<Symbol> const_inputs={}) const;

  const FunctionSchema& schema() const {
    if (!schema_)
      findSchema();
    return *schema_;
  }
  const FunctionSchema* maybeSchema() const;

  void dump() const;

  virtual ~Node() = default;

 private:
  enum class MoveSide { BEFORE, AFTER };
  bool tryMove(Node* movePoint, MoveSide moveSide);
  void move(Node* movePoint, MoveSide moveSide);

  std::pair<Value*, const Argument&> findInput(Symbol name);
  void findSchema() const;
  // Lookup iterator in use list of _input i_ that corresponds to its use of _this_
  TORCH_API use_list::iterator findUseForInput(size_t i);

  // remove the use of input i, this sets input i to nullptr, but
  // is only used internally to Node before setting it to a new value
  // or erasing the entry from the list.
  TORCH_API Value* dropInput(size_t i);

  bool inBlockList() const {
    if(next() == nullptr) {
      JIT_ASSERT(prev() == nullptr);
    }
    return next() != nullptr;
  }

  TORCH_API void removeFromList();
  TORCH_API void lint() const;

  void assignTopoPosition();

protected:
  // subclasses must override
  // this function is used by createClone to initialize a new version
  // of a node in another graph. It should allocate a new instance of the same
  // concrete type as 'this', but in graph 'g' which might be different
  // than graph_
  virtual Node * allocNewInstance(Graph * g) {
    return new Node(g, kind());
  }
  // create a copy of all properties of Node s into this.
  // subclasses should extend if they have additional information to copy.
  // 'this' will be allocated with s->allocNewInstance(g) so it should have
  // the same concrete type as 's'
  //
  TORCH_API virtual void cloneFrom(Node * s);
};

struct Block {
  friend struct Node;
  friend struct Graph;
  TH_DISALLOW_COPY_AND_ASSIGN(Block);
  TORCH_API Block(Graph * graph_, Node * node_);
  at::ArrayRef<Value*> inputs() {
    return input_->outputs();
  }
  at::ArrayRef<const Value*> inputs() const {
    const auto & inputs = input_->outputs();
    return {inputs.data(), inputs.size()};
  }
  at::ArrayRef<Value*> outputs() {
    return output_->inputs();
  }
  at::ArrayRef<const Value*> outputs() const {
    return static_cast<const Node*>(output_)->inputs();
  }
  graph_node_list nodes() {
    return {output_, kNextDirection};
  }
  const_graph_node_list nodes() const {
    return {output_, kNextDirection};
  }
  Node * return_node() {
    return output_;
  }
  const Node * return_node() const {
    return output_;
  }
  Node * param_node() {
    return input_;
  }
  const Node * param_node() const {
    return input_;
  }
  Value * addInput(std::string name="") {
    Value * v = input_->addOutput();
    v->setUniqueName(name);
    return v;
  }
  Value* insertInput(size_t i, std::string name = "") {
    Value* v = input_->insertOutput(i);
    v->setUniqueName(name);
    return v;
  }
  void eraseInput(size_t i) {
    input_->eraseOutput(i);
  }
  size_t registerOutput(Value * v) {
    output_->addInput(v);
    return outputs().size() - 1;
  }
  size_t insertOutput(size_t i, Value* n) {
    output_->insertInput(i, n);
    return i;
  }
  void eraseOutput(size_t i) {
    output_->removeInput(i);
  }
  Node * appendNode(Node * n) {
    JIT_ASSERT(n->graph_ == graph_ && !n->inBlockList());
    n->insertBefore(output_);
    return n;
  }

  Node * prependNode(Node * n) {
    JIT_ASSERT(n->graph_ == graph_ && !n->inBlockList());
    n->insertAfter(output_);
    return n;
  }
  Graph * owningGraph() {
    return graph_;
  }
  const Graph * owningGraph() const {
    return graph_;
  }
  Node * owningNode() {
    return owning_node_;
  }
  const Node * owningNode() const {
    return owning_node_;
  }
  // clone all inputs, nodes, and outputs from src and append them
  // to the inputs, nodes, and outputs of this block
  // value_map is used whenever a node in src references a free variable
  // in src to look up its corresponding value
  TORCH_API void cloneFrom(Block * src, std::function<Value*(Value*)> value_map);
private:
  void reIndexTopology();

  // should only be called in the constructor
  Node* initOutput(Node* p) {
    p->next() = p;
    p->prev() = p;
    return p;
  }

  // get rid of all nodes
  // destroys in reverse order so that uses internal to this block
  // do not have to be removed before you can destroy the block
  void destroy();

  Graph * const graph_;
  // holds outputs in a way that can be reflected
  // as a Use object
  // also used as the beginning/end of the circular node list to avoid
  // having corner cases where the list is empty.
  Node * const output_;
  Node * const input_;
  Node * const owning_node_; // either the node that has this block or nullptr for root
};

struct Graph {
TH_DISALLOW_COPY_AND_ASSIGN(Graph);
friend struct Node;
friend struct Value;
friend struct Block;
private:

  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes

  std::unordered_set<const Node*> all_nodes;
  std::unordered_set<const Value*> all_values;
  std::unordered_set<const Block*> all_blocks;
  size_t next_unique_;

  std::unordered_map<std::string, Value*> unique_names_;

  ScopePtr current_scope_;

  Block* const block_;
  // when insertNode() is called, the node is inserted before this node
  // by default this is set to append to the top level block
  Node* insert_before_;

public:

  Graph(ScopePtr scope_root)
  : next_unique_(0)
  , current_scope_(std::move(scope_root))
  , block_(new Block(this, nullptr))
  , insert_before_(return_node()) {}

  Graph() : Graph(c10::make_intrusive<Scope>()) {}

  at::ArrayRef<Value*> inputs() {
    return block_->inputs();
  }
  at::ArrayRef<const Value*> inputs() const {
    const auto & block = *block_;
    return block.inputs();
  }
  at::ArrayRef<Value*> outputs() {
    return block_->outputs();
  }
  at::ArrayRef<const Value*> outputs() const {
    const auto & block = *block_;
    return block.outputs();
  }
  graph_node_list nodes() {
    return block_->nodes();
  }
  const_graph_node_list nodes() const {
    const auto & block = *block_;
    return block.nodes();
  }
  Node * return_node() {
    return block_->return_node();
  }
  const Node * return_node() const {
    return block_->return_node();
  }
  void push_scope(const std::string& scope_name) {
    current_scope_ = current_scope_->push(Symbol::scope(scope_name));
  }
  void pop_scope() {
    current_scope_ = current_scope_->parent();
  }
  ScopePtr current_scope() {
    return current_scope_;
  }
  void set_current_scope(ScopePtr scope) {
    current_scope_ = scope;
  }
  Value * addInput(std::string name="") {
    return block_->addInput(std::move(name));
  }
  Value* insertInput(size_t i, std::string name = "") {
    return block_->insertInput(i, std::move(name));
  }
  void eraseInput(size_t i) {
    block_->eraseInput(i);
  }
  void eraseOutput(size_t i) {
    block_->eraseOutput(i);
  }
  const std::unordered_map<std::string, Value*>& uniqueNames() const {
    return unique_names_;
  }

  size_t registerOutput(Value * n) {
    return block_->registerOutput(n);
  }

  TORCH_API Node * create(NodeKind kind, size_t num_outputs=1);
  TORCH_API Node * create(NodeKind kind, ArrayRef<Value*> inputs, size_t num_outputs=1);

  TORCH_API Node* createUndefined();
  TORCH_API Node* createNoneGenerator();
  TORCH_API Node* createFusionGroup();
  TORCH_API Node* createTuple(at::ArrayRef<Value*> values);
  TORCH_API Node* createTupleUnpack(Value * v);
  TORCH_API Node* createTupleIndex(Value * tup, int64_t index);
  TORCH_API Node* createTupleSlice(Value * tup, int64_t beg, int64_t end);
  TORCH_API Node* createList(const TypePtr& elem_type, at::ArrayRef<Value*> values);
  TORCH_API Node* createListUnpack(Value *v, size_t size);
  TORCH_API Node* createNumToTensor(Value* value);
  TORCH_API Node* createBoolToTensor(Value* value);
  TORCH_API Node* createTensorToNum(const TypePtr& type, Value* value);
  TORCH_API Node* createImplicitTensorToNum(const TypePtr& type, Value* value);
  TORCH_API Node* createTensorToBool(Value* value);
  TORCH_API Node* createIntToFloat(Value* value);
  TORCH_API Node* createFloatToInt(Value* value);
  TORCH_API Node* createStringToFloat(Value* value);
  Node* createPythonOp(
      THPObjectPtr&& pyobj,
      const std::string& cconv,
      pyobj_list&& scalar_args);
  // clone n, making a new node in _this_ graph.
  // use node_map to translate inputs of n to inputs of the cloned node
  // if copy_blocks is false, it will not recursively clone the nested blocks
  // this node contains.
  TORCH_API Node * createClone(Node * n, std::function<Value*(Value*)> value_map, bool copy_blocks=true);

  TORCH_API Value* insertConstant(
      IValue val,
      c10::optional<SourceRange> loc = c10::nullopt,
      c10::optional<ScopePtr> scope = c10::nullopt);


  // schema-driven insert
  // this inserts a node into the graph with inputs determined from args and kwargs using Python
  // argument matching rules, and checks that the op matches a known schema
  // if this node successfully completes, it guarentees the node is a correctly-formed invocation
  // of opname
  Value* insert(
      Symbol opname,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs = {},
      c10::optional<SourceRange> range = {});

  Node * appendNode(Node * n) {
    return block_->appendNode(n);
  }

  Node * prependNode(Node * n) {
    return block_->prependNode(n);
  }

  // insert before insert_before_ node
  // initialized to insert at the end of the top level block
  // can be changed with setInsertPoint()
  Node * insertNode(Node * n) {
    JIT_ASSERT(insert_before_->inBlockList() && "insert point node is no longer in a block list");
    return n->insertBefore(insert_before_);
  }
  // set where nodes are inserted to append to the end of this block
  void setInsertPoint(Block * b) {
    JIT_ASSERT(b->owningGraph() == this);
    insert_before_ = b->return_node();
  }
  // set where nodes are inserted to insert _before_ this node
  // for implementation simplicity we only support inserting before a node for now
  void setInsertPoint(Node * n) {
    JIT_ASSERT(n->owningGraph() == this && n->inBlockList());
    insert_before_ = n;
  }
  Node * insertPoint() {
    return insert_before_;
  }

  // the top level block
  Block * block() {
    return block_;
  }
  const Block * block() const {
    return block_;
  }

  // Checks well-formedness and invariants of graph
  TORCH_API void lint() const;
  // for use in debugger
  TORCH_API void dump() const;

  TORCH_API ~Graph();

  TORCH_API std::string toString() const;

  friend TORCH_API std::ostream& operator<<(std::ostream & out, const Graph & g);

  TORCH_API std::ostream& prettyPrint(std::ostream & out);
  TORCH_API void dumpPretty();

  TORCH_API std::shared_ptr<Graph> copy();

private:

  TORCH_API void freeNode(Node * n);
  TORCH_API void freeValue(Value * v);
  TORCH_API void freeBlock(Block * b);
};

struct WithInsertPoint : public ResourceGuard {
  WithInsertPoint(Node * n)
  : ResourceGuard([this] {
    prev->owningGraph()->setInsertPoint(prev);
  })
  , prev(n->owningGraph()->insertPoint()) {
    n->owningGraph()->setInsertPoint(n);
  }
  WithInsertPoint(Block * b)
  : WithInsertPoint(b->return_node()) {}
private:
  Node * prev;
};

struct WithCurrentScope : public ResourceGuard {
  WithCurrentScope(Graph & g, ScopePtr scope)
  : ResourceGuard([&g, this]() {
    g.set_current_scope(prev_scope);
  })
  , prev_scope(g.current_scope()) {
    g.set_current_scope(scope);
  }
private:
  ScopePtr prev_scope;
};

inline Value::Value(Node * node_, size_t offset_)
: node_(node_),
  offset_(offset_),
  unique_(node_->graph_->next_unique_++),
  type_(DynamicType::get()) {
  node_->graph_->all_values.emplace(this);
}

inline Value* Value::setType(const TypePtr type) {
  JIT_ASSERT(type);
  type_ = type;
  for (Use & use : uses_) {
    use.user->schema_ = nullptr;
  }
  return this;
}

inline Graph * Value::owningGraph() {
  return node()->owningGraph();
}

inline const Graph * Value::owningGraph() const {
  return node()->owningGraph();
}

// Helper macros for constructing switch statements over Node types
// instead of heavy-weight visitors
// read 'between' these defines to see how they turn into a big switch
// statement

// Mutable case
// The IFM/ELSEIFM indicate that subclass *refinement* occurs.
// This is only valid for node types for which we have subclasses.
#define IR_IFM(x,Kind) GENERIC_IF(,prim::Kind,x,Kind)
#define IR_ELSEIFM(Kind) GENERIC_ELSEIF(,prim::Kind,Kind)

#define IR_IFM_CONST(x,Kind) GENERIC_IF(const,prim::Kind,x,Kind)
#define IR_ELSEIFM_CONST(Kind) GENERIC_ELSEIF(const,prim::Kind,Kind)

#define IR_IF(x, Kind)           \
  auto&& __match_key = x;        \
  switch (__match_key->kind()) { \
    case ::c10::prim::Kind: {    \
      auto* value = __match_key; \
      (void)value;
#define IR_ELSEIF(Kind)        \
  }                            \
  break;                       \
  case ::c10::prim::Kind: {    \
    auto* value = __match_key; \
    (void)value;

#define IR_ELSE() GENERIC_ELSE()
#define IR_END() GENERIC_END()

/* example:
  Node * n = ...;
  IR_IF(n,Select)
    cout << "Select of" << value->input() << "\n";
  IR_ELSEIF(PythonOp)
    cout << value->pyobj << "\n";
  IR_ELSEIF(Add)
    cout << "Add" << \n";
  IR_ELSE() // optional
    cout << "something else\n";
  IR_END()
*/

/************* All nodes not required to be defined before Graph **************/

 // execute a Python function, used for Ops we can't optimize but that we want to optimize around
struct PythonOp : public Node {
  static constexpr Symbol Kind = prim::PythonOp;

  PythonOp(Graph * graph)
  : Node(graph,prim::PythonOp) {}
  PythonOp* init(
      THPObjectPtr&& pyobj,
      const std::string& cconv,
      pyobj_list&& scalar_args) {
    this->pyobj = std::move(pyobj);
    this->scalar_args = std::move(scalar_args);
    this->cconv = cconv;
    return this;
  }
  // The Python object which contains the implementation of this function.
  // This is either a class (non-legacy) or an object (legacy).  See
  // TraceInterpreterState for execution semantics.
  THPObjectPtr pyobj;
  // The calling convention for the Python function.
  // 'c' -- constant argument
  // 'd' -- dynamic argument
  std::string cconv;
  // Scalar arguments to the Python function.  Not necessarily passed to
  // the function in this order; see cconv for the correct order.
  std::vector<THPObjectPtr> scalar_args;
  virtual std::string name() const = 0;
  virtual void writeScalars(std::ostream& out) const = 0;
  void cloneFrom(Node * other_) override = 0;
  Node * allocNewInstance(Graph * g) override = 0;
  // recover the autograd.Function instance, if this PythonOp's function
  // was originally SomeFunction.apply
  // used in ONNX for discovering symbolics
  virtual c10::optional<THPObjectPtr> autogradFunction() const = 0;
};
// patched in when python bindings are loaded
TORCH_API PythonOp* allocPythonOp(Graph* g);
TORCH_API void setAllocPythonOp(PythonOp* (*v)(Graph* g));
inline Node* Graph::createPythonOp(
    THPObjectPtr&& pyobj,
    const std::string& cconv,
    pyobj_list&& scalar_args) {
  auto op = allocPythonOp(this);
  return op->init(
      std::move(pyobj),
      cconv,
      std::move(scalar_args));
}

TORCH_API void LintGraph(std::shared_ptr<Graph>& graph);

}} // namespace torch::jit
