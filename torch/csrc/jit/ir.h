#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <atomic>
#include <algorithm>
#include <unordered_set>
#include <functional>
#include <cstdint>

#include <ATen/ATen.h>

#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/utils/python_stub.h"

#include "ATen/ArrayRef.h"
#include "torch/csrc/jit/generic_if.h"
#include "torch/csrc/assertions.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/resource_guard.h"
#include "torch/csrc/jit/type.h"
#include "torch/csrc/jit/graph_node_list.h"
#include "torch/csrc/jit/variable_flags.h"
#include "torch/csrc/jit/source_location.h"
#include "torch/csrc/utils/functional.h"

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

std::ostream& operator<<(std::ostream & out, const Graph & g);
std::ostream& operator<<(std::ostream & out, const Type & t);
std::ostream& operator<<(std::ostream & out, const Node & t);

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
};
static inline bool operator==(const Use & a, const Use & b) {
  return a.user == b.user && a.offset == b.offset;
}

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


// Scope is a node of a trie that represents the tree of nested scopes.
// Individual scopes are pushed and popped from Graph, which holds a
// pointer to the current scope. Each Node in Graph holds a pointer
// to the scope that was current when the node was created.
// The trie never needs to shrink, it only grows until it is disposed
// of when Graph is deallocated. Hence, pointers to scopes held by nodes
// will always be valid as long as Graph is alive.
struct Scope {
private:
  Scope* parent_;
  Symbol name_;
  std::vector<std::unique_ptr<Scope> > children_;
public:
  Scope() {
    name_ = Symbol::scope("");
    parent_ = NULL;
  }
  Scope(Scope* parent, Symbol name) {
    name_ = name;
    parent_ = parent;
  }
  Scope* push(Symbol name) {
    children_.push_back(std::unique_ptr<Scope>(new Scope(this, name)));
    return children_.back().get();
  }
  Scope* parent() {
    if (parent_ == NULL) {
      throw std::runtime_error("Cannot get parent from Scope with no parent");
    }
    return parent_;
  }
  bool isRoot() {
    return parent_ == NULL;
  }
  Scope* getRoot() {
    Scope* current = this;
    while (current->parent_) {
      current = current->parent_;
    }
    return current;
  }
  Symbol name() {
    return name_;
  }
  std::string namesFromRoot(const std::string& separator="/") {
    // TODO: I think the answer is we shouldn't have used Symbol here
    std::string out = this->name_.toUnqualString();
    if (this->isRoot()) {
      return out;
    }
    Scope* parent = this->parent_;
    while (!parent->isRoot()) {
      out = std::string(parent->name_.toUnqualString()) + separator + out;
      parent = parent->parent_;
    }
    return out;
  }
};

// the list types are intentionally simple, but we type-def
// them here so if we need to change them, refactoring will be easier
using node_list = std::vector<Node*>;
using value_list = std::vector<Value*>;
using use_list = std::vector<Use>;
using pyobj_list = std::vector<THPObjectPtr>;
template<typename T>
using ArrayRef = at::ArrayRef<T>;
using NodeKind = Symbol;

struct Value {
  TH_DISALLOW_COPY_AND_ASSIGN(Value);
  Value(Node * node_, size_t offset_);
private:
  friend struct Node;
  friend struct Graph;
  Node * node_;
  size_t offset_;
  size_t unique_ = 0;          // unique id
  size_t stage_ = 0;           // 0-forward, 1-backward, 2-double-backward,...
  use_list uses_;
  std::string unique_name_;
  TypePtr type_;
public:
  Value* setType(const TypePtr type) {
    JIT_ASSERT(type);
    type_ = type;
    return this;
  }
  void inferTypeFrom(const at::Tensor& output) {
    setType(std::make_shared<TensorType>(output));
  }
  const TypePtr & type() const {
    JIT_ASSERT(type_ != nullptr);
    return type_;
  }
  bool isHandle() const {
    return type()->kind() == TypeKind::HandleType;
  }
  bool isTensor() const {
    return type()->kind() == TypeKind::TensorType;
  }
  size_t unique() const {
    return unique_;
  }
  Value* setUniqueName(const std::string & name);
  std::string uniqueName() const {
    if (unique_name_ != "")
      return unique_name_;
    return std::to_string(unique());
  }
  Value* setStage(size_t s) {
    stage_ = s;
    return this;
  }
  size_t stage() const {
    return stage_;
  }
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

  void replaceFirstUseWith(Value * newValue);

  // Replaces all uses of this node with 'newValue'.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%3, %3)
  // Execute: %3.replaceAllUsesWith(%6)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%6)
  //          %5 = h(%6, %6)
  void replaceAllUsesWith(Value * newValue);

  Value* copyMetadata(Value * from) {
    setType(from->type());
    if (from->unique_name_ != "")
      setUniqueName(from->uniqueName());
    return this;
  }

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
  Node* & next() { return next_in_graph[kNextDirection]; }
  Node* & prev() { return next_in_graph[kPrevDirection]; }
  Node* const & next() const { return next_in_graph[kNextDirection]; }
  Node* const & prev() const { return next_in_graph[kPrevDirection]; }

  const NodeKind kind_;
  std::vector<Value*> inputs_;
  std::vector<Value*> outputs_;
  // subblocks
  std::vector<Block*> blocks_;
  Graph* graph_;
  Block* owning_block_;
  std::shared_ptr<SourceLocation> source_location_;
  size_t stage_;
  Scope* scope_;
protected:
  Node(Graph * graph_, NodeKind kind_); //defined after graph
public:
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
  size_t stage() const {
    return stage_;
  }
  Node* setStage(size_t s) {
    stage_ = s;
    return this;
  }
  Scope* scope() {
    return scope_;
  }
  void setScope(Scope* scope) {
    scope_ = scope;
  }
  std::string scopeName() const {
    if (scope_ == NULL) {
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
  bool hasUses() const {
    for(auto o : outputs()) {
      if(o->uses().size() > 0)
        return true;
    }
    return false;
  }
  void replaceAllUsesWith(Node * n) {
    JIT_ASSERT(outputs().size() == n->outputs().size());
    size_t nOutputs = outputs().size();
    for(size_t i = 0; i < nOutputs; i++) {
      outputs()[i]->replaceAllUsesWith(n->outputs()[i]);
    }
  }
  // lots of things like chunk have a single input or singel output, so we have a
  // helper to make accessing it easier
  Value * input() {
    JIT_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  Value * output() {
    JIT_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  const  Value * input() const {
    JIT_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  // Access a particular input.  This is a checked index.
  Value * input(size_t i) {
    return inputs_.at(i);
  }
  const Value * input(size_t i) const {
    return inputs_.at(i);
  }

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
  Value* addInput(Value * node) {
    JIT_ASSERT(graph_ == node->owningGraph());
    node->uses_.emplace_back(this, inputs_.size());
    inputs_.push_back(node);
    return node;
  }

  // Add 'node' as an input to 'this' at the specified position in the
  // arguments. Returns the added node for ease of chaining.
  Value* insertInput(size_t i, Value* node) {
    JIT_ASSERT(graph_ == node->owningGraph());
    // First we update the offsets for all existing inputs that will reside
    // after the one we're inserting. Concretely, these are the inputs at
    // indices [i, # input). Since we're inserting one input before all of
    // these inputs, increment their use offsets for this Node by 1
    for (size_t use_itr = i; use_itr < inputs_.size(); ++use_itr) {
      // See Note [User node does not uniquely identify use]
      auto use = findUseForInput(use_itr);
      use->offset += 1;
    }
    // Insert the actual input at the specified index
    inputs_.insert(inputs_.begin() + i, node);
    // Register the new use of the value we're inserted as an input.
    node->uses_.emplace_back(this, i);
    return node;
  }

  // Replace the input of 'this' at position 'i' with
  // 'newValue', returning the old node.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.replaceInput(1, %4)
  // Result:  %3 = f(%1, %4)
  Value * replaceInput(size_t i, Value * newValue) {
    JIT_ASSERT(newValue->owningGraph() == graph_);
    Value * old = dropInput(i);
    inputs_[i] = newValue;
    newValue->uses_.emplace_back(this, i);
    return old;
  }

  // Replace all occurrences of 'from' in the inputs of this
  // node with 'to'. Corresponds to llvm's replaceUsesOfWith.
  //
  // Given:   %3 = f(%1, %2, %1)
  // Execute: %3.replaceInputWith(%1, %4)
  // Result:  %3 = f(%4, %2, %4)
  void replaceInputWith(Value * from, Value * to) {
    JIT_ASSERT(from->owningGraph() == graph_);
    JIT_ASSERT(to->owningGraph() == graph_);
    size_t i = 0;
    for(auto input : inputs()) {
      if(input == from)
        replaceInput(i, to);
      i++;
    }
  }

  Value* addOutput() {
    outputs_.push_back(new Value(this, outputs_.size()));
    return outputs_.back();
  }

  Value* insertOutput(size_t i) {
    outputs_.insert(outputs_.begin() + i, new Value(this, i));
    for (size_t itr = i + 1; itr < outputs_.size(); ++itr) {
      outputs_[itr]->setOffset(outputs_[itr]->offset() + 1);
    }
    return outputs_.at(i);
  }

  void eraseOutput(size_t i);

  Block * addBlock();
  void eraseBlock(size_t i);

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

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertBefore(%4)
  // Result:  %3 = f(%1, %2)
  //          %5 = h(%1)
  //          %4 = g(%3)
  Node* insertBefore(Node * n) {
    JIT_ASSERT(n->inBlockList());
    insertAfter(n->prev());
    return this;
  }

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
  Node* insertAfter(Node * n) {
    JIT_ASSERT(!inBlockList() && n->inBlockList());
    JIT_ASSERT(n->owningBlock());
    this->owning_block_ = n->owningBlock();
    Node * next = n->next();
    n->next() = this;
    this->prev() = n;
    this->next() = next;
    next->prev() = this;
    return this;
  }

  // Move 'this' (already in the graph) after 'n' in the topological order.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.moveAfter(%3)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  //
  void moveAfter(Node * n) {
    removeFromList();
    insertAfter(n);
  }

  // Move a node 'n' (already in the graph) before 'this' in the topological order.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %3.moveBefore(%2)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  void moveBefore(Node * n) {
    removeFromList();
    insertBefore(n);
  }

  // Remove the input at 'i' from this node.
  //
  // WARNING: This is O(n) in the number of inputs, so avoid repeatedly calling
  // removeInput.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeInput(1)
  // Result: %3 = f(%1)
  void removeInput(size_t i) {
    dropInput(i);
    // everything after this input shifts left,
    // so we need to update their use offsets to match
    for(size_t j = i+1; j < inputs_.size(); j++) {
      auto it = findUseForInput(j);
      it->offset--;
    }
    inputs_.erase(inputs_.begin() + i);
  }

  // Remove all inputs from a node.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeAllInputs()
  // Result: %3 = f()
  void removeAllInputs() {
    for(size_t i = 0; i < inputs().size(); ++i)
      dropInput(i);
    inputs_.clear();
  }

  // iterators of the node list starting at this node
  // useful for resuming a search starting at this node
  graph_node_list_iterator iterator();
  graph_node_list_iterator reverseIterator();
  const_graph_node_list_iterator iterator() const;
  const_graph_node_list_iterator reverseIterator() const;

  // Remove 'this' from the instruction list and deallocate it.
  //
  // Invariant: no outputs of 'this' may have any uses.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.destroy()
  // Result: %3 = g(%1)
  void destroy();

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
    JIT_ASSERTM(T::Kind == kind(), "expected a %s but found a %s", T::Kind.toDisplayString(), kind().toDisplayString());
    return static_cast<T*>(this);
  }

  virtual ~Node() {}
private:
  // Lookup iterator in use list of _input i_ that corresponds to its use of _this_
  use_list::iterator findUseForInput(size_t i) {
    auto & input_uses = inputs_[i]->uses_;
    // O(N) on the use list, but unless we get nodes with +100 uses
    // vector traversal still is probably faster than linked list
    auto use_it = std::find(input_uses.begin(), input_uses.end(), Use(this, i));
    JIT_ASSERT(use_it != input_uses.end());
    return use_it;
  }

  // remove the use of input i, this sets input i to nullptr, but
  // is only used internally to Node before setting it to a new value
  // or erasing the entry from the list.
  Value* dropInput(size_t i) {
    JIT_ASSERT(i < inputs_.size());
    auto input_node = inputs_[i];
    auto use_it = findUseForInput(i);
    input_node->uses_.erase(use_it);
    inputs_[i] = nullptr;
    return input_node;
  }

  bool inBlockList() const {
    if(next() == nullptr) {
      JIT_ASSERT(prev() == nullptr);
    }
    return next() != nullptr;
  }
  void removeFromList() {
    JIT_ASSERT(inBlockList());
    this->owning_block_ = nullptr;
    Node * next = this->next();
    Node * prev = this->prev();
    prev->next() = next;
    next->prev() = prev;
    this->next() = nullptr;
    this->prev() = nullptr;
  }
  void lint() const;
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
  // NB: This does NOT clone stages.  You're expected to set the stage correctly
  // if you are going to preserve it.
  virtual void cloneFrom(Node * s);
};

struct Block {
  friend struct Node;
  friend struct Graph;
  TH_DISALLOW_COPY_AND_ASSIGN(Block);
  Block(Graph * graph_, Node * node_);
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
    return graph_node_list(output_, kNextDirection);
  }
  const_graph_node_list nodes() const {
    return const_graph_node_list(output_, kNextDirection);
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
    if (name != "") v->setUniqueName(name);
    return v;
  }
  Value* insertInput(size_t i, std::string name = "") {
    Value* v = input_->insertOutput(i);
    if (name != "")
      v->setUniqueName(name);
    return v;
  }
  void eraseInput(size_t i) {
    input_->eraseOutput(i);
  }
  size_t registerOutput(Value * n) {
    output_->addInput(n);
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
  Node * owningNode() {
    return owning_node_;
  }
  // clone all inputs, nodes, and outputs from src and append them
  // to the inputs, nodes, and outputs of this block
  // value_map is used whenever a node in src references a free variable
  // in src to look up its corresponding value
  void cloneFrom(Block * src, std::function<Value*(Value*)> value_map);
private:
  // should only be called in the constructor
  Node* initOutput(Node* p) {
    p->next() = p;
    p->prev() = p;
    p->setStage(std::numeric_limits<size_t>::max());
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

  std::unordered_set<std::string> unique_names_;

  size_t new_node_stage_;

  std::shared_ptr<Scope> scope_root_;
  Scope * current_scope_;

  Block* const block_;
  // when insertNode() is called, the node is inserted before this node
  // by default this is set to append to the top level block
  Node* insert_before_;

public:

  Graph(std::shared_ptr<Scope> scope_root)
  : next_unique_(0)
  , new_node_stage_(0)
  , scope_root_(scope_root)
  , current_scope_(scope_root_.get())
  , block_(new Block(this, nullptr))
  , insert_before_(return_node()) {}

  Graph()
  : Graph( std::make_shared<Scope>()) {}

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
  Scope * current_scope() {
    return current_scope_;
  }
  void set_current_scope(Scope* scope) {
    if (scope->getRoot() != scope_root_.get()) {
      throw std::runtime_error("trying to set a scope as current that does not belong to the Graph's scope trie");
    }
    current_scope_ = scope;
  }
  std::shared_ptr<Scope> scope_root() {
    return scope_root_;
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
  void advanceStage() {
    new_node_stage_++;
  }
  void setStage(size_t new_stage) {
    new_node_stage_ = new_stage;
  }
  size_t stage() const {
    return new_node_stage_;
  }
  ResourceGuard setStageTemporary(size_t s) {
    auto prev_stage = new_node_stage_;
    new_node_stage_ = s;
    return ResourceGuard([prev_stage, this]() { this->new_node_stage_ = prev_stage; });
  }

  size_t registerOutput(Value * n) {
    return block_->registerOutput(n);
  }

  Node * create(NodeKind kind, size_t num_outputs=1) {
    // NB: Node constructor adds node to all_nodes
    auto n = new Node(this, kind);
    for(size_t i = 0; i < num_outputs; i++)
      n->addOutput();
    return n;
  }

  Node * create(NodeKind kind, ArrayRef<Value*> inputs, size_t num_outputs=1) {
    auto n = create(kind, num_outputs);
    for(auto i : inputs)
      n->addInput(i);
    return n;
  }

  Node * createUndefined() {
    return create(prim::Undefined);
  }
  Node * createConstant(const at::Tensor& ref) {
    JIT_ASSERT(ref.defined());
    AutoGPU guard(ref.type().is_cuda() ? ref.get_device() : -1);
    auto n = create(prim::Constant);
    n->t_(attr::value, ref.clone());
    return n;
  }
  Node * createFusionGroup(int device) {
    auto n = create(prim::FusionGroup, 0);
    n->g_(attr::Subgraph,std::make_shared<Graph>(scope_root_));
    n->i_(attr::device, device);
    return n;
  }
  Node* createTuple(at::ArrayRef<Value*> values) {
    auto types = fmap(values, [](Value* v) { return v->type(); });
    auto tt = std::make_shared<TupleType>(std::move(types));
    auto n = create(prim::TupleConstruct, values);
    n->output()->setType(tt);
    return n;
  }
  Node* createTupleUnpack(Value * v) {
    TupleType* tt = v->type()->expect<TupleType>();
    auto n = create(prim::TupleUnpack, {v}, 0);
    for(auto & element : tt->elements()) {
      n->addOutput()->setType(element);
    }
    return n;
  }
  Node* createPythonOp(
      THPObjectPtr&& pyobj,
      const std::string& cconv,
      bool is_legacy,
      std::vector<VariableFlags>&& var_flags,
      pyobj_list&& scalar_args,
      bool tracing_autograd_python_function = true);
  Node * createCppOp(const std::shared_ptr<torch::autograd::Function> & fn, std::vector<VariableFlags> && var_flags);
  // clone n, making a new node in _this_ graph.
  // use node_map to translate inputs of n to inputs of the cloned node
  // if copy_blocks is false, it will not recursively clone the nested blocks
  // this node contains.
  Node * createClone(Node * n, std::function<Value*(Value*)> value_map, bool copy_blocks=true) {
    //n can be from a different graph
    Node * r = n->allocNewInstance(this);
    for(auto o : n->outputs()) {
      r->addOutput()->copyMetadata(o);
    }
    r->cloneFrom(n);
    for(auto i : n->inputs()) {
      r->addInput(value_map(i));
    }
    if(copy_blocks) {
      for(auto b : n->blocks()) {
        r->addBlock()->cloneFrom(b, value_map);
      }
    }
    return r;
  }

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
  void lint() const;
  // for use in debugger
  void dump() const;

  ~Graph() {
    for (const Node * n : all_nodes)
      delete n;
    for (const Value * v : all_values)
      delete v;
    for (const Block * b : all_blocks)
      delete b;
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
  }

  friend std::ostream& operator<<(std::ostream & out, const Graph & g);
  std::shared_ptr<Graph> copy();

private:

  void freeNode(Node * n) {
    auto it = all_nodes.find(n);
    JIT_ASSERT(it != all_nodes.end());
    delete *it;
    all_nodes.erase(it);
  }
  void freeValue(Value * v) {
    auto it = all_values.find(v);
    JIT_ASSERT(it != all_values.end());
    delete *it;
    all_values.erase(it);
  }
  void freeBlock(Block * b) {
    auto it = all_blocks.find(b);
    JIT_ASSERT(it != all_blocks.end());
    delete *it;
    all_blocks.erase(it);
  }
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
  WithCurrentScope(Graph & g, Scope* scope)
  : ResourceGuard([&g, this]() {
    g.set_current_scope(prev_scope);
  })
  , prev_scope(g.current_scope()) {
    g.set_current_scope(scope);
  }
private:
  Scope * prev_scope;
};

inline Value::Value(Node * node_, size_t offset_)
: node_(node_),
  offset_(offset_),
  unique_(node_->graph_->next_unique_++),
  stage_(node_->graph_->new_node_stage_),
  type_(DynamicType::get()) {
  node_->graph_->all_values.emplace(this);
}

inline Graph * Value::owningGraph() {
  return node()->owningGraph();
}

inline const Graph * Value::owningGraph() const {
  return node()->owningGraph();
}

inline void Value::replaceFirstUseWith(Value * newValue) {
  JIT_ASSERT(owningGraph() == newValue->owningGraph());
  auto u = uses()[0];
  u.user->inputs_[u.offset] = newValue;
  newValue->uses_.push_back(u);
  uses_.erase(uses_.begin());
}

inline void Value::replaceAllUsesWith(Value * newValue) {
  while (!uses().empty()) {
    replaceFirstUseWith(newValue);
  }
}

inline Node::Node(Graph * graph_, NodeKind kind_) :
  kind_(kind_),
  graph_(graph_),
  owning_block_(nullptr),
  stage_(graph_->new_node_stage_),
  scope_(graph_->current_scope_) {
  graph_->all_nodes.emplace(this);
}

inline void Node::eraseOutput(size_t i) {
  JIT_ASSERT(i < outputs_.size());
  JIT_ASSERT(outputs_[i]->uses().size() == 0);
  Value * n = outputs_[i];
  outputs_.erase(outputs_.begin() + i);
  owningGraph()->freeValue(n);
  for(size_t j = i; j < outputs_.size(); j++) {
    outputs_[j]->offset_--;
  }
}

inline Block * Node::addBlock() {
  blocks_.push_back(new Block(owningGraph(), this));
  return blocks_.back();
}

inline void Node::eraseBlock(size_t i) {
  JIT_ASSERT(i < blocks_.size());
  Block * n = blocks_[i];
  blocks_.erase(blocks_.begin() + i);
  n->destroy();
}

inline void Node::destroy() {
  while(outputs().size() > 0)
    eraseOutput(outputs().size() - 1);
  while(blocks().size() > 0)
    eraseBlock(blocks().size() - 1);
  removeAllInputs();
  if(inBlockList())
    removeFromList();
  graph_->freeNode(this);
}

inline void Node::cloneFrom(Node * s) {
	setSourceLocation(s->getSourceLocation());
	if (s->owningGraph()->scope_root_ == owningGraph()->scope_root_) {
		scope_ = s->scope_;
	}
	copyAttributes(*s);
}

inline Value* Value::setUniqueName(const std::string & orig_name) {
  if (orig_name.find_first_not_of("0123456789") == std::string::npos) {
    throw std::runtime_error("names may not be integers: " + orig_name);
  }
  auto & names = node()->owningGraph()->unique_names_;
  auto name = orig_name;
  for(size_t i = 1; names.find(name) != names.end(); i++) {
    std::stringstream ss;
    ss << orig_name << "." << i;
    name = ss.str();
  }
  names.insert(name);
  unique_name_ = std::move(name);
  return this;
}

inline Block::Block(Graph * graph_, Node * node_)
: graph_(graph_)
, output_(initOutput(graph_->create(prim::Return, 0)))
, input_(graph_->create(prim::Param,0))
, owning_node_(node_) {
  graph_->all_blocks.emplace(this);
  output_->owning_block_ = this;
  input_->owning_block_ = this;
}

inline void Block::destroy() {
  // we cannot destroy the output because it is used as the sentinel
  // for the nodes() list and has to remain valid for the loop
  output_->removeAllInputs();
  for(auto it = this->nodes().reverse().begin(),
      end = this->nodes().reverse().end();
      it != end; ++it) {
    it.destroyCurrent();
  }
  output_->destroy();
  input_->destroy();
  graph_->freeBlock(this);
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

#define IR_IF(x, Kind) \
  auto && __match_key = x; \
  switch(__match_key->kind()) { \
    case ::torch::jit::prim::Kind: { \
      auto * value = __match_key; (void) value;
#define IR_ELSEIF(Kind) \
    } break; \
    case ::torch::jit::prim::Kind: { \
      auto * value = __match_key; (void) value;

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
      bool is_legacy,
      std::vector<VariableFlags>&& var_flags,
      pyobj_list&& scalar_args,
      bool tracing_autograd_python_function = true) {
    this->pyobj = std::move(pyobj);
    this->scalar_args = std::move(scalar_args);
    this->cconv = cconv;
    this->var_flags = std::move(var_flags);
    this->is_legacy = is_legacy;
    this->tracing_autograd_python_function = tracing_autograd_python_function;
    return this;
  }
  virtual Node * allocNewInstance(Graph * g) override {
    return new PythonOp(g);
  }
  //TODO: make this non-autograd specific
  //remove is_legacy, avoid THPObjectPtr to avoid big PyTorch dependency

  // The Python object which contains the implementation of this function.
  // This is either a class (non-legacy) or an object (legacy).  See
  // TraceInterpreterState for execution semantics.
  THPObjectPtr pyobj;
  // The calling convention for the Python function.
  // 's' -- python scalar argument
  // 't' -- tensor argument
  std::string cconv;
  bool is_legacy;
  bool tracing_autograd_python_function;
  // Scalar arguments to the Python function.  Not necessarily passed to
  // the function in this order; see cconv for the correct order.
  std::vector<THPObjectPtr> scalar_args;
  std::vector<VariableFlags> var_flags;
  std::string name() const;
  virtual void cloneFrom(Node * other_) override;
};
inline Node* Graph::createPythonOp(
    THPObjectPtr&& pyobj,
    const std::string& cconv,
    bool is_legacy,
    std::vector<VariableFlags>&& var_flags,
    pyobj_list&& scalar_args,
    bool tracing_autograd_python_function) {
  auto op = new PythonOp(this);
  return op->init(
      std::move(pyobj),
      cconv,
      is_legacy,
      std::move(var_flags),
      std::move(scalar_args),
      tracing_autograd_python_function);
}

// A Cpp operator is an operator which dispatches directly to an autograd function.
// TODO: These are not executable without reentrant engine.
struct CppOp : public Node {
  static constexpr Symbol Kind = prim::CppOp;
  CppOp(Graph * g)
  : Node(g,prim::CppOp) {}
  std::shared_ptr<torch::autograd::Function> fn;
  std::vector<VariableFlags> var_flags;
  std::string name() const;
  CppOp* init(std::shared_ptr<torch::autograd::Function> fn, std::vector<VariableFlags> && var_flags) {
    JIT_ASSERT(fn);
    this->fn = std::move(fn);
    this->var_flags = std::move(var_flags);
    return this;
  }
  virtual Node * allocNewInstance(Graph * g) override {
    return new CppOp(g);
  }
  virtual void cloneFrom(Node * other_) override {
    Node::cloneFrom(other_);
    auto other = other_->cast<CppOp>();
    this->fn = other->fn;
    this->var_flags = other->var_flags;
  }
};
inline Node * Graph::createCppOp(const std::shared_ptr<torch::autograd::Function> & fn, std::vector<VariableFlags> && var_flags) {
  auto op = new CppOp(this);
  return op->init(fn, std::move(var_flags));
}

inline graph_node_list_iterator Node::iterator() {
  return graph_node_list_iterator(this, 0);
}
inline graph_node_list_iterator Node::reverseIterator() {
  return iterator().reverse();
}
inline const_graph_node_list_iterator Node::iterator() const {
  return const_graph_node_list_iterator(this, 0);
}
inline const_graph_node_list_iterator Node::reverseIterator() const {
  return iterator().reverse();
}

void LintGraph(std::shared_ptr<Graph>& graph);

}} // namespace torch::jit
