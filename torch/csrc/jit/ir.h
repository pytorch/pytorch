#pragma once

// TODO: Remove Python dependency with layer of indirection

#include <Python.h>

#include <iostream>
#include <memory>
#include <vector>
#include <atomic>
#include <algorithm>
#include <unordered_set>
#include <cstdint>

#include <ATen/ATen.h>

#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/auto_gpu.h"

#include "torch/csrc/jit/DisallowCopy.h"
#include "ATen/ArrayRef.h"
#include "torch/csrc/jit/assert.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/attributes.h"

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
// and dependencies on a list of values. The "prim-ops", so to speak.
struct Node;

#define TH_FORALL_TYPES(_) \
_(MultiType) \
_(TensorType) \
_(HandleType)

enum class TypeKind {
#define DEFINE_TYPE(T) T,
  TH_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

struct Type;
using TypePtr = std::shared_ptr<Type>;

struct Type {
private:
  TypeKind kind_;

protected:
  Type(TypeKind kind)
    : kind_(kind) {}

public:
  TypeKind kind() const {
    return kind_;
  }

  // Dynamically cast this object to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid..
  template<typename T>
  T* cast() {
    if (T::Kind == kind())
      return static_cast<T*>(this);
    return nullptr;
  }
  template<typename T>
  T* expect() {
    JIT_ASSERT(T::Kind == kind())
    return static_cast<T*>(this);
  }
};

// This node represents a single Tensor value
struct TensorType : public Type {
  friend struct Type;
  TensorType(const at::Tensor& tensor)
    : Type(TypeKind::TensorType), scalar_type_(tensor.type().scalarType()) {
      auto ndim = tensor.dim();
      sizes_.resize(ndim);
      strides_.resize(ndim);
      // NOTE: This is not memcpy! These are assignments.
      std::copy(tensor.sizes().begin(), tensor.sizes().end(), sizes_.begin());
      std::copy(tensor.strides().begin(), tensor.strides().end(), strides_.begin());
  }
  TensorType(at::ScalarType type, const std::vector<int64_t> & sizes)
  : Type(TypeKind::TensorType), scalar_type_(type), sizes_(sizes) {
    strides_.resize(sizes_.size());
    strides_.back() = 1;
    for(size_t i = sizes_.size() - 1; i > 0; i--) {
      strides_[i-1] = strides_[i]*sizes_[i];
    }
  }
  static const TypeKind Kind = TypeKind::TensorType;
  at::ScalarType scalarType() const {
    return scalar_type_;
  }
  const std::vector<std::int64_t>& sizes() const {
    return sizes_;
  }
  const std::vector<std::int64_t>& strides() const {
    return strides_;
  }
  TypePtr contiguous() const {
    return std::make_shared<TensorType>(scalar_type_,sizes_);
  }
private:
  at::ScalarType scalar_type_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
};

// Type of multireturn nodes. Note that it doesn't mean that they must always
// have multiple outputs, but each output will be represented with a select node.
struct MultiType : public Type {
  friend struct Type;

  MultiType()
    : Type(TypeKind::MultiType) {}

public:
  static const TypeKind Kind = TypeKind::MultiType;
};

// This value represents an opaque handle to external state.
// Operators that produce/consume values of this type agree on
// the format.

/* Example Usage: passing state to opaque autograd Functions:
graph(%1, %8) {
  %2.0, %2.1 = ^AddConstant(2, False)(%1) // first output is Type::Handle, containing ctx
  %4.0, %4.1 = ^Add(False)(%2.1, %1) // first output is Type::Handle, containing ctx
  %6.0, %6.1 = ^Abs()(%4.1) // first output is Type::Handle, containing ctx
  ---------------- stage 1 ----------------
  %13 = AutogradOp[AbsBackward](%6.0, %8) // first argument is Type::Handle, consuming ctx
  %15 = AutogradOp[AddBackward](%4.0, %13.0) // first argument is Type::Handle, consuming ctx
  %18 = AutogradOp[AddConstantBackward](%2.0, %15.1) // first argument is Type::Handle, consuming ctx
  %20 = AutogradOp[N5torch8autograd3AddE](%18.0, %18.0)
  return (%6.0, %20.0);
}
*/
struct HandleType : public Type {
  friend struct Type;

  HandleType()
    : Type(TypeKind::HandleType) {}

public:
  static const TypeKind Kind = TypeKind::HandleType;
};

// Each use is represented by this type, see Node::uses()
// 'user' is the consumer of the node, offset is the index into
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

// Param represents an input to the Graph, it has no inputs itself.
// Graph holds a list of parameters.
struct Param;

// the list types are intentionally simple, but we type-def
// them here so if we need to change them, refactoring will be easier
using node_list = std::vector<Node*>;
using param_list = node_list;
using use_list = std::vector<Use>;
using pyobj_list = std::vector<THPObjectPtr>;
template<typename T>
using ArrayRef = at::ArrayRef<T>;
using NodeKind = Symbol;
struct graph_node_list;
struct graph_node_list_iterator;

inline TypePtr getInitialType(NodeKind kind) {
  static TypePtr multiType = std::make_shared<MultiType>();
  switch(kind) {
    case kPythonOp:
    case kCppOp:
    case kEval:
    case kChunk:
    case kFusionGroup:
      return multiType;
    default:
      return nullptr;
  }
}

static constexpr int kNextDirection = 0;
static constexpr int kPrevDirection = 1;

struct Node : public Attributes<Node> {
  TH_DISALLOW_COPY_AND_ASSIGN(Node);
  friend struct Graph;
  friend graph_node_list;
  friend graph_node_list_iterator;
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

  Node * next_in_graph[2] = {nullptr,nullptr};
  Node* & next() { return next_in_graph[kNextDirection]; }
  Node* & prev() { return next_in_graph[kPrevDirection]; }

  const NodeKind kind_;
  std::vector<Node*> inputs_;
  use_list uses_;
  Graph* graph_;
  size_t unique_ = 0;          // unique id
  size_t stage_ = 0;           // 0-forward, 1-backward, 2-double-backward,...
protected:
  TypePtr type_;
  Node(Graph * graph_, NodeKind kind_); //defined after graph
public:
  NodeKind kind() const {
    return kind_;
  }
  const TypePtr & type() const {
    JIT_ASSERT(type_ != nullptr);
    return type_;
  }
  const TypePtr & typeOption() const {
    return type_;
  }
  bool hasMultipleOutputs() const {
    return hasType() && type()->kind() == TypeKind::MultiType;
  }
  bool hasType() const {
    return type_ != nullptr;
  }
  Node* setType(const TypePtr type) {
    type_ = type;
    return this;
  }
  void inferTypeFrom(const at::Tensor& output) {
    setType(std::make_shared<TensorType>(output));
  }
  Graph * owningGraph() {
    return graph_;
  }
  size_t unique() {
    return unique_;
  }
  void setStage(size_t s) {
    stage_ = s;
  }
  size_t stage() {
    return stage_;
  }
  const std::vector<Node*>& inputs() {
    return inputs_;
  }
  // lots of things like select/chunk have a single input, so we have a
  // helper to make accessing it easier
  Node * input() {
    JIT_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  // select is used so frequently enought it is reasonable to have a helper
  // to access the offset.
  size_t offset() {
    return size_t(i(kOffset));
  }
  const use_list & uses() {
    return uses_;
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
  Node* addInput(Node * node) {
    JIT_ASSERT(graph_ == node->graph_);
    node->uses_.emplace_back(this, inputs_.size());
    inputs_.push_back(node);
    return node;
  }

  // Replace the input of 'this' at position 'i' with
  // 'newValue', returning the old node.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.replaceInput(1, %4)
  // Result:  %3 = f(%1, %4)
  Node * replaceInput(size_t i, Node * newValue) {
    JIT_ASSERT(newValue->graph_ == graph_);
    Node * old = dropInput(i);
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
  void replaceInputWith(Node * from, Node * to) {
    JIT_ASSERT(from->graph_ == graph_);
    JIT_ASSERT(to->graph_ == graph_);
    size_t i = 0;
    for(auto input : inputs()) {
      if(input == from)
        replaceInput(i, to);
      i++;
    }
  }

  // Replaces all uses of this node with 'newValue'.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%3, %3)
  // Execute: %3.replaceAllUsesWith(%6)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%6)
  //          %5 = h(%6, %6)
  void replaceAllUsesWith(Node * newValue) {
    JIT_ASSERT(graph_ == newValue->graph_);
    for(auto u : uses()) {
      u.user->inputs_[u.offset] = newValue;
      newValue->uses_.push_back(u);
    }
    uses_.clear();
  }

  // Insert unattached 'this' node after 'n' in the topological order.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertBefore(%4)
  // Result:  %3 = f(%1, %2)
  //          %5 = h(%1)
  //          %4 = g(%3)
  void insertBefore(Node * n) {
    JIT_ASSERT(n->inGraphList());
    insertAfter(n->prev());
  }

  // Insert unattached 'this' node after 'n' in the topological order.
  //
  // Given: %3 = f(%1, %2)
  //        %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertAfter(%4)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%1)
  void insertAfter(Node * n) {
    JIT_ASSERT(!inGraphList() && n->inGraphList());
    Node * next = n->next();
    n->next() = this;
    this->prev() = n;
    this->next() = next;
    next->prev() = this;
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

  // Remove 'this' from the instruction list and deallocate it.
  //
  // Invariant: 'this' must not have any uses.
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
  template<typename T>
  T* cast() {
    if(T::Kind == kind())
      return static_cast<T*>(this);
    return nullptr;
  }
  template<typename T>
  T* expect() {
    JIT_ASSERTM(T::Kind == kind(), "expected a %s but found a %s", symbolToString(T::Kind), symbolToString(kind()));
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
  Node* dropInput(size_t i) {
    JIT_ASSERT(i < inputs_.size());
    auto input_node = inputs_[i];
    auto use_it = findUseForInput(i);
    input_node->uses_.erase(use_it);
    inputs_[i] = nullptr;
    return input_node;
  }

  bool inGraphList() {
    JIT_ASSERT(next() != nullptr || prev() == nullptr);
    return next() != nullptr;
  }
  void removeFromList() {
    JIT_ASSERT(inGraphList());
    Node * next = this->next();
    Node * prev = this->prev();
    prev->next() = next;
    next->prev() = prev;
    this->next() = nullptr;
    this->prev() = nullptr;
  }
  void lint();
protected:
  // subclasses must override
  // this function is used by createClone to initialize a new version
  // of a node in another graph. It should allocate a new instance of the same
  // concrete type as 'this', but in graph 'g' which might be different
  // than graph_
  virtual Node * allocNewInstance(Graph * g) {
    return new Node(g,kind());
  }
  // create a copy of all properties of Node s into this.
  // subclasses should extend if they have additional informaiton to copy.
  // 'this' will be allocated with s->allocNewInstance(g) so it should have
  // the same concrete type as 's'
  virtual void cloneFrom(Node * s) {
    copyAttributes(*s);
  }
};

struct graph_node_list_iterator {
  graph_node_list_iterator()
  : cur(nullptr), d(kNextDirection) {}
  graph_node_list_iterator(Node * cur, int d)
  : cur(cur), d(d) {}
  graph_node_list_iterator(const graph_node_list_iterator & rhs)
  : cur(rhs.cur),d(rhs.d) {}
  Node * operator*() { return cur; }
  Node * operator->() { return cur; }
  graph_node_list_iterator & operator++() {
    JIT_ASSERT(cur);
    cur = cur->next_in_graph[d];
    return *this;
  }
  graph_node_list_iterator operator++(int) {
    graph_node_list_iterator old = *this;
    ++(*this);
    return old;
  }
  // erase cur without invalidating this iterator
  // named differently from destroy so that ->/. bugs do not
  // silently cause the wrong one to be called.
  // iterator will point to the previous entry after call
  void destroyCurrent() {
    Node * n = cur;
    cur = cur->next_in_graph[d == kNextDirection ? kPrevDirection : kNextDirection];
    n->destroy();
  }
  graph_node_list_iterator reverse() {
    return graph_node_list_iterator(cur, d == kNextDirection ? kPrevDirection : kNextDirection);
  }
private:
  Node * cur;
  int d; //direction 0 is forward 1 is reverse, see next_in_graph
};

inline graph_node_list_iterator Node::iterator() {
  return graph_node_list_iterator(this,0);
}
inline graph_node_list_iterator Node::reverseIterator() {
  return iterator().reverse();
}

struct graph_node_list {
  using iterator = graph_node_list_iterator;
  graph_node_list_iterator begin() {
    return graph_node_list_iterator(head->next_in_graph[d],d);
  }
  graph_node_list_iterator end() {
    return graph_node_list_iterator(head,d);
  }
  graph_node_list_iterator rbegin() {
    return reverse().begin();
  }
  graph_node_list_iterator rend() {
    return reverse().end();
  }
  graph_node_list reverse() {
    return graph_node_list(head, d == kNextDirection ? kPrevDirection : kNextDirection);
  }
  graph_node_list(Node * head, int d)
  : head(head), d(d) {}
private:
  Node * head;
  int d;
};

static inline bool operator==(graph_node_list_iterator a,graph_node_list_iterator b) {
  return *a == *b;
}

static inline bool operator!=(graph_node_list_iterator a,graph_node_list_iterator b) {
  return *a != *b;
}

struct Graph {
TH_DISALLOW_COPY_AND_ASSIGN(Graph);
friend struct Node;
private:
  param_list inputs_;

  // holds outputs in a way that can be reflected
  // as a Use object
  // also used as the beginning/end of the circular node list to avoid
  // having corner cases where the list is empty.
  Node * output_;

  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes

  std::unordered_set<Node*> all_nodes;
  size_t next_unique_;

  size_t new_node_stage_;

public:
  Graph()
  : next_unique_(0)
  , new_node_stage_(0) {
    output_ = create(kReturn);
    output_->next() = output_;
    output_->prev() = output_;
    output_->stage_ = -1; // >= than all stages
  }

  const param_list & inputs() {
    return inputs_;
  }
  const node_list & outputs() {
    return output_->inputs();
  }
  graph_node_list nodes() {
    return graph_node_list(output_,kNextDirection);
  }
  Node * return_node() {
    return output_;
  }

  Node * addInput() {
    Node* p = create(kParam);
    inputs_.push_back(p);
    return p;
  }

  void advanceStage() {
    new_node_stage_++;
  }
  size_t stage() {
    return new_node_stage_;
  }

  void eraseInput(size_t i) {
    JIT_ASSERT(i < inputs_.size());
    JIT_ASSERT(inputs_[i]->uses().size() == 0);
    Node * n = inputs_[i];
    inputs_.erase(inputs_.begin() + i);
    freeNode(n);
  }

  size_t registerOutput(Node * n) {
    output_->addInput(n);
    return outputs().size() - 1;
  }

  Node * create(NodeKind kind) {
    return new Node(this,kind);
  }

  Node * create(NodeKind kind, ArrayRef<Node*> inputs) {
    auto n = new Node(this,kind);
    for(auto i : inputs)
      n->addInput(i);
    return n;
  }

  // Select nodes are used to handle multiple returns for the ops that actually return
  // multiple values like PythonOp
  // By convention, there is a unique select node for each output of an op
  // so you can iterate over uses of a multi-return op to get all the select nodes.
  // in this case
  // number_of_outputs = op.uses().size()
  // this will change if Tuples ever become first class.

  Node * createSelect(Node * n, int64_t offset) {
    auto r = create(kSelect,{n});
    r->i_(kOffset,offset);
    return r;
  }

  Node * createChunk(Node * input, int64_t numChunks, int64_t dim) {
    auto n = create(kChunk,{input});
    n->i_(kNumChunks,numChunks);
    n->i_(kDim, dim);
    return n;
  }
  Node * createConstant(const at::Tensor& ref) {
    AutoGPU guard(ref.type().isCuda() ? ref.get_device() : -1);
    auto n = create(kConstant);
    n->t_(kValue,ref.clone());
    return n;
  }
  Node * createFusionGroup() {
    auto n = create(kFusionGroup);
    n->g_(kSubgraph,std::make_shared<Graph>());
    return n;
  }
  Node * createPythonOp(THPObjectPtr&& pyobj, const std::string & cconv, bool is_legacy, pyobj_list&& scalar_args);
  Node * createCppOp(const std::shared_ptr<torch::autograd::Function> & fn);
  // clone n, making a new node in _this_ graph.
  // use node_map to translate inputs of n to inputs of the cloned node
  Node * createClone(Node * n, std::function<Node*(Node*)> node_map) {
    //n can be from a different graph
    Node * r = n->allocNewInstance(this);
    r->type_ = n->type_;
    r->cloneFrom(n);
    for(auto i : n->inputs()) {
      r->addInput(node_map(i));
    }
    return r;
  }

  Node * appendNode(Node * n) {
    JIT_ASSERT(n->graph_ == this && !n->inGraphList());
    n->insertBefore(output_);
    return n;
  }

  Node * prependNode(Node * n) {
    JIT_ASSERT(n->graph_ == this && !n->inGraphList());
    n->insertAfter(output_);
    return n;
  }

  // Checks well-formedness and invariants of graph
  void lint();

  ~Graph() {
    for (Node * n : all_nodes)
      delete n;
  }

private:

  void freeNode(Node * n) {
    auto it = all_nodes.find(n);
    JIT_ASSERT(it != all_nodes.end());
    delete *it;
    all_nodes.erase(it);
  }
};

inline Node::Node(Graph * graph_, NodeKind kind_) :
  kind_(kind_),
  graph_(graph_),
  unique_(graph_->next_unique_++),
  stage_(graph_->new_node_stage_),
  type_(getInitialType(kind_)) {
  graph_->all_nodes.emplace(this);
}

inline void Node::destroy() {
  JIT_ASSERT(inGraphList());
  JIT_ASSERTM(uses().size() == 0, "attempting to erase a Node that still has uses.");
  removeAllInputs();
  removeFromList();
  graph_->freeNode(this);
}

// Helper macros for constructing switch statements over Node types
// instead of heavy-weight visitors
// read 'between' these defines to see how they turn into a big switch
// statement

// TODO: I'm pretty sure Constness can be done with C++ templates, ala
// std::is_const, but no time to work it out...
#define GENERIC_IF(Constness, FullKind, x, Kind) \
  auto && __match_key = x; \
  switch(__match_key->kind()) { \
    case FullKind: { \
      auto * value = static_cast<Constness ::torch::jit::Kind*>(__match_key); (void) value;
#define GENERIC_ELSEIF(Constness, FullKind, Kind) \
    } break; \
    case FullKind: { \
      auto * value = static_cast<Constness ::torch::jit::Kind*>(__match_key); (void) value;
#define GENERIC_ELSE() \
    } break; \
    default: {
#define GENERIC_END() \
    } break; \
  };

// Mutable case
#define IR_IFM(x,Kind) GENERIC_IF(,k##Kind,x,Kind)
#define IR_ELSEIFM(Kind) GENERIC_ELSEIF(,k##Kind,Kind)
#define IR_ELSE() GENERIC_ELSE()
#define IR_END() GENERIC_END()

#define IR_IF(x, Kind) \
  auto && __match_key = x; \
  switch(__match_key->kind()) { \
    case ::torch::jit::k##Kind: { \
      auto * value = __match_key; (void) value;
#define IR_ELSEIF(Kind) \
    } break; \
    case ::torch::jit::k##Kind: { \
      auto * value = __match_key; (void) value;

// Immutable case
#define TYPE_IF(x,Kind) GENERIC_IF(const,TypeKind::Kind,x,Kind)
#define TYPE_ELSEIF(Kind) GENERIC_ELSEIF(const,TypeKind::Kind,Kind)
#define TYPE_ELSE() GENERIC_ELSE()
#define TYPE_END() GENERIC_END()

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

std::ostream& operator<<(std::ostream & out, Graph & g);
std::ostream& operator<<(std::ostream & out, const Type & t);
std::ostream& operator<<(std::ostream & out, Node & t);

/************* All nodes not required to be defined before Graph **************/

 // execute a Python function, used for Ops we can't optimize but that we want to optimize around
struct PythonOp : public Node {
  static const NodeKind Kind = kPythonOp;
  PythonOp(Graph * graph)
  : Node(graph,kPythonOp) {}
  PythonOp* init(THPObjectPtr&& pyobj, const std::string & cconv, bool is_legacy, pyobj_list&& scalar_args) {
    this->pyobj = std::move(pyobj);
    this->scalar_args = std::move(scalar_args);
    this->cconv = cconv;
    this->is_legacy = is_legacy;
    return this;
  }
  virtual Node * allocNewInstance(Graph * g) override {
    return new PythonOp(g);
  }
  //TODO: make this non-autograd specific
  //remove is_legacy, avoid THPObjectPtr to avoid big PyTorch dependency

  // The Python object which contains the implementation of this function.
  // This is either a class (non-legacy) or an object (legacy).  See
  // TraceInterpreter for execution semantics.
  THPObjectPtr pyobj;
  // The calling convention for the Python function.
  // 's' -- python scalar argument
  // 't' -- tensor argument
  std::string cconv;
  bool is_legacy;
  // Scalar arguments to the Python function.  Not necessarily passed to
  // the function in this order; see cconv for the correct order.
  std::vector<THPObjectPtr> scalar_args;
  std::string name();
  virtual void cloneFrom(Node * other_) override {
    Node::cloneFrom(other_);
    auto other = other_->cast<PythonOp>();
    this->cconv = other->cconv;
    this->is_legacy = other->is_legacy;
    Py_INCREF(other->pyobj.get());
    this->pyobj = THPObjectPtr(other->pyobj.get());
    for(auto & sa : other->scalar_args) {
      Py_INCREF(sa.get());
      this->scalar_args.emplace_back(sa.get());
    }
  }
};
inline Node * Graph::createPythonOp(THPObjectPtr&& pyobj, const std::string & cconv, bool is_legacy, pyobj_list&& scalar_args) {
  auto op = new PythonOp(this);
  return op->init(std::move(pyobj),cconv,is_legacy,std::move(scalar_args));
}

// A Cpp operator is an operator which dispatches directly to an autograd function.
// TODO: These are not executable without reentrant engine.
struct CppOp : public Node {
  static const NodeKind Kind = kCppOp;
  CppOp(Graph * g)
  : Node(g,kCppOp) {}
  std::shared_ptr<torch::autograd::Function> fn;
  std::string name();
  CppOp* init(std::shared_ptr<torch::autograd::Function> fn) {
    this->fn = std::move(fn);
    return this;
  }
  virtual Node * allocNewInstance(Graph * g) override {
    return new CppOp(g);
  }
};
inline Node * Graph::createCppOp(const std::shared_ptr<torch::autograd::Function> & fn) {
  auto op = new CppOp(this);
  return op->init(fn);
}

void LintGraph(std::shared_ptr<Graph>& graph);

}} // namespace torch::jit
