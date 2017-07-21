#pragma once

// TODO: Remove Python dependency with layer of indirection

#include <iostream>

#include <Python.h>
#include <memory>
#include <vector>
#include <atomic>
#include <algorithm>
#include <unordered_set>

#include "torch/csrc/utils/object_ptr.h"

#include "torch/csrc/jit/DisallowCopy.h"
#include "ATen/ArrayRef.h"
#include "torch/csrc/jit/assert.h"

namespace torch { namespace jit {

// Graph represents one "function" of computation.
// It uses a simple ownership model where the graph owns all the nodes inside it.
// All references inside the graph are raw pointers.
// Destroying the Graph will invalidate any pointers to nodes in the graph.
struct Graph;

// Node is the base class of the IR graph. It represents one computation
// and dependencies on a list of values. The "prim-ops", so to speak.
struct Node;

struct Type {}; // we will need a type, but for now it does nothing...

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


// defined using x-macros so that we can generate toString easily
#define TH_FORALL_NODES(_) \
_(PythonOp) \
_(Param) \
_(Select) \
_(Return) \
_(Add) \
_(SimpleMap) \
_(FusionGroup)

enum class NodeKind {
#define DEFINE_NODE(n) n,
TH_FORALL_NODES(DEFINE_NODE)
#undef DEFINE_NODE
};

struct graph_node_list_iterator;
struct graph_node_list;

struct Node {
  TH_DISALLOW_COPY_AND_ASSIGN(Node);
  friend struct Graph;
  friend struct graph_node_list_iterator;
  friend struct graph_node_list;
private:
  // each node but Return/Param
  // is associated with exactly one place in the node list...
  // of the graph_
  // this circular is a doubly-linked list, the Return node is used as the sentinel for the beginning and end of the list
  // such that the list never has null pointers
  // next_in_graph[0] is next pointer
  // next_in_graph[1] is prev pointer
  // using an array to allow the same iterator class for forward and reverse node lists
  Node * next_in_graph[2];
  Node* & next() { return next_in_graph[0]; }
  Node* & prev() { return next_in_graph[1]; }

  const NodeKind kind_;
  Type * type_;
  std::vector<Node*> inputs_;
  use_list uses_;
  Graph* graph_;
  size_t unique_ = 0;
  // what stage of computation 0-forward, 1-backward, 2-double-backward,...
  size_t stage_ = 0;
protected:
  Node(NodeKind kind_)
  : next_in_graph{nullptr,nullptr}, kind_(kind_), type_(nullptr) {}
public:
  NodeKind kind() {
    return kind_;
  }
  Type * type() {
    return type_;
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

  // Graphs

  // Add 'node' as an input to 'this' at the end of existing
  // arguments.  Returns the added node for ease of chaining.
  //
  // Precondition: 'node' must be topologically before 'this'.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.addInput(%4)
  // Result:  %3 = f(%1, %2, %4)
  Node* addInput(Node * node) {
    JIT_ASSERT(graph_ == node->graph_);
    node->uses_.emplace_back(this,inputs_.size());
    inputs_.push_back(node);
    return node;
  }

  // Replace the input of 'this' at position 'i' with
  // 'newValue', returning the old node.
  //
  // Precondition: 'newValue' must be topologically before 'this'.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.replaceInput(1, %4)
  // Result:  %3 = f(%1, %4)
  Node * replaceInput(size_t i, Node * newValue) {
    JIT_ASSERT(newValue->graph_ == graph_);
    Node * old = dropInput(i);
    inputs_[i] = newValue;
    newValue->uses_.emplace_back(this,i);
    return old;
  }

  // Replace all occurrences of 'from' in the inputs of this
  // node with 'to'. Corresponds to llvm's replaceUsesOfWith.
  //
  // Precondition: 'to' must be topologically before 'this'.
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

  const use_list & uses() {
    return uses_;
  }
  
  // Replaces all uses of this node with 'newValue'.
  //
  // Precondition: 'newValue' must be topologically before all uses
  // of 'this'.  A sound approximation is that 'newVAlue' is topologically
  // before 'this'.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%3, %3)
  // Execute: %3.replaceAllUsesWith(%6)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%6)
  //          %5 = h(%6, %6)
  void replaceAllUsesWith(Node * newValue) {
    assert(graph_ == newValue->graph_);
    for(auto u : uses()) {
      u.user->inputs_[u.offset] = newValue;
      newValue->uses_.push_back(u);
    }
    uses_.clear();
  }

  // Insert node 'n' before this one in the topological order.
  //
  // Precondition: All inputs of 'n' must be topologically before
  // 'this'.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %4.insertBefore(%5)
  // Result:  %3 = f(%1, %2)
  //          %5 = h(%1)
  //          %4 = g(%3)
  void insertBefore(Node * n) {
    JIT_ASSERT(n->inGraphList());
    insertAfter(n->prev());
  }
  void insertAfter(Node * n) {
    JIT_ASSERT(!inGraphList() && n->inGraphList());
    Node * next = n->next();
    n->next() = this;
    this->prev() = n;
    this->next() = next;
    next->prev() = this;
  }
  void moveAfter(Node * n) {
    JIT_ASSERT(inGraphList());
    removeFromList();
    insertAfter(n);
  }
  void moveBefore(Node * n) {
    JIT_ASSERT(inGraphList());
    removeFromList();
    insertBefore(n);
  }
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
  void removeAllInputs() {
    for(size_t i = 0; i < inputs().size(); ++i)
      dropInput(i);
    inputs_.clear();
  }
  // iterators of the node list starting at this node
  // useful for resuming a search starting at this node
  graph_node_list_iterator iterator();
  graph_node_list_iterator reverseIterator();
  void eraseFromParent();
  // dynamic cast: if(auto s = n.cast<Select>()) { ... }
  template<typename T>
  T* cast() {
    if(T::Kind == kind())
      return static_cast<T*>(this);
    return nullptr;
  }
  virtual ~Node() {}
  //initialize this Node by copying properties of 'other'
  //translation of inputs is handled automatically in Graph::clone.
private:
  // lookup iterator in use list of _input i_ that corresponds to its use of _this_
  use_list::iterator findUseForInput(size_t i) {
    Node * old_node = inputs_[i];
    Use use(this,i);
    //O(N) on the use list, but unless we get nodes with +100 uses
    //vector traversal still is probably faster than linked list
    auto use_it = std::find(old_node->uses_.begin(),old_node->uses_.end(),use);
    JIT_ASSERT(use_it != old_node->uses().end());
    return use_it;
  }
  // remove the use of input i, this sets input i to nullptr, but
  // is only used internally to Node before setting it to a new value
  // or erasing the entry from the list.
  Node* dropInput(size_t i) {
    JIT_ASSERT(i < inputs_.size());
    auto old_node = inputs_[i];
    auto use_it = findUseForInput(i);
    old_node->uses_.erase(use_it);
    inputs_[i] = nullptr;
    return old_node;
  }
  bool inGraphList() {
    return next() && prev();
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
protected:
  virtual Node * allocClone(Graph * in_graph) = 0;
};

struct graph_node_list_iterator {
  graph_node_list_iterator()
  : cur(nullptr), d(0) {}
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
  // named differently from eraseFromParent so that ->/. bugs do not
  // silently cause the wrong one to be called.
  // iterator will point to the previous entry after call
  void eraseCurrentFromParent() {
    Node * d = cur;
    cur = cur->next_in_graph[d == 0 ? 1 : 0];
    d->eraseFromParent();
  }
private:
  Node * cur;
  int d; //direction 0 is forward 1 is reverse, see next_in_graph
};

inline graph_node_list_iterator Node::iterator() {
  return graph_node_list_iterator(this,0);
}

inline graph_node_list_iterator Node::reverseIterator() {
  return graph_node_list_iterator(this,1);
}

struct graph_node_list {
  graph_node_list_iterator begin() {
    return graph_node_list_iterator(head->next_in_graph[d],d);
  }
  graph_node_list_iterator end() {
    return graph_node_list_iterator(head,d);
  }
  graph_node_list reverse() {
    return graph_node_list(head, d == 0 ? 1 : 0);
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

/******************* Nodes required inside a Graph ****************************/

// NodeWithKind handles the mapping between concrete node type and
// its NodeKind tag.
//
// It also sets up the clone infrastructure
// using CRTP so that we can alloc new clones and dispatch to custom clone code
// without significant boilerplate

template<typename Self, NodeKind K>
struct NodeWithKind : public Node {
  friend class Graph; /* so it can access allocClone() */
  static const NodeKind Kind = K;
  NodeWithKind()
  : Node(K) {}
  // virtual so we can easily define a default here
  // defined using CRTP so cloneFrom doesn't need casts.
  // called from allocClone
  virtual void cloneFrom(Self * s) {}
protected:
  // allocate a new Node with the same type as this node, and
  // get it initialized in in_graph
  // in_graph may not be the same graph as this->graph_ because we might be
  // cloning the node into a new graph
  // defined here because we need to know Self to allocate a new node.
  // user-defined cloneFrom is called.
  virtual Node * allocClone(Graph * in_graph);
};

// helper to define simple primitive Ops.
template<typename Self, NodeKind K>
struct Primitive : public NodeWithKind<Self,K> {
  void init() {}
  void init(ArrayRef<Node*> inputs) {
    for(auto i : inputs)
      this->addInput(i);
  }
};


// the outputs of the Graph are represented as an Node so that its inputs
// can be tracked as Uses.
struct Return : public Primitive<Return,NodeKind::Return> {};

// an input tensor to the graph
struct Param : public NodeWithKind<Param,NodeKind::Param> {
  void init() {}
};

struct Graph {
TH_DISALLOW_COPY_AND_ASSIGN(Graph);
friend struct Node;
template<typename Self, NodeKind K>
friend struct NodeWithKind;
private:
  param_list inputs_;

  // holds outputs in a way that can be reflected
  // as a Use object
  // also used as the beginning/end of the circular node list to avoid
  // having corner cases where the list is empty.
  Return * output_;

  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes

  //allows fast removal of nodes when they are deleted.
  std::unordered_set<Node*> all_nodes;
  size_t next_unique;

public:
  Graph()
  : next_unique(0) {
    output_ = create<Return>();
    // initialize output_ as the head of our double-linked circular list.
    output_->next() = output_;
    output_->prev() = output_;
  }

  Param * addInput() {
    Param* p = create<Param>();
    inputs_.push_back(p);
    return p;
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

  // like make_shared, forward arguments to node initializers
  // while also correctly allocating the node to live in this graph
  // e.g. g.create<Select>(another,0);
  template<typename T, typename... Args >
  T * create(Args&&... args) {
    // default construction of all nodes
    T* r = new T();
    // common initialization for all nodes when they live in this graph
    initNewNodeForGraph(r);
    // custom per-node initialization
    r->init(std::forward<Args>(args)...);
    return r;
  }
  // clone n, making a new node in _this_ graph.
  // use node_map to translate inputs of n to inputs of the cloned node
  Node * createClone(Node * n, std::function<Node*(Node*)> node_map) {
    //n can be from a different graph
    Node * r = n->allocClone(this);
    for(auto i : n->inputs()) {
      r->addInput(node_map(i));
    }
    return r;
  }
  Node * addNode(Node * n) {
    JIT_ASSERT(n->graph_ == this && !n->inGraphList());
    n->insertBefore(output_);
    return n;
  }
  Node * prependNode(Node * n) {
    JIT_ASSERT(n->graph_ == this && !n->inGraphList());
    n->insertAfter(output_);
    return n;
  }

  template<typename T, typename... Args >
  Node * addNode(Args&&... args) {
    T* n = create<T>(std::forward<Args>(args)...);
    return addNode(n);
  }
  Node * addClone(Node * n, std::function<Node*(Node*)> node_map) {
    Node * r = createClone(n,node_map);
    addNode(n);
    return r;
  }
  const param_list & inputs() {
    return inputs_;
  }
  const node_list & outputs() {
    return output_->inputs();
  }
  graph_node_list nodes() {
    return graph_node_list(output_,0);
  }
  ~Graph() {
    for(auto n : all_nodes)
      delete n;
  }
private:
  // per graph initialization for any node
  // called from NodeWithKind::allocClone and Graph::create
  void initNewNodeForGraph(Node * r) {
    r->graph_ = this;
    r->unique_ = next_unique++;
    all_nodes.insert(r);
  }
  void freeNode(Node * n) {
    all_nodes.erase(n);
    delete n;
  }
};

inline void Node::eraseFromParent() {
  JIT_ASSERT(inGraphList());
  JIT_ASSERTM(uses().size() == 0, "attempting to erase a Node that still has uses.");
  removeAllInputs();
  removeFromList();
  graph_->freeNode(this);
}

template<typename Self, NodeKind K>
Node * NodeWithKind<Self,K>::allocClone(Graph * in_graph) {
  auto s = new Self();
  in_graph->initNewNodeForGraph(s);
  // static cast is needed because the compiler doesn't know NodeWithKind is a CRTP.
  s->cloneFrom(static_cast<Self*>(this));
  return s;
}

// Helper macros for constructing switch statements over Node types
// instead of heavy-weight visitors
// read 'between' these defines to see how they turn into a big switch
// statement

#define IR_IF(x,Kind) \
  auto && __match_key = x; \
  switch(__match_key->kind()) { \
    case NodeKind::Kind: { \
      Kind * value = static_cast<Kind*>(__match_key); (void) value;
#define IR_ELSEIF(Kind) \
    } break; \
    case NodeKind::Kind: { \
      Kind * value = static_cast<Kind*>(__match_key); (void) value;
#define IR_ELSE() \
    } break; \
    default: {
#define IR_END() \
    } break; \
  };

/* example:
  Node * n = ...;
  IR_IF(n,Select)
    cout << "Select of" << value->base() << "\n";
  IR_ELSEIF(PythonOp)
    cout << value->pyobj << "\n";
  IR_ELSEIF(Add)
    cout << "Add" << \n";
  IR_ELSE() // optional
    cout << "something else\n";
  IR_END()
*/

std::ostream& operator<<(std::ostream & out, Graph & g);

static inline const char * toString(NodeKind kind) {
  switch(kind) {
#define DEFINE_CASE(Kind) \
    case NodeKind::Kind: return #Kind;
    TH_FORALL_NODES(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      __builtin_unreachable();
  }
}


/************* All nodes not required to be defined before Graph **************/

 // execute a Python function, used for Ops we can't optimize but that we want to optimize around
struct PythonOp : public NodeWithKind<PythonOp,NodeKind::PythonOp> {
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
  void init(THPObjectPtr&& pyobj, const std::string & cconv, bool is_legacy, pyobj_list&& scalar_args) {
    this->pyobj = std::move(pyobj);
    this->scalar_args = std::move(scalar_args);
    this->cconv = cconv;
    this->is_legacy = is_legacy;
  }
  virtual void cloneFrom(PythonOp * other) override {
    throw std::runtime_error("cannot clone PythonOp because of THPObjectPtr");
  }
};

// Select nodes are used to handle multiple returns for the ops that actually return
// multiple values like PythonOp
// By convension, there is a unique select node for each output of an op
// so you can iterate over uses of a multi-return op to get all the select nodes.
// in this case
// number_of_outputs = op.uses().size()
// this will change if Tuples ever become first class.
struct Select : public NodeWithKind<Select,NodeKind::Select> {
  void init(Node * node, size_t offset) {
    addInput(node);
    this->offset_ = offset;
  }
  // which multi-return op is it?
  Node * base() {
    return inputs()[0];
  }
  // which output is it?
  size_t offset() {
    return offset_;
  }
  virtual void cloneFrom(Select * other) override {
    this->offset_ = other->offset_;
  }
private:
  size_t offset_;
};

// NB: non-nullary constructors don't get forwarded to the
// parents, so you have to spell out the constructors you want explicitly.

// example primitive
struct Add : public Primitive<Add,NodeKind::Add> {};

// Temporary node that represents single-return fusable math operators
// until we have actual operators to reflect them.
struct SimpleMap: public NodeWithKind<SimpleMap, NodeKind::SimpleMap> {
  std::string op; //'Tanh'
  void init(const std::string & op, ArrayRef<Node*> inputs) {
    this->op = op;
    for(auto i : inputs)
      addInput(i);
  }
  virtual void cloneFrom(SimpleMap * n) override {
    this->op = n->op;
  }
};

struct FusionGroup : public NodeWithKind<FusionGroup,NodeKind::FusionGroup> {
  void init() {
    subgraph_ = std::make_shared<Graph>();
  }
  virtual void cloneFrom(FusionGroup * other) {
    subgraph_ = other->subgraph_;
  }
  Graph & subgraph() {
    return *subgraph_;
  }
private:
  std::shared_ptr<Graph> subgraph_;
};

}}
