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
// It uses a simple ownship model where the graph owns all the nodes inside it.
// All refererences inside the graph are raw pointers.
// Destroying the Graph will invalidate any pointers to nodes in the graph.
struct Graph;

//Node is the base class of the IR graph. It represents one computation
// and dependencies on a list of values. The "prim-ops", so to speak.
struct Node;

// Graphs and Nodes are immutable after construction.
// That is, once a node is added to a graph, or an input is added to
// a node, they cannot be removed. We allow incremental addition
// of nodes to graphs and inputs to nodes to make construction easier.
// This design simplifies use-def tracking since we never need to
// patch the use-list and can build it incrementally.

// Transforms are functional, building new graphs for each phase, using
// environments/hash-tables to link from old to new.

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
_(SimpleMap)

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
  Graph* const graph_;
  size_t unique_ = 0;
  // what stage of computation 0-forward, 1-backward, 2-double-backward,...
  size_t stage_ = 0;
protected:
  Node(NodeKind kind_, Graph* graph_)
  : next_in_graph{nullptr,nullptr}, kind_(kind_), type_(nullptr), graph_(graph_)  {}
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
  Node* addInput(Node * node) {
    JIT_ASSERT(graph_ == node->graph_);
    node->uses_.emplace_back(this,inputs_.size());
    inputs_.push_back(node);
    return node;
  }
  //returns the old input
  Node * replaceInput(size_t i, Node * newValue) {
    JIT_ASSERT(i < inputs_.size() && (!newValue || newValue->graph_ == graph_));
    auto old_node = inputs_[i];
    Use use(this,i);
    if(old_node) {
      //O(N) on the use list, but unless we get nodes with +100 uses
      //vector traversal still is probably faster than linked list
      auto use_it = std::find(old_node->uses_.begin(),old_node->uses_.end(),use);
      JIT_ASSERT(use_it != old_node->uses().end());
      old_node->uses_.erase(use_it);
    }
    if(newValue) {
      newValue->uses_.push_back(use);
    }
    return old_node;
  }
  // llvm's replaceUsesOfWith
  void replaceInputWith(Node * from, Node * to) {
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
  void replaceAllUsesWith(Node * newValue) {
    assert(graph_ == newValue->graph_);
    for(auto u : uses()) {
      u.user->inputs_[u.offset] = newValue;
      newValue->uses_.push_back(u);
    }
    uses_.clear();
  }
private:
  bool inGraphList() {
    return next() && prev();
  }
public:
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
private:
  void dropAllInputs() {
    for(size_t i = 0; i < inputs().size(); ++i)
      replaceInput(i, nullptr);
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
public:
  void eraseFromParent();
  // dynamic cast: if(auto s = n.cast<Select>()) { ... }
  template<typename T>
  T* cast() {
    if(T::Kind == kind())
      return static_cast<T*>(this);
    return nullptr;
  }
};

 // execute a Python function, used for Ops we can't optimize but that we want to optimize around
struct PythonOp : public Node {
  static const NodeKind Kind = NodeKind::PythonOp;
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
  PythonOp(Graph* graph, THPObjectPtr&& pyobj, const std::string & cconv, bool is_legacy, pyobj_list&& scalar_args)
    : Node(Kind, graph)
    , pyobj(std::move(pyobj))
    , cconv(cconv)
    , is_legacy(is_legacy)
    , scalar_args(std::move(scalar_args))
    {}
};

 // an input tensor to the graph
struct Param : public Node {
  static const NodeKind Kind = NodeKind::Param;
  Param(Graph* graph)
  : Node(Kind, graph)
  {}
};

// Select nodes are used to handle multiple returns for the ops that actually return
// multiple values like PythonOp
// By convension, there is a unique select node for each output of an op
// so you can iterate over uses of a multi-return op to get all the select nodes.
// in this case
// number_of_outputs = op.uses().size()
// this will change if Tuples ever become first class.
struct Select : public Node {
  static const NodeKind Kind = NodeKind::Select;
  Select(Graph* graph, Node * node, size_t offset)
  : Node(Kind, graph), offset_(offset) {
    addInput(node);
  }
  // which multi-return op is it?
  Node * base() {
    return inputs()[0];
  }
  // which output is it?
  size_t offset() {
    return offset_;
  }
private:
  size_t offset_;
};

// helper to define simple primitive Ops.
template<NodeKind K>
struct Primitive : public Node {
  static const NodeKind Kind = K;
  Primitive(Graph* graph)
  : Node(Kind, graph) {}
  Primitive(Graph* graph, ArrayRef<Node*> inputs)
  : Primitive(graph) {
    for(auto i : inputs)
      addInput(i);
  }
};

// NB: non-nullary constructors don't get forwarded to the
// parents, so you have to spell out the constructors you want explicitly.

// example primitive
struct Add : public Primitive<NodeKind::Add> {
  Add(Graph* graph)
  : Primitive(graph) {}
  Add(Graph* graph, ArrayRef<Node*> inputs)
  : Primitive(graph, inputs) {}
};

// Temporary node that represents single-return fusable math operators
// until we have actual operators to reflect them.
struct SimpleMap: public Primitive<NodeKind::SimpleMap> {
  std::string op; //'Tanh'
  SimpleMap(Graph * graph, const std::string & op)
  : Primitive(graph), op(op) {}
  SimpleMap(Graph * graph, const std::string & op, ArrayRef<Node*> inputs)
  : Primitive(graph,inputs), op(op) {}
};

// the outputs of the Graph are represented as an Node so that its inputs
// can be tracked as Uses.
struct Return : public Primitive<NodeKind::Return> {
  Return(Graph* graph)
  : Primitive(graph) {}
  Return(Graph* graph, ArrayRef<Node*> inputs)
  : Primitive(graph, inputs) {}
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


struct Graph {
TH_DISALLOW_COPY_AND_ASSIGN(Graph);
friend struct Node;
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

  void registerOutput(Node * n) {
    output_->addInput(n);
  }

  // like make_shared, forward arguments to node constructors
  // while also associating it with the graph
  // e.g. g.create<Select>(another,0);
  template<typename T, typename... Args >
  T * create(Args&&... args) {
    T* r = new T(this, std::forward<Args>(args)...);
    r->unique_ = next_unique++;
    all_nodes.insert(r);
    return r;
  }
  Node * addNode(Node * n) {
    JIT_ASSERT(n->graph_ == this && !n->inGraphList());
    n->insertBefore(output_);
    return n;
  }
  template<typename T, typename... Args >
  Node * addNode(Args&&... args) {
    T* n = create<T>(std::forward<Args>(args)...);
    return addNode(n);
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
};

inline void Node::eraseFromParent() {
  JIT_ASSERT(inGraphList());
  JIT_ASSERTM(uses().size() == 0, "attempting to erase a Node that still has uses.");
  dropAllInputs();
  removeFromList();
  graph_->all_nodes.erase(this);
  delete this;
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

}}
