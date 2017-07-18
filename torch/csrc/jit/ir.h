#pragma once

// TODO: Remove Python dependency with layer of indirection

#include <iostream>

#include <Python.h>
#include <memory>
#include <vector>
#include <cassert>
#include <atomic>

#include "torch/csrc/utils/object_ptr.h"

#include "torch/csrc/jit/DisallowCopy.h"
#include "ATen/ArrayRef.h"

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
_(Add)

enum class NodeKind {
#define DEFINE_NODE(n) n,
TH_FORALL_NODES(DEFINE_NODE)
#undef DEFINE_NODE
};

struct Node {
  TH_DISALLOW_COPY_AND_ASSIGN(Node);
  friend struct Graph;
private:
  const NodeKind kind_;
  Type * type_;
  std::vector<Node*> inputs_;
  use_list uses_;
  Graph * graph_ = nullptr;
  size_t unique_ = 0;
  // what stage of computation 0-forward, 1-backward, 2-double-backward,...
  size_t stage_ = 0;
protected:
  Node(NodeKind kind_)
  : kind_(kind_), type_(nullptr) {}
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
    assert(graph_ == node->graph_);
    node->uses_.emplace_back(this,inputs_.size());
    inputs_.push_back(node);
    return node;
  }
  const use_list & uses() {
    return uses_;
  }
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

  PythonOp(THPObjectPtr&& pyobj, const std::string & cconv, bool is_legacy, pyobj_list&& scalar_args)
    : Node(NodeKind::PythonOp)
    , pyobj(std::move(pyobj))
    , cconv(cconv)
    , is_legacy(is_legacy)
    , scalar_args(std::move(scalar_args))
    {}
};

 // an input tensor to the graph
struct Param : public Node {
  static const NodeKind Kind = NodeKind::Param;
  Param()
  : Node(Kind) {}
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
  Select(Node * node, size_t offset)
  : Node(Kind), offset_(offset) {
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
  Primitive()
  : Node(Kind) {}
  Primitive(ArrayRef<Node*> inputs)
  : Primitive() {
    for(auto i : inputs)
      addInput(i);
  }
};
// example primitive
struct Add : public Primitive<NodeKind::Add> {};

// the outputs of the Graph are represented as an Node so that its inputs
// can be tracked as Uses.
struct Return : public Primitive<NodeKind::Return> {};

struct Graph {
TH_DISALLOW_COPY_AND_ASSIGN(Graph);
private:
  param_list inputs_;
  // a canonical topological order of the computation ops
  // that makes printing the computation easier.
  node_list nodes_;
  // holds outputs in a way that can be reflected
  // as a Use object
  Return * output_;

  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes
  std::vector<Node*> all_nodes;

  // like make_shared, forward arguments to node constructors
  // while also associating it with the graph
  // e.g. g.create<Select>(another,0);
  template<typename T, typename... Args >
  T * create(Args&&... args) {
    T* r = new T(std::forward<Args>(args)...);
    r->unique_ = all_nodes.size();
    r->graph_ = this;
    all_nodes.push_back(r);
    return r;
  }
public:
  Graph() {
    output_ = create<Return>();
  }

  Param * addInput() {
    Param* p = create<Param>();
    inputs_.push_back(p);
    return p;
  }

  void registerOutput(Node * n) {
    output_->addInput(n);
  }

  template<typename T, typename... Args >
  Node * addNode(Args&&... args) {
    T* n = create<T>(std::forward<Args>(args)...);
    nodes_.push_back(n);
    return n;
  }
  const param_list & inputs() {
    return inputs_;
  }
  const node_list & outputs() {
    return output_->inputs();
  }
  const node_list & nodes() {
    return nodes_;
  }
  ~Graph() {
    for(auto n : all_nodes)
      delete n;
  }
};

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
