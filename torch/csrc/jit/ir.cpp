#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/autograd/function.h"

#include "pybind11/pybind11.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <stack>
#include <sstream>

namespace py = pybind11;

namespace torch { namespace jit {

std::string getPythonName(const PyObject* obj, bool is_legacy) {
  AutoGIL gil;
  if (is_legacy) {
    return std::string(obj->ob_type->tp_name);
  } else {
    // NB: hypothetically __name__ could mutate the Python
    // object in a externally visible way. Please don't!
    auto wobj = const_cast<PyObject*>(obj);
    THPObjectPtr name{PyObject_GetAttrString(wobj, "__name__")};
    return THPUtils_unpackString(name.get());
  }
}
void printNodeRef(std::ostream & out, Node * n) {
  out << "%" << n->uniqueName();
}

std::ostream& operator<<(std::ostream & out, const node_list & nodes) {
  size_t i = 0;
  for(auto n : nodes) {
    if(i++ > 0)
      out << ", ";
    printNodeRef(out, n);
  }
  return out;
}

static std::ostream& operator<<(std::ostream & out, THPObjectPtr& obj) {
  auto pyobj = py::handle(obj.get());
  if (py::isinstance<py::tuple>(pyobj)) {
    // This special-case for printing tuples handles a problem where
    // str((2L, 3L)) outputs "(2L, 3L)" in Python 2 but "(2, 3)"
    // in Python 3.  In order to suppress the L-suffix, we must
    // manually print the string ourselves, calling str() on the
    // sub-elements.
    //
    // This is a fairly fragile fix (What if you have nested tuples
    // in tuples? What if you have dictionaries?) but it seems to hit
    // the cases that are triggered in practice in onnx-pytorch.  Revisit
    // this code if this is not the case.
    //
    // By the way, one non-solution for this problem is to monkeypatch
    // tuple.__str__; this doesn't work because Python doesn't allow
    // monkeypatching methods of built-in types.
    auto pytuple = pyobj.cast<py::tuple>();
    out << "(";
    size_t i = 0;
    for (auto& o : pytuple) {
      if (i > 0) {
        out << ", ";
      }
      THPObjectPtr str(py::str(o).release().ptr());
      out << THPUtils_unpackString(str.get());
      i++;
    }
    if (i == 1) {
      out << ",";
    }
    out << ")";
    return out;
  } else {
    THPObjectPtr str { PyObject_Str(obj.get()) };
    return out << THPUtils_unpackString(str.get());
  }
}

std::string PythonOp::name() {
  return getPythonName(pyobj.get(),is_legacy);
}

std::string CppOp::name() {
  return fn->name();
}

static void emitUses(std::ostream & out, Node * n) {
  size_t i = 0;
  for(auto u : n->uses()) {
    if(i++ > 0)
      out << ", ";
    printNodeRef(out, u.user);
    out << ".i" << u.offset;
  }
}

struct node_list_with_types {
  const node_list& nodes;
  bool use_newlines;
  node_list_with_types(const node_list& nodes, bool use_newlines = false)
    : nodes(nodes), use_newlines(use_newlines) {}
};
std::ostream& operator<<(std::ostream & out, node_list_with_types l) {
  size_t i = 0;
  size_t prev_stage = 0;
  for(auto n : l.nodes) {
    if(i++ > 0) {
      if (l.use_newlines) {
        // TODO: Indent here is hard-coded for "graph(": un-hard-code it
        out << "\n      ";
        if (n->stage() != prev_stage) {
          out << "-------- stage " << n->stage() << " --------\n      ";
          prev_stage = n->stage();
        }
      } else {
        out << ", ";
      }
    }
    printNodeRef(out, n);
    out << " : ";
    if(n->hasType())
      out << *n->type();
    else
      out << "UNKNOWN_TYPE";
  }
  return out;
}
template<typename T>
void printPrimList(std::ostream & out, const std::vector<T> & items) {
  out << "[";
  int i = 0;
  for(auto & item : items) {
    if(i++ > 0)
      out << ", ";
    out << item;
  }
  out << "]";
}
void printAttributes(std::ostream & out, Node * n) {
  out << "[";
  auto names = n->attributeNames();
  int i = 0;
  for(auto name : names) {
    if(i++ > 0)
      out << ", ";
    out << symbolToString(name) <<"=";
    switch(n->kindOf(name)) {
      case AttributeKind::f:
        out << n->f(name);
        break;
      case AttributeKind::fs:
        printPrimList(out,n->fs(name));
        break;
      case AttributeKind::i:
        out << n->i(name);
        break;
      case AttributeKind::is:
        printPrimList(out,n->is(name));
        break;
      case AttributeKind::s:
        out << n->s(name);
        break;
      case AttributeKind::ss:
        printPrimList(out,n->ss(name));
        break;
      case AttributeKind::t:
        out << "<Tensor>";
        break;
      case AttributeKind::ts:
        out << "[<Tensors>]";
        break;
      case AttributeKind::g:
        out << "<Graph>";
        break;
      case AttributeKind::gs:
        out << "[<Graphs>]";
        break;
    }
  }
  out << "]";
}

std::ostream& printNode(std::ostream & out, Node * n, std::vector<Node*> * groups) {
  node_list outputs = n->outputs();
  out << node_list_with_types(outputs);
  out << " = ";
  IR_IFM(n,PythonOp)
    out << "^" << value->name();
    out << "(";
    int i = 0;
    for (auto& scalar : value->scalar_args) {
      if (i++ > 0)
        out << ", ";
      out << scalar;
    }
    out << ")";
  IR_ELSEIF(FusionGroup)
    if(groups) {
      out << "fusion_group_" << groups->size();
      groups->push_back(value);
    } else {
      out << "fusion_group[" << *n->g(kSubgraph) << "]";
    }
  IR_ELSEIFM(CppOp)
    out << "CppOp[" << value->name() << "]";
  IR_ELSE()
    out << symbolToString(n->kind());
    if(n->hasAttributes()) {
      printAttributes(out,n);
    }
  IR_END()
  out << "(" << n->inputs() << "), uses = [";
  if(n->hasMultipleOutputs()) {
    size_t i = 0;
    for(auto u : n->uses()) {
      if(i++ > 0)
        out << ", ";
      out << "[";
      emitUses(out,u.user);
      out << "]";
    }
  } else {
    emitUses(out,n);
  }
  out << "];\n";
  return out;
}

std::ostream& operator<<(std::ostream & out, Node & n) {
  return printNode(out, &n, nullptr);
}

std::ostream& operator<<(std::ostream & out, Graph & g) {
  // Uncomment this to debug all_nodes issues
  /*
  {
    size_t i = 0;
    for (auto& n : g.all_nodes) {
      if (i++ > 0) out << ", ";
      out << *n;
    }
    out << "\n";
  }
  */
  out << "graph(" << node_list_with_types(g.inputs(), true) << ") {\n";
  std::vector<Node*> groups;
  size_t prev_stage = 0;
  for(auto n : g.nodes()) {
    if(n->kind() != kSelect) { //improve readibility by printing selects inline
      if (n->stage() != prev_stage) {
        out << "  ---------------- stage " << n->stage() << " ----------------\n";
        prev_stage = n->stage();
      }
      out << "  ";
      printNode(out, n, &groups);
    }
  }
  out << "  return (" << g.outputs() << ");\n}\n";
  size_t i = 0;
  for(auto fg : groups) {
    out << "with fusion_group_" <<i++ << " = " << *fg->g(kSubgraph);
  }
  return out;
}

using node_set = std::set<const Node*>;
#define ALL_OF(container) container.begin(), container.end()

// These functions purposely operate on the internal members directly, to force
// you to think about how the invariants change if you change the data
// representation (even if the external API does not change.)

// NB: This assert is written to assume you don't have any unattached
// nodes.  Unattached nodes can occur while manipulations to the
// graph are occurring.
void Node::lint() const {
  // Node invariants
  // - if node should live in list, nodes_iter is consistent
  // - Inputs are all marked as a use by the nodes they refer to
  // - Stage is consistent (stage is >= all input stages)
  // - Owning graph is non-null and consistent
  // - The "Select" invariant, when the node is MultiReturn
  //
  // The handle invariant:
  //    If a node takes a handle as an input, it is always the
  //    LAST input of the node.  There is at most one handle input.

  {
    size_t i = 0;
    for (auto input : inputs_) {
      // WARNING: O(n^2)
      JIT_ASSERT(std::find(ALL_OF(input->uses_), Use(const_cast<Node*>(this), i)) != input->uses_.end());
      JIT_ASSERT(stage_ >= input->stage_);
      JIT_ASSERT(graph_->all_nodes.count(this) == 1);
      // Handle invariant
      if (i != inputs_.size() - 1) {
        JIT_ASSERT(!input->hasType() || input->type()->kind() != TypeKind::HandleType);
      }
      i++;
    }
  }

  {
    size_t i = 0;
    for (auto use : uses_) {
      // Use invariants
      // - Use is consistent with inputs
      // - Every user node is live (checked in Graph)
      JIT_ASSERT(use.user->inputs_[use.offset] == this);
      // Select invariant
      // - Multi-return nodes only have select uses
      // - uses = [Select 0, Select 1, Select 2, ...]
      if (type_ && type_->kind() == TypeKind::MultiType) {
        JIT_ASSERT(use.offset == 0);
        IR_IF(use.user, Select)
          JIT_ASSERT(value->offset() == i);
        IR_ELSE()
          JIT_ASSERT(0);
        IR_END()
      }
      i++;
    }
  }

  // Node subclass invariants
  // - Return uses is zero
  // - Param inputs is zero
  // - Select inputs is one
  // - Python operator cconv is correct

  IR_IF(this,Constant)
    JIT_ASSERT(inputs_.size() == 0);
  IR_ELSEIF(Return)
    JIT_ASSERT(uses_.size() == 0);
  IR_ELSEIF(Param)
    JIT_ASSERT(inputs_.size() == 0);
  IR_ELSEIF(Select)
    JIT_ASSERT(inputs_.size() == 1);
  IR_ELSEIFM_CONST(PythonOp)
    std::size_t n_scalars = 0, n_tensors = 0;
    for (auto c : value->cconv) {
      if (c == 's') {
        n_scalars++;
      } else if (c == 't') {
        n_tensors++;
      } else {
        JIT_ASSERT(0);
      }
      JIT_ASSERT(static_cast<bool>(value->pyobj));
    }
    JIT_ASSERT(n_scalars == value->scalar_args.size());
    JIT_ASSERT(n_tensors == inputs_.size());
  IR_ELSEIFM_CONST(CppOp)
    // TODO: add invariants
  IR_ELSEIF(Eval)
    // TODO: add invariants
  // TODO: It's not good for these ops to be top-level, it makes cases longer.
  IR_ELSEIF(Add)
    JIT_ASSERT(inputs_.size() == 2);
  IR_ELSEIF(Mul)
    JIT_ASSERT(inputs_.size() == 2);
  IR_ELSEIF(Neg)
    JIT_ASSERT(inputs_.size() == 1);
  IR_ELSEIF(Sigmoid)
    JIT_ASSERT(inputs_.size() == 1);
  IR_ELSEIF(Tanh)
    JIT_ASSERT(inputs_.size() == 1);
  IR_ELSEIF(FusionGroup)
    // TODO: Typecheck the parameters
    value->g(kSubgraph)->lint();
  IR_ELSEIF(Split)
    JIT_ASSERT(inputs_.size() == 1);
  IR_END()

}

void Graph::lint() const {
  // Graph invariants

  // Uncomment the following to see the graph
  // std::cout << *this << std::endl;

  // nodes
  // - nodes_ is a valid topological ordering for inputs
  // - No repeated nodes
  // - Params and return do NOT occur in nodes
  // - next_unique_ is greater than all uniques in graph
  // - uniques in all_nodes are unique

  std::unordered_set<const Node*> in_scope;
  std::unordered_set<size_t> seen_uniques;
  auto check_node = [&](const Node* n) {
    auto b = in_scope.insert(n);
    JIT_ASSERT(b.second);
    auto b2 = seen_uniques.insert(n->unique_);
    JIT_ASSERT(b2.second);
    JIT_ASSERT(n->unique_ < next_unique_);
  };

  for (auto input : inputs_) {
    JIT_ASSERT(input->kind_ == kParam);
    input->lint();
    check_node(input);
  }
  for (auto n : nodes()) {
    n->lint();
    JIT_ASSERT(n->kind_ != kParam);
    JIT_ASSERT(n->kind_ != kReturn);
    for (auto input : n->inputs_) {
      if (in_scope.count(input) != 1) {
        if (n->kind_ == kSelect) {
          JIT_ASSERTM(0, "%%%d (select node) not in scope; you probably forget to append it to the graph (you won't see this in the graph rendering)", input->unique_);
        } else {
          JIT_ASSERTM(0, "%%%d not in scope", input->unique_);
        }
      }
    }
    for (auto use : n->uses_) {
      JIT_ASSERT(in_scope.count(use.user) == 0);
      JIT_ASSERT(all_nodes.count(use.user) == 1);
    }
    check_node(n);
  }
  JIT_ASSERT(output_->kind() == kReturn);
  output_->lint();
  for (auto output : output_->inputs_) {
    JIT_ASSERT(in_scope.count(output) == 1);
  }
  check_node(output_);

  // all_nodes
  // - inputs_, output_ and nodes_ are all included in all_nodes
  // - all_nodes does not contain dead nodes??? (likely to be temporarily
  // suspended).  Weaker: all_nodes contains all inputs and returns
  // - only one return node???

  node_set all_nodes_set(ALL_OF(all_nodes)); // NB: all_nodes is *unordered*
  node_set nodes_set(ALL_OF(nodes()));
  node_set inputs_set(ALL_OF(inputs_));
  node_set output_set{output_};
  // TODO: Make a more type safe std::includes wrapper which disallows use on
  // non-ordered containers
  JIT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(nodes_set)));
  JIT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(inputs_set)));
  JIT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(output_set)));

  node_set sum_set;
  sum_set.insert(ALL_OF(nodes_set));
  sum_set.insert(ALL_OF(inputs_set));
  sum_set.insert(ALL_OF(output_set));
  JIT_ASSERT(std::includes(ALL_OF(sum_set), ALL_OF(all_nodes_set)));

  // graph->stage() should be equal to max(node.stage for node in graph->nodes())
  if (nodes().begin() == nodes().end()) {
    JIT_ASSERT(stage() == 0);
  } else {
    JIT_ASSERT(stage() == nodes().rbegin()->stage());
  }
}

void Graph::dump() {
  std::cout << *this << "\n";
}

void LintGraph(std::shared_ptr<Graph>& graph) {
  graph->lint();
}

}}
