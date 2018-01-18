#include <Python.h>
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
#include <algorithm>
#include <string>

namespace py = pybind11;

namespace torch { namespace jit {

constexpr int max_tensor_display_size = 10;

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
void printValueRef(std::ostream & out, const Value * n) {
  out << "%" << n->uniqueName();
}

template <typename T>
std::ostream& operator<<(std::ostream & out, const std::vector<T> & nodes) {
  out << at::ArrayRef<T>{nodes};
  return out;
}

template <typename T>
std::ostream& operator<<(std::ostream & out, const at::ArrayRef<T> & nodes) {
  size_t i = 0;
  for(auto n : nodes) {
    if(i++ > 0)
      out << ", ";
    printValueRef(out, n);
  }
  return out;
}
std::ostream& printPyObject(std::ostream & out, const THPObjectPtr& obj) {
  AutoGIL gil;
  auto pyobj = py::handle(const_cast<PyObject*>(obj.get()));
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
    return out << THPUtils_unpackString(py::str(pyobj).ptr());
  }
}

std::string PythonOp::name() const {
  return getPythonName(pyobj.get(),is_legacy);
}

std::string CppOp::name() const {
  return fn->name();
}

struct const_value_list_with_types {
  const std::vector<const Value*>& values;
  bool use_newlines;
  const_value_list_with_types(const std::vector<const Value*>& values, bool use_newlines = false)
    : values(values), use_newlines(use_newlines) {}
};
std::ostream& operator<<(std::ostream & out, const_value_list_with_types l) {
  size_t i = 0;
  size_t prev_stage = 0;
  for(auto n : l.values) {
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
    printValueRef(out, n);
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
void printAttributes(std::ostream & out, const Node * n) {
  out << "[";
  auto names = n->attributeNames();
  int i = 0;
  for(auto name : names) {
    if(i++ > 0)
      out << ", ";
    out << name.toString() <<"=";
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
        {
          at::Tensor t = n->t(name);
          // 1-elem tensors are usually boxed scalars, so print them like it
          if (t.numel() == 1) {
            auto scalar = at::Scalar(t.view({})).local();
            out << "{";
            if (scalar.isFloatingPoint()) {
              out << scalar.toDouble();
            } else {
              out << scalar.toLong();
            }
            out << "}";
          } else if (t.numel() <= max_tensor_display_size) {
            // TODO: This is awful code.  Also it doesn't work on Windows.
            std::ostringstream tensor_ss;
            tensor_ss << t;
            std::string tensor_s{tensor_ss.str()};
            // Remove newlines
            std::replace(tensor_s.begin(), tensor_s.end(), '\n', ' ');
            out << tensor_s;
          } else {
            out << "<Tensor>";
          }
          break;
        }
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

std::ostream& printNode(std::ostream & out, const Node * n, std::vector<const Node*> * groups) {
  auto outputs = n->outputs();
  out << const_value_list_with_types(outputs);
  out << " = ";
  IR_IFM_CONST(n,PythonOp)
    out << "^" << value->name();
    out << "(";
    int i = 0;
    for (auto& scalar : value->scalar_args) {
      if (i++ > 0)
        out << ", ";
      printPyObject(out, scalar);
    }
    out << ")";
  IR_ELSEIF(FusionGroup)
    if(groups) {
      out << "fusion_group_" << groups->size();
      groups->push_back(value);
    } else {
      out << "fusion_group[" << *n->g(kSubgraph) << "]";
    }
  IR_ELSEIFM_CONST(CppOp)
    out << "CppOp[" << value->name() << "]";
  IR_ELSE()
    out << n->kind().toString();
    if(n->hasAttributes()) {
      printAttributes(out,n);
    }
  IR_END()
  out << "(" << n->inputs() << ")";
  std::string scopeName = n->scopeName();
  if (scopeName.empty()) {
    out << "\n";
  }
  else {
    out << ", ";
    out << "scope: " << scopeName << "\n";
  }
  return out;
}

std::ostream& operator<<(std::ostream & out, const Node & n) {
  return printNode(out, &n, nullptr);
}

std::ostream& operator<<(std::ostream & out, const Graph & g) {
  out << "graph(" << const_value_list_with_types(g.inputs(), true) << ") {\n";
  std::vector<const Node*> groups;
  size_t prev_stage = 0;
  for(auto n : g.nodes()) {
    if (n->stage() != prev_stage) {
      out << "  ---------------- stage " << n->stage() << " ----------------\n";
      prev_stage = n->stage();
    }
    out << "  ";
    printNode(out, n, &groups);
  }
  out << "  return (" << g.outputs() << ");\n}\n";
  size_t i = 0;
  for(auto fg : groups) {
    out << "with fusion_group_" <<i++ << " = " << *fg->g(kSubgraph);
  }
  /*
  // Uncomment this to debug all_nodes issues
  {
    out << "\n";
    out << "all_nodes:\n";
    for (auto& n : g.all_nodes) {
      printNode(out, const_cast<Node*>(n), nullptr);
    }
  }
  */
  return out;
}

static void checkSameDevice(const Node* node) {
  bool has_device = false;
  int device;
  auto checkValue = [&](const Value* v) {
    if(v->hasType()) {
      if(TensorType* type = v->type()->cast<TensorType>()) {
        if(!has_device) {
          has_device = true;
          device = type->device();
        } else {
          JIT_ASSERT(device == type->device());
        }
      }
    }
  };
  for(auto input : node->inputs()) {
    checkValue(input);
  }
  for(auto output : node->outputs()) {
    checkValue(output);
  }
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

  for(auto o : outputs()) {
    size_t i = 0;
    for (auto use : o->uses()) {
      // Use invariants
      // - Use is consistent with inputs
      // - Every user node is live (checked in Graph)
      JIT_ASSERT(use.user->inputs_[use.offset] == o);
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
    JIT_ASSERT(outputs().size() == 0);
  IR_ELSEIF(Param)
    JIT_ASSERT(inputs_.size() == 0);
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
  IR_ELSEIF(FusionGroup)
    checkSameDevice(value);
    // TODO: Typecheck the parameters
    value->g(kSubgraph)->lint();
  IR_END()

}

// TODO: When lint fails, give better indication about which
// instruction triggered the failure.
void Graph::lint() const {
  // Graph invariants

  // Uncomment the following to see the graph
  // std::cout << *const_cast<Graph*>(this);

  // nodes
  // - nodes_ is a valid topological ordering for inputs
  // - No repeated nodes
  // - Params and return do NOT occur in nodes
  // - next_unique_ is greater than all uniques in graph
  // - uniques in all_nodes are unique
  // - every use will occur later in the topsort

  std::unordered_set<const Value*> in_scope;
  std::unordered_set<const Node*> node_in_scope;
  std::unordered_set<size_t> seen_uniques;
  std::unordered_map<const Node*, int64_t> anticipated_uses;
  auto check_value = [&](const Value* v) {
    auto b = in_scope.insert(v);
    JIT_ASSERT(b.second);  // insertion took place
    auto b2 = seen_uniques.insert(v->unique());
    JIT_ASSERT(b2.second);  // insertion took place
    JIT_ASSERT(v->unique() < next_unique_);

    for (auto use : v->uses()) {
      JIT_ASSERT(node_in_scope.count(use.user) == 0);
      JIT_ASSERT(all_nodes.count(use.user) == 1);
      anticipated_uses[use.user]++;  // int default constructs to 0
    }
  };
  auto check_node = [&](const Node* n) {
    for (auto input : n->inputs_) {
      if (in_scope.count(input) != 1) {
        JIT_ASSERTM(0, "%%%d not in scope", input->unique());
      }
    }
    JIT_ASSERT(anticipated_uses[n] == static_cast<int64_t>(n->inputs_.size()));
    anticipated_uses[n] = -1;  // we saw the anticipated user!
    auto node_inserted = node_in_scope.insert(n);
    JIT_ASSERT(node_inserted.second);  // insertion took place
    size_t i = 0;
    for(auto o : n->outputs()) {
      JIT_ASSERT(o->node() == n);
      JIT_ASSERT(i++ == o->offset_);
      check_value(o);
    }
    n->lint();
  };

  for (auto input : inputs()) {
    check_value(input);
    JIT_ASSERT(input->node()->kind_ == kParam);
  }

  for (auto n : nodes()) {
    JIT_ASSERT(n->kind_ != kParam);
    JIT_ASSERT(n->kind_ != kReturn);
    check_node(n);
  }

  JIT_ASSERT(output_->kind() == kReturn);
  check_node(output_);

  for (auto kv : anticipated_uses) {
    JIT_ASSERT(kv.second == -1);
  }

  // all_nodes
  // - inputs_, output_ and nodes_ are all included in all_nodes
  // - all_nodes does not contain dead nodes??? (likely to be temporarily
  // suspended).  Weaker: all_nodes contains all inputs and returns
  // - only one return node???

  node_set all_nodes_set(ALL_OF(all_nodes)); // NB: all_nodes is *unordered*
  node_set nodes_set(ALL_OF(nodes()));
  node_set inputs_set {input_};
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
  if (begin() == end()) {
    JIT_ASSERT(stage() == 0);
  } else {
    JIT_ASSERT(stage() == rbegin()->stage());
  }
}

void Graph::dump() const {
  std::cout << *this << "\n";
}

void LintGraph(std::shared_ptr<Graph>& graph) {
  graph->lint();
}


void PythonOp::cloneFrom(Node * other_) {
  Node::cloneFrom(other_);
  auto other = other_->cast<PythonOp>();
  this->cconv = other->cconv;
  this->is_legacy = other->is_legacy;
  Py_INCREF(other->pyobj.get());
  this->pyobj = THPObjectPtr(other->pyobj.get());
  this->var_flags = other->var_flags;
  for(auto & sa : other->scalar_args) {
    Py_INCREF(sa.get());
    this->scalar_args.emplace_back(sa.get());
  }
}

}}
