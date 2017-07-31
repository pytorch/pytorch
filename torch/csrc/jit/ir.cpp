#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <stack>
#include <sstream>

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
std::ostream& operator<<(std::ostream & out, Node & n) {
  if(auto s = n.cast<Select>())
    out << "%" << s->base()->unique() << "." << s->offset();
  else
    out << "%" << n.unique();
  return out;
}
std::ostream& operator<<(std::ostream & out, const node_list & nodes) {
  size_t i = 0;
  for(auto n : nodes) {
    if(i++ > 0)
      out << ", ";
    out << *n;
  }
  return out;
}

static std::ostream& operator<<(std::ostream & out, THPObjectPtr& obj) {
   THPObjectPtr repr { PyObject_Repr(obj.get()) };
   return out << THPUtils_unpackString(repr.get());
}

std::string PythonOp::name() {
  return getPythonName(pyobj.get(),is_legacy);
}

std::ostream& operator<<(std::ostream & out, Graph & g) {
  out << "graph(" << g.inputs() << ") {\n";
  std::vector<FusionGroup*> groups;
  for(auto n : g.nodes()) {
    if(!n->cast<Select>()) { //improve readibility by printing selects inline
      out << "  %" << n->unique() << " = ";
      IR_IF(n,PythonOp)
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
        out << "fusion_group_" << groups.size();
        groups.push_back(value);
      IR_ELSE()
        out << toString(n->kind());
      IR_END()
      out << "(" << n->inputs() << "), uses = [";
      size_t i = 0;
      for(auto u : n->uses()) {
        if(i++ > 0)
          out << ", ";
        out << *u.user << ".i" << u.offset;
      }
      out << "];\n";
    }
  }
  out << "  return (" << g.outputs() << ");\n}\n";
  size_t i = 0;
  for(auto fg : groups) {
    out << "with fusion_group_" <<i++ << " = " << fg->subgraph();
  }
  return out;
}

using node_set = std::set<Node*>;
#define ALL_OF(container) container.begin(), container.end()

// These functions purposely operate on the internal members directly, to force
// you to think about how the invariants change if you change the data
// representation (even if the external API does not change.)

// NB: This assert is written to assume you don't have any unattached
// nodes.  Unattached nodes can occur while manipulations to the
// graph are occurring.
void Node::lint() {
  // Node invariants
  // - if node should live in list, nodes_iter is consistent
  // - Inputs are all marked as a use by the nodes they refer to
  // - Stage is consistent (stage is >= all input stages)
  // - Owning graph is non-null and consistent
  // - The "Select" invariant, when the node is MultiReturn

  if (kind_ != NodeKind::Param && kind_ != NodeKind::Return) {
    JIT_ASSERT(*nodes_iter_ == this);
  }

  {
    size_t i = 0;
    for (auto input : inputs_) {
      // WARNING: O(n^2)
      JIT_ASSERT(std::find(ALL_OF(input->uses_), Use(this, i)) != input->uses_.end());
      JIT_ASSERT(stage_ >= input->stage_);
      JIT_ASSERT(graph_->all_nodes.count(this) == 1);
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
      if (type_->kind() == TypeKind::Multi) {
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

  IR_IF(this, Return)
    JIT_ASSERT(uses_.size() == 0);
  IR_ELSEIF(Constant)
    JIT_ASSERT(inputs_.size() == 0);
  IR_ELSEIF(Param)
    JIT_ASSERT(inputs_.size() == 0);
  IR_ELSEIF(Select)
    JIT_ASSERT(inputs_.size() == 1);
  IR_ELSEIF(PythonOp)
    int n_scalars = 0, n_tensors = 0;
    for (auto c : value->cconv) {
      if (c == 's') {
        n_scalars++;
      } else if (c == 't') {
        n_tensors++;
      } else {
        JIT_ASSERT(0);
      }
      JIT_ASSERT(value->pyobj != nullptr);
    }
    JIT_ASSERT(n_scalars == value->scalar_args.size());
    JIT_ASSERT(n_tensors == inputs_.size());
  // TODO: It's not good for these ops to be top-level, it makes cases longer.
  IR_ELSEIF(Add)
    JIT_ASSERT(inputs_.size() == 2);
  IR_ELSEIF(Mul)
    JIT_ASSERT(inputs_.size() == 2);
  IR_ELSEIF(Negate)
    JIT_ASSERT(inputs_.size() == 1);
  IR_ELSEIF(Sigmoid)
    JIT_ASSERT(inputs_.size() == 1);
  IR_ELSEIF(Tanh)
    JIT_ASSERT(inputs_.size() == 1);
  IR_ELSEIF(FusionGroup)
    // TODO: Typecheck the parameters
    value->subgraph().lint();
  IR_END()

}

void Graph::lint() {
  // Graph invariants

  // std::cout << *this << std::endl;

  // nodes
  // - nodes_ is a valid topological ordering for inputs
  // - No repeated nodes
  // - Params and return do NOT occur in nodes
  // - next_unique_ is greater than all uniques in graph
  // - uniques in all_nodes are unique

  std::unordered_set<Node*> in_scope;
  std::unordered_set<size_t> seen_uniques;
  auto check_node = [&](Node* n) {
    auto b = in_scope.insert(n);
    JIT_ASSERT(b.second);
    auto b2 = seen_uniques.insert(n->unique_);
    JIT_ASSERT(b2.second);
    JIT_ASSERT(n->unique_ < next_unique_);
  };

  for (auto input : inputs_) {
    JIT_ASSERT(input->kind_ == NodeKind::Param);
    input->lint();
    check_node(input);
  }
  for (auto n : nodes_) {
    n->lint();
    JIT_ASSERT(n->kind_ != NodeKind::Param);
    JIT_ASSERT(n->kind_ != NodeKind::Return);
    for (auto input : n->inputs_) {
      JIT_ASSERT(in_scope.count(input) == 1);
    }
    for (auto use : n->uses_) {
      JIT_ASSERT(in_scope.count(use.user) == 0);
      JIT_ASSERT(all_nodes.count(use.user) == 1);
    }
    check_node(n);
  }
  JIT_ASSERT(output_->kind() == NodeKind::Return);
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
  node_set nodes_set(ALL_OF(nodes_));
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

}

std::unique_ptr<Graph> LintGraph(std::unique_ptr<Graph> graph) {
  graph->lint();
  return graph;
}

}}
