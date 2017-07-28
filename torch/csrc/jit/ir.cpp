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

// These functions purposely operate on the internal members directly, to force
// you to think about how the invariants change if you change the data
// representation (even if the external API does not change.)

void Use::lint() {
  // Use invariants
  // - Use is consistent with inputs
  // - Every user node is live (in all_nodes)
}

void Node::lint() {
  // Node invariants
  // - nodes_iter is consistent with node
  // - Inputs are all live (checked by topological order) and marked
  //   as a use
  // - Stage is consistent (stage is >= all input stages)
  // - Owning graph is non-null and consistent
  // - The "Select" invariant, when the node is MultiReturn

  // Node subclass invariants
  // - Return uses is zero
  // - Param inputs is zero
  // - Select inputs is one
  // - Python operator cconv is correct

}

using node_set = std::set<Node*>;
#define ALL_OF(container) container.begin(), container.end()

void Graph::lint() {
  // Graph invariants

  // std::cout << *this << std::endl;

  // nodes
  // - nodes_ is a valid topological ordering for inputs
  // - No repeated nodes
  // - Params and return do NOT occur in nodes

  std::unordered_set<Node*> in_scope;
  for (auto input : inputs_) {
    JIT_ASSERT(input->kind() == NodeKind::Param);
    input->lint();
    auto b = in_scope.insert(input);
    JIT_ASSERT(b.second);
  }
  for (auto n : nodes_) {
    n->lint();
    for (auto input : n->inputs_) {
      JIT_ASSERT(in_scope.count(input) == 1);
    }
    auto b = in_scope.insert(n);
    JIT_ASSERT(b.second); // no repeated nodes
  }
  JIT_ASSERT(output_->kind() == NodeKind::Return);
  output_->lint();
  for (auto output : output_->inputs_) {
    JIT_ASSERT(in_scope.count(output) == 1);
  }

  // all_nodes
  // - inputs_, output_ and nodes_ are all included in all_nodes
  // - all_nodes does not contain dead nodes??? (likely to be temporarily
  // suspended).  Weaker: all_nodes contains all inputs and returns
  // - only one return node???

  node_set nodes_set(ALL_OF(nodes_));
  node_set inputs_set(ALL_OF(inputs_));
  node_set output_set{output_};
  // This assert is currently failing
  // JIT_ASSERT(std::includes(ALL_OF(all_nodes), ALL_OF(nodes_set)));
  // JIT_ASSERT(std::includes(ALL_OF(all_nodes), ALL_OF(inputs_set)));
  // JIT_ASSERT(std::includes(ALL_OF(all_nodes), ALL_OF(output_set)));

  node_set all_set;

  // Uniques
  // - next_unique_ is greater than all uniques in graph
  // - uniques in all_nodes are unique

}

}}
