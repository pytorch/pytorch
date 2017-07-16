#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
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

std::ostream& operator<<(std::ostream & out, const node_list & nodes) {
  size_t i = 0;
  for(auto n : nodes) {
    if(i++ > 0)
      out << ", ";
    if(auto s = n->cast<Select>())
      out << "%" << s->base()->unique() << "." << s->offset();
    else
      out << "%" << n->unique();
  }
  return out;
}

static std::ostream& operator<<(std::ostream & out, THPObjectPtr& obj) {
   THPObjectPtr repr { PyObject_Repr(obj.get()) };
   return out << THPUtils_unpackString(repr.get());
}

std::ostream& operator<<(std::ostream & out, Graph & g) {
  out << "graph(" << g.inputs() << ") {\n";
  for(auto n : g.nodes()) {
    out << "  %" << n->unique() << " = ";
    IR_IF(n,PythonOp)
      out << getPythonName(value->pyobj.get(),value->is_legacy);
      for (auto& scalar : value->scalar_args) {
        out << " " << scalar;
      }
      out << "(" << value->inputs() << ");\n";
    IR_ELSEIF(Select)
    IR_ELSE()
      out << toString(n->kind()) << "(" << n->inputs() << ");\n";
    IR_END()
    }
  out << "  return (" << g.outputs() << ");\n}\n";
  return out;
}

}}
