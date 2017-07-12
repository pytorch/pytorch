#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <iostream>

namespace torch { namespace autograd {

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

std::ostream& operator<<(std::ostream & out, const Value & l) {
  return out << "%" << l.unique;
}
std::ostream & operator<<(std::ostream & out, const value_list & values) {
  bool first = true;
  for(auto & v : values) {
    if(!first)
      out << ", ";
    first = false;
    out << *v;
  }
  return out;
}


static std::ostream& operator<<(std::ostream & out, THPObjectPtr& obj) {
   THPObjectPtr repr { PyObject_Repr(obj.get()) };
   return out << THPUtils_unpackString(repr.get());
}

std::ostream& operator<<(std::ostream & out, const Graph & g) {
  for(auto n : g.nodes) {
    out << n->outputs;
    out << " = ";
    switch(n->kind()) {
      case Node::Id::PythonOp:
        auto & value = (PythonOp&)*n;
        out << getPythonName(value.pyobj.get(), value.is_legacy);
        if (value.is_legacy) {
          out << " (legacy)";
        }
        for (auto& scalar : value.scalar_args) {
          out << " " << scalar;
        }
        break;
    }
    out << "(" << n->inputs << ")\n";
  }
  out << "return (" << g.outputs << ")\n";
  return out;
}

}}
