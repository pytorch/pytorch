#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"

#include <iostream>

namespace torch { namespace autograd {

std::string InputNode::name() const {
  return std::string("Variable");
}

std::string PyNode::name() const {
  AutoGIL gil;
  auto wobj = const_cast<PyObject*>(pyobj.get());
  if (is_legacy) {
    return std::string(wobj->ob_type->tp_name);
  } else {
    // NB: hypothetically __name__ could mutate the Python
    // object in a externally visible way. Please don't!
    THPObjectPtr name{PyObject_GetAttrString(wobj, "__name__")};
    // TODO: missing error check
    return std::string(PyString_AsString(name));
  }
}

// This printer is awful and I should be ashamed
void printGraph(const Node* n, int i) {
    if (!n) {
        std::cout << std::string(i, ' ') << "leaf" << std::endl;
        return;
    }
    std::cout << std::string(i, ' ') << n->name() << std::endl;
    for (auto o : n->inputs) {
        std::cout << std::string(i, ' ') << o.output_nr << std::endl;
        printGraph(o.node.get(), i+1);
    }
}

}}
