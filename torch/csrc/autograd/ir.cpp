#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <iostream>

namespace torch { namespace autograd {

std::string getExprName(Expr* expr) {
  switch (expr->_id) {
    case Expr::Id::PyApply: return "PyApply";
    case Expr::Id::Let: return "Let";
    case Expr::Id::Tuple: return "Tuple";
  }
  __builtin_unreachable();
}

std::string getPythonName(const PyObject* obj, bool is_legacy) {
  AutoGIL gil;
  if (is_legacy) {
    return std::string(obj->ob_type->tp_name);
  } else {
    // NB: hypothetically __name__ could mutate the Python
    // object in a externally visible way. Please don't!
    auto wobj = const_cast<PyObject*>(obj);
    THPObjectPtr name{PyObject_GetAttrString(wobj, "__name__")};
    // TODO: missing error check
    return std::string(PyString_AsString(name));
  }
}

// TODO: proper pretty-printer

class Printer : public ExprVisitor<Printer> {
  std::ostream& s;

public:
  Printer(std::ostream& s) : s(s) {}

  void printPyObject(THPObjectPtr& obj) {
    THPObjectPtr repr { PyObject_Repr(obj.get()) };
    s << THPUtils_unpackString(repr.get());
  }

  void visitLocal(std::shared_ptr<Local> a) {
    s << "%" << a->unique;
  }

  // Expr
  void visitLet(std::shared_ptr<Let> e, int indent) {
    bool first = true;
    s << std::string(indent, ' ');
    for (auto l : e->bind.lvals) {
      if (first) {
        first = false;
      } else {
        s << ", ";
      }
      visitLocal(l);
    }
    s << " = ";
    visitExpr(e->bind.rval, indent + 2);
    s << std::endl;
    visitExpr(e->expr, indent);
  }
  void visitPyApply(std::shared_ptr<PyApply> e, int indent) {
    s << getPythonName(e->pyobj.get(), e->is_legacy);
    bool first = true;
    for (auto& scalar : e->scalar_args) {
      if (first) {
        first = false;
        s << " ";
      } else {
        s << ", ";
      }
      printPyObject(scalar);
    }
    for (auto& a : e->tensor_args) {
      if (first) {
        first = false;
        s << " ";
      } else {
        s << ", ";
      }
      visitLocal(a);
    }
    if (e->is_legacy) {
      s << " (legacy)";
    }
  }
  void visitTuple(std::shared_ptr<Tuple> e, int indent) {
    s << std::string(indent, ' ');
    s << "(";
    bool first = true;
    for (auto l : e->locals) {
      if (first) {
        first = false;
      } else {
        s << ", ";
      }
      visitLocal(l);
    }
    s << ")";
  }
};

void printExpr(std::shared_ptr<Expr> e) {
  Printer(std::cout).visitExpr(e, 0);
}

void printExpr(std::shared_ptr<Expr> e, std::ostream& s) {
  Printer(s).visitExpr(e, 0);
}

}}
