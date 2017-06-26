#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <iostream>

namespace torch { namespace autograd {

std::string getArgName(Arg* arg) {
  switch (arg->_id) {
    case Arg::Id::Local: return "Local";
    case Arg::Id::PyConst: return "PyConst";
  }
  __builtin_unreachable();
}

std::string getExprName(Expr* expr) {
  switch (expr->_id) {
    case Expr::Id::PyApply: return "PyApply";
    case Expr::Id::Let: return "Let";
    case Expr::Id::Locals: return "Locals";
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

class Printer : public ExprVisitor<Printer>, public ArgVisitor<Printer> {
  std::ostream& s;

public:
  Printer(std::ostream& s) : s(s) {}

  // Arg
  void visitLocal(std::shared_ptr<Local> a) {
    s << "%" << a->unique;
  }
  void visitPyConst(std::shared_ptr<PyConst> a) {
    THPObjectPtr repr { PyObject_Repr(a->pyobj.get()) };
    s << THPUtils_unpackString(repr.get());
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
    for (auto a : e->args) {
      if (first) {
        first = false;
        s << " ";
      } else {
        s << ", ";
      }
      visitArg(a);
    }
    if (e->is_legacy) {
      s << " (legacy)";
    }
  }
  void visitLocals(std::shared_ptr<Locals> e, int indent) {
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
