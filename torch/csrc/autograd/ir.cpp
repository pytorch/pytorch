#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <iostream>

namespace torch { namespace autograd {

std::string getOperatorName(const Operator& o) {
  switch (o._id) {
    case Operator::Id::PythonOp: return "PythonOp";
  }
  __builtin_unreachable();
}

std::string getExprName(const Expr& expr) {
  switch (expr._id) {
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
    return THPUtils_unpackString(name.get());
  }
}

// TODO: proper pretty-printer

class Printer : public ExprVisitor<Printer>, public OperatorVisitor<Printer> {
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

  // Operator
  void visitPythonOp(std::shared_ptr<PythonOp> e) {
    s << getPythonName(e->pyobj.get(), e->is_legacy);
    if (e->is_legacy) {
      s << " (legacy)";
    }
    for (auto& scalar : e->scalar_args) {
      s << " ";
      printPyObject(scalar);
    }
  }

  // Instruction
  void visitInstruction(std::shared_ptr<Instruction> i) {
    visitOperator(i->op);
    bool first = true;
    for (auto& l : i->args) {
      if (first) {
        s << " ";
      } else {
        s << ", ";
      }
      visitLocal(l);
    }
  }

  // Expr
  void visitLet(std::shared_ptr<Let> e) {
    bool first = true;
    for (auto l : e->bind.lvals) {
      if (first) {
        first = false;
      } else {
        s << ", ";
      }
      visitLocal(l);
    }
    s << " = ";
    visitInstruction(e->bind.rval);
    s << std::endl;
    visitExpr(e->expr);
  }
  void visitTuple(std::shared_ptr<Tuple> e) {
    s << "ret (";
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

  // Graph
  void visitGraph(std::shared_ptr<Graph> g) {
    s << "graph";
    bool first = true;
    for (auto& l : g->params) {
      if (first) {
        s << " ";
      } else {
        s << ", ";
      }
      visitLocal(l);
    }
    s << " {" << std::endl;
    visitExpr(g->body);
    s << std::endl;
    s << "}";
  }
};

void printExpr(std::shared_ptr<Expr> e) {
  Printer(std::cout).visitExpr(e);
}

void printExpr(std::shared_ptr<Expr> e, std::ostream& s) {
  Printer(s).visitExpr(e);
}

void printGraph(std::shared_ptr<Graph> e, std::ostream& s) {
  Printer(s).visitGraph(e);
}

}}
