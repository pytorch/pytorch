#include "torch/csrc/jit/tensorexpr/ir_printer.h"

namespace torch {
namespace jit {
namespace tensorexpr {

void IRPrinter::print(ExprHandle expr) {
  expr.node()->accept(this);
}

void IRPrinter::print(const Expr& expr) {
  expr.accept(this);
}

void IRPrinter::print(const Stmt& stmt) {
  stmt.accept(this);
}

// TODO: change whether to include the parenthesis to the parent expression,
// we need to look at the operator precedence to make the output simpler.
template <typename Op>
void visitBinaryOp(
    const BinaryOpNode<Op>* v,
    const std::string& op_str,
    IRPrinter* printer) {
  std::ostream& os = printer->os();
  os << "(";
  v->lhs()->accept(printer);
  os << " " << op_str << " ";
  v->rhs()->accept(printer);
  os << ")";
}

void IRPrinter::visit(const Add* v) {
  visitBinaryOp(v, "+", this);
}

void IRPrinter::visit(const Sub* v) {
  visitBinaryOp(v, "-", this);
}

void IRPrinter::visit(const Mul* v) {
  visitBinaryOp(v, "*", this);
}

void IRPrinter::visit(const Div* v) {
  visitBinaryOp(v, "/", this);
}

void IRPrinter::visit(const And* v) {
  visitBinaryOp(v, "&", this);
}

void IRPrinter::visit(const Xor* v) {
  visitBinaryOp(v, "^", this);
}

void IRPrinter::visit(const Lshift* v) {
  visitBinaryOp(v, "<<", this);
}

void IRPrinter::visit(const Rshift* v) {
  visitBinaryOp(v, ">>", this);
}

void IRPrinter::visit(const Mod* v) {
  if (v->dtype() == kInt32) {
    visitBinaryOp(v, "%", this);
  } else if (v->dtype() == kFloat32) {
    os() << "mod(" << v->lhs() << ", " << v->rhs() << ")";
  } else {
    throw std::runtime_error("invalid dtype: " + std::to_string(v->dtype()));
  }
}

void IRPrinter::visit(const Max* v) {
  os() << "Max(";
  v->lhs()->accept(this);
  os() << ", ";
  v->rhs()->accept(this);
  os() << ", " << (unsigned int)v->propagate_nans() << ")";
}

void IRPrinter::visit(const Min* v) {
  os() << "Min(";
  v->lhs()->accept(this);
  os() << ", ";
  v->rhs()->accept(this);
  os() << ", " << (unsigned int)v->propagate_nans() << ")";
}

void IRPrinter::visit(const CompareSelect* v) {
  CompareSelectOperation cmp_op = v->compare_select_op();
  os() << "(";
  v->lhs()->accept(this);
  switch (cmp_op) {
    case CompareSelectOperation::kEQ:
      os() << "==";
      break;
    case CompareSelectOperation::kNE:
      os() << "!=";
      break;
    case CompareSelectOperation::kGT:
      os() << ">";
      break;
    case CompareSelectOperation::kGE:
      os() << ">=";
      break;
    case CompareSelectOperation::kLT:
      os() << "<";
      break;
    case CompareSelectOperation::kLE:
      os() << "<=";
      break;
    default:
      throw std::runtime_error("invalid compare select operator");
  }
  v->rhs()->accept(this);
  os() << ")";
}

void IRPrinter::visit(const IntImm* v) {
  os() << v->value();
}

void IRPrinter::visit(const FloatImm* v) {
  std::ostringstream oss;
  oss << v->value();
  std::string s = oss.str();
  if (s.find('.') == std::string::npos) {
    s += ".f";
  } else {
    s += "f";
  }
  os() << s;
}

void IRPrinter::visit(const Cast* v) {
  auto dtype = v->dtype();
  os() << dtype << "(";
  v->src_value()->accept(this);
  os() << ")";
}

void IRPrinter::visit(const Var* v) {
  os() << name_manager_.get_unique_name(v);
}

void IRPrinter::visit(const Let* v) {
  os() << "(let ";
  v->var()->accept(this);
  os() << " = ";
  v->value()->accept(this);
  os() << " in ";
  v->body()->accept(this);
  os() << ")";
}

void IRPrinter::visit(const LetStmt* v) {
  const Var* var = v->var();
  os() << var->dtype().ToCppString() << " " << *var << " = " << *v->value() << "; "
       << std::endl;
  v->body()->accept(this);
}

void IRPrinter::visit(const Ramp* v) {
  os() << "Ramp(" << v->base() << ", " << v->stride() << ", " << v->lanes()
       << ")";
}

void IRPrinter::visit(const Load* v) {
  // TODO: support the mask case
  os() << *v->base_handle() << "[" << *v->index() << "]";
}

void IRPrinter::visit(const For* v) {
  const Var* var = v->var();
  VarHandle vv(var);
  os() << "for (" << var->dtype().ToCppString() << " " << vv << " = "
       << ExprHandle(v->start()) << "; " << vv << " < " << ExprHandle(v->stop()) << "; " << vv
       << "++) {";
  std::string loop_options_str = v->loop_options().ToString();
  if (!loop_options_str.empty()) {
    os() << " // " << loop_options_str;
  }
  os() << std::endl;
  if (v->body()) {
    os() << *v->body() << std::endl;
  }
  os() << "}";
}

void IRPrinter::visit(const Block* v) {
  for (int i = 0; i < v->nstmts(); ++i) {
    os() << *v->stmt(i) << std::endl;
  }
}

void IRPrinter::visit(const Store* v) {
  // TODO: handle the mask
  os() << *v->base_handle() << "[" << *v->index() << "] = " << *v->value() << ";";
}

void IRPrinter::visit(const Broadcast* v) {
  os() << "Broadcast(" << v->value() << ", " << v->lanes() << ")";
}

void IRPrinter::visit(const IfThenElse* v) {
  os() << "IfThenElse(" << *v->condition() << ", " << *v->true_value() << ", "
       << *v->false_value() << ")";
}

void IRPrinter::visit(const BaseCallNode* v) {
  os() << v->func_name() << "(";
  for (int i = 0; i < v->nparams(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->param(i);
  }
  os() << ")";
}

void IRPrinter::visit(const Allocate* v) {
  os() << "Allocate(" << *v->buffer_var() << ", " << v->dtype();
  os() << ", {";
  const std::vector<const Expr*>& dims = v->dims();
  for (size_t i = 0; i < dims.size(); i++) {
    if (i != 0) {
      os() << ", ";
    }
    os() << *dims[i];
  }
  os() << "});";
}

void IRPrinter::visit(const Free* v) {
  os() << "Free(" << *v->buffer_var() << ");";
}

void IRPrinter::visit(const Cond* v) {
  const Expr* cond = v->condition();
  Stmt* true_stmt = v->true_stmt();
  Stmt* false_stmt = v->false_stmt();
  if (!true_stmt) {
    os() << "if(!" << *cond << ") {" << std::endl;
    os() << *false_stmt << std::endl;
    os() << "}";
  } else {
    os() << "if(" << *cond << ") {" << std::endl;
    os() << *true_stmt << std::endl;
    os() << "}";
    if (false_stmt) {
      os() << " else {" << std::endl;
      os() << *false_stmt << std::endl;
      os() << "}";
    }
  }
}

std::ostream& operator<<(std::ostream& stream, const ExprHandle& expr) {
  IRPrinter::PrinterStream* printer_stream =
      dynamic_cast<IRPrinter::PrinterStream*>(&stream);
  if (printer_stream != nullptr) {
    expr.node()->accept(printer_stream->printer());
  } else {
    IRPrinter p(stream);
    p.print(expr);
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Expr& expr) {
  IRPrinter::PrinterStream* printer_stream =
      dynamic_cast<IRPrinter::PrinterStream*>(&stream);
  if (printer_stream != nullptr) {
    expr.accept(printer_stream->printer());
  } else {
    IRPrinter p(stream);
    p.print(expr);
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Stmt& stmt) {
  IRPrinter::PrinterStream* printer_stream =
      dynamic_cast<IRPrinter::PrinterStream*>(&stream);
  if (printer_stream != nullptr) {
    stmt.accept(printer_stream->printer());
  } else {
    IRPrinter p(stream);
    p.print(stmt);
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, Stmt* stmt) {
  IRPrinter::PrinterStream* printer_stream =
      dynamic_cast<IRPrinter::PrinterStream*>(&stream);
  if (printer_stream != nullptr) {
    stmt->accept(printer_stream->printer());
  } else {
    IRPrinter p(stream);
    p.print(*stmt);
  }
  return stream;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
