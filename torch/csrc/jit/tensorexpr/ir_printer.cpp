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
    IRPrinter* printer,
    bool parens = true) {
  std::ostream& os = printer->os();
  int self_prec = getPrecedence(v->expr_type());
  int lhs_prec = getPrecedence(v->lhs()->expr_type());
  int rhs_prec = getPrecedence(v->rhs()->expr_type());

  if (lhs_prec >= self_prec) {
    os << "(";
  }
  v->lhs()->accept(printer);
  if (lhs_prec >= self_prec) {
    os << ")";
  }

  os << " " << op_str << " ";

  if (rhs_prec >= self_prec) {
    os << "(";
  }
  v->rhs()->accept(printer);
  if (rhs_prec >= self_prec) {
    os << ")";
  }
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

void IRPrinter::visit(const Or* v) {
  visitBinaryOp(v, "|", this);
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
  if (v->dtype().is_integral()) {
    visitBinaryOp(v, "%", this);
  } else if (v->dtype().is_floating_point()) {
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
  int self_prec = getPrecedence(v->expr_type());
  int lhs_prec = getPrecedence(v->lhs()->expr_type());
  int rhs_prec = getPrecedence(v->rhs()->expr_type());

  if (lhs_prec >= self_prec) {
    os() << "(";
  }
  v->lhs()->accept(this);
  if (lhs_prec >= self_prec) {
    os() << ")";
  }
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

  if (rhs_prec >= self_prec) {
    os() << "(";
  }
  v->rhs()->accept(this);
  if (rhs_prec >= self_prec) {
    os() << ")";
  }
  os() << " ? ";

  auto withParens = [&](const Expr* e) {
    auto prec = getPrecedence(e->expr_type());
    if (prec >= self_prec) {
      os() << "(";
    }
    e->accept(this);
    if (prec >= self_prec) {
      os() << "(";
    }
  };
  withParens(v->ret_val1());
  os() << " : ";
  withParens(v->ret_val2());
}

static void formatFPSuffix(std::ostream& os, double v) {
  // No suffix for doubles.
}

template <typename T>
static void formatFPSuffix(std::ostream& os, T v) {
  os << (v == std::ceil(v) ? ".f" : "f");
}

template <
    typename T,
    std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static void formatImm(std::ostream& os, T v) {
  const int precision = 16;
  if (std::isnan(v)) {
    os << "NAN";
  } else if (std::isinf(v)) {
    os << (v > 0 ? "POS_INFINITY" : "NEG_INFINITY");
  } else {
    os << std::setprecision(precision) << v;
    formatFPSuffix(os, v);
  }
}

template <
    typename T,
    std::enable_if_t<!std::is_floating_point<T>::value>* = nullptr>
static void formatImm(std::ostream& os, T v) {
  os << v;
}

// NOLINTNEXTLINE
#define IMM_PRINT_VISIT(Type, Name)           \
  void IRPrinter::visit(const Name##Imm* v) { \
    formatImm(os(), v->value());              \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_PRINT_VISIT);
#undef IMM_PRINT_VISIT

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
  int self_prec = getPrecedence(v->expr_type());
  int value_prec = getPrecedence(v->value()->expr_type());
  int body_prec = getPrecedence(v->body()->expr_type());
  os() << "let ";
  v->var()->accept(this);
  os() << " = ";

  if (value_prec >= self_prec) {
    os() << "(";
  }
  v->value()->accept(this);
  if (value_prec >= self_prec) {
    os() << ")";
  }

  os() << " in ";

  if (body_prec >= self_prec) {
    os() << "(";
  }
  v->body()->accept(this);
  if (body_prec >= self_prec) {
    os() << ")";
  }
}

void IRPrinter::visit(const LetStmt* v) {
  const Var* var = v->var();
  os() << var->dtype().ToCppString() << " " << *var << " = " << *v->value()
       << "; " << std::endl;
  v->body()->accept(this);
}

void IRPrinter::visit(const Ramp* v) {
  os() << "Ramp(" << *v->base() << ", " << *v->stride() << ", " << v->lanes()
       << ")";
}

void IRPrinter::visit(const Load* v) {
  // TODO: support the mask case
  os() << *v->base_handle() << "[" << *v->index() << "]";
}

void IRPrinter::visit(const For* v) {
  const Var* var = v->var();
  VarHandle vv(var);
  emitIndent();
  os() << "for (" << var->dtype().ToCppString() << " " << vv << " = "
       << ExprHandle(v->start()) << "; " << vv << " < " << ExprHandle(v->stop())
       << "; " << vv << "++) {";
  std::string loop_options_str = v->loop_options().ToString();
  if (!loop_options_str.empty()) {
    os() << " // " << loop_options_str;
  }
  os() << std::endl;
  if (v->body()) {
    indent_++;
    os() << *v->body();
    indent_--;
  }
  emitIndent();
  os() << "}";
}

void IRPrinter::visit(const Block* v) {
  for (Stmt* s : v->stmts()) {
    os() << *s << std::endl;
  }
}

void IRPrinter::visit(const Store* v) {
  // TODO: handle the mask
  emitIndent();
  os() << *v->base_handle() << "[" << *v->index() << "] = " << *v->value()
       << ";";
}

void IRPrinter::visit(const Broadcast* v) {
  os() << "Broadcast(" << *v->value() << ", " << v->lanes() << ")";
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
  emitIndent();
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
  emitIndent();
  os() << "Free(" << *v->buffer_var() << ");";
}

void IRPrinter::visit(const Cond* v) {
  const Expr* cond = v->condition();
  Stmt* true_stmt = v->true_stmt();
  Stmt* false_stmt = v->false_stmt();
  if (!true_stmt) {
    emitIndent();
    os() << "if (!" << *cond << ") {" << std::endl;
    indent_++;
    os() << *false_stmt << std::endl;
    indent_--;
    emitIndent();
    os() << "}";
  } else {
    emitIndent();
    os() << "if (" << *cond << ") {" << std::endl;
    indent_++;
    os() << *true_stmt << std::endl;
    indent_--;
    emitIndent();
    os() << "}";
    if (false_stmt) {
      os() << " else {" << std::endl;
      indent_++;
      os() << *false_stmt << std::endl;
      indent_--;
      emitIndent();
      os() << "}";
    }
  }
}

void IRPrinter::emitIndent() {
  os() << std::setw(2 * indent_) << "";
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

void print(const Expr* expr) {
  if (expr) {
    IRPrinter p(std::cout);
    p.print(*expr);
  } else {
    std::cout << "(null expr)";
  }
}

void print(const Stmt* stmt) {
  if (stmt) {
    IRPrinter p(std::cout);
    p.print(*stmt);
  } else {
    std::cout << "(null stmt)\n";
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
