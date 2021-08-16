#include <torch/csrc/jit/tensorexpr/ir_printer.h>

#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace tensorexpr {

std::string IRPrinter::dtypeToCppString(const Dtype& dtype) {
  return dtype.ToCppString();
}

void IRPrinter::print(ExprHandle expr) {
  expr.node()->accept(this);
}

void IRPrinter::print(Expr& expr) {
  expr.accept(this);
}

void IRPrinter::print(Stmt& stmt) {
  stmt.accept(this);
}

// TODO: change whether to include the parenthesis to the parent expression,
// we need to look at the operator precedence to make the output simpler.
template <typename Op>
void visitBinaryOp(
    BinaryOpNode<Op>* v,
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

void IRPrinter::visit(Add* v) {
  visitBinaryOp(v, "+", this);
}

void IRPrinter::visit(Sub* v) {
  visitBinaryOp(v, "-", this);
}

void IRPrinter::visit(Mul* v) {
  visitBinaryOp(v, "*", this);
}

void IRPrinter::visit(Div* v) {
  visitBinaryOp(v, "/", this);
}

void IRPrinter::visit(And* v) {
  visitBinaryOp(v, "&", this);
}

void IRPrinter::visit(Or* v) {
  visitBinaryOp(v, "|", this);
}

void IRPrinter::visit(Xor* v) {
  visitBinaryOp(v, "^", this);
}

void IRPrinter::visit(Lshift* v) {
  visitBinaryOp(v, "<<", this);
}

void IRPrinter::visit(Rshift* v) {
  visitBinaryOp(v, ">>", this);
}

void IRPrinter::visit(Mod* v) {
  if (v->dtype().is_integral()) {
    visitBinaryOp(v, "%", this);
  } else if (v->dtype().is_floating_point()) {
    os() << "mod(" << *v->lhs() << ", " << *v->rhs() << ")";
  } else {
    throw std::runtime_error("invalid dtype: " + std::to_string(v->dtype()));
  }
}

void IRPrinter::visit(Max* v) {
  os() << "Max(";
  v->lhs()->accept(this);
  os() << ", ";
  v->rhs()->accept(this);
  os() << ", " << (unsigned int)v->propagate_nans() << ")";
}

void IRPrinter::visit(Min* v) {
  os() << "Min(";
  v->lhs()->accept(this);
  os() << ", ";
  v->rhs()->accept(this);
  os() << ", " << (unsigned int)v->propagate_nans() << ")";
}

void IRPrinter::visit(CompareSelect* v) {
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

  auto withParens = [&](Expr* e) {
    auto prec = getPrecedence(e->expr_type());
    if (prec >= self_prec) {
      os() << "(";
    }
    e->accept(this);
    if (prec >= self_prec) {
      os() << ")";
    }
  };
  withParens(v->ret_val1());
  os() << " : ";
  withParens(v->ret_val2());
}

static void formatFPSuffix(std::ostream& os, double v) {
  os << (v == std::ceil(v) ? ".0" : "");
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
  os << +v;
}

// NOLINTNEXTLINE
#define IMM_PRINT_VISIT(Type, Name)           \
  void IRPrinter::visit(const Name##Imm* v) { \
    formatImm(os(), v->value());              \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_PRINT_VISIT);
#undef IMM_PRINT_VISIT

void IRPrinter::visit(Cast* v) {
  auto dtype = v->dtype();
  os() << dtypeToCppString(dtype) << "(";
  v->src_value()->accept(this);
  os() << ")";
}

void IRPrinter::visit(Var* v) {
  os() << name_manager_.get_unique_name(v);
}

void IRPrinter::visit(Ramp* v) {
  os() << "Ramp(" << *v->base() << ", " << *v->stride() << ", " << v->lanes()
       << ")";
}

void IRPrinter::visit(Load* v) {
  // TODO: support the mask case
  if (v->indices().size() == 0) {
    os() << *v->base_handle();
  } else {
    os() << *v->base_handle() << "[";
    size_t i = 0;
    for (Expr* ind : v->indices()) {
      if (i++) {
        os() << ", ";
      }
      ind->accept(this);
    }
    if (v->indices().empty()) {
      os() << "0";
    }
    os() << "]";
  }
}

void IRPrinter::visit(Broadcast* v) {
  os() << "Broadcast(" << *v->value() << ", " << v->lanes() << ")";
}

void IRPrinter::visit(IfThenElse* v) {
  os() << "IfThenElse(" << *v->condition() << ", " << *v->true_value() << ", "
       << *v->false_value() << ")";
}

void IRPrinter::visit(Intrinsics* v) {
  os() << v->func_name() << "(";
  for (auto i : c10::irange(v->nparams())) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->param(i);
  }
  os() << ")";
}

void IRPrinter::visit(Term* v) {
  os() << "Term(";
  v->scalar()->accept(this);
  for (auto* t : v->variables()) {
    os() << ",";
    t->accept(this);
  }
  os() << ")";
}

void IRPrinter::visit(Polynomial* v) {
  bool first = true;
  os() << "Polynomial(";
  for (auto* t : v->variables()) {
    if (!first) {
      os() << " + ";
    }
    first = false;
    t->accept(this);
  }

  if (!first) {
    os() << " + ";
  }
  v->scalar()->accept(this);
  os() << ")";
}

void IRPrinter::visit(RoundOff* v) {
  os() << "RoundOff(";
  v->lhs()->accept(this);
  os() << ", ";
  v->rhs()->accept(this);
  os() << ")";
}

void IRPrinter::visit(MaxTerm* v) {
  os() << "MaxTerm(";
  if (v->scalar()) {
    v->scalar()->accept(this);
    os() << ", ";
  }
  for (size_t i = 0; i < v->variables().size(); ++i) {
    v->variables()[i]->accept(this);
    if (i < v->variables().size() - 1) {
      os() << ", ";
    }
  }
  os() << ")";
}

void IRPrinter::visit(MinTerm* v) {
  os() << "MinTerm(";
  if (v->scalar()) {
    v->scalar()->accept(this);
    os() << ", ";
  }
  for (size_t i = 0; i < v->variables().size(); ++i) {
    v->variables()[i]->accept(this);
    if (i < v->variables().size() - 1) {
      os() << ", ";
    }
  }
  os() << ")";
}

void IRPrinter::visit(ReduceOp* v) {
  os() << "ReduceOp(";
  os() << *v->body() << ", ";

  bool first = true;
  os() << "reduce_args={";
  for (auto* d : v->reduce_args()) {
    if (!first) {
      os() << ", ";
    }
    os() << d->name_hint();
    first = false;
  }
  os() << "})";
}

// === Stmt visitors below ===

// Newlines and indentation are handled solely by the `Block` printer.  For
// each statement in a `Block` the printer will insert indentation before
// the statement and a newline after the statement.

void IRPrinter::visit(Store* v) {
  // TODO: handle the mask
  if (v->indices().size() == 0) {
    os() << *v->base_handle() << " = " << *v->value() << ";";
    return;
  }

  os() << *v->base_handle() << "[";
  size_t i = 0;
  for (Expr* ind : v->indices()) {
    if (i++) {
      os() << ", ";
    }
    ind->accept(this);
  }
  if (v->indices().empty()) {
    os() << "0";
  }
  os() << "] = " << *v->value() << ";";
}

void IRPrinter::visit(For* v) {
  Var* var = v->var();
  VarHandle vv(var);
  os() << "for (" << dtypeToCppString(var->dtype()) << " " << vv << " = "
       << ExprHandle(v->start()) << "; " << vv << " < " << ExprHandle(v->stop())
       << "; " << vv << "++) ";
  std::string loop_options_str = v->loop_options().ToString();
  if (!loop_options_str.empty()) {
    os() << " /* " << loop_options_str << " */";
  }
  if (v->body()) {
    os() << *v->body();
  } else {
    os() << "{}";
  }
}

void IRPrinter::visit(Block* v) {
  os() << "{\n";
  indent_++;

  for (Stmt* s : *v) {
    emitIndent();
    os() << *s << "\n";
  }
  indent_--;
  emitIndent();
  os() << "}";
}

void IRPrinter::visit(Allocate* v) {
  os() << "Allocate(" << *v->buffer_var()
       << "); // dtype=" << dtypeToCppString(v->dtype());
  os() << ", dims=[";
  const std::vector<Expr*>& dims = v->dims();
  for (auto i : c10::irange(dims.size())) {
    if (i != 0) {
      os() << ", ";
    }
    os() << *dims[i];
  }
  os() << "]";
}

void IRPrinter::visit(Free* v) {
  os() << "Free(" << *v->buffer_var() << ");";
}

void IRPrinter::visit(Let* v) {
  os() << dtypeToCppString(v->dtype()) << " " << *v->var();
  os() << " = " << *v->value();
  os() << ";";
}

void IRPrinter::visit(Cond* v) {
  Expr* cond = v->condition();
  Stmt* true_stmt = v->true_stmt();
  Stmt* false_stmt = v->false_stmt();
  if (!true_stmt) {
    os() << "if (!" << *cond << ") ";
    os() << *false_stmt;
  } else {
    os() << "if (" << *cond << ") ";
    os() << *true_stmt;
    if (false_stmt) {
      os() << " else ";
      os() << *false_stmt;
    }
  }
}

void IRPrinter::visit(AtomicAdd* v) {
  os() << "atomicAdd(&" << *v->base_handle() << "[";
  size_t i = 0;
  for (Expr* ind : v->indices()) {
    if (i++) {
      os() << ", ";
    }
    ind->accept(this);
  }
  if (v->indices().empty()) {
    os() << "0";
  }
  os() << "], " << *v->value() << ");";
}

void IRPrinter::visit(SyncThreads* v) {
  os() << "__syncthreads();";
}

void IRPrinter::visit(ExternalCall* v) {
  os() << *v->buf() << " = " << v->func_name() << "(";

  os() << "buf_args={";
  int i = 0;
  for (Buf* buf_arg : v->buf_args()) {
    if (i++ > 0) {
      os() << ", ";
    }
    os() << *buf_arg;
  }

  os() << "}, args={";
  i = 0;
  for (Expr* arg : v->args()) {
    if (i++ > 0) {
      os() << ", ";
    }
    os() << *arg;
  }
  os() << "})";
}

void IRPrinter::emitIndent() {
  os() << std::setw(2 * indent_) << "";
}

std::ostream& operator<<(std::ostream& stream, const ExprHandle& expr) {
  IRPrinter::PrinterStream* printer_stream =
      dynamic_cast<IRPrinter::PrinterStream*>(&stream);
  ExprHandle& mutable_expr = const_cast<ExprHandle&>(expr);
  if (printer_stream != nullptr) {
    mutable_expr.node()->accept(printer_stream->printer());
  } else {
    IRPrinter p(stream);
    p.print(mutable_expr);
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Expr& expr) {
  IRPrinter::PrinterStream* printer_stream =
      dynamic_cast<IRPrinter::PrinterStream*>(&stream);
  Expr& mutable_expr = const_cast<Expr&>(expr);
  if (printer_stream != nullptr) {
    mutable_expr.accept(printer_stream->printer());
  } else {
    IRPrinter p(stream);
    p.print(mutable_expr);
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Stmt& stmt) {
  IRPrinter::PrinterStream* printer_stream =
      dynamic_cast<IRPrinter::PrinterStream*>(&stream);
  Stmt& mutable_stmt = const_cast<Stmt&>(stmt);
  if (printer_stream != nullptr) {
    mutable_stmt.accept(printer_stream->printer());
  } else {
    IRPrinter p(stream);
    p.print(mutable_stmt);
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Tensor& t) {
  stream << std::to_string(&t);
  return stream;
}

void print(const Expr* expr) {
  if (expr) {
    Expr* mutable_expr = const_cast<Expr*>(expr);
    IRPrinter p(std::cout);
    p.print(*mutable_expr);
  } else {
    std::cout << "(null expr)";
  }
  std::cout << "\n";
}

void print(const Stmt* stmt) {
  if (stmt) {
    Stmt* mutable_stmt = const_cast<Stmt*>(stmt);
    IRPrinter p(std::cout);
    p.print(*mutable_stmt);
  } else {
    std::cout << "(null stmt)\n";
  }
}

void print(const Tensor* t) {
  std::cout << std::to_string(t);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace std {
std::string to_string(const Expr* expr) {
  std::ostringstream oss;
  oss << *expr;
  return oss.str();
}

std::string to_string(const Stmt* stmt) {
  std::ostringstream oss;
  oss << *stmt;
  return oss.str();
}

std::string to_string(const Tensor* t) {
  if (!t) {
    return "(null tensor)\n";
  }
  std::ostringstream oss;
  // TODO: move this to Buf printer
  oss << "Tensor " << t->buf()->name_hint() << "[";
  for (auto i : c10::irange(t->buf()->ndim())) {
    if (i != 0) {
      oss << ", ";
    }
    oss << *t->buf()->dim(i);
  }
  oss << "]:\n" << *t->stmt() << "\n";
  return oss.str();
}
} // namespace std
