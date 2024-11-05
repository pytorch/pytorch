#include <torch/csrc/jit/tensorexpr/ir_printer.h>

#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <c10/util/irange.h>

#include <iostream>

namespace torch::jit::tensorexpr {

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
std::string IRPrinter::to_string(CompareSelectOperation op) {
  switch (op) {
    case CompareSelectOperation::kEQ:
      return "==";
    case CompareSelectOperation::kNE:
      return "!=";
    case CompareSelectOperation::kGT:
      return ">";
    case CompareSelectOperation::kGE:
      return ">=";
    case CompareSelectOperation::kLT:
      return "<";
    case CompareSelectOperation::kLE:
      return "<=";
    default:
      throw std::runtime_error("invalid compare select operator");
  }
}

void IRPrinter::PrinterStream::initialize_imbue() {
  // Similar to https://github.com/pytorch/pytorch/issues/79583:
  // global locale can be set to something other than "C", which can add
  // extra commas in the printed numbers.
  static std::locale c_locale("C");
  // note: IRPrinter is a subclass of ostream, so imbue is a member function.
  imbue(c_locale);
}

// TODO: change whether to include the parenthesis to the parent expression,
// we need to look at the operator precedence to make the output simpler.
template <
    typename Op,
    std::enable_if_t<std::is_same_v<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>>* = nullptr>
void visitBinaryOp(
    NodePtr<Op> v,
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

void IRPrinter::visit(const AddPtr& v) {
  visitBinaryOp(v, "+", this);
}

void IRPrinter::visit(const SubPtr& v) {
  visitBinaryOp(v, "-", this);
}

void IRPrinter::visit(const MulPtr& v) {
  visitBinaryOp(v, "*", this);
}

void IRPrinter::visit(const DivPtr& v) {
  visitBinaryOp(v, "/", this);
}

void IRPrinter::visit(const AndPtr& v) {
  visitBinaryOp(v, "&", this);
}

void IRPrinter::visit(const OrPtr& v) {
  visitBinaryOp(v, "|", this);
}

void IRPrinter::visit(const XorPtr& v) {
  visitBinaryOp(v, "^", this);
}

void IRPrinter::visit(const LshiftPtr& v) {
  visitBinaryOp(v, "<<", this);
}

void IRPrinter::visit(const RshiftPtr& v) {
  visitBinaryOp(v, ">>", this);
}

void IRPrinter::visit(const ModPtr& v) {
  if (v->dtype().is_integral()) {
    visitBinaryOp(v, "%", this);
  } else if (v->dtype().is_floating_point()) {
    os() << "mod(" << *v->lhs() << ", " << *v->rhs() << ")";
  } else {
    throw std::runtime_error("invalid dtype: " + std::to_string(v->dtype()));
  }
}

void IRPrinter::visit(const MaxPtr& v) {
  os() << "Max(";
  v->lhs()->accept(this);
  os() << ", ";
  v->rhs()->accept(this);
  os() << ", " << (unsigned int)v->propagate_nans() << ")";
}

void IRPrinter::visit(const MinPtr& v) {
  os() << "Min(";
  v->lhs()->accept(this);
  os() << ", ";
  v->rhs()->accept(this);
  os() << ", " << (unsigned int)v->propagate_nans() << ")";
}

void IRPrinter::visit(const CompareSelectPtr& v) {
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

  os() << to_string(cmp_op);

  if (rhs_prec >= self_prec) {
    os() << "(";
  }
  v->rhs()->accept(this);
  if (rhs_prec >= self_prec) {
    os() << ")";
  }
  os() << " ? ";

  auto withParens = [&](const ExprPtr& e) {
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

template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
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

static void formatIntSuffix(std::ostream& os, int64_t v) {
  os << "ll";
}

template <typename T>
static void formatIntSuffix(std::ostream& os, T v) {}

template <typename T, std::enable_if_t<!std::is_floating_point_v<T>>* = nullptr>
static void formatImm(std::ostream& os, T v) {
  os << +v;
  formatIntSuffix(os, v);
}

// NOLINTNEXTLINE
#define IMM_PRINT_VISIT(Type, Name)              \
  void IRPrinter::visit(const Name##ImmPtr& v) { \
    formatImm(os(), v->value());                 \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_PRINT_VISIT)
#undef IMM_PRINT_VISIT

void IRPrinter::visit(const CastPtr& v) {
  auto dtype = v->dtype();
  os() << dtypeToCppString(dtype) << "(";
  v->src_value()->accept(this);
  os() << ")";
}

void IRPrinter::visit(const BitCastPtr& v) {
  auto dtype = v->dtype();
  os() << "BitCast<" << dtype.ToCppString() << ">(";
  v->src_value()->accept(this);
  os() << ")";
}

void IRPrinter::visit(const VarPtr& v) {
  os() << name_manager_.get_unique_name(v);
}

void IRPrinter::visit(const BufPtr& v) {
  auto dtype = v->dtype();
  os() << *v->base_handle();
  os() << "(dtype=" << dtypeToCppString(dtype);
  if (v->qscale()) {
    os() << ", qscale=";
    v->qscale()->accept(this);
  }
  if (v->qscale()) {
    os() << ", qzero=";
    v->qzero()->accept(this);
  }
  os() << ", sizes=[";
  size_t i = 0;
  for (const ExprPtr& s : v->dims()) {
    if (i++) {
      os() << ", ";
    }
    s->accept(this);
  }
  os() << "]";
  os() << ", strides=[";
  i = 0;
  for (const ExprPtr& s : v->strides()) {
    if (i++) {
      os() << ", ";
    }
    s->accept(this);
  }
  os() << "]";

  os() << ")";
}

void IRPrinter::visit(const RampPtr& v) {
  os() << "Ramp(" << *v->base() << ", " << *v->stride() << ", " << v->lanes()
       << ")";
}

void IRPrinter::visit(const LoadPtr& v) {
  // TODO: support the mask case
  if (v->indices().empty()) {
    os() << *v->base_handle();
  } else {
    os() << *v->base_handle() << "[";
    size_t i = 0;
    for (const ExprPtr& ind : v->indices()) {
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

void IRPrinter::visit(const BroadcastPtr& v) {
  os() << "Broadcast(" << *v->value() << ", " << v->lanes() << ")";
}

void IRPrinter::visit(const IfThenElsePtr& v) {
  os() << "IfThenElse(" << *v->condition() << ", " << *v->true_value() << ", "
       << *v->false_value() << ")";
}

void IRPrinter::visit(const IntrinsicsPtr& v) {
  os() << v->func_name() << "(";
  for (const auto i : c10::irange(v->nparams())) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->param(i);
  }
  os() << ")";
}

void IRPrinter::visit(const TermPtr& v) {
  os() << "Term(";
  v->scalar()->accept(this);
  for (const auto& t : v->variables()) {
    os() << ",";
    t->accept(this);
  }
  os() << ")";
}

void IRPrinter::visit(const PolynomialPtr& v) {
  bool first = true;
  os() << "Polynomial(";
  for (const auto& t : v->variables()) {
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

void IRPrinter::visit(const RoundOffPtr& v) {
  os() << "RoundOff(";
  v->lhs()->accept(this);
  os() << ", ";
  v->rhs()->accept(this);
  os() << ")";
}

void IRPrinter::visit(const MaxTermPtr& v) {
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

void IRPrinter::visit(const MinTermPtr& v) {
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

void IRPrinter::visit(const ReduceOpPtr& v) {
  os() << "ReduceOp(";
  os() << *v->body() << ", ";

  bool first = true;
  os() << "reduce_args={";
  for (const auto& d : v->reduce_args()) {
    if (!first) {
      os() << ", ";
    }
    os() << *d;
    first = false;
  }
  os() << "})";
}

// === Stmt visitors below ===

// Newlines and indentation are handled solely by the `Block` printer.  For
// each statement in a `Block` the printer will insert indentation before
// the statement and a newline after the statement.

void IRPrinter::visit(const StorePtr& v) {
  // TODO: handle the mask
  if (v->indices().empty()) {
    os() << *v->base_handle() << " = " << *v->value() << ";";
    return;
  }

  os() << *v->base_handle() << "[";
  size_t i = 0;
  for (const ExprPtr& ind : v->indices()) {
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

void IRPrinter::visit(const ForPtr& v) {
  VarPtr var = v->var();
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

void IRPrinter::visit(const BlockPtr& v) {
  os() << "{\n";
  indent_++;

  for (const StmtPtr& s : *v) {
    emitIndent();
    os() << *s << "\n";
  }
  indent_--;
  emitIndent();
  os() << "}";
}

void IRPrinter::visit(const AllocatePtr& v) {
  os() << "Allocate(" << *v->buffer_var()
       << "); // dtype=" << dtypeToCppString(v->dtype());
  os() << ", dims=[";
  const std::vector<ExprPtr>& dims = v->dims();
  for (const auto i : c10::irange(dims.size())) {
    if (i != 0) {
      os() << ", ";
    }
    os() << *dims[i];
  }
  os() << "]";
}

void IRPrinter::visit(const FreePtr& v) {
  os() << "Free(" << *v->buffer_var() << ");";
}

void IRPrinter::visit(const FreeExtPtr& v) {
  os() << "FreeExt(bufs={";
  int i = 0;
  for (const auto& buf : v->bufs()) {
    if (i++ > 0) {
      os() << ", ";
    }
    os() << *buf;
  }

  os() << "});";
}

void IRPrinter::visit(const PlacementAllocatePtr& v) {
  os() << "Alias(" << *v->buf()->base_handle() << ","
       << *v->buf_to_reuse()->base_handle() << ");";
}

void IRPrinter::visit(const LetPtr& v) {
  os() << dtypeToCppString(v->var()->dtype()) << " " << *v->var();
  os() << " = " << *v->value() << ";";
}

void IRPrinter::visit(const CondPtr& v) {
  ExprPtr cond = v->condition();
  StmtPtr true_stmt = v->true_stmt();
  StmtPtr false_stmt = v->false_stmt();
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

void IRPrinter::visit(const AtomicAddPtr& v) {
  os() << "atomicAdd(&" << *v->base_handle() << "[";
  size_t i = 0;
  for (const ExprPtr& ind : v->indices()) {
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

void IRPrinter::visit(const SyncThreadsPtr& v) {
  os() << "__syncthreads();";
}

void IRPrinter::visit(const ExternalCallPtr& v) {
  os() << *v->buf() << " = " << v->func_name() << "(";

  os() << "buf_args={";
  int i = 0;
  for (const BufPtr& buf_arg : v->buf_args()) {
    if (i++ > 0) {
      os() << ", ";
    }
    os() << *buf_arg;
  }

  os() << "}, args={";
  i = 0;
  for (const ExprPtr& arg : v->args()) {
    if (i++ > 0) {
      os() << ", ";
    }
    os() << *arg;
  }
  os() << "})";
}

void IRPrinter::visit(const ExternalCallWithAllocPtr& v) {
  int i = 0;
  for (const auto& buf_out_arg : v->buf_out_args()) {
    if (i++ > 0) {
      os() << ", ";
    }
    os() << *buf_out_arg;
  }

  os() << " := " << v->func_name() << "(";

  os() << "buf_args={";
  i = 0;
  for (const auto& buf_arg : v->buf_args()) {
    if (i++ > 0) {
      os() << ", ";
    }
    os() << *buf_arg;
  }

  os() << "}, args={";
  i = 0;
  for (const auto& arg : v->args()) {
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
  stream << std::to_string(t);
  return stream;
}

void print(const ExprPtr& expr) {
  if (expr) {
    IRPrinter p(std::cout);
    p.print(*expr);
  } else {
    std::cout << "(null expr)";
  }
  std::cout << "\n";
}

void print(const StmtPtr& stmt) {
  if (stmt) {
    IRPrinter p(std::cout);
    p.print(*stmt);
  } else {
    std::cout << "(null stmt)\n";
  }
}

void print(const Tensor& t) {
  std::cout << std::to_string(t);
}

} // namespace torch::jit::tensorexpr

namespace std {
std::string to_string(const ExprPtr& expr) {
  std::ostringstream oss;
  oss << *expr;
  return oss.str();
}

std::string to_string(const StmtPtr& stmt) {
  std::ostringstream oss;
  oss << *stmt;
  return oss.str();
}

std::string to_string(const Tensor& t) {
  std::ostringstream oss;
  // TODO: move this to Buf printer
  oss << "Tensor " << t.buf()->name_hint() << "[";
  for (const auto i : c10::irange(t.buf()->ndim())) {
    if (i != 0) {
      oss << ", ";
    }
    oss << *t.buf()->dim(i);
  }
  oss << "]:\n" << *t.stmt() << "\n";
  return oss.str();
}
} // namespace std
