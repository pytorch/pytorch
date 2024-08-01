#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

#include <torch/csrc/jit/tensorexpr/cpp_codegen.h>
#include <torch/csrc/jit/tensorexpr/cpp_intrinsics.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/types.h>

namespace torch::jit::tensorexpr {

// Rewrites the variables' name according to valid C++ naming convention.
// E.g. in Graph IR, variable name may contain '.', in C++, they are replaced
// with '_'.
class CppVarNameRewriter : public IRVisitor {
 public:
  void visit(const VarPtr& v) override {
    constexpr char kDot = '.';
    constexpr char kUnderscore = '_';
    if (v->name_hint().find(kDot) == std::string::npos) {
      return;
    }
    std::string name = v->name_hint();
    std::replace(name.begin(), name.end(), kDot, kUnderscore);
    v->set_name_hint(std::move(name));
  }

  void visit(const BufPtr& v) override {
    v->base_handle()->accept(this);
  }
};

static std::string declareExternalFunction(const std::string& func_name) {
  return "void " + func_name +
      "("
      "int64_t bufs_num, "
      "void** buf_data, "
      "int64_t* buf_ranks, "
      "int64_t* buf_dims, "
      "int8_t* buf_dtypes, "
      "int64_t args_num, "
      "int64_t* extra_args);";
}

CppPrinter::CppPrinter(std::ostream* os) : IRPrinter(*os), lane_(0) {}

CppPrinter::~CppPrinter() = default;

void CppPrinter::printPrologue() {
  os() << "#include <cassert>" << '\n';
  os() << "#include <cmath>" << '\n';
  os() << "#include <algorithm>" << '\n';
  os() << "#include <type_traits>" << '\n';
  os() << '\n';

  os() << "#define POS_INFINITY INFINITY" << '\n';
  os() << "#define NEG_INFINITY -INFINITY" << '\n';
  os() << '\n';

  os() << cpp_intrinsics_definition << '\n';
  os() << '\n';

  os() << "namespace torch {" << '\n';
  os() << "namespace jit {" << '\n';
  os() << "namespace tensorexpr {" << '\n';
  for (auto const& it : getNNCFunctionRegistry()) {
    os() << declareExternalFunction(it.first) << '\n';
  }
  os() << "} // namespace tensorexpr" << '\n';
  os() << "} // namespace jit" << '\n';
  os() << "} // namespace torch" << '\n';
  os() << '\n';

  os() << "using namespace torch::jit::tensorexpr;" << '\n';
  os() << '\n';
}

template <typename T>
inline std::enable_if_t<!std::is_floating_point_v<T>, void> visit_mod(
    std::ostream& os,
    const ExprPtr lhs,
    const ExprPtr rhs) {
  os << *lhs << " % " << *rhs;
}

template <typename T>
inline std::enable_if_t<std::is_floating_point_v<T>, void> visit_mod(
    std::ostream& os,
    const ExprPtr lhs,
    const ExprPtr rhs) {
  os << "std::fmod(" << *lhs << ", " << *rhs << ")";
}

template <typename T>
inline std::
    enable_if_t<std::is_floating_point_v<T> || std::is_integral_v<T>, void>
    visit_max(std::ostream& os, const ExprPtr lhs, const ExprPtr rhs) {
  os << "std::max(" << *lhs << ", " << *rhs << ")";
}

template <typename T>
inline std::
    enable_if_t<!std::is_floating_point_v<T> && !std::is_integral_v<T>, void>
    visit_max(std::ostream& os, const ExprPtr lhs, const ExprPtr rhs) {
  os << "(" << *lhs << " < " << *rhs << ") ? " << *rhs << " : " << *lhs;
}

template <typename T>
inline std::
    enable_if_t<std::is_floating_point_v<T> || std::is_integral_v<T>, void>
    visit_min(std::ostream& os, const ExprPtr lhs, const ExprPtr rhs) {
  os << "std::min(" << *lhs << ", " << *rhs << ")";
}

template <typename T>
inline std::
    enable_if_t<!std::is_floating_point_v<T> && !std::is_integral_v<T>, void>
    visit_min(std::ostream& os, const ExprPtr lhs, const ExprPtr rhs) {
  os << *lhs << " < " << *rhs << " ? " << *lhs << " : " << *rhs;
}

template <typename T>
void visit_binary_op(
    std::ostream& os,
    const ExprPtr lhs,
    const ExprPtr rhs,
    IRNodeType op_type) {
  switch (op_type) {
    case IRNodeType::kMod:
      visit_mod<T>(os, lhs, rhs);
      break;
    case IRNodeType::kMax:
      visit_max<T>(os, lhs, rhs);
      break;
    case IRNodeType::kMin:
      visit_min<T>(os, lhs, rhs);
      break;
    default:
      throw std::runtime_error("invalid op type");
  }
}

template <typename Op>
void dispatch_binary_op(std::ostream& os, const BinaryOpNode<Op>* v) {
  switch (v->lhs()->dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                      \
  case ScalarType::Name:                                           \
    visit_binary_op<Type>(os, v->lhs(), v->rhs(), v->expr_type()); \
    break;
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw unsupported_dtype();
  }
}

void CppPrinter::visit(const RampPtr& v) {
  visit(alloc<Add>(v->base(), alloc<Mul>(alloc<IntImm>(lane_), v->stride())));
}

void CppPrinter::visit(const BroadcastPtr& v) {
  v->value()->accept(this);
}

void CppPrinter::visit(const ModPtr& v) {
  dispatch_binary_op(os(), v.get());
}

void CppPrinter::visit(const MaxPtr& v) {
  dispatch_binary_op(os(), v.get());
}

void CppPrinter::visit(const MinPtr& v) {
  dispatch_binary_op(os(), v.get());
}

void CppPrinter::visit(const CompareSelectPtr& v) {
  os() << "((" << *v->lhs() << " "
       << IRPrinter::to_string(v->compare_select_op()) << " " << *v->rhs()
       << ") ? " << *v->ret_val1() << " : " << *v->ret_val2() << ")";
}

void CppPrinter::visit(const IfThenElsePtr& v) {
  os() << "((" << *v->condition() << ") ? " << *v->true_value() << " : "
       << *v->false_value() << ")";
}

void CppPrinter::visit(const AllocatePtr& v) {
  size_t size = v->dtype().byte_size();
  for (const auto& dim : v->dims()) {
    IntImmPtr d = to<IntImm>(dim);
    if (d) {
      size *= d->value();
    } else {
      throw std::runtime_error("Only IntImm dimensions are supported for now");
    }
  }

  emitIndent();
  os() << v->dtype().ToCppString() << "* " << (*v->buffer_var())
       << " = static_cast<" << v->dtype().ToCppString() << "*>(malloc(" << size
       << "));" << '\n';
}

void CppPrinter::visit(const FreePtr& v) {
  emitIndent();
  os() << "free(" << *v->buffer_var() << ");" << '\n';
}

void CppPrinter::visit(const LoadPtr& v) {
  auto flat_idx =
      flatten_index(v->buf()->dims(), v->indices(), v->buf()->strides());
  os() << *v->base_handle() << "[" << *flat_idx << "]";
}

void CppPrinter::visit(const StorePtr& v) {
  auto flat_idx =
      flatten_index(v->buf()->dims(), v->indices(), v->buf()->strides());
  const int lanes = v->value()->dtype().lanes();
  for (int lane = 0; lane < lanes; lane++) {
    lane_ = lane;
    emitIndent();
    os() << *v->base_handle() << "[" << *flat_idx << "] = " << *v->value()
         << ";" << '\n';
  }
}

void CppPrinter::visit(const CastPtr& v) {
  os() << "static_cast<" << v->dtype().ToCppString() << ">(" << *v->src_value()
       << ")";
}

void CppPrinter::visit(const BitCastPtr& v) {
  os() << "std::bitcast<" << v->src_value()->dtype().ToCppString() << ", "
       << v->dtype().ToCppString() << ">(" << *v->src_value() << ")";
}

void CppPrinter::visit(const IntrinsicsPtr& v) {
  if (v->op_type() == kRand || v->op_type() == kSigmoid) {
    throw std::runtime_error("kRand and kSigmoid are not supported");
  }

  os() << "std::" << v->func_name() << "(";
  for (int i = 0; i < v->nparams(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->param(i);
  }
  os() << ")";
}

void CppPrinter::visit(const ExternalCallPtr& v) {
  // The generated code needs to link against functions defined
  // in external_functions.cpp.

  auto& func_registry = getNNCFunctionRegistry();
  if (!func_registry.count(v->func_name())) {
    throw unimplemented_lowering(v);
  }

  std::vector<BufPtr> bufs(v->buf_args());
  bufs.insert(bufs.begin(), v->buf());
  auto for_buf = [&](const std::function<void(const BufPtr)>& print_buf) {
    for (size_t i = 0; i < bufs.size(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      print_buf(bufs[i]);
    }
  };

  emitIndent();
  os() << "{" << '\n';
  indent_++;

  emitIndent();
  os() << "void* buf_ptrs[]{";
  for_buf([&](const BufPtr& b) { os() << *b->base_handle(); });
  os() << "};" << '\n';

  emitIndent();
  os() << "int64_t buf_ranks[]{";
  for_buf([&](const BufPtr& b) { os() << b->ndim(); });
  os() << "};" << '\n';

  emitIndent();
  os() << "int64_t buf_dims[]{";
  for_buf([&](const BufPtr& buf) {
    for (size_t i = 0; i < buf->ndim(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      os() << *buf->dim(i);
    }
  });
  os() << "};" << '\n';

  emitIndent();
  os() << "int8_t buf_dtypes[]{";
  for_buf([&](const BufPtr& buf) {
    os() << static_cast<int>(buf->dtype().scalar_type());
  });
  os() << "};" << '\n';

  emitIndent();
  os() << "int64_t extra_args[]{";
  for (size_t i = 0; i < v->args().size(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->args()[i];
  }
  os() << "};" << '\n';

  emitIndent();
  os() << v->func_name() << "(" << '\n';
  emitIndent();
  os() << "    " << bufs.size() << "," << '\n';
  emitIndent();
  os() << "    buf_ptrs," << '\n';
  emitIndent();
  os() << "    buf_ranks," << '\n';
  emitIndent();
  os() << "    buf_dims," << '\n';
  emitIndent();
  os() << "    buf_dtypes," << '\n';
  emitIndent();
  os() << "    " << v->args().size() << "," << '\n';
  emitIndent();
  os() << "    extra_args);" << '\n';

  indent_--;
  emitIndent();
  os() << "}" << '\n';
}

void CppPrinter::visit(const LetPtr& v) {
  if (v->var()->dtype().lanes() == 1) {
    emitIndent();
    os() << v->var()->dtype().ToCppString() << " " << *v->var() << " = "
         << *v->value() << ";" << '\n';
  } else {
    vector_vars_[v->var()] = v->value();
  }
}

void CppPrinter::visit(const VarPtr& v) {
  if (v->dtype().lanes() == 1) {
    os() << name_manager()->get_unique_name(v);
  } else {
    os() << *vector_vars_.at(v);
  }
}

CppCodeGen::CppCodeGen(
    StmtPtr stmt,
    const std::vector<BufferArg>& buffer_args,
    at::Device device,
    const std::string& kernel_func_name)
    : CodeGen(std::move(stmt), buffer_args, device, kernel_func_name) {
  init();
}

void CppCodeGen::init() {
  printer_ = std::make_unique<CppPrinter>(&oss_);
  var_name_rewriter_ = std::make_unique<CppVarNameRewriter>();

  apply_visitor(var_name_rewriter_.get());

  printer_->printPrologue();
  os() << "void " << kernel_func_name() << "(";
  const std::vector<BufferArg> buffer_args = this->buffer_args();
  for (size_t i = 0; i < buffer_args.size(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    const BufferArg& buffer_arg = buffer_args[i];
    const VarPtr var = buffer_arg.var();
    Dtype dtype = buffer_arg.dtype();
    os() << dtype.ToCppString() << (buffer_arg.isVar() ? " " : "* ") << *var;
  }
  os() << ")";
  stmt()->accept(printer_.get());
  os() << '\n';
}

CppCodeGen::~CppCodeGen() = default;

void CppCodeGen::call(const std::vector<CallArg>& args) {
  // TODO: compile the generated C++ kernel into a library,
  // and call the library here.
  os() << "int main() {}" << '\n';
}

void CppCodeGen::call_raw(const std::vector<void*>& args) {
  // TODO: compile the generated C++ kernel into a library,
  // and call the library here.
  os() << "int main() {}" << '\n';
}

RegisterCodeGen<CppCodeGen> cpp_codegen_reg("cpp_codegen");

} // namespace torch::jit::tensorexpr
