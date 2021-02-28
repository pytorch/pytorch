#include <type_traits>

#include <torch/csrc/jit/tensorexpr/cpp_codegen.h>
#include <torch/csrc/jit/tensorexpr/cpp_tensor.h>
#include <torch/csrc/jit/tensorexpr/cpp_vector.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/types.h>

namespace torch {
namespace jit {
namespace tensorexpr {

void CppPrinter::visit(const Ramp* v) {
  const IntImm* base = dynamic_cast<const IntImm*>(v->base());
  const IntImm* stride = dynamic_cast<const IntImm*>(v->stride());
  if (base == nullptr || stride == nullptr) {
    throw std::runtime_error("Ramp only supports IntImm as base and stride");
  }
  os() << "Ramp(" << *base << ", " << *stride << ", " << v->lanes() << ")";
}

void CppPrinter::visit(const Broadcast* v) {
  os() << "Broadcast<" << v->value()->dtype().ToCppString() << ">("
       << *v->value() << ", " << v->lanes() << ")";
}

template <typename T>
inline typename std::enable_if<!std::is_floating_point<T>::value, void>::type
visit_mod(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << *lhs << " % " << *rhs;
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, void>::type
visit_mod(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << "std::fmod(" << *lhs << ", " << *rhs << ")";
}

template <typename T>
inline typename std::enable_if<
    std::is_floating_point<T>::value || std::is_integral<T>::value,
    void>::type
visit_max(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << "std::max(" << *lhs << ", " << *rhs << ")";
}

template <typename T>
inline typename std::enable_if<
    !std::is_floating_point<T>::value && !std::is_integral<T>::value,
    void>::type
visit_max(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << "(" << *lhs << " < " << *rhs << ") ? " << *rhs << " : " << *lhs;
}

template <typename T>
inline typename std::enable_if<
    std::is_floating_point<T>::value || std::is_integral<T>::value,
    void>::type
visit_min(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << "std::min(" << *lhs << ", " << *rhs << ")";
}

template <typename T>
inline typename std::enable_if<
    !std::is_floating_point<T>::value && !std::is_integral<T>::value,
    void>::type
visit_min(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << *lhs << " < " << *rhs << " ? " << *lhs << " : " << *rhs;
}

template <typename T>
void visit_binary_op(
    std::ostream& os,
    const Expr* lhs,
    const Expr* rhs,
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
    AT_FORALL_SCALAR_TYPES_AND2(Half, Bool, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw unsupported_dtype();
  }
}

void CppPrinter::visit(const Mod* v) {
  if (v->lhs()->dtype().lanes() == 1) {
    dispatch_binary_op(os(), v);
  } else {
    os() << *v->lhs() << " % " << *v->rhs();
  }
}

void CppPrinter::visit(const Max* v) {
  if (v->lhs()->dtype().lanes() == 1) {
    dispatch_binary_op(os(), v);
  } else {
    os() << "Max(" << *v->lhs() << ", " << *v->rhs() << ")";
  }
}

void CppPrinter::visit(const Min* v) {
  if (v->lhs()->dtype().lanes() == 1) {
    dispatch_binary_op(os(), v);
  } else {
    os() << "Min(" << *v->lhs() << ", " << *v->rhs() << ")";
  }
}

std::string CppPrinter::to_lambda(
    CompareSelectOperation op,
    const std::string& ty) {
  std::stringstream ss;
  ss << "[](" << ty << " lhs, " << ty << " rhs) { "
     << "return lhs " << to_string(op) << " rhs;"
     << " }";
  return ss.str();
}

void CppPrinter::visit(const CompareSelect* v) {
  if (v->lhs()->dtype().lanes() == 1) {
    os() << "((" << *v->lhs() << " "
         << IRPrinter::to_string(v->compare_select_op()) << " " << *v->rhs()
         << ") ? " << *v->ret_val1() << " : " << *v->ret_val2() << ")";
  } else {
    std::string input_ty = v->lhs()->dtype().ToCppString();
    std::string return_ty = v->ret_val1()->dtype().ToCppString();
    os() << "CompareSelect<" << input_ty << ", " << return_ty << ">("
         << to_lambda(v->compare_select_op(), input_ty) << ", " << *v->lhs()
         << ", " << *v->rhs() << ", " << *v->ret_val1() << ", "
         << *v->ret_val2() << ")";
  }
}

void CppPrinter::visit(const IfThenElse* v) {
  os() << "((" << *v->condition() << ") ? " << *v->true_value() << " : "
       << *v->false_value() << ")";
}

void CppPrinter::visit(const Allocate* v) {
  emitIndent();
  os() << "Tensor<" << v->dtype().ToCppString() << "> " << *v->buffer_var()
       << "({";
  for (size_t i = 0; i < v->dims().size(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->dims()[i];
  }
  os() << "});" << std::endl;
}

void CppPrinter::visit(const Free* v) {
  emitIndent();
  os() << *v->buffer_var() << ".free();" << std::endl;
}

void CppPrinter::visit(const Load* v) {
  if (v->indices().size() > 1) {
    os() << *v->base_handle() << "[{";
    for (size_t i = 0; i < v->indices().size(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      os() << *v->indices()[i];
    }
    os() << "}]";
  } else if (v->flat_index()->dtype().lanes() == 1) {
    os() << *v->base_handle() << "[" << *v->flat_index() << "]";
  } else {
    os() << *v->base_handle() << ".load(" << *v->flat_index() << ", "
         << *v->mask() << ")";
  }
}

void CppPrinter::visit(const Store* v) {
  emitIndent();
  if (v->indices().size() > 1) {
    os() << *v->base_handle() << "[{";
    for (size_t i = 0; i < v->indices().size(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      os() << *v->indices()[i];
    }
    os() << "}] = " << *v->value() << ";";
  } else if (v->flat_index()->dtype().lanes() == 1) {
    os() << *v->base_handle() << "[" << *v->flat_index()
         << "] = " << *v->value() << ";";
  } else {
    os() << *v->base_handle() << ".store(" << *v->flat_index() << ", "
         << *v->value() << ", " << *v->mask() << ");";
  }
  os() << std::endl;
}

void CppPrinter::visit(const Cast* v) {
  if (v->src_value()->dtype().lanes() == 1) {
    os() << "static_cast<" << v->dtype().ToCppString() << ">("
         << *v->src_value() << ")";
  } else {
    os() << "Cast<" << v->src_value()->dtype().ToCppString() << ", "
         << v->dtype().ToCppString() << ">(" << *v->src_value() << ")";
  }
}

void CppPrinter::visit(const BitCast* v) {
  os() << "BitCast<" << v->src_value()->dtype().ToCppString() << ", "
       << v->dtype().ToCppString() << ">(" << *v->src_value() << ")";
}

void CppPrinter::visit(const Intrinsics* v) {
  switch (v->op_type()) {
    case kRand:
    case kSigmoid:
      throw std::runtime_error("kRand and kSigmoid are not supported");
  }

  if (v->param(0)->dtype().lanes() == 1) {
    os() << "std::" << v->func_name() << "(";
    for (int i = 0; i < v->nparams(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      os() << *v->param(i);
    }
    os() << ")";
  } else {
    ScalarType ty = v->param(0)->dtype().scalar_type();
    for (int i = 1; i < v->nparams(); ++i) {
      ty = promoteTypes(ty, v->param(i)->dtype().scalar_type());
    }
    const std::string input_type = Dtype(ty).ToCppString();
    const std::string ret_type = (v->op_type() == kIsNan)
        ? "int"
        : (is_integral(ty) ? "double" : input_type);

    os() << "ComputeIntrinsics<" << input_type << ", " << ret_type << ">(";
    os() << "std::" << v->func_name() << ", ";
    for (int i = 0; i < v->nparams(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      os() << *v->param(i);
    }
    os() << ")";
  }
}

void CppPrinter::visit(const ExternalCall* v) {
  // The generated code needs to link against functions defined
  // in external_functions.cpp.

  auto& func_registry = getNNCFunctionRegistry();
  if (!func_registry.count(v->func_name())) {
    throw unimplemented_lowering(v);
  }

  std::vector<const Buf*> bufs(v->buf_args());
  bufs.insert(bufs.begin(), v->buf());
  auto for_buf = [&](std::function<void(const Buf*)> print_buf) {
    for (size_t i = 0; i < bufs.size(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      print_buf(bufs[i]);
    }
  };

  emitIndent();
  os() << "{" << std::endl;
  indent_++;

  emitIndent();
  os() << "std::vector<void*> buf_ptrs{";
  for_buf([&](const Buf* b) { os() << "&" << *b->base_handle(); });
  os() << "};" << std::endl;

  emitIndent();
  os() << "std::vector<int64_t> buf_ranks{";
  for_buf([&](const Buf* b) { os() << b->ndim(); });
  os() << "};" << std::endl;

  emitIndent();
  os() << "std::vector<int64_t> buf_dims{";
  for_buf([&](const Buf* buf) {
    for (size_t i = 0; i < buf->ndim(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      os() << *buf->dim(i);
    }
  });
  os() << "};" << std::endl;

  emitIndent();
  os() << "std::vector<int8_t> buf_dtypes{";
  for_buf([&](const Buf* buf) {
    os() << static_cast<int>(buf->dtype().scalar_type());
  });
  os() << "};" << std::endl;

  emitIndent();
  os() << "std::vector<int64_t> extra_args{";
  for (size_t i = 0; i < v->args().size(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->args()[i];
  }
  os() << "};" << std::endl;

  emitIndent();
  os() << v->func_name() << "(" << std::endl;
  emitIndent();
  os() << "    buf_ptrs.size()," << std::endl;
  emitIndent();
  os() << "    buf_ptrs.data()," << std::endl;
  emitIndent();
  os() << "    buf_ranks.data()," << std::endl;
  emitIndent();
  os() << "    buf_dims.data()," << std::endl;
  emitIndent();
  os() << "    buf_dtypes.data()," << std::endl;
  emitIndent();
  os() << "    extra_args.size()," << std::endl;
  emitIndent();
  os() << "    extra_args.data());" << std::endl;

  indent_--;
  emitIndent();
  os() << "}";
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
