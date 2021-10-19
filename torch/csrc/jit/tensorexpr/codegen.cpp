#include <torch/csrc/jit/tensorexpr/codegen.h>

#include <sstream>

namespace torch {
namespace jit {
namespace tensorexpr {

RegisterCodeGenList::StmtFactoryMethod RegisterCodeGenList::
    FindStmtFactoryMethod(const std::string& name) {
  auto iter = stmt_factory_methods_.find(name);
  if (iter == stmt_factory_methods_.end()) {
    std::ostringstream oss;
    oss << "Invalid stmt codegen name: " << name << ". ";
    oss << "Existing codegen names: [";
    int index = 0;
    for (auto& entry : stmt_factory_methods_) {
      if (index != 0) {
        oss << ", ";
      }
      oss << entry.first;
      index++;
    }
    oss << "]";
    throw std::runtime_error(oss.str());
  }
  return iter->second;
}

void RegisterCodeGenList::AddStmtFactoryMethod(
    const std::string& name,
    const StmtFactoryMethod& stmt_factory_method) {
  stmt_factory_methods_[name] = stmt_factory_method;
}

std::unique_ptr<CodeGen> CreateCodeGen(
    const std::string& name,
    StmtPtr stmt,
    const std::vector<CodeGen::BufferArg>& params,
    at::Device device,
    const std::string& kernel_func_name) {
  RegisterCodeGenList::StmtFactoryMethod method =
      RegisterCodeGenList::GetInstance().FindStmtFactoryMethod(name);
  return method(stmt, params, device, kernel_func_name);
}

ExprPtr GenericIntrinsicsExpander::mutate(IntrinsicsPtr v) {
  if (v->op_type() == kSigmoid) {
    auto x = v->param(0)->accept_mutator(this);
    auto one = expr_to_vec(
        ExprHandle(getImmediateByType(v->dtype(), 1.0)), v->dtype().lanes());
    auto zero = expr_to_vec(
        ExprHandle(getImmediateByType(v->dtype(), 0.0)), v->dtype().lanes());
    ExprHandle y = one / (one + exp(zero - ExprHandle(x)));
    return y.node();
  }
  return IRMutator::mutate(v);
}

void* CodeGen::argToPtr(const BufferArg& bufferArg, const CallArg& callArg) {
  if (!bufferArg.isVar()) {
    return callArg.data();
  }

  switch (bufferArg.dtype().scalar_type()) {
#define TYPE_CASE(_1, Name) \
  case ScalarType::Name:    \
    return callArg.Name##Ptr();

    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE

    default:
      throw unsupported_dtype();
  }
  return nullptr;
}

void CodeGen::call_with_numel(void** args, int64_t numel) {
  TORCH_INTERNAL_ASSERT(
      false, "This codegen backend does not implement call_with_numel");
}
} // namespace tensorexpr
} // namespace jit
} // namespace torch
