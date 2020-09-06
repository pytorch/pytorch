#include "torch/csrc/jit/runtime/operator.h"
#include "torch/csrc/jit/runtime/custom_operator.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"

#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ${generated_comment}

// This file contains manual unboxing wrappers for ops that aren't
// use_c10_dispatcher: full because the templated unboxing logic in c10 doesn't
// support them yet. The ultimate goal is to make all ops use the templated
// unboxing and delete this codegen file.

// NOTE [Sharded File]: This file is generated in a sharded fashion to speed up
// incremental rebuilds. See the comment at the top of
// templates/VariableType.cpp for an analogous, in-depth discussion.

namespace torch { namespace jit {

using autograd::Variable;
using autograd::variable_list;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using at::TensorOptions;
using at::DeviceGuard;
using at::MemoryFormat;

using ::c10::fmap;
using ::c10::filter;
using c10::OperatorKernel;
using c10::OperatorHandle;
using c10::KernelFunction;
using c10::RegistrationHandleRAII;
using c10::Stack;

namespace {

template<class Return, class... Args>
Return callUnboxedKernel(OperatorKernel* unboxedKernel, Args... args) {
  using FuncType = Return (Args...);
  auto* typedUnboxedKernel = static_cast<c10::impl::WrapFunctionIntoRuntimeFunctor<FuncType*>*>(unboxedKernel);
  return (*typedUnboxedKernel)(std::forward<Args>(args)...);
}

// TODO: remove the toOptionalTensor and toListOfOptionalTensor
// when we remove the undefined tensor semantic from TH

// XXX: This function is to specialize IValue for tensor type in
// interpreter, it should only be used in this file
at::Tensor toOptionalTensor(const IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

// XXX: This function is to specialize IValue for list of optional
// tensor type in interpreter, it should only be used in this file
std::vector<Tensor> toListOfOptionalTensor(const IValue& v) {
  // v is a list of optional tensor, loop over as generic list
  auto vlist = v.toListRef();
  std::vector<Tensor> res;

  for (const IValue &v: vlist) {
    res.emplace_back(toOptionalTensor(v));
  }
  return res;
}

template<size_t N>
std::array<bool, N> as_bool_array(const c10::List<bool>& list) {
  std::array<bool, N> res;
  AT_ASSERT(list.size() == N);
  std::copy(list.begin(), list.end(), res.begin());
  return res;
}

KernelFunction::InternalBoxedKernelFunction *DUMMY_OPERATION =
  [](c10::OperatorKernel *, const c10::OperatorHandle &, std::vector<c10::IValue> *) -> void {
    TORCH_CHECK(false, "Operator has been stripped in the custom build.")
  };

class Registerer final {
public:
  Registerer&& op(const std::string& schemaStr, KernelFunction::InternalBoxedKernelFunction* boxed_kernel_wrapper) && {
    static auto& dispatcher = c10::Dispatcher::singleton();
    auto schema = parseSchema(schemaStr);
    schema.setAliasAnalysis(AliasAnalysisKind::FROM_SCHEMA);
    c10::OperatorName name = schema.operator_name();
    RegistrationHandleRAII registration = dispatcher.registerName(name);
    auto op = dispatcher.findOp(name).value();
    registrationHandles_.push_back(std::move(registration));
    dispatcher.setManuallyBoxedKernelFor_(op, boxed_kernel_wrapper);
    return std::move(*this);
  }

  Registerer() = default;
  Registerer(const Registerer&) = delete;
  Registerer& operator=(const Registerer&) = delete;
  Registerer(Registerer&&) noexcept = default;
  Registerer& operator=(Registerer&&) noexcept = default;
private:
  std::vector<RegistrationHandleRAII> registrationHandles_;
};

static auto registry = Registerer()
  // Generated operators
  ${constructors}
  ;

} // anon namespace


}} // namespace torch::jit
