#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/script/function_schema_parser.h"

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

// NOTE [Sharded File]: This file is generated in a sharded fashion to speed up
// incremental rebuilds. See the comment at the top of
// templates/VariableType.cpp for an analogous, in-depth discussion.
//
// Note that unlike VariableType.cpp, when sharding this file we take
// care to generate all overloads of a particular name in a single
// file and in a particular order. See gen_jit_dispatch.py for
// details.

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
  auto* typedUnboxedKernel = static_cast<c10::detail::WrapRuntimeKernelFunctor<FuncType*>*>(unboxedKernel);
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

c10::OperatorOptions atenOperatorOptions() {
  c10::OperatorOptions result;
  result.setAliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA);
  return result;
}

int (*DUMMY_OPERATION)(Stack&) = [](Stack& stack) -> int {
  TORCH_CHECK(false, "Operator has been stripped in the custom build.")
  return 0;
};

class Registerer final {
public:
  Registerer&& op(const std::string& schema, KernelFunction::InternalBoxedKernelFunction* boxed_kernel_wrapper) && {
    static auto& dispatcher = c10::Dispatcher::singleton();
    std::pair<RegistrationHandleRAII, OperatorHandle> registration = dispatcher.registerSchema(parseSchema(schema), atenOperatorOptions());
    registrationHandles_.push_back(std::move(registration.first));
    dispatcher.setManuallyBoxedKernelFor_(registration.second, boxed_kernel_wrapper);
    return std::move(*this);
  }

  Registerer&& jitOnlyOp(const std::string& schema, std::function<int (Stack*)> boxed_kernel_wrapper) && {
    torch::jit::registerOperator(
      torch::jit::Operator(
        schema,
        Operation([boxed_kernel_wrapper = std::move(boxed_kernel_wrapper)] (Stack& stack) -> int {
          return boxed_kernel_wrapper(&stack);
        }),
        atenOperatorOptions()
      )
    );
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
  .jitOnlyOp("aten::get_device(Tensor self) -> int",
    [](Stack* stack) {
      RECORD_FUNCTION("get_device", std::vector<c10::IValue>());
      auto result =
          at::get_device((std::move(peek(*stack, 0, 1))).toTensor());
      drop(*stack, 1);
      pack(*stack, std::move(result));
      return 0;
    })
  .jitOnlyOp("aten::storage_offset(Tensor self) -> int",
    [](Stack* stack) {
      RECORD_FUNCTION("storage_offset", std::vector<c10::IValue>());
      auto result =
          ((std::move(peek(*stack, 0, 1))).toTensor()).storage_offset();
      drop(*stack, 1);
      pack(*stack, std::move(result));
       return 0;
    })
  .jitOnlyOp("aten::is_contiguous(Tensor self) -> bool",
    [](Stack* stack) {
      RECORD_FUNCTION("is_contiguous", std::vector<c10::IValue>());
      auto result =
          ((std::move(peek(*stack, 0, 1))).toTensor()).is_contiguous();
      drop(*stack, 1);
      pack(*stack, std::move(result));
      return 0;
    })

  // Generated operators
  ${constructors}
  ;

} // anon namespace


}} // namespace torch::jit
