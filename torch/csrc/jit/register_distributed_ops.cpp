#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/distributed/rpc/torchscript_functions.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pybind_utils.h>

using at::Scalar;
using at::Tensor;
namespace dist_autograd = torch::distributed::autograd;
namespace dist_rpc = torch::distributed::rpc;

namespace torch {
namespace jit {

using namespace distributed::rpc;

namespace {
at::Tensor toOptionalTensor(const c10::IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

at::Tensor optional_to_tensor(c10::optional<at::Tensor> v) {
  return v.has_value() ? *v : at::Tensor();
}

c10::OperatorOptions aliasAnalysisFromSchema() {
  c10::OperatorOptions result;
  result.setAliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA);
  return result;
}

c10::OperatorOptions aliasAnalysisSpecialCase() {
  c10::OperatorOptions result;
  result.setAliasAnalysis(c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE);
  return result;
}

RegisterOperators reg_rpc_ops({
    Operator(
        "aten::to_here(RRef(t) self) -> t",
        [](Stack& stack) {
          auto rref = pop(stack).toRRef();
          IValue res;
          if (rref->isOwner()) {
            res = c10::dynamic_intrusive_pointer_cast<dist_rpc::OwnerRRef>(rref)
                      ->getValue();
          } else {
            res = c10::dynamic_intrusive_pointer_cast<dist_rpc::UserRRef>(rref)
                      ->toHere();
          }
          push(stack, std::move(res));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::is_owner(RRef(t) self) -> bool",
        [](Stack& stack) {
          auto rref = pop(stack).toRRef();
          push(stack, rref->isOwner());
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
         prim::rpc_async,
         [](const Node* node) -> Operation {
           int num_inputs = node->inputs().size();
           return [num_inputs](Stack& stack) {
             // Get inputs from the stack.
             auto stackIter = stack.end() - num_inputs;
             auto& dstWorkerNameIValue = *stackIter++;
             auto& qualifiedNameIValue = *stackIter++;
             IValue emptyTuple(c10::ivalue::Tuple::create({}));
             IValue emptyDict{c10::impl::GenericDict(AnyType::get(), AnyType::get())};
             // Equavalent to Python statment
             // `args = args if args is not None else ()`.
             auto& argsTupleIValue = num_inputs >= 3 ? *stackIter++ : emptyTuple;
             // `kwargs = kwargs if kwargs is not None else {}`.
             auto& kwargsDictIValue = num_inputs >= 4 ? *stackIter++ : emptyDict;
             TORCH_INTERNAL_ASSERT(dstWorkerNameIValue.isString());
             TORCH_INTERNAL_ASSERT(qualifiedNameIValue.isString());
             TORCH_INTERNAL_ASSERT(argsTupleIValue.isTuple());
             TORCH_INTERNAL_ASSERT(kwargsDictIValue.isGenericDict());
             LOG(ERROR) << "argsTupleIValue: " << argsTupleIValue;
             LOG(ERROR) << "kwargsDictIValue: " << kwargsDictIValue;

             // Get FunctionSchema for qualifiedName.
             auto qualifiedName = c10::QualifiedName(qualifiedNameIValue.toStringRef());
             std::shared_ptr<script::CompilationUnit> cuPtr;
             {
               py::gil_scoped_acquire acquire;
               cuPtr = get_python_cu();
             }
             auto& functionSchema = cuPtr->get_function(qualifiedName).getSchema();

             // Build stack for the user function.
             // It's the same logic as createStackForSchema.
             Stack userFuncStack;
             auto argsTuplePtr = argsTupleIValue.toTuple();
             auto kwargsDict = kwargsDictIValue.toGenericDict();
             auto numArgsAndKwargs = argsTuplePtr->elements().size() + kwargsDict.size();
             userFuncStack.reserve(numArgsAndKwargs);

             // Push args.
             for (auto& elem : argsTupleIValue.toTuple()->elements()) {
               push(userFuncStack, std::move(elem));
             }

             // Push kwargs.
             size_t consumed_kwargs = 0;
             for (size_t i = userFuncStack.size(); i < functionSchema.arguments().size(); ++i) {
               const auto& arg = functionSchema.arguments()[i];
               const auto& argName = arg.name();
               if (kwargsDict.contains(argName)) {
                 push(userFuncStack, kwargsDict.at(argName));
                 consumed_kwargs += 1;
               } else if (arg.default_value()) {
                 push(userFuncStack, *arg.default_value());
               } else {
                 throw std::runtime_error(c10::str(
                     functionSchema.name(),
                     "() is missing value for argument '",
                     argName,
                     "'. Declaration: ",
                     functionSchema));
               }
             }
             TORCH_INTERNAL_ASSERT(
                 kwargsDict.size() == consumed_kwargs,
                 "There is unknown kwarg given.");

             // Send RPC request.
             auto futureIValuePtr = rpcTorchscript(
                dstWorkerNameIValue.toStringRef(),
                qualifiedName,
                functionSchema,
                userFuncStack
             );

             // Push output to the stack.
             drop(stack, num_inputs);
             stack.push_back(futureIValuePtr);
             return 0;
           };
         },
         aliasAnalysisSpecialCase())});

auto reg_distributed_ops =
   torch::RegisterOperators()
       .op("aten::get_gradients(int context_id) -> Dict(Tensor, Tensor)",
           torch::RegisterOperators::options()
               .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA)
               .catchAllKernel([](int64_t context_id) {
                 const auto& autogradContext =
                     dist_autograd::DistAutogradContainer::getInstance()
                         .retrieveContext(context_id);
                 return autogradContext->getGradients();
               }));

} // namespace
} // namespace jit
} // namespace torch
