#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/distributed/rpc/torchscript_functions.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

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

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

c10::AliasAnalysisKind aliasAnalysisSpecialCase() {
  return c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
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
         "aten::is_confirmed(RRef(t) self) -> bool",
         [](Stack& stack) {
           auto rref = pop(stack).toRRef();
           push(stack, rref->isConfirmed());
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
             IValue emptyDict{
                 c10::impl::GenericDict(AnyType::get(), AnyType::get())};
             // Equavalent to Python statment
             // `args = args if args is not None else ()`.
             auto& argsTupleIValue =
                 num_inputs >= 3 ? *stackIter++ : emptyTuple;
             // `kwargs = kwargs if kwargs is not None else {}`.
             auto& kwargsDictIValue =
                 num_inputs >= 4 ? *stackIter++ : emptyDict;
             TORCH_INTERNAL_ASSERT(dstWorkerNameIValue.isString());
             TORCH_INTERNAL_ASSERT(qualifiedNameIValue.isString());
             TORCH_INTERNAL_ASSERT(argsTupleIValue.isTuple());
             TORCH_INTERNAL_ASSERT(kwargsDictIValue.isGenericDict());

             // Get FunctionSchema for qualifiedName.
             auto qualifiedName =
                 c10::QualifiedName(qualifiedNameIValue.toStringRef());
             std::shared_ptr<CompilationUnit> cuPtr;
             {
               py::gil_scoped_acquire acquire;
               cuPtr = get_python_cu();
             }
             auto& functionSchema =
                 cuPtr->get_function(qualifiedName).getSchema();

             // Build Stack for the user callable.
             // It's similar to
             // Stack createStackForSchema(FunctionSchema, py::args, py::kwargs).
             // Instead, it's
             // Stack createStackForSchema(FunctionSchema, IValue<Tuple>, IValue<Dict>).
             Stack userCallableStack;
             userCallableStack.reserve(functionSchema.arguments().size());

             // Move args from Tuple IValue to Stack.
             for (auto& elem : argsTupleIValue.toTuple()->elements()) {
               push(userCallableStack, std::move(elem));
             }

             // Move kwargs from Dict IValue to Stack.
             size_t consumed_kwargs = 0;
             auto kwargsDict = kwargsDictIValue.toGenericDict();
             for (size_t i = userCallableStack.size();
                  i < functionSchema.arguments().size();
                  ++i) {
               const auto& arg = functionSchema.arguments()[i];
               const auto& argName = arg.name();
               if (kwargsDict.contains(argName)) {
                 push(userCallableStack, kwargsDict.at(argName));
                 consumed_kwargs += 1;
               } else if (arg.default_value()) {
                 push(userCallableStack, *arg.default_value());
               } else {
                 throw std::runtime_error(c10::str(
                     functionSchema.name(),
                     "() is missing value for argument '",
                     argName,
                     "'. Declaration: ",
                     functionSchema));
               }
             }
             // Raise exception showing the unexpected kwargs.
             if (consumed_kwargs != kwargsDict.size()) {
               std::vector<std::string> names;
               for (const auto& entry : kwargsDict) {
                 const IValue& keyIValue = entry.key();
                 const string& keyStr = keyIValue.toStringRef();
                 names.emplace_back(keyStr);
               }
               functionSchema.findErrorInKwargs(names);
             }

             // Send RPC request.
             auto futureIValuePtr = rpcTorchscript(
                 dstWorkerNameIValue.toStringRef(),
                 qualifiedName,
                 functionSchema,
                 userCallableStack);

             // Push output to the stack.
             drop(stack, num_inputs);
             stack.emplace_back(std::move(futureIValuePtr));
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
