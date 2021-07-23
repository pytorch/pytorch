#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/distributed/autograd/autograd.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/distributed/rpc/torchscript_functions.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/library.h>

#include <fmt/format.h>
#include <stdexcept>

using at::Scalar;
using at::Tensor;
namespace dist_autograd = torch::distributed::autograd;
namespace dist_rpc = torch::distributed::rpc;

namespace torch {
namespace jit {

namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static auto workerInfo =
    torch::class_<dist_rpc::WorkerInfo>("dist_rpc", "WorkerInfo")
        .def(torch::init<std::string, int64_t>());

// prepare the rpc input arguments and call the C++ impls
void prepare_and_call_rpc_op(
    Stack* stack,
    int num_inputs,
    const std::string& rpc_op) {
  // Get inputs from the stack.
  auto stackIter = stack->end() - num_inputs;
  auto& dstWorkerIValue = *stackIter++;
  auto& qualifiedNameIValue = *stackIter++;
  IValue emptyTuple(c10::ivalue::Tuple::create({}));
  IValue emptyDict{c10::impl::GenericDict(AnyType::get(), AnyType::get())};
  // Equivalent to Python statement
  // `args = args if args is not None else ()`.
  auto& argsTupleIValue = num_inputs >= 3 ? *stackIter++ : emptyTuple;
  // `kwargs = kwargs if kwargs is not None else {}`.
  auto& kwargsDictIValue = num_inputs >= 4 ? *stackIter++ : emptyDict;

  // IValue corresponding to placeholder for RPC timeout. Used if no
  // rpc timeout is specified by user.
  IValue noTimeout(torch::distributed::rpc::kUnsetRpcTimeout);
  const auto rpcMaxInputs = 5;
  auto& timeoutIValue = num_inputs >= rpcMaxInputs ? *stackIter++ : noTimeout;
  TORCH_INTERNAL_ASSERT(
      dstWorkerIValue.isString() ||
      c10::getCustomClassType<c10::intrusive_ptr<dist_rpc::WorkerInfo>>() ==
          dstWorkerIValue.type());
  TORCH_INTERNAL_ASSERT(qualifiedNameIValue.isString());
  TORCH_INTERNAL_ASSERT(argsTupleIValue.isTuple());
  TORCH_INTERNAL_ASSERT(kwargsDictIValue.isGenericDict());
  TORCH_INTERNAL_ASSERT(timeoutIValue.isDouble());

  // Get FunctionSchema for qualifiedName.
  auto qualifiedName = c10::QualifiedName(qualifiedNameIValue.toStringRef());
  std::shared_ptr<CompilationUnit> cuPtr;
  {
    py::gil_scoped_acquire acquire;
    cuPtr = get_python_cu();
  }
  auto& functionSchema = cuPtr->get_function(qualifiedName).getSchema();

  // Build the stack for the user callable.
  // It's similar to
  // Stack createStackForSchema(FunctionSchema, py::args,
  // py::kwargs). Instead, it's Stack
  // createStackForSchema(FunctionSchema, IValue<Tuple>,
  // IValue<Dict>).
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
    throw std::runtime_error(functionSchema.findErrorInKwargs(names));
  }

  // Get destination WorkerName.
  std::string dstWorkerNameStr;
  if (dstWorkerIValue.isString()) {
    // ivalue::ConstantString::str_ is a const member, which can't be
    // moved, copy it here.
    dstWorkerNameStr = dstWorkerIValue.toStringRef();
  } else {
    dstWorkerNameStr =
        dstWorkerIValue.toCustomClass<dist_rpc::WorkerInfo>()->name_;
  }
  // Get RPC timeout, if specified by user.
  const auto rpcTimeout = timeoutIValue.toDouble();

  if (rpc_op == "rpc_async") {
    // Send RPC request.
    auto futureIValuePtr = dist_rpc::rpcTorchscript(
        dstWorkerNameStr,
        qualifiedName,
        functionSchema,
        userCallableStack,
        rpcTimeout);
    // Push output to the stack.
    drop(stack, num_inputs);
    stack->emplace_back(std::move(futureIValuePtr));
  } else if (rpc_op == "rpc_sync") {
    // Send RPC request.
    auto futureIValuePtr = dist_rpc::rpcTorchscript(
        dstWorkerNameStr,
        qualifiedName,
        functionSchema,
        userCallableStack,
        rpcTimeout);
    futureIValuePtr->wait();
    if (futureIValuePtr->hasError()) {
      // throw error if future hasError
      throw std::runtime_error(futureIValuePtr->tryRetrieveErrorMessage());
    } else {
      auto res = futureIValuePtr->value();
      // Push output to the stack.
      drop(stack, num_inputs);
      stack->emplace_back(std::move(res));
    }
  } else if (rpc_op == "rpc_remote") {
    auto rrefPtr = dist_rpc::remoteTorchscript(
        dstWorkerNameStr,
        qualifiedName,
        functionSchema,
        userCallableStack,
        rpcTimeout);
    // Push output to the stack.
    drop(stack, num_inputs);
    stack->emplace_back(
        c10::static_intrusive_pointer_cast<c10::RRefInterface>(rrefPtr));
  } else {
    throw std::runtime_error(
        c10::str(rpc_op, "() is not supported in TorchScript!'"));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
RegisterOperators reg_rpc_ops(
    {Operator(
         fmt::format(
             "aten::to_here(RRef(t) self, float timeout = {}) -> t(*)",
             torch::distributed::rpc::kDefaultRpcTimeoutSeconds),
         [](Stack* stack) {
           auto timeout = pop(stack).toDouble();
           auto rref = pop(stack).toRRef();
           IValue res;
           if (rref->isOwner()) {
             res =
                 c10::dynamic_intrusive_pointer_cast<dist_rpc::OwnerRRef>(rref)
                     ->getValue();
           } else {
             res = c10::dynamic_intrusive_pointer_cast<dist_rpc::UserRRef>(rref)
                       ->toHere(timeout);
           }
           push(stack, std::move(res));
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::local_value(RRef(t) self) -> t(*)",
         [](Stack* stack) {
           auto rref = pop(stack).toRRef();
           TORCH_CHECK(
               rref->isOwner(),
               "Can't call RRef.local_value() on a non-owner RRef.");
           IValue res =
               c10::static_intrusive_pointer_cast<dist_rpc::OwnerRRef>(rref)
                   ->getValue();
           push(stack, std::move(res));
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::is_owner(RRef(t) self) -> bool",
         [](Stack* stack) {
           auto rref = pop(stack).toRRef();
           push(stack, rref->isOwner());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::owner(RRef(t) self) -> __torch__.torch.classes.dist_rpc.WorkerInfo",
         [](Stack* stack) {
           auto rref = pop(stack).toRRef();
           push(
               stack,
               torch::make_custom_class<distributed::rpc::WorkerInfo>(
                   rref->ownerName(), rref->owner()));
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::owner_name(RRef(t) self) -> str",
         [](Stack* stack) {
           auto rref = pop(stack).toRRef();
           push(stack, rref->ownerName());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::confirmed_by_owner(RRef(t) self) -> bool",
         [](Stack* stack) {
           auto rref = pop(stack).toRRef();
           push(stack, rref->confirmedByOwner());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::dist_backward(int context_id, Tensor[] roots, bool retain_graph=False) -> ()",
         [](Stack* stack) {
           bool retain_graph = pop(stack).toBool();
           auto roots_list = pop(stack).toTensorList();
           int64_t context_id = pop(stack).toInt();
           torch::autograd::variable_list roots(
               roots_list.begin(), roots_list.end());
           dist_autograd::backward(context_id, roots, retain_graph);
         },
         aliasAnalysisConservative()),
     Operator(
         prim::rpc_sync,
         [](const Node* node) -> Operation {
           int num_inputs = node->inputs().size();
           return [num_inputs](Stack* stack) {
             prepare_and_call_rpc_op(stack, num_inputs, "rpc_sync");
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::rpc_remote,
         [](const Node* node) -> Operation {
           int num_inputs = node->inputs().size();
           return [num_inputs](Stack* stack) {
             prepare_and_call_rpc_op(stack, num_inputs, "rpc_remote");
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::rpc_async,
         [](const Node* node) -> Operation {
           int num_inputs = node->inputs().size();
           return [num_inputs](Stack* stack) {
             prepare_and_call_rpc_op(stack, num_inputs, "rpc_async");
           };
         },
         aliasAnalysisSpecialCase())});

// Implementations located in
// torch/csrc/jit/runtime/register_distributed_ops.cpp
TORCH_LIBRARY_IMPL(aten, CatchAll, m) {
  m.impl("get_gradients", [](int64_t context_id) {
    const auto& autogradContext =
        dist_autograd::DistAutogradContainer::getInstance().retrieveContext(
            context_id);
    return autogradContext->getGradients();
  });
}

} // namespace
} // namespace jit
} // namespace torch
