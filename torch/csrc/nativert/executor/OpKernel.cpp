#include "torch/csrc/nativert/executor/OpKernel.h"

#include <c10/util/Logging.h>
#include <fmt/ostream.h>

#include "torch/csrc/nativert/common/ConfigUtils.h"
#include "torch/csrc/nativert/common/Enumerate.h"
#include "torch/csrc/nativert/common/String.h"
#include "torch/csrc/nativert/executor/ExecutionFrame.h"

#ifdef __SIGRID_USE_GPU__
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#endif

namespace torch::nativert {

c10::OperatorHandle getOperatorForTarget(
    std::string_view target,
    const Node* node) {
  // target could come as either "torch.ops.aten.add.default" or
  // "aten.add.default"
  std::vector<std::string_view> atoms = split(target, '.');

  size_t numAtoms = atoms.size();
  if (numAtoms < 3) {
    TORCH_CHECK(false, "Invalid target: ", target);
  }

  const std::string_view ns = atoms[numAtoms - 3];
  const std::string_view opName = atoms[numAtoms - 2];
  const std::string_view overloadName = atoms[numAtoms - 1];

  const auto operatorName = fmt::format("{}::{}", ns, opName);
  std::string normalizedOverloadName;
  if (overloadName == "default") {
    normalizedOverloadName = "";
  } else {
    normalizedOverloadName = overloadName;
  }

  auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(
      operatorName.c_str(), normalizedOverloadName.c_str());

  return handle;
}

std::string readableArgs(
    const c10::FunctionSchema& schema,
    const std::vector<c10::IValue>& stack) {
  auto schemaArgs = schema.arguments();
  std::stringstream ss;
  for (const auto& [i, arg] : enumerate(stack)) {
    ss << "arg" << i << " " << schemaArgs[i].name() << ": " << arg.tagKind()
       << " ";
    if (arg.isTensor()) {
      auto t = arg.toTensor();
      ss << t.dtype() << t.sizes() << t.device();
    } else if (arg.isTensorList()) {
      auto tl = arg.toTensorVector();
      ss << "[";
      for (const auto& t : tl) {
        ss << t.dtype() << t.sizes() << t.device() << ", ";
      }
      ss << "]";
    } else if (arg.isNone()) {
      // pass
    } else {
      ss << arg;
    }
    ss << "\n";
  }
  return ss.str();
}

const bool OpKernel::blockingEnabled_ =
    maybeGetEnv("TORCH_NATIVE_RUNTIME_CUDA_LAUNCH_BLOCKING").value_or("0") == "1";

void OpKernel::compute(ExecutionFrame& executionFrame) const {
  VLOG(2) << "Executing: " << *node_;

  computeInternal(executionFrame);

#ifdef __SIGRID_USE_GPU__
  if (device_.has_value() && device_->is_cuda() && blockingEnabled_) {
    AT_CUDA_CHECK(cudaDeviceSynchronize());
    AT_CUDA_CHECK(cudaGetLastError());
  }
#endif

  VLOG(2) << "Completed: " << *node_;
}

Arguments prefillStackWithStaticArgs(
    const Node* node,
    const c10::FunctionSchema& schema) {
  std::vector<c10::IValue> stackWithStaticArgs;
  std::vector<Value*> dynamicArgs;
  const auto& schemaArgs = schema.arguments();
  stackWithStaticArgs.resize(schemaArgs.size());
  dynamicArgs.resize(schemaArgs.size());

  // initialized stackWithStaticArgs_ with static inputs
  for (const auto& [idx, schemaArg] : enumerate(schemaArgs)) {
    const auto& argName = schemaArg.name();

    // Check if this is a dynamic input to the op.
    const auto input = node->tryGetInput(argName);
    if (input != nullptr) {
      stackWithStaticArgs.at(idx) = c10::IValue();
      dynamicArgs.at(idx) = input->value;
      continue;
    }

    // Check if this is a statically known input to the op.
    const auto attribute = node->tryGetAttribute(argName);
    if (attribute != nullptr) {
      stackWithStaticArgs.at(idx) = constantToIValue(attribute->value);
      continue;
    }

    // Otherwise, this must have a default value
    if (schemaArg.default_value().has_value()) {
      stackWithStaticArgs.at(idx) = schemaArg.default_value().value();
      continue;
    }

    TORCH_CHECK(
        false,
        "Cannot initialize argument ",
        argName,
        " for node ",
        *node,
        " with schema ",
        schema);
  }
  return Arguments{std::move(stackWithStaticArgs), std::move(dynamicArgs)};
}

void fillDynamicInputs(
    const ExecutionFrame& executionFrame,
    const Arguments& arguments,
    std::vector<c10::IValue>& stack) {
  // fill the stack with dynamic values from execution frame,
  // including tensor, tensors, symint, symints

  for (auto [idx, value] : arguments.getDynamicArgs()) {
    CHECK(idx < stack.size()) << "invalid idx";
    CHECK(stack.at(idx).isNone()) << "stack[idx] shouldn't have been populated";
    if (value->type() == Type::TensorList) {
      // TODO: This for passing List<Tensor> as an input to op that takes a
      // List<Optional<Tensor>>.
      // this is awful, but if I don't cast it to a vector and back to a
      // list, I get list covariance problems where List<Tensor> is not a
      // subtype of List<Optional<Tensor>>, which pops up when trying to
      // execute aten.index.Tensor. See the code in:
      // https://fburl.com/code/t1poq3z3. Our lists should be covariant
      // because they are static, but IValues don't know that :(
      stack[idx] = executionFrame.getIValue(value->id()).toTensorList().vec();
    } else if (value->type() == Type::None) {
      stack[idx] = c10::IValue();
    } else {
      stack[idx] = executionFrame.getIValue(value->id());
    }
  }
}

} // namespace torch::nativert
