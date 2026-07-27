#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>

namespace torch::distributed::rpc {

// This function sends an rpc call to run torchscript function, currently the
// torchscript function could only be a user defined python function with
// "@torch.jit.script" annotation. The torchscript function could not be
// a class constructor, class method, instance method or a script module.
//   dst: destination worker name
//   qualifiedName: torchscript function qualified name string like
//                  "moduleName::torchscriptFunctionName", e.g,
//                  "dist_autograd_test::my_py_add"
//   stack: a bag of IValue args passed to torchscriptFunctionName
// It returns c10::intrusive_ptr<ivalue::Future>
c10::intrusive_ptr<c10::ivalue::Future> TORCH_API rpcTorchscript(
    const std::string& dstWorkerName,
    const c10::QualifiedName& qualifiedName,
    const c10::FunctionSchema& functionSchema,
    std::vector<c10::IValue> stack,
    const float rpcTimeoutSeconds = torch::distributed::rpc::kUnsetRpcTimeout,
    const bool isAsyncExecution = false);

c10::intrusive_ptr<RRef> TORCH_API remoteTorchscript(
    const std::string& dstWorkerName,
    const c10::QualifiedName& qualifiedName,
    const c10::FunctionSchema& functionSchema,
    std::vector<c10::IValue>& stack,
    const float rpcTimeoutSeconds = torch::distributed::rpc::kUnsetRpcTimeout,
    const bool isAsyncExecution = false);

} // namespace torch::distributed::rpc
