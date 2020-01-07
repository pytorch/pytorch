#pragma once

#include <ATen/core/ivalue.h>

namespace torch {
namespace distributed {
namespace rpc {

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
c10::intrusive_ptr<c10::ivalue::Future> rpcTorchscriptCall(
    const std::string& dst,
    const c10::QualifiedName& qualifiedName,
    std::vector<c10::IValue>& stack);

} // namespace rpc
} // namespace distributed
} // namespace torch
