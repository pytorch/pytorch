#pragma once

#include <ATen/core/ivalue.h>

namespace torch {
namespace distributed {
namespace rpc {

// This function sends a rpc call to run torchscript function, currently the
// torchscript function could only be a user defined python function with
// "@torch.jit.script" annotation. The torchscript function could not be
// a class constructor, class method, instance method or a script module.
//   dst: destination worker name
//   qualifiedName: torchscript function qualified name string like
//                  "moduleName::fnName", e.g, "dist_autograd_test::my_py_add"
//   stack: a bags of IValue args passed to fnName
// It returns IValue that is c10::intrusive_ptr<ivalue::Future>
c10::IValue rpcTorchscriptCall(
    const std::string& dst,
    const c10::QualifiedName& qualifiedName,
    std::vector<c10::IValue>& stack);

} // namespace rpc
} // namespace distributed
} // namespace torch
