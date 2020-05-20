#pragma once

#include <torch/csrc/distributed/rpc/py_rref.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

// Converts an internal FutureMessage type into a user-facing FutureIValue type
// by creating a new FutureIValue and call its markCompleted as a callback in
// the given FutureMessage.
// If hasValue is true, the Message will be converted into a py::object and then
// wrap it with an IValue. If hasValue is false, this FutureIValue is only used
// for signaling and launching callbacks. In this case, the message will be
// discarded and then set the FutureIValue using an empty IValue or the given
// FutureError if there is an error.
c10::intrusive_ptr<JitFuture> wrapFutureMessageInJitFuture(
    const std::shared_ptr<FutureMessage>& futureResponseMessage,
    bool hasValue = true);

c10::intrusive_ptr<JitFuture> pyRpcBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs,
    const float rpcTimeoutSeconds);

c10::intrusive_ptr<JitFuture> pyRpcPythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const float rpcTimeoutSeconds);

c10::intrusive_ptr<JitFuture> pyRpcTorchscript(
    const std::string& dstWorkerName,
    const std::string& qualifiedNameStr,
    const float rpcTimeoutSeconds,
    const py::args& argsTuple,
    const py::kwargs& kwargsDict);

PyRRef pyRemoteBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs);

PyRRef pyRemotePythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors);

PyRRef pyRemoteTorchscript(
    const std::string& dstWorkerName,
    const std::string& qualifiedNameStr,
    const py::args& args,
    const py::kwargs& kwargs);

} // namespace rpc
} // namespace distributed
} // namespace torch
