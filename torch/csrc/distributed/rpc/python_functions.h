#pragma once

#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/rpc/py_rref.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

py::object toPyObj(const Message& message);

std::shared_ptr<FutureMessage> pyRpcBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const std::shared_ptr<torch::autograd::profiler::RecordFunctionAsync>& rf,
    const py::args& args,
    const py::kwargs& kwargs);

std::shared_ptr<FutureMessage> pyRpcPythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const std::shared_ptr<torch::autograd::profiler::RecordFunctionAsync>& rf);

PyRRef pyRemoteBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const std::shared_ptr<torch::autograd::profiler::RecordFunctionAsync>& rf,
    const py::args& args,
    const py::kwargs& kwargs);

PyRRef pyRemotePythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const std::shared_ptr<torch::autograd::profiler::RecordFunctionAsync>& rf);

} // namespace rpc
} // namespace distributed
} // namespace torch
