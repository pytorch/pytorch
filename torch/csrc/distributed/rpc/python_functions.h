#pragma once

#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/rpc/py_rref.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

std::shared_ptr<jit::PythonFutureWrapper> pyRpcBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf,
    const py::args& args,
    const py::kwargs& kwargs);

std::shared_ptr<jit::PythonFutureWrapper> pyRpcPythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf);

std::shared_ptr<jit::PythonFutureWrapper> pyRpcTorchscript(
    const std::string& dstWorkerName,
    const std::string& qualifiedNameStr,
    const py::tuple& argsTuple,
    const py::dict& kwargsDict);

PyRRef pyRemoteBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf,
    const py::args& args,
    const py::kwargs& kwargs);

PyRRef pyRemotePythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf);

PyRRef pyRemoteTorchscript(
    const std::string& dstWorkerName,
    const std::string& qualifiedNameStr,
    const py::args& args,
    const py::kwargs& kwargs);

} // namespace rpc
} // namespace distributed
} // namespace torch
