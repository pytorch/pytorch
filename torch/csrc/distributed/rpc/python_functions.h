#pragma once

#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/py_rref.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

py::object to_py_obj(const Message& message);

std::shared_ptr<FutureMessage> py_rpc_builtin(
    RpcAgent& agent,
    const WorkerId& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs);

std::shared_ptr<FutureMessage> py_rpc_python_udf(
    RpcAgent& agent,
    const WorkerId& dst,
    const std::string& pickledPythonUDF);

PyRRef py_remote_builtin(
    RpcAgent& agent,
    const WorkerId& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs);

PyRRef py_remote_python_udf(
    RpcAgent& agent,
    const WorkerId& dst,
    const std::string& pickledPythonUDF);

} // namespace rpc
} // namespace distributed
} // namespace torch
