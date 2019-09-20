#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace PythonRpcHandler {
// initialize python module object and function objects in which python user
// defined function (UDF) will run there
void init();
// execute python UDF, result is pickled to binary string
std::vector<char> generatePythonUDFResult(const Message& message);
py::object runPythonUDF(const std::string& pickledPythonUDF);
std::string serialize(const py::object& obj);
py::object deserialize(const std::string& serializedObj);
// returned python UDF result is pickled binary string, so run python
// function to unpickle the python UDF result and return pyObject to user
py::object loadPythonUDFResult(const Message& message);
} // namespace PythonRpcHandler

} // namespace rpc
} // namespace distributed
} // namespace torch
