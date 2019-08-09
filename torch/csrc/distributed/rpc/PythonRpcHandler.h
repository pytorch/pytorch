#pragma once

#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/utils/pybind.h>
#include <Python.h>

namespace torch {
namespace distributed {
namespace rpc {

class PythonRpcHandler {
public:
  static void init();
  static void cleanUp();

  static std::vector<char> generatePythonUDFResult(const Message& request);
  static py::object loadPythonUDFResult(const Message& message);

private:
  static PyObject* module_;
  static PyObject* runUDFFunction_;
  static PyObject* loadResultFunction_;
  static bool pyInitializerCalled_;
};

}
}
}
