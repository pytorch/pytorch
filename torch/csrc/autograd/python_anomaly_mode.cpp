#include <c10/util/Exception.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

#include <iostream>

namespace torch {
namespace autograd {

void PyAnomalyMetadata::store_stack() {
  pybind11::gil_scoped_acquire gil;
  THPObjectPtr mod(PyImport_ImportModule("torch.fx.traceback"));
  if (!mod) {
    throw python_error();
  }

  THPObjectPtr list(PyObject_CallMethod(mod.get(), "format_stack", ""));
  if (!list) {
    throw python_error();
  }

  if (PyDict_SetItemString(dict(), ANOMALY_TRACE_KEY, list.get())) {
    throw python_error();
  }
}

void PyAnomalyMetadata::print_stack(const std::string& current_node_name) {
  pybind11::gil_scoped_acquire gil;
  if (!PyDict_Check(dict())) {
    throw std::runtime_error("Anomaly metadata is not a python dictionary.");
  }
  PyObject* trace_stack = PyDict_GetItemString(dict(), ANOMALY_TRACE_KEY);
  _print_stack(trace_stack, current_node_name, false);
  PyObject* pyparent(PyDict_GetItemString(dict(), ANOMALY_PARENT_KEY));

  // if there is no "parent_" in metadata, then it means this metadata's node
  // is the root and stop printing the traceback
  while (pyparent) {
    THPObjectPtr parent_metadata(PyObject_GetAttrString(pyparent, "metadata"));
    if (!parent_metadata) {
      throw python_error();
    }
    THPObjectPtr parent_name_pyobj(PyObject_CallMethod(pyparent, "name", ""));
    if (!parent_name_pyobj) {
      throw python_error();
    }
    const char* parent_name_char = PyUnicode_AsUTF8(parent_name_pyobj.get());
    if (!parent_name_char) {
      throw python_error();
    }
    const std::string parent_name(parent_name_char);
    PyObject* parent_stack =
        PyDict_GetItemString(parent_metadata.get(), ANOMALY_TRACE_KEY);
    _print_stack(parent_stack, parent_name, true);
    // get the parent of this node, if this node is a root, pyparent is simply
    // null
    pyparent = PyDict_GetItemString(parent_metadata.get(), ANOMALY_PARENT_KEY);
  }
}

void PyAnomalyMetadata::assign_parent(
    const std::shared_ptr<Node>& parent_node) {
  // assign the python object of parent_node in metadata["parent_"]
  // if parent_node is nullptr, then do nothing (it can mean that "parent_" key
  // is not in metadata)

  pybind11::gil_scoped_acquire gil;
  if (!parent_node)
    return;

  THPObjectPtr parent_node_(functionToPyObject(parent_node));
  if (!parent_node_) {
    throw python_error();
  }
  if (PyDict_SetItemString(dict(), ANOMALY_PARENT_KEY, parent_node_.get())) {
    throw python_error();
  }
}

void _print_stack(
    PyObject* stack,
    const std::string& current_node_name,
    bool is_parent) {
  if (!stack) {
    TORCH_WARN(
        "Error detected in ",
        current_node_name,
        ". ",
        "No forward pass information available. Enable detect anomaly "
        "during forward pass for more information.");
    return;
  }

  THPObjectPtr empty_string(PyUnicode_FromString(""));
  if (!empty_string) {
    throw python_error();
  }

  // stack is a list of Python strings ending with newlines. Use join to convert
  // to a single string.
  THPObjectPtr msg(PyUnicode_Join(empty_string, stack));
  if (!msg) {
    throw python_error();
  }

  if (!is_parent) {
    TORCH_WARN(
        "Error detected in ",
        current_node_name,
        ". ",
        "Traceback of forward call that caused the error:\n",
        THPUtils_unpackString(msg.get()));
  } else {
    TORCH_WARN(
        "\n\n",
        "Previous calculation was induced by ",
        current_node_name,
        ". "
        "Traceback of forward call that induced the previous calculation:\n",
        THPUtils_unpackString(msg.get()));
  }
}

} // namespace autograd
} // namespace torch
