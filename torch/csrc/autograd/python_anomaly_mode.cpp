#include <c10/util/Exception.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/profiler/combined_traceback.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

#include <sstream>

namespace torch::autograd {

namespace {
// Filter function matching the logic used by MemoryViz.js frameFilter
bool shouldOmitFrame(const std::string& funcname, const std::string& filename) {
  // Functions to omit - matches MemoryViz.js omitFunctions
  static const std::vector<std::string> omitFunctions = {
      "unwind::unwind",
      "CapturedTraceback::gather",
      "gather_with_cpp",
      "_start",
      "__libc_start_main",
      "PyEval_",
      "PyObject_",
      "PyFunction_",
  };

  // Filenames to omit - matches MemoryViz.js omitFilenames
  static const std::vector<std::string> omitFilenames = {
      "core/boxing",
      "/Register",
      "/Redispatch",
      "pythonrun.c",
      "Modules/main.c",
      "Objects/call.c",
      "Objects/methodobject.c",
      "pycore_ceval.h",
      "ceval.c",
      "cpython/abstract.h",
  };

  for (const auto& of : omitFunctions) {
    if (funcname.find(of) != std::string::npos) {
      return true;
    }
  }

  for (const auto& of : omitFilenames) {
    if (filename.find(of) != std::string::npos) {
      return true;
    }
  }

  return false;
}
} // namespace

void PyAnomalyMetadata::store_stack() {
  if (AnomalyMode::should_use_mixed_stack()) {
    // Use CapturedTraceback for mixed Python/C++/TorchScript traces
    captured_traceback_ = torch::CapturedTraceback::gather(
        /*python=*/true, /*script=*/true, /*cpp=*/true);
    return;
  }

  // Existing behavior: use torch.fx.traceback.format_stack()
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
  // If we have a CapturedTraceback (mixed_stack mode), symbolize and print it
  if (captured_traceback_) {
    std::vector<torch::CapturedTraceback*> to_symbolize = {
        captured_traceback_.get()};
    torch::SymbolizedTracebacks symbolized = torch::symbolize(to_symbolize);

    std::vector<std::string> filtered_frames;
    for (uint64_t frame_idx : symbolized.tracebacks.at(0)) {
      const auto& frame = symbolized.all_frames.at(frame_idx);
      // Use the same filtering logic as MemoryViz.js frameFilter
      if (shouldOmitFrame(frame.funcname, frame.filename)) {
        continue;
      }
      std::ostringstream frame_oss;
      frame_oss << "  File \"" << frame.filename << "\", line " << frame.lineno
                << ", in " << frame.funcname;
      filtered_frames.push_back(frame_oss.str());
    }

    // Skip the first 3 frames as they are internal to anomaly mode
    // (store_stack, Node constructor, etc.)
    size_t start_idx = std::min(static_cast<size_t>(3), filtered_frames.size());

    std::ostringstream oss;
    for (size_t i = start_idx; i < filtered_frames.size(); ++i) {
      oss << filtered_frames[i] << "\n";
    }

    TORCH_WARN(
        "Error detected in ",
        current_node_name,
        ". ",
        "Traceback of forward call that caused the error:\n",
        oss.str());
    // TODO: Add parent traceback tracking for mixed mode. As it is rarely used
    // (due to lack of popularity of higher order derivative)
    return;
  }

  pybind11::gil_scoped_acquire gil;
  if (!PyDict_Check(dict())) {
    TORCH_CHECK(false, "Anomaly metadata is not a python dictionary.");
  }
  PyObject* trace_stack = nullptr;
  if (PyDict_GetItemStringRef(dict(), ANOMALY_TRACE_KEY, &trace_stack) < 0) {
    throw python_error();
  }
  _print_stack(trace_stack, current_node_name, false);
  PyObject* pyparent = nullptr;
  if (PyDict_GetItemStringRef(dict(), ANOMALY_PARENT_KEY, &pyparent) < 0) {
    throw python_error();
  }

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
    PyObject* parent_stack = nullptr;
    if (PyDict_GetItemStringRef(
            parent_metadata.get(), ANOMALY_TRACE_KEY, &parent_stack) < 0) {
      throw python_error();
    }
    _print_stack(parent_stack, parent_name, true);
    // get the parent of this node, if this node is a root, pyparent is simply
    // null
    if (PyDict_GetItemStringRef(
            parent_metadata.get(), ANOMALY_PARENT_KEY, &pyparent) < 0) {
      throw python_error();
    }
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

std::shared_ptr<torch::CapturedTraceback> PyAnomalyMetadata::
    captured_traceback() const {
  return captured_traceback_;
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

} // namespace torch::autograd
