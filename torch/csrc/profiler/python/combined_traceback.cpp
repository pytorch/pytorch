#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/profiler/python/combined_traceback.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
namespace py = pybind11;

namespace torch {
// Locking:
// We need to free PyCodeObjects when ~StackContext runs, but
// CUDACachingAllocator may hold its device lock when ~StackContext runs.

// Because the thread calling the allocator _may_ hold the GIL,
// attempting to lock the GIL in ~StackContext can deadlock:
// T0: GIL Lock -> Call Allocator    ->| Waiting Device Lock
// T1: Call Allocator -> Device Lock ->| Waiting GIL Lock
// Instead the destructor defers freeing stack frames by putting them in
// to_free_frames. We still need a lock to manage this vector, but
// we can ensure an overall lock ordering of GIL -> device_lock ->
// to_free_frames_mutex because ::gather is called outside of the device lock.

namespace {
static std::mutex to_free_frames_mutex;
static std::vector<CapturedTraceback::PyFrame> to_free_frames;
struct PythonTraceback : public CapturedTraceback::Python {
  bool canGather() override {
    // Check if it's safe to gather Python frames from the current thread.
    // Returns false for pure C++ threads that cannot acquire the GIL.
    if (!Py_IsInitialized()) {
      return false;
    }
    // Already holding GIL - safe to gather
    if (PyGILState_Check() == 1) {
      return true;
    }
    // Thread is registered with Python - can acquire GIL
    if (PyGILState_GetThisThreadState() != nullptr) {
      return true;
    }
    // Pure C++ thread with no Python state - cannot acquire GIL
    return false;
  }
  std::vector<CapturedTraceback::PyFrame> gather() override {
    std::vector<CapturedTraceback::PyFrame> frames;
    py::gil_scoped_acquire acquire;
    {
      std::lock_guard<std::mutex> lock(to_free_frames_mutex);
      for (CapturedTraceback::PyFrame f : to_free_frames) {
        Py_XDECREF(f.code);
      }
      to_free_frames.clear();
    }
    PyFrameObject* f = PyEval_GetFrame();
    Py_XINCREF(f);
    while (f) {
      frames.emplace_back(
          CapturedTraceback::PyFrame{PyFrame_GetCode(f), PyFrame_GetLasti(f)});
      auto f_back = PyFrame_GetBack(f);
      Py_XDECREF(f);
      f = f_back;
    }
    return frames;
  }
  void release(std::vector<CapturedTraceback::PyFrame>& frames) override {
    std::lock_guard<std::mutex> lock(to_free_frames_mutex);
    to_free_frames.insert(to_free_frames.end(), frames.begin(), frames.end());
  }
  using void_visitproc = int (*)(void* self, void* arg);
  int traverse(
      std::vector<CapturedTraceback::PyFrame>& frames,
      void_visitproc visit,
      void* arg) override {
    for (auto& f : frames) {
      Py_VISIT(f.code);
    }
    return 0;
  }
  int clear(std::vector<CapturedTraceback::PyFrame>& frames) override {
    for (auto& f : frames) {
      Py_CLEAR(f.code);
    }
    return 0;
  }
  void appendSymbolized(
      const std::vector<CapturedTraceback::PyFrame>& to_symbolize,
      SymbolizedTracebacks& result) override {
    py::gil_scoped_acquire acquire;
    py::str line_s = "line";
    py::str name_s = "name";
    py::str filename_s = "filename";

    auto torch = py::module::import("torch");
    py::object stack_frames_for_code;
    if (py::hasattr(torch, "_inductor")) {
      py::object inductor = torch.attr("_inductor");
      if (py::hasattr(inductor, "codecache")) {
        stack_frames_for_code = inductor.attr("codecache")
                                    .attr("PyCodeCache")
                                    .attr("stack_frames_for_code");
      }
    }
    for (const auto& f : to_symbolize) {
      auto f_code = (PyCodeObject*)f.code;
      py::handle filename = f_code->co_filename;
      py::handle funcname = f_code->co_name;
      auto lineno = PyCode_Addr2Line(f_code, f.lasti);
      result.tracebacks.emplace_back();
      result.tracebacks.back().push_back(result.all_frames.size());
      result.all_frames.emplace_back(unwind::Frame{
          py::cast<std::string>(filename),
          py::cast<std::string>(funcname),
          (uint64_t)lineno});
      // find all the additional frames associated with inductor generated
      // code
      if (stack_frames_for_code.ptr()) {
        py::object extra = stack_frames_for_code(filename, lineno);
        if (!extra.is_none()) {
          for (py::handle h : extra) {
            result.tracebacks.back().push_back(result.all_frames.size());
            result.all_frames.emplace_back(unwind::Frame{
                py::cast<std::string>(h[filename_s]),
                py::cast<std::string>(h[name_s]),
                py::cast<uint64_t>(h[line_s])});
          }
        }
      }
    }
  }

  // Extract forward traceback from the current autograd node's anomaly
  // metadata. Returns a vector of strings representing the forward stack trace,
  // or empty if not available.
  std::vector<std::string> gatherForwardTraceback() override {
    std::vector<std::string> result;

    // Get the currently executing backward node
    auto node = torch::autograd::get_current_node();
    if (!node) {
      return result;
    }

    // Get metadata from the node.
    // Note: metadata() may create new metadata if it doesn't exist, but we need
    // to check the dict for ANOMALY_TRACE_KEY anyway to know if forward tracing
    // was actually enabled during forward pass.
    auto* base_metadata = node->metadata();
    if (!base_metadata) {
      return result;
    }

    // Check if the metadata is a Python anomaly metadata (which contains the
    // dict)
    auto* metadata =
        dynamic_cast<torch::autograd::PyAnomalyMetadata*>(base_metadata);
    if (!metadata) {
      return result;
    }

    // Get the traceback from the metadata dict.
    // This runs from a CUDA allocator callback, so a Python exception may
    // already be pending (e.g. the forward function just raised). The compat
    // shim for PyDict_GetItemRef on Python < 3.13 uses PyErr_Occurred() to
    // distinguish "not found" from "error", so a stale pending exception would
    // be misread as a lookup failure and then cleared, destroying the real
    // exception. Save/restore the exception state to avoid that.
    py::gil_scoped_acquire gil;

    PyObject* exc_type = nullptr;
    PyObject* exc_value = nullptr;
    PyObject* exc_tb = nullptr;
    PyErr_Fetch(&exc_type, &exc_value, &exc_tb);

    PyObject* dict = metadata->dict();
    if (!dict || !PyDict_Check(dict)) {
      PyErr_Restore(exc_type, exc_value, exc_tb);
      return result;
    }

    PyObject* traceback = nullptr;
    if (PyDict_GetItemStringRef(
            dict,
            torch::autograd::PyAnomalyMetadata::ANOMALY_TRACE_KEY,
            &traceback) < 0) {
      PyErr_Clear();
      PyErr_Restore(exc_type, exc_value, exc_tb);
      return result;
    }

    if (!traceback || !PyList_Check(traceback)) {
      Py_XDECREF(traceback);
      PyErr_Restore(exc_type, exc_value, exc_tb);
      return result;
    }

    // Convert Python list of strings to vector of strings
    Py_ssize_t size = PyList_Size(traceback);
    result.reserve(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject* item = PyList_GetItem(traceback, i); // borrowed reference
      if (item && PyUnicode_Check(item)) {
        const char* str = PyUnicode_AsUTF8(item);
        if (str) {
          result.emplace_back(str);
        }
      }
    }

    Py_DECREF(traceback);
    PyErr_Restore(exc_type, exc_value, exc_tb);
    return result;
  }
};

} // namespace

std::vector<nlohmann::json> json_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize) {
  std::unordered_map<CapturedTraceback*, uint64_t> cached_frames;
  std::vector<CapturedTraceback*> unique_frames;
  for (const auto& sc : to_symbolize) {
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      cached_frames.try_emplace(sc, unique_frames.size());
      unique_frames.push_back(sc);
    }
  }
  auto s = symbolize(unique_frames);

  std::string line_s = "line";
  std::string name_s = "name";
  std::string filename_s = "filename";
  std::vector<nlohmann::json> all_frames;

  for (const auto& f : s.all_frames) {
    nlohmann::json d;
    d[name_s] = f.funcname;
    d[filename_s] = f.filename;
    d[line_s] = f.lineno;
    all_frames.emplace_back(std::move(d));
  }

  std::vector<nlohmann::json> py_unique_frames;
  for (const auto& t : s.tracebacks) {
    nlohmann::json l;
    for (const auto& e : t) {
      l.emplace_back(all_frames.at(e));
    }
    py_unique_frames.push_back(std::move(l));
  }

  std::vector<nlohmann::json> result;
  result.reserve(to_symbolize.size());
  for (const auto& sc : to_symbolize) {
    result.push_back(py_unique_frames.at(cached_frames.at(sc)));
  }
  return result;
}

std::vector<py::object> py_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize) {
  // we dedup repeated to_symbolize objects to prevent
  // creating a bunch of duplicated frame objects
  std::unordered_map<CapturedTraceback*, uint64_t> cached_frames;
  std::vector<CapturedTraceback*> unique_frames;
  for (const auto& sc : to_symbolize) {
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      cached_frames.insert({sc, unique_frames.size()});
      unique_frames.push_back(sc);
    }
  }
  auto s = symbolize(unique_frames);

  py::str line_s = "line";
  py::str name_s = "name";
  py::str filename_s = "filename";
  std::vector<py::dict> all_frames;
  for (const auto& f : s.all_frames) {
    py::dict d;
    d[name_s] = f.funcname;
    d[filename_s] = f.filename;
    d[line_s] = f.lineno;
    all_frames.emplace_back(std::move(d));
  }

  std::vector<py::object> py_unique_frames;
  for (const auto& t : s.tracebacks) {
    py::list l;
    for (const auto& e : t) {
      l.append(all_frames.at(e));
    }
    py_unique_frames.push_back(std::move(l));
  }

  std::vector<py::object> result;
  result.reserve(to_symbolize.size());
  for (const auto& sc : to_symbolize) {
    result.push_back(py_unique_frames.at(cached_frames.at(sc)));
  }
  return result;
}

void freeDeadCapturedTracebackFrames() {
  std::lock_guard<std::mutex> lock(to_free_frames_mutex);
  for (CapturedTraceback::PyFrame f : to_free_frames) {
    Py_XDECREF(f.code);
  }
  to_free_frames.clear();
}

void installCapturedTracebackPython() {
  CapturedTraceback::addPythonUnwinder(new PythonTraceback());
}

} // namespace torch
