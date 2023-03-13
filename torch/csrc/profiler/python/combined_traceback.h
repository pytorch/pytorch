#include <torch/csrc/profiler/combined_traceback.h>

#include <pybind11/pybind11.h>

namespace torch {

// symbolize combined traceback objects, converting them into lists of
// dictionaries that are easily consumed in python.
std::vector<pybind11::object> py_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize);

void installCapturedTracebackPython();

} // namespace torch
