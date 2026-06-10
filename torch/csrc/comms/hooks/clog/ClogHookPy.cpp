// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/comms/TorchComm.hpp>
#include <torch/csrc/comms/hooks/clog/ClogHook.hpp>

namespace py = pybind11;
using namespace torch::comms;

void init_clog_hook_bindings(py::module_& m) {
  py::class_<ClogHook, std::shared_ptr<ClogHook>>(
      m,
      "clog",
      R"(
clog logs collective operation signatures and lifecycle events
to a pipe-delimited log file.

Example:
    >>> from torch.comms.hooks import clog
    >>> logger = clog(output="/tmp/clog.log", events=["ALL"])
    >>> logger.register_with_comm(comm_a)
    >>> logger.register_with_comm(comm_b)
    >>> # ... run collectives ...
      )")
      .def(
          py::init<
              std::string,
              std::vector<std::string>,
              std::vector<std::string>>(),
          R"(
          Create a clog logger.

          Args:
              output: File path for log output.
              events: Events to log (LIFECYCLE, ALL). Enqueue events are always logged.
              verbose: Optional fields (buffers).
          )",
          py::arg("output"),
          py::arg("events"),
          py::arg("verbose") = std::vector<std::string>{})
      .def(
          "register_with_comm",
          &ClogHook::registerWithComm,
          R"(
          Register this hook with a TorchComm communicator.

          Args:
              comm: The communicator to register with.
          )",
          py::arg("comm"),
          py::call_guard<py::gil_scoped_release>());
}
