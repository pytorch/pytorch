// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/comms/hooks/fr/FlightRecorder.hpp>

namespace py = pybind11;
using namespace torch::comms::fr;

void init_flight_recorder_bindings(py::module_& m) {
  // Bind FlightRecorderHook class
  py::class_<FlightRecorderHook, std::shared_ptr<FlightRecorderHook>>(
      m,
      "FlightRecorderHook",
      R"(
FlightRecorderHook tracks all collective operations in flight for TorchComm communicators.

The output format matches the OSS FlightRecorder format from PyTorch's
distributed module, so traces can be analyzed using the same fr_trace
analysis tools.

Example:
    >>> from torch.comms.hooks import fr
    >>> comm = torch.comms.new_comm("nccl", device, "world")
    >>> recorder = fr.FlightRecorderHook(max_entries=1024)
    >>> recorder.register_with_comm(comm)
    >>> # ... run some collectives ...
    >>> json_trace = recorder.dump_json()

For testing, use ``isolated=True`` to create a separate FlightRecorder instance
that is not shared with other hooks::

    >>> recorder = fr.FlightRecorderHook(max_entries=100, isolated=True)
      )")
      .def(
          py::init<size_t, bool>(),
          R"(
          Create a FlightRecorderHook with specified buffer size.

          Args:
              max_entries: Maximum number of entries in the ring buffer.
                          Older entries are overwritten when full.
              isolated: If True, creates an isolated FlightRecorder instance
                       for this hook instead of using the global singleton.
          )",
          py::arg("max_entries") = 2048,
          py::arg("isolated") = false)
      .def(
          "register_with_comm",
          &FlightRecorderHook::registerWithComm,
          R"(
          Register this hook with a TorchComm communicator.

          Args:
              comm: The communicator to register with.
          )",
          py::arg("comm"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "dump_json",
          &FlightRecorderHook::dump_json,
          R"(
          Dump all entries as a JSON string in the OSS FlightRecorder format.

          This format is compatible with the fr_trace analyzer tools from
          torch.distributed.flight_recorder.

          Args:
              include_completed: If False, only return entries that are not completed.

          Returns:
              A JSON string containing the flight recorder trace.
          )",
          py::arg("include_completed") = true,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "reset",
          &FlightRecorderHook::reset,
          "Clear all entries and reset sequence counters.",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "is_enabled",
          &FlightRecorderHook::isEnabled,
          "Check if the hook has registered communicators.")
      .def(
          "__len__",
          &FlightRecorderHook::size,
          "Get the current number of entries.")
      .def(
          "size",
          &FlightRecorderHook::size,
          "Get the current number of entries.")
      .def(
          "dump_file",
          &FlightRecorderHook::dump_file,
          R"(
          Dump the flight recorder trace and write it to a file.

          The output location is controlled by the TORCHCOMM_FR_DUMP_TEMP_FILE
          environment variable. Files are written as <prefix><rank>.

          Args:
              rank: The rank to use for the file name.
              include_completed: If False, only dump entries that are not completed.
          )",
          py::arg("rank"),
          py::arg("include_completed") = true,
          py::call_guard<py::gil_scoped_release>());
}
