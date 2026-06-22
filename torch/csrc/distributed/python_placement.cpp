#include <torch/csrc/distributed/python_placement.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/distributed/Placement.h>
#include <torch/csrc/utils/pybind.h>

using namespace pybind11::literals;

namespace torch::distributed {
namespace {
const auto placement_class_docstring =
    R"(The base class for the Placement type, where it describes how a DTensor is placed onto the
``DeviceMesh``. ``Placement`` and ``DeviceMesh`` together could describe the DTensor Layout.
It is the base class of the three main DTensor Placement types: ``Shard``, ``Replicate``,
and ``Partial``.

This class is not meant to be used directly, mainly served as a typing stub.
)";
} // namespace

void initPlacementBindings(PyObject* module) {
  auto py_module = py::reinterpret_borrow<py::module>(module);
  auto distributed_module = py_module.def_submodule("_distributed");

  // Use OpaqueBase as the metaclass to allow isinstance(fake_obj, Placement) to
  // work.
  py::object opaque_base_module = py::module_::import("torch._opaque_base");
  py::object opaque_base = opaque_base_module.attr("OpaqueBaseMeta");

  auto placement_cls =
      py::class_<Placement>(
          distributed_module,
          "Placement",
          py::metaclass(opaque_base),
          placement_class_docstring)
          .def(py::init<>()) // Allow construction of Python subclasses.
          .def(
              "is_partial",
              &Placement::is_partial,
              py::arg("reduce_op") = py::none())
          .def("is_replicate", &Placement::is_replicate)
          .def("is_shard", &Placement::is_shard, py::arg("dim") = py::none());

  auto shard_cls =
      py::class_<Shard, Placement>(
          distributed_module, "Shard", py::metaclass(opaque_base))
          .def(py::init<int64_t>(), py::arg("dim"))
          .def_readonly("dim", &Shard::dim)
          .def("is_shard", &Shard::is_shard, py::arg("dim") = py::none())
          .def(
              "__eq__",
              [](const Shard& lhs, const Shard& rhs) { return lhs == rhs; },
              py::is_operator())
          // Note: we need to use dicts for pickling to match the old
          // dataclasses.
          .def(py::pickle(
              [](const Shard& shard) { return py::dict("dim"_a = shard.dim); },
              [](const py::dict& d) {
                return Shard(py::cast<int64_t>(d["dim"]));
              }));

  auto strided_shard_cls =
      py::class_<StridedShard, Placement>(
          distributed_module, "StridedShard", py::metaclass(opaque_base))
          .def(
              py::init<int64_t, int64_t>(),
              py::arg("dim"),
              py::kw_only(),
              py::arg("split_factor"))
          .def_readonly("dim", &StridedShard::dim)
          .def_readonly("split_factor", &StridedShard::split_factor)
          .def(
              "__eq__",
              [](const StridedShard& lhs, const StridedShard& rhs) {
                return lhs == rhs;
              },
              py::is_operator())
          .def(py::pickle(
              [](const StridedShard& shard) {
                return py::dict(
                    "dim"_a = shard.dim, "split_factor"_a = shard.split_factor);
              },
              [](const py::dict& d) {
                return StridedShard(
                    py::cast<int64_t>(d["dim"]),
                    py::cast<int64_t>(d["split_factor"]));
              }));

  auto replicate_cls =
      py::class_<Replicate, Placement>(
          distributed_module, "Replicate", py::metaclass(opaque_base))
          .def(py::init())
          .def("is_replicate", &Replicate::is_replicate)
          .def(
              "__eq__",
              [](const Replicate& lhs, const Replicate& rhs) {
                return lhs == rhs;
              },
              py::is_operator())
          .def(py::pickle(
              // I observed SIGSEGV when trying to use None as the
              // pickled state, though AFAICT that matches the
              // behavior of
              // object().__reduce__().
              // test_placement_types.test_type_identification will repro if an
              // enterprising reader wants to get this fixed.
              [](const Replicate& repl) { return py::dict(); },
              [](const py::dict&) { return Replicate(); }));

  auto partial_cls =
      py::class_<Partial, Placement>(
          distributed_module, "Partial", py::metaclass(opaque_base))
          .def(py::init<>())
          .def(py::init<std::optional<std::string>>(), py::arg("reduce_op"))
          .def_readonly("reduce_op", &Partial::reduce_op)
          .def(
              "is_partial",
              &Partial::is_partial,
              py::arg("reduce_op") = py::none())
          .def(
              "__eq__",
              [](const Partial& lhs, const Partial& rhs) { return lhs == rhs; },
              py::is_operator())
          .def(py::pickle(
              [](const Partial& part) {
                return py::dict("reduce_op"_a = part.reduce_op);
              },
              [](const py::dict& d) {
                return Partial(py::cast<std::string>(d["reduce_op"]));
              }));
}
} // namespace torch::distributed
