// TiledIR pybind11 bindings. Owned by the tiling worker.
//
// Defines the GLOBAL `bind_tiled_ir(pybind11::module_&)` helper OUTSIDE the
// loops_ir namespace. The core worker's bindings.cpp forward-declares and
// calls it after core/kernel registration; do not modify bindings.cpp.
//
// Exposed surface (all snakecase):
//   TileOptions(block_size=256, num_warps=4) with readonly
//     block_size/num_warps.
//   TiledInstruction factories (program_id/arange/const_int/const_float/
//     broadcast/int_binary/cmp_lt/select/float_binary/relu/load/store/cast)
//     plus readonly op/id/dtype/shape/operands/int_value/float_value/
//     buffer/is_vector. Op names are plain strings (stable across the C++
//     boundary); the full op list lives in tiled_ir.h.
//   TiledBlock(name) with add_instruction/set_terminator plus readonly
//     name/instructions/terminator.
//   TiledFunction(...) with add_block plus readonly name/parameters/blocks/
//     grid_size/block_size/num_warps/enable_fp_fusion/disable_ftz/numerics.
//   TiledProgram(numerics="strict_eager") with add_function plus
//   verify()/dump()
//     and readonly functions/numerics.
//   lower_to_tiled(kernel_program, options=TileOptions()).
//
// C++ errors surface as ValueError (std::invalid_argument), matching core.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include "kernel_ir.h"
#include "tiled_ir.h"

namespace py = pybind11;
using namespace torch::inductor::loops_ir;

void bind_tiled_ir(py::module_& m) {
  py::class_<TileOptions>(m, "TileOptions")
      .def(
          py::init<int64_t, int64_t, bool, bool>(),
          py::arg("block_size") = 256,
          py::arg("num_warps") = 4,
          py::arg("enable_fp_fusion") = false,
          py::arg("disable_ftz") = true)
      .def_readonly("block_size", &TileOptions::block_size)
      .def_readonly("num_warps", &TileOptions::num_warps)
      .def_readonly("enable_fp_fusion", &TileOptions::enable_fp_fusion)
      .def_readonly("disable_ftz", &TileOptions::disable_ftz);

  py::class_<TiledInstruction>(m, "TiledInstruction")
      .def_static("program_id", &TiledInstruction::ProgramId, py::arg("id"))
      .def_static(
          "arange", &TiledInstruction::Arange, py::arg("id"), py::arg("lanes"))
      .def_static(
          "const_int",
          &TiledInstruction::ConstInt,
          py::arg("id"),
          py::arg("value"))
      .def_static(
          "const_float",
          &TiledInstruction::ConstFloat,
          py::arg("id"),
          py::arg("value"))
      .def_static(
          "broadcast",
          &TiledInstruction::Broadcast,
          py::arg("id"),
          py::arg("operand"),
          py::arg("dtype"),
          py::arg("lanes"))
      .def_static(
          "int_binary",
          &TiledInstruction::IntBinary,
          py::arg("op"),
          py::arg("id"),
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("lanes"))
      .def_static(
          "cmp_lt",
          &TiledInstruction::CmpLt,
          py::arg("id"),
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("lanes"))
      .def_static(
          "select",
          &TiledInstruction::Select,
          py::arg("id"),
          py::arg("cond"),
          py::arg("on_true"),
          py::arg("on_false"),
          py::arg("dtype"),
          py::arg("lanes"))
      .def_static(
          "float_binary",
          &TiledInstruction::FloatBinary,
          py::arg("op"),
          py::arg("id"),
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("lanes"))
      .def_static(
          "relu",
          &TiledInstruction::Relu,
          py::arg("id"),
          py::arg("arg"),
          py::arg("lanes"))
      .def_static(
          "load",
          &TiledInstruction::Load,
          py::arg("id"),
          py::arg("buffer"),
          py::arg("offset"),
          py::arg("mask"),
          py::arg("lanes"))
      .def_static(
          "store",
          &TiledInstruction::Store,
          py::arg("buffer"),
          py::arg("value"),
          py::arg("offset"),
          py::arg("mask"))
      .def_static(
          "cast",
          &TiledInstruction::Cast,
          py::arg("id"),
          py::arg("arg"),
          py::arg("dtype"),
          py::arg("lanes"))
      .def_property_readonly("op", &TiledInstruction::op)
      .def_property_readonly("id", &TiledInstruction::id)
      .def_property_readonly("dtype", &TiledInstruction::dtype)
      .def_property_readonly("shape", &TiledInstruction::shape)
      .def_property_readonly("operands", &TiledInstruction::operands)
      .def_property_readonly("int_value", &TiledInstruction::int_value)
      .def_property_readonly("float_value", &TiledInstruction::float_value)
      .def_property_readonly("buffer", &TiledInstruction::buffer)
      .def_property_readonly("is_vector", &TiledInstruction::isVector);

  py::class_<TiledBlock>(m, "TiledBlock")
      .def(py::init<std::string>(), py::arg("name"))
      .def(
          "add_instruction",
          &TiledBlock::addInstruction,
          py::arg("instruction"))
      .def("set_terminator", &TiledBlock::setTerminator, py::arg("terminator"))
      .def_property_readonly("name", &TiledBlock::name)
      .def_property_readonly("instructions", &TiledBlock::instructions)
      .def_property_readonly("terminator", &TiledBlock::terminator);

  py::class_<TiledFunction>(m, "TiledFunction")
      .def(
          py::init<
              std::string,
              std::vector<Buffer>,
              int64_t,
              int64_t,
              int64_t,
              bool,
              bool,
              std::string>(),
          py::arg("name"),
          py::arg("parameters"),
          py::arg("grid_size"),
          py::arg("block_size"),
          py::arg("num_warps"),
          py::arg("enable_fp_fusion") = false,
          py::arg("disable_ftz") = true,
          py::arg("numerics") = "strict_eager")
      .def("add_block", &TiledFunction::addBlock, py::arg("block"))
      .def_property_readonly("name", &TiledFunction::name)
      .def_property_readonly("parameters", &TiledFunction::parameters)
      .def_property_readonly("blocks", &TiledFunction::blocks)
      .def_property_readonly("grid_size", &TiledFunction::grid_size)
      .def_property_readonly("block_size", &TiledFunction::block_size)
      .def_property_readonly("num_warps", &TiledFunction::num_warps)
      .def_property_readonly(
          "enable_fp_fusion", &TiledFunction::enable_fp_fusion)
      .def_property_readonly("disable_ftz", &TiledFunction::disable_ftz)
      .def_property_readonly("numerics", &TiledFunction::numerics);

  py::class_<TiledProgram>(m, "TiledProgram")
      .def(py::init<std::string>(), py::arg("numerics") = "strict_eager")
      .def("add_function", &TiledProgram::addFunction, py::arg("function"))
      .def("verify", &TiledProgram::verify)
      .def("dump", &TiledProgram::dump)
      .def_property_readonly("functions", &TiledProgram::functions)
      .def_property_readonly("numerics", &TiledProgram::numerics);

  m.def(
      "lower_to_tiled",
      &lower_to_tiled,
      py::arg("kernel_program"),
      py::arg("options") = TileOptions());
}
