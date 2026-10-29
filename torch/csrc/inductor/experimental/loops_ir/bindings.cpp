// pybind11 bindings. Standalone module TORCH_EXTENSION_NAME; only pybind11
// headers plus ir.h / passes.h / kernel_ir.h. C++ errors surface as
// ValueError (std::invalid_argument).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ir.h"
#include "kernel_ir.h"
#include "pass_manager.h"
#include "passes.h"

namespace pybind11 {
class module_;
} // namespace pybind11

// Defined by the tiling worker in tiled_bindings.cpp (not edited here).
void bind_tiled_ir(pybind11::module_& m);

namespace py = pybind11;
using namespace torch::inductor::loops_ir;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Standalone inductor loops IR (typed IR + index tree)";

  py::class_<Index>(m, "Index")
      .def_static("constant", &Index::Constant, py::arg("value"))
      .def_static("variable", &Index::Variable, py::arg("name"))
      .def_static(
          "binary",
          &Index::Binary,
          py::arg("op"),
          py::arg("lhs"),
          py::arg("rhs"))
      .def("to_string", &Index::toString)
      .def("__str__", &Index::toString)
      .def("__repr__", [](const Index& self) {
        return "Index(" + self.toString() + ")";
      });

  py::class_<Buffer>(m, "Buffer")
      .def(
          py::init<std::string, std::string, int64_t>(),
          py::arg("name"),
          py::arg("dtype"),
          py::arg("elements"))
      .def_readwrite("name", &Buffer::name)
      .def_readwrite("dtype", &Buffer::dtype)
      .def_readwrite("elements", &Buffer::elements);

  py::class_<Descriptor>(m, "Descriptor")
      .def(
          py::init<
              std::string,
              std::string,
              std::vector<int64_t>,
              std::vector<int64_t>,
              int64_t>(),
          py::arg("name"),
          py::arg("buffer"),
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("offset") = 0)
      .def_readwrite("name", &Descriptor::name)
      .def_readwrite("buffer", &Descriptor::buffer)
      .def_readwrite("sizes", &Descriptor::sizes)
      .def_readwrite("strides", &Descriptor::strides)
      .def_readwrite("offset", &Descriptor::offset);

  py::class_<View>(m, "View")
      .def(
          py::init<
              std::string,
              std::string,
              std::vector<int64_t>,
              std::vector<int64_t>,
              int64_t>(),
          py::arg("name"),
          py::arg("source"),
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("offset") = 0)
      .def_readwrite("name", &View::name)
      .def_readwrite("source", &View::source)
      .def_readwrite("sizes", &View::sizes)
      .def_readwrite("strides", &View::strides)
      .def_readwrite("offset", &View::offset);

  py::class_<Range>(m, "Range")
      .def(
          py::init<std::string, int64_t, int64_t>(),
          py::arg("name"),
          py::arg("start"),
          py::arg("stop"))
      .def_readwrite("name", &Range::name)
      .def_readwrite("start", &Range::start)
      .def_readwrite("stop", &Range::stop);

  py::class_<Accumulator>(m, "Accumulator")
      .def(
          py::init<std::string, std::string, double, std::string>(),
          py::arg("name"),
          py::arg("dtype"),
          py::arg("identity"),
          py::arg("combine") = "add")
      .def_readwrite("name", &Accumulator::name)
      .def_readwrite("dtype", &Accumulator::dtype)
      .def_readwrite("identity", &Accumulator::identity)
      .def_readwrite("combine", &Accumulator::combine);

  py::class_<CallArg>(m, "CallArg")
      .def_static("none", &CallArg::MakeNone)
      .def_static("tensor", &CallArg::MakeTensor, py::arg("name"))
      .def_static("integer", &CallArg::MakeInteger, py::arg("value"))
      .def_static("floating", &CallArg::MakeFloating, py::arg("value"))
      .def_static("boolean", &CallArg::MakeBoolean, py::arg("value"))
      .def_static("string", &CallArg::MakeString, py::arg("value"))
      .def_static("dtype", &CallArg::MakeDtype, py::arg("name"))
      .def_static("layout", &CallArg::MakeLayout, py::arg("name"))
      .def_static("memory_format", &CallArg::MakeMemoryFormat, py::arg("name"))
      .def_static("list", &CallArg::MakeList, py::arg("items"))
      .def_static("tuple", &CallArg::MakeTuple, py::arg("items"))
      .def_property_readonly("kind_name", &CallArg::kindName)
      .def_property_readonly("tensor_name", &CallArg::tensorName)
      .def_property_readonly("int_value", &CallArg::intValue)
      .def_property_readonly("float_value", &CallArg::floatValue)
      .def_property_readonly("bool_value", &CallArg::boolValue)
      .def_property_readonly("text", &CallArg::text)
      .def_property_readonly("items", &CallArg::items);

  py::class_<CallKeyword>(m, "CallKeyword")
      .def(py::init<std::string, CallArg>(), py::arg("name"), py::arg("value"))
      .def_property_readonly("name", &CallKeyword::name)
      .def_property_readonly("value", &CallKeyword::value);

  py::class_<Instruction>(m, "Instruction")
      .def_static(
          "constant", &Instruction::Constant, py::arg("id"), py::arg("value"))
      .def_static(
          "load",
          &Instruction::Load,
          py::arg("id"),
          py::arg("buffer"),
          py::arg("offset"))
      .def_static(
          "view_read",
          &Instruction::ViewRead,
          py::arg("id"),
          py::arg("view"),
          py::arg("coordinates"))
      .def_static(
          "binary",
          &Instruction::Binary,
          py::arg("id"),
          py::arg("op"),
          py::arg("lhs"),
          py::arg("rhs"))
      .def_static("relu", &Instruction::Relu, py::arg("id"), py::arg("arg"))
      .def_static("accumulate", &Instruction::Accumulate, py::arg("args"))
      .def_static(
          "read_accumulator",
          &Instruction::ReadAccumulator,
          py::arg("id"),
          py::arg("accumulator"))
      .def_static(
          "store",
          &Instruction::Store,
          py::arg("buffer"),
          py::arg("offset"),
          py::arg("value"))
      .def_property_readonly("kind", &Instruction::kindName)
      .def_property_readonly("id", &Instruction::id)
      .def_property_readonly("value", &Instruction::value)
      .def_property_readonly("buffer", &Instruction::buffer)
      .def_property_readonly("offset", &Instruction::offset)
      .def_property_readonly("view", &Instruction::view)
      .def_property_readonly("coordinates", &Instruction::coordinates)
      .def_property_readonly("op", &Instruction::op)
      .def_property_readonly("lhs", &Instruction::lhs)
      .def_property_readonly("rhs", &Instruction::rhs)
      .def_property_readonly("arg", &Instruction::arg)
      .def_property_readonly("args", &Instruction::args)
      .def_property_readonly("accumulator", &Instruction::accumulator);

  py::class_<Compute>(m, "Compute")
      .def(
          py::init<
              std::string,
              std::vector<std::string>,
              std::vector<Buffer>,
              std::vector<Range>,
              std::vector<Range>,
              std::vector<Accumulator>>(),
          py::arg("name"),
          py::arg("inputs"),
          py::arg("results"),
          py::arg("pointwise_ranges"),
          py::arg("reduction_ranges"),
          py::arg("accumulators"))
      .def(
          "add_reduction_instruction",
          &Compute::addReductionInstruction,
          py::arg("instruction"))
      .def(
          "add_post_instruction",
          &Compute::addPostInstruction,
          py::arg("instruction"))
      .def_property_readonly("name", &Compute::name)
      .def_property_readonly("inputs", &Compute::inputs)
      .def_property_readonly("results", &Compute::results)
      .def_property_readonly("pointwise_ranges", &Compute::pointwiseRanges)
      .def_property_readonly("reduction_ranges", &Compute::reductionRanges)
      .def_property_readonly("accumulators", &Compute::accumulators)
      .def_property_readonly("reduction_body", &Compute::reductionBody)
      .def_property_readonly("post_body", &Compute::postBody);

  py::class_<ExternCall>(m, "ExternCall")
      .def(
          py::init<
              std::string,
              std::string,
              std::vector<std::string>,
              std::vector<Buffer>,
              std::vector<double>,
              std::string,
              bool,
              std::vector<Descriptor>>(),
          py::arg("name"),
          py::arg("target"),
          py::arg("inputs"),
          py::arg("results"),
          py::arg("scalar_args"),
          py::arg("effects") = "unknown",
          py::arg("may_alias_inputs") = true,
          py::arg("descriptors") = std::vector<Descriptor>{})
      .def_static(
          "structured",
          &ExternCall::Structured,
          py::arg("name"),
          py::arg("target"),
          py::arg("arguments"),
          py::arg("keyword_args"),
          py::arg("results"),
          py::arg("descriptors"),
          py::arg("effects") = "unknown",
          py::arg("may_alias_inputs") = true)
      .def_property_readonly("name", &ExternCall::name)
      .def_property_readonly("target", &ExternCall::target)
      .def_property_readonly("inputs", &ExternCall::inputs)
      .def_property_readonly("results", &ExternCall::results)
      .def_property_readonly("scalar_args", &ExternCall::scalarArgs)
      .def_property_readonly("effects", &ExternCall::effects)
      .def_property_readonly("may_alias_inputs", &ExternCall::mayAliasInputs)
      .def_property_readonly("descriptors", &ExternCall::descriptors)
      .def_property_readonly("arguments", &ExternCall::arguments)
      .def_property_readonly("keyword_args", &ExternCall::keywordArgs);

  py::class_<Program>(m, "Program")
      .def(
          py::init<std::string, std::string>(),
          py::arg("name"),
          py::arg("numerics") = "strict_eager")
      .def("add_input", &Program::addInput, py::arg("buffer"))
      .def(
          "add_input_descriptor",
          &Program::addInputDescriptor,
          py::arg("descriptor"))
      .def("add_view", &Program::addView, py::arg("view"))
      .def("add_compute", &Program::addCompute, py::arg("compute"))
      .def("add_extern_call", &Program::addExternCall, py::arg("extern_call"))
      .def("add_output", &Program::addOutput, py::arg("descriptor"))
      .def("verify", &Program::verify)
      .def("dump", &Program::dump)
      .def_property_readonly("name", &Program::name)
      .def_property_readonly("inputs", &Program::inputs)
      .def_property_readonly("input_descriptors", &Program::inputDescriptors)
      .def_property_readonly("nodes", &Program::nodes)
      .def_property_readonly("outputs", &Program::outputs)
      .def_property_readonly("numerics", &Program::numerics);

  py::class_<KernelFunction>(m, "KernelFunction")
      .def(
          py::init<std::string, std::vector<Compute>, std::string>(),
          py::arg("name"),
          py::arg("regions"),
          py::arg("numerics") = "strict_eager")
      .def_property_readonly("name", &KernelFunction::name)
      .def_property_readonly("regions", &KernelFunction::regions)
      .def_property_readonly("numerics", &KernelFunction::numerics);

  py::class_<KernelProgram>(m, "KernelProgram")
      .def(
          py::init<std::string, std::string>(),
          py::arg("name"),
          py::arg("numerics") = "strict_eager")
      .def("add_input", &KernelProgram::addInput, py::arg("buffer"))
      .def(
          "add_input_descriptor",
          &KernelProgram::addInputDescriptor,
          py::arg("descriptor"))
      .def("add_view", &KernelProgram::addView, py::arg("view"))
      .def("add_function", &KernelProgram::addFunction, py::arg("function"))
      .def(
          "add_extern_call",
          &KernelProgram::addExternCall,
          py::arg("extern_call"))
      .def("add_output", &KernelProgram::addOutput, py::arg("descriptor"))
      .def("verify", &KernelProgram::verify)
      .def("dump", &KernelProgram::dump)
      .def("build_dependencies", &KernelProgram::buildDependencies)
      .def_property_readonly("name", &KernelProgram::name)
      .def_property_readonly("inputs", &KernelProgram::inputs)
      .def_property_readonly(
          "input_descriptors", &KernelProgram::inputDescriptors)
      .def_property_readonly("views", &KernelProgram::views)
      .def_property_readonly("outputs", &KernelProgram::outputs)
      .def_property_readonly("nodes", &KernelProgram::nodes)
      .def_property_readonly("dependencies", &KernelProgram::dependencies)
      .def_property_readonly("numerics", &KernelProgram::numerics)
      .def_property_readonly("buffers", &KernelProgram::buffers);

  m.def("optimize_program", &optimize_program, py::arg("program"));
  m.def("lower_to_kernels", &lower_to_kernels, py::arg("program"));
  m.def("fuse_kernels", &fuse_kernels, py::arg("program"));
  m.def(
      "prove_dense_injective_store",
      &prove_dense_injective_store,
      py::arg("compute"),
      py::arg("result"));
  m.def("run_pass", &run_pass, py::arg("name"), py::arg("program"));
  m.def("view_inline", &view_inline, py::arg("program"));
  m.def("index_simplify", &index_simplify, py::arg("program"));
  m.def("unit_range_removal", &unit_range_removal, py::arg("program"));
  m.def("local_cse_dce", &local_cse_dce, py::arg("program"));
  m.def("dead_prune", &dead_prune, py::arg("program"));
  m.def("pass_names", &pass_names);
  m.def("optimize_program_dumps", &optimize_program_dumps, py::arg("program"));

  // Tiling worker's bindings (separate file). Registered here — before the
  // formal-pipeline defs below — so TileOptions exists when run_pipeline's
  // default argument is converted. Adds only new types; core/kernel
  // registration above is unaffected.
  bind_tiled_ir(m);

  // Formal pipeline API. PipelineResult exposes readonly
  // optimized_program (Program), kernels_before_fusion (KernelProgram),
  // kernel_program (fused KernelProgram), tiled_program (TiledProgram),
  // trace (per-pass entries).
  py::class_<pipeline::PassTraceEntry>(m, "PassTraceEntry")
      .def_property_readonly(
          "name", [](const pipeline::PassTraceEntry& e) { return e.name; })
      .def_property_readonly(
          "version",
          [](const pipeline::PassTraceEntry& e) { return e.version; })
      .def_property_readonly(
          "input_ir",
          [](const pipeline::PassTraceEntry& e) { return e.inputIr; })
      .def_property_readonly(
          "output_ir",
          [](const pipeline::PassTraceEntry& e) { return e.outputIr; })
      .def_property_readonly(
          "numerics",
          [](const pipeline::PassTraceEntry& e) { return e.numerics; })
      .def_property_readonly(
          "input_fingerprint",
          [](const pipeline::PassTraceEntry& e) { return e.inputFingerprint; })
      .def_property_readonly(
          "output_fingerprint",
          [](const pipeline::PassTraceEntry& e) { return e.outputFingerprint; })
      .def_property_readonly(
          "analyses_computed",
          [](const pipeline::PassTraceEntry& e) { return e.analysesComputed; })
      .def_property_readonly(
          "analyses_invalidated",
          [](const pipeline::PassTraceEntry& e) {
            return e.analysesInvalidated;
          })
      .def_property_readonly(
          "dump", [](const pipeline::PassTraceEntry& e) { return e.dump; });

  py::class_<pipeline::PipelineResult>(m, "PipelineResult")
      .def_property_readonly(
          "optimized_program",
          [](const pipeline::PipelineResult& r) { return r.optimizedProgram; })
      .def_property_readonly(
          "kernels_before_fusion",
          [](const pipeline::PipelineResult& r) {
            return r.kernelBeforeFusion;
          })
      .def_property_readonly(
          "kernel_program",
          [](const pipeline::PipelineResult& r) { return r.kernelProgram; })
      .def_property_readonly(
          "tiled_program",
          [](const pipeline::PipelineResult& r) { return r.tiledProgram; })
      .def_property_readonly(
          "trace", [](const pipeline::PipelineResult& r) { return r.trace; });

  m.def(
      "run_pipeline",
      [](const Program& program,
         const TileOptions& tileOptions,
         bool captureDumps) {
        return pipeline::runPipeline(program, tileOptions, captureDumps);
      },
      py::arg("program"),
      py::arg("tile_options") = TileOptions(),
      py::arg("capture_dumps") = true);
  m.def("pipeline_pass_names", &pipeline::pipelinePassNames);
  m.def("_registry_self_test", &pipeline::registrySelfTest);
  m.def("pass_descriptors", []() {
    py::list out;
    for (const auto& d : pipeline::passDescriptors()) {
      py::dict entry;
      entry["name"] = d.name;
      entry["version"] = d.version;
      entry["input_ir"] = analyses::toString(d.inputIr);
      entry["output_ir"] = analyses::toString(d.outputIr);
      py::list required;
      for (auto k : d.required) {
        required.append(analyses::toString(k));
      }
      entry["required_analyses"] = required;
      py::list preserved;
      for (auto k : d.preserved) {
        preserved.append(analyses::toString(k));
      }
      entry["preserved_analyses"] = preserved;
      py::list invalidated;
      for (auto k : d.invalidated) {
        invalidated.append(analyses::toString(k));
      }
      entry["invalidated_analyses"] = invalidated;
      entry["strict_safe"] = d.strictSafe;
      entry["consults_tile_options"] = d.consultsTileOptions;
      entry["capability"] = d.capability;
      out.append(std::move(entry));
    }
    return out;
  });
}
