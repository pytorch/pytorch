#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>

#ifdef USE_CUDA
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ops/composite.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_cache.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_definition.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_interface.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <iostream>
#include <tuple>

namespace torch {
namespace jit {

void initNvFuserPythonBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  //! Top Level nvFuser Python submodule
  auto nvfuser = m.def_submodule("_nvfuser");

  //! DataTypes supported by nvFuser in the FusionDefinition
  py::enum_<Nvf::DataType>(nvfuser, "DataType")
      .value("Double", Nvf::DataType::Double)
      .value("Float", Nvf::DataType::Float)
      .value("Half", Nvf::DataType::Half)
      .value("Int", Nvf::DataType::Int)
      .value("Int32", Nvf::DataType::Int32)
      .value("Bool", Nvf::DataType::Bool)
      .value("BFloat16", Nvf::DataType::BFloat16)
      .value("ComplexFloat", Nvf::DataType::ComplexFloat)
      .value("ComplexDouble", Nvf::DataType::ComplexDouble)
      .value("Null", Nvf::DataType::Null);

  nvfuser.def(
      "compute_contiguity",
      [](const std::vector<int64_t>& sizes,
         const std::vector<int64_t>& strides) {
        py::tuple contiguity(sizes.size());
        TORCH_CHECK(
            sizes.size() == strides.size(),
            "compute_contiguity: Sizes and strides must have the same number of dimensions");
        if (sizes.size() == 0) {
          return contiguity;
        }
        contiguity[sizes.size() - 1] = strides.back() == 1;
        for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i) {
          contiguity[i] = strides[i] == strides[i + 1] * sizes[i + 1];
        }
        return contiguity;
      });

  //! Binding the FusionCache that holds a cache of Fusions
  //! This is only bound to provide an interface to get the number of fusions
  //! that are cached.
  py::class_<nvfuser::FusionCache> fusion_cache(nvfuser, "FusionCache");
  fusion_cache
      .def_static(
          "get",
          &nvfuser::FusionCache::get,
          py::arg("max_fusions") = int(8192),
          py::return_value_policy::reference)
      .def("num_fusions", &nvfuser::FusionCache::numFusions)
      .def("print_stats", [](nvfuser::FusionCache& self) {
        self.print(std::cout);
      });

  py::class_<nvfuser::FusionInterface> fusion(nvfuser, "Fusion");
  fusion.def(py::init<>())
      .def(py::init<size_t>(), py::arg("fusion_id"))
      .def("define", &nvfuser::FusionInterface::define)
      .def("defined", &nvfuser::FusionInterface::defined)
      .def(
          "execute",
          [](nvfuser::FusionInterface& self, const py::iterable& iter) {
            std::vector<IValue> inputs;
            for (py::handle obj : iter) {
              inputs.push_back(toIValue(obj, c10::AnyType::get()));
            }
            return self.execute(inputs);
          },
          py::return_value_policy::reference)
      .def("id", &nvfuser::FusionInterface::id)
      .def("print", &nvfuser::FusionInterface::print);

  //! These are the FusionDefinition supported object types that are either
  //! defined as inputs or the output of an operation.
  py::class_<nvfuser::Tensor>(nvfuser, "Tensor");
  py::class_<nvfuser::Scalar>(nvfuser, "Scalar");

  //! The FusionDefinition is a context manager in Python where the user will
  //! define the set the operations and connections between operations for
  //! nvFuser to create.
  py::class_<nvfuser::FusionDefinition> fusion_def(nvfuser, "FusionDefinition");
  fusion_def
      .def(
          py::init<nvfuser::FusionInterface*, int>(),
          py::arg("fusion"),
          py::arg("max_length") = int(1024))
      .def_readwrite("ops", &nvfuser::FusionDefinition::ops)
      .def(
          "__enter__",
          [](nvfuser::FusionDefinition& self) -> nvfuser::FusionDefinition* {
            // Instrumentation to mark the beginning of a FusionDefinition
            Nvf::inst::Trace::instance()->beginEvent(
                "FusionDefinition Context Manager");
            return self.enter();
          })
      .def(
          "__exit__",
          [](nvfuser::FusionDefinition& self,
             void* exc_type,
             void* exc_value,
             void* traceback) {
            self.exit();
            // Mark the end of a FusionDefinition Context Manager
            Nvf::inst::Trace::instance()->endEvent(nullptr);
          })
      .def(
          "__str__",
          [](nvfuser::FusionDefinition& self) {
            std::stringstream ss;
            self.print(ss);
            return ss.str();
          })
      .def(
          "add_output",
          [](nvfuser::FusionDefinition& self, nvfuser::Scalar output) {
            FUSER_PERF_SCOPE("FusionDefinition.add_output (scalar)");
            self.defineRecord(new nvfuser::OutputRecord<Nvf::Val>(
                {self.recordingState(output())}));
          })
      .def(
          "add_output",
          [](nvfuser::FusionDefinition& self, nvfuser::Tensor output) {
            FUSER_PERF_SCOPE("FusionDefinition.add_output (tensor)");
            self.defineRecord(new nvfuser::OutputRecord<Nvf::TensorView>(
                {self.recordingState(output())}));
          })
      .def(
          "define_tensor",
          [](nvfuser::FusionDefinition& self,
             size_t ndims,
             Nvf::DataType dtype = Nvf::DataType::Float,
             bool is_cpu = false) -> nvfuser::Tensor {
            FUSER_PERF_SCOPE("FusionDefinition.define_tensor (simple)");
            std::vector<int64_t> maybe_symbolic_sizes(ndims, -1);
            ;
            std::vector<bool> contig_info(ndims, false);

            nvfuser::Tensor out = self.defineTensor();
            self.defineRecord(new nvfuser::TensorRecord(
                {self.recordingState(out())},
                std::move(maybe_symbolic_sizes),
                std::move(contig_info),
                dtype,
                is_cpu));

            return out;
          },
          py::arg("ndims"),
          py::arg("dtype") = Nvf::DataType::Float,
          py::arg("is_cpu") = false,
          py::return_value_policy::reference)
      .def(
          "define_tensor",
          [](nvfuser::FusionDefinition& self,
             std::vector<int64_t>& symbolic_sizes,
             std::vector<bool>& contiguous,
             Nvf::DataType dtype = Nvf::DataType::Float,
             bool is_cpu = false) -> nvfuser::Tensor {
            FUSER_PERF_SCOPE("FusionDefinition.define_tensor (default)");

            for (size_t i = 0; i < symbolic_sizes.size(); ++i) {
              TORCH_CHECK(
                  symbolic_sizes[i] == -1 || symbolic_sizes[i] == 1,
                  "The value ",
                  symbolic_sizes[i],
                  " at index ",
                  i,
                  " was neither broadcast(1) or symbolic(-1).");
            }

            nvfuser::Tensor out = self.defineTensor();
            self.defineRecord(new nvfuser::TensorRecord(
                {self.recordingState(out())},
                symbolic_sizes,
                contiguous,
                dtype,
                is_cpu));

            return out;
          },
          py::arg("symbolic_sizes"),
          py::arg("contiguous"),
          py::arg("dtype") = Nvf::DataType::Float,
          py::arg("is_cpu") = false,
          py::return_value_policy::reference)
      .def(
          "define_tensor",
          [](nvfuser::FusionDefinition& self,
             std::vector<int64_t>& sizes,
             std::vector<int64_t>& strides,
             Nvf::DataType dtype = Nvf::DataType::Float,
             bool is_cpu = false) -> nvfuser::Tensor {
            FUSER_PERF_SCOPE("FusionDefinition.define_tensor (integration)");
            TORCH_CHECK(
                sizes.size() == strides.size(),
                "The number of sizes does not match the number of strides.",
                sizes.size(),
                strides.size());

            // TensorViewBuilder assumes any dim with a compile time constant
            // size == 1 is a "maybe broadcast" axis, symbolic sizes are
            // identified by -1, and size == 0 is not supported.

            // Translate to TensorViewBuilder's view of the world.
            std::vector<int64_t> maybe_symbolic_sizes;
            maybe_symbolic_sizes.reserve(sizes.size());
            for (const auto i : c10::irange(sizes.size())) {
              TORCH_INTERNAL_ASSERT(
                  sizes[i] > 0,
                  "Size of ",
                  sizes[i],
                  " is not supported in nvFuser. Expected size > 0.");
              if (sizes[i] == 1) {
                maybe_symbolic_sizes.push_back(1);
              } else {
                maybe_symbolic_sizes.push_back(-1);
              }
            }

            std::vector<bool> contig_info(strides.size(), false);
            for (int i = contig_info.size() - 1; i >= 0; --i) {
              if (i == static_cast<int>(contig_info.size() - 1)) {
                contig_info[i] = (strides[i] == 1);
              } else {
                contig_info[i] =
                    (strides[i] == (strides[i + 1] * sizes[i + 1]));
              }
            }

            nvfuser::Tensor out = self.defineTensor();
            self.defineRecord(new nvfuser::TensorRecord(
                {self.recordingState(out())},
                std::move(maybe_symbolic_sizes),
                std::move(contig_info),
                dtype,
                is_cpu));

            return out;
          },
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("dtype") = Nvf::DataType::Float,
          py::arg("is_cpu") = false,
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](nvfuser::FusionDefinition& self, double val) -> nvfuser::Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_constant (double)");
            nvfuser::Scalar out = self.defineScalar();
            self.defineRecord(new nvfuser::ConstantRecord<Nvf::Double, double>(
                {self.recordingState(out())}, val));
            return out;
          },
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](nvfuser::FusionDefinition& self,
             std::complex<double> val) -> nvfuser::Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_constant (complex)");
            nvfuser::Scalar out = self.defineScalar();
            self.defineRecord(
                new nvfuser::
                    ConstantRecord<Nvf::ComplexDouble, c10::complex<double>>(
                        {self.recordingState(out())},
                        static_cast<c10::complex<double>>(val)));
            return out;
          },
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](nvfuser::FusionDefinition& self, bool val) -> nvfuser::Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_constant (bool)");
            nvfuser::Scalar out = self.defineScalar();
            self.defineRecord(new nvfuser::ConstantRecord<Nvf::Bool, bool>(
                {self.recordingState(out())}, val));
            return out;
          },
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](nvfuser::FusionDefinition& self, int64_t val) -> nvfuser::Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_constant (int)");
            nvfuser::Scalar out = self.defineScalar();
            self.defineRecord(new nvfuser::ConstantRecord<Nvf::Int, int64_t>(
                {self.recordingState(out())}, val));
            return out;
          },
          py::return_value_policy::reference)
      .def(
          "define_scalar",
          [](nvfuser::FusionDefinition& self,
             Nvf::DataType dtype = Nvf::DataType::Double) -> nvfuser::Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_scalar");
            nvfuser::Scalar out = self.defineScalar();
            self.defineRecord(
                new nvfuser::ScalarRecord({self.recordingState(out())}, dtype));
            return out;
          },
          py::arg("dtype") = Nvf::DataType::Double,
          py::return_value_policy::reference);

  //! The Operators class is a nested class of FusionDefinition to allow the
  //! user to query the class for the list of operators.
  //!
  //! Example:
  //!   help(FusionDefinition.Operators)
  //!
  //! Additional operators are expected to be defined below as needed.  They
  //! may require defining a new RecordFunctor child class if they are unique.
  py::class_<nvfuser::FusionDefinition::Operators> nvf_ops(
      fusion_def, "Operators");
  nvf_ops.def(py::init<nvfuser::FusionDefinition*>());

  // ******************** INSERT OP BINDINGS BELOW HERE ********************
#define OP_PREFIX "Operators."
#define NVFUSER_PYTHON_BINDING_UNARY_OP(op_str, op_name)               \
  nvf_ops.def(                                                         \
      op_str,                                                          \
      [](nvfuser::FusionDefinition::Operators& self,                   \
         nvfuser::Tensor input) -> nvfuser::Tensor {                   \
        FUSER_PERF_SCOPE("Operators." op_str);                         \
        nvfuser::FusionDefinition* fd = self.fusion_definition;        \
        nvfuser::Tensor output = fd->defineTensor();                   \
        fd->defineRecord(                                              \
            new nvfuser::OpRecord<Nvf::TensorView*, Nvf::TensorView*>( \
                {fd->recordingState(input())},                         \
                {fd->recordingState(output())},                        \
                ("ops." op_str),                                       \
                static_cast<Nvf::TensorView* (*)(Nvf::TensorView*)>(   \
                    Nvf::op_name)));                                   \
        return output;                                                 \
      },                                                               \
      py::return_value_policy::reference);                             \
  nvf_ops.def(                                                         \
      op_str,                                                          \
      [](nvfuser::FusionDefinition::Operators& self,                   \
         nvfuser::Scalar input) -> nvfuser::Scalar {                   \
        FUSER_PERF_SCOPE("Operators." op_str);                         \
        nvfuser::FusionDefinition* fd = self.fusion_definition;        \
        nvfuser::Scalar output = fd->defineScalar();                   \
        fd->defineRecord(new nvfuser::OpRecord<Nvf::Val*, Nvf::Val*>(  \
            {fd->recordingState(input())},                             \
            {fd->recordingState(output())},                            \
            ("ops." op_str),                                           \
            static_cast<Nvf::Val* (*)(Nvf::Val*)>(Nvf::op_name)));     \
        return output;                                                 \
      },                                                               \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_UNARY_OP("abs", abs)
  NVFUSER_PYTHON_BINDING_UNARY_OP("acos", acos)
  NVFUSER_PYTHON_BINDING_UNARY_OP("asin", asin)
  NVFUSER_PYTHON_BINDING_UNARY_OP("atan", atan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("atanh", atanh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("ceil", ceil)
  NVFUSER_PYTHON_BINDING_UNARY_OP("cos", cos)
  NVFUSER_PYTHON_BINDING_UNARY_OP("cosh", cosh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("exp", exp)
  NVFUSER_PYTHON_BINDING_UNARY_OP("expm1", expm1)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erf", erf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erfc", erfc)
  NVFUSER_PYTHON_BINDING_UNARY_OP("floor", floor)
  NVFUSER_PYTHON_BINDING_UNARY_OP("frac", frac)
  NVFUSER_PYTHON_BINDING_UNARY_OP("lgamma", lgamma)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log", log)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log10", log10)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log1p", log1p)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log2", log2)
  NVFUSER_PYTHON_BINDING_UNARY_OP("neg", neg)
  NVFUSER_PYTHON_BINDING_UNARY_OP("bitwise_not", bitwise_not)
  NVFUSER_PYTHON_BINDING_UNARY_OP("relu", relu)
  NVFUSER_PYTHON_BINDING_UNARY_OP("rand_like", randlike)
  NVFUSER_PYTHON_BINDING_UNARY_OP("reciprocal", reciprocal)
  NVFUSER_PYTHON_BINDING_UNARY_OP("round", round)
  NVFUSER_PYTHON_BINDING_UNARY_OP("rsqrt", rsqrt)
  NVFUSER_PYTHON_BINDING_UNARY_OP("set", set)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sign", sign)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sigmoid", sigmoid)
  NVFUSER_PYTHON_BINDING_UNARY_OP("silu", silu)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sin", sin)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sinh", sinh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sqrt", sqrt)
  NVFUSER_PYTHON_BINDING_UNARY_OP("tan", tan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("tanh", tanh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("trunc", trunc)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isfinite", isfinite)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isinf", isinf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isnan", isnan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isneginf", isneginf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isposinf", isposinf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isreal", isreal)
  NVFUSER_PYTHON_BINDING_UNARY_OP("real", real)
  NVFUSER_PYTHON_BINDING_UNARY_OP("imag", imag)
#undef NVFUSER_PYTHON_BINDING_UNARY_OP

#define NVFUSER_PYTHON_BINDING_BINARY_OP(op_str, op_name)                   \
  nvf_ops.def(                                                              \
      op_str,                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                        \
         nvfuser::Tensor arg1,                                              \
         nvfuser::Tensor arg2) -> nvfuser::Tensor {                         \
        FUSER_PERF_SCOPE("Operators." op_str);                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;             \
        nvfuser::Tensor output = fd->defineTensor();                        \
        fd->defineRecord(new nvfuser::OpRecord<                             \
                         Nvf::TensorView*,                                  \
                         Nvf::TensorView*,                                  \
                         Nvf::TensorView*>(                                 \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},       \
            {fd->recordingState(output())},                                 \
            ("ops." op_str),                                                \
            static_cast<                                                    \
                Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::TensorView*)>(  \
                Nvf::op_name)));                                            \
        return output;                                                      \
      },                                                                    \
      py::return_value_policy::reference);                                  \
  nvf_ops.def(                                                              \
      op_str,                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                        \
         nvfuser::Tensor arg1,                                              \
         nvfuser::Scalar arg2) -> nvfuser::Tensor {                         \
        FUSER_PERF_SCOPE("Operators." op_str);                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;             \
        nvfuser::Tensor output = fd->defineTensor();                        \
        fd->defineRecord(new nvfuser::OpRecord<                             \
                         Nvf::TensorView*,                                  \
                         Nvf::TensorView*,                                  \
                         Nvf::Val*>(                                        \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},       \
            {fd->recordingState(output())},                                 \
            ("ops." op_str),                                                \
            static_cast<Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::Val*)>( \
                Nvf::op_name)));                                            \
        return output;                                                      \
      },                                                                    \
      py::return_value_policy::reference);                                  \
  nvf_ops.def(                                                              \
      op_str,                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                        \
         nvfuser::Scalar arg1,                                              \
         nvfuser::Tensor arg2) -> nvfuser::Tensor {                         \
        FUSER_PERF_SCOPE("Operators." op_str);                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;             \
        nvfuser::Tensor output = fd->defineTensor();                        \
        fd->defineRecord(new nvfuser::OpRecord<                             \
                         Nvf::TensorView*,                                  \
                         Nvf::Val*,                                         \
                         Nvf::TensorView*>(                                 \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},       \
            {fd->recordingState(output())},                                 \
            ("ops." op_str),                                                \
            static_cast<Nvf::TensorView* (*)(Nvf::Val*, Nvf::TensorView*)>( \
                Nvf::op_name)));                                            \
        return output;                                                      \
      },                                                                    \
      py::return_value_policy::reference);                                  \
  nvf_ops.def(                                                              \
      op_str,                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                        \
         nvfuser::Scalar arg1,                                              \
         nvfuser::Scalar arg2) -> nvfuser::Scalar {                         \
        FUSER_PERF_SCOPE("Operators." op_str);                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;             \
        nvfuser::Scalar output = fd->defineScalar();                        \
        fd->defineRecord(                                                   \
            new nvfuser::OpRecord<Nvf::Val*, Nvf::Val*, Nvf::Val*>(         \
                {fd->recordingState(arg1()), fd->recordingState(arg2())},   \
                {fd->recordingState(output())},                             \
                ("ops." op_str),                                            \
                static_cast<Nvf::Val* (*)(Nvf::Val*, Nvf::Val*)>(           \
                    Nvf::op_name)));                                        \
        return output;                                                      \
      },                                                                    \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_OP("add", add)
  NVFUSER_PYTHON_BINDING_BINARY_OP("atan2", atan2)
  NVFUSER_PYTHON_BINDING_BINARY_OP("div", div)
  NVFUSER_PYTHON_BINDING_BINARY_OP("fmod", fmod)
  NVFUSER_PYTHON_BINDING_BINARY_OP("mul", mul)
  NVFUSER_PYTHON_BINDING_BINARY_OP("pow", pow)
  NVFUSER_PYTHON_BINDING_BINARY_OP("remainder", remainder)
  NVFUSER_PYTHON_BINDING_BINARY_OP("sub", sub)
  NVFUSER_PYTHON_BINDING_BINARY_OP("mod", mod)
  NVFUSER_PYTHON_BINDING_BINARY_OP("eq", eq)
  NVFUSER_PYTHON_BINDING_BINARY_OP("ge", ge)
  NVFUSER_PYTHON_BINDING_BINARY_OP("gt", gt)
  NVFUSER_PYTHON_BINDING_BINARY_OP("le", le)
  NVFUSER_PYTHON_BINDING_BINARY_OP("lt", lt)
  NVFUSER_PYTHON_BINDING_BINARY_OP("ne", ne)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_and", bitwise_and)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_or", bitwise_or)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_xor", bitwise_xor)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_left_shift", bitwise_left_shift)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_right_shift", bitwise_left_shift)
#undef NVFUSER_PYTHON_BINDING_BINARY_OP

#define NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP(op_str, op_name)                 \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Tensor arg1,                                                       \
         nvfuser::Tensor arg2,                                                       \
         nvfuser::Scalar arg3) -> nvfuser::Tensor {                                  \
        FUSER_PERF_SCOPE("Operators." op_str);                                       \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                      \
        nvfuser::Tensor output = fd->defineTensor();                                 \
        fd->defineRecord(new nvfuser::OpRecord<                                      \
                         Nvf::TensorView*,                                           \
                         Nvf::TensorView*,                                           \
                         Nvf::TensorView*,                                           \
                         Nvf::Val*>(                                                 \
            {fd->recordingState(arg1()),                                             \
             fd->recordingState(arg2()),                                             \
             fd->recordingState(arg3())},                                            \
            {fd->recordingState(output())},                                          \
            ("ops." op_str),                                                         \
            static_cast<                                                             \
                Nvf::                                                                \
                    TensorView* (*)(Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*)>( \
                Nvf::op_name)));                                                     \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);                                           \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Tensor arg1,                                                       \
         nvfuser::Scalar arg2,                                                       \
         nvfuser::Scalar arg3) -> nvfuser::Tensor {                                  \
        FUSER_PERF_SCOPE("Operators." op_str);                                       \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                      \
        nvfuser::Tensor output = fd->defineTensor();                                 \
        fd->defineRecord(new nvfuser::OpRecord<                                      \
                         Nvf::TensorView*,                                           \
                         Nvf::TensorView*,                                           \
                         Nvf::Val*,                                                  \
                         Nvf::Val*>(                                                 \
            {fd->recordingState(arg1()),                                             \
             fd->recordingState(arg2()),                                             \
             fd->recordingState(arg3())},                                            \
            {fd->recordingState(output())},                                          \
            ("ops." op_str),                                                         \
            static_cast<                                                             \
                Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::Val*, Nvf::Val*)>(       \
                Nvf::op_name)));                                                     \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);                                           \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Scalar arg1,                                                       \
         nvfuser::Tensor arg2,                                                       \
         nvfuser::Scalar arg3) -> nvfuser::Tensor {                                  \
        FUSER_PERF_SCOPE("Operators." op_str);                                       \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                      \
        nvfuser::Tensor output = fd->defineTensor();                                 \
        fd->defineRecord(new nvfuser::OpRecord<                                      \
                         Nvf::TensorView*,                                           \
                         Nvf::Val*,                                                  \
                         Nvf::TensorView*,                                           \
                         Nvf::Val*>(                                                 \
            {fd->recordingState(arg1()),                                             \
             fd->recordingState(arg2()),                                             \
             fd->recordingState(arg3())},                                            \
            {fd->recordingState(output())},                                          \
            ("ops." op_str),                                                         \
            static_cast<                                                             \
                Nvf::TensorView* (*)(Nvf::Val*, Nvf::TensorView*, Nvf::Val*)>(       \
                Nvf::op_name)));                                                     \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);                                           \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Scalar arg1,                                                       \
         nvfuser::Scalar arg2,                                                       \
         nvfuser::Scalar arg3) -> nvfuser::Scalar {                                  \
        FUSER_PERF_SCOPE("Operators." op_str);                                       \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                      \
        nvfuser::Scalar output = fd->defineScalar();                                 \
        fd->defineRecord(                                                            \
            new nvfuser::OpRecord<Nvf::Val*, Nvf::Val*, Nvf::Val*, Nvf::Val*>(       \
                {fd->recordingState(arg1()),                                         \
                 fd->recordingState(arg2()),                                         \
                 fd->recordingState(arg3())},                                        \
                {fd->recordingState(output())},                                      \
                ("ops." op_str),                                                     \
                static_cast<Nvf::Val* (*)(Nvf::Val*, Nvf::Val*, Nvf::Val*)>(         \
                    Nvf::op_name)));                                                 \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP("add_alpha", add_alpha)
  NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP("sub_alpha", sub_alpha)
#undef NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_OP(op_str, op_name)                                  \
  nvf_ops.def(                                                                              \
      op_str,                                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                                        \
         nvfuser::Scalar arg1,                                                              \
         nvfuser::Scalar arg2,                                                              \
         nvfuser::Scalar arg3) -> nvfuser::Scalar {                                         \
        FUSER_PERF_SCOPE("Operators." op_str);                                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                             \
        nvfuser::Scalar output = fd->defineScalar();                                        \
        fd->defineRecord(                                                                   \
            new nvfuser::OpRecord<Nvf::Val*, Nvf::Val*, Nvf::Val*, Nvf::Val*>(              \
                {fd->recordingState(arg1()),                                                \
                 fd->recordingState(arg2()),                                                \
                 fd->recordingState(arg3())},                                               \
                {fd->recordingState(output())},                                             \
                ("ops." op_str),                                                            \
                static_cast<Nvf::Val* (*)(Nvf::Val*, Nvf::Val*, Nvf::Val*)>(                \
                    Nvf::op_name)));                                                        \
        return output;                                                                      \
      },                                                                                    \
      py::return_value_policy::reference);                                                  \
  nvf_ops.def(                                                                              \
      op_str,                                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                                        \
         nvfuser::Tensor arg1,                                                              \
         nvfuser::Tensor arg2,                                                              \
         nvfuser::Tensor arg3) -> nvfuser::Tensor {                                         \
        FUSER_PERF_SCOPE("Operators." op_str);                                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                             \
        nvfuser::Tensor output = fd->defineTensor();                                        \
        fd->defineRecord(new nvfuser::OpRecord<                                             \
                         Nvf::TensorView*,                                                  \
                         Nvf::TensorView*,                                                  \
                         Nvf::TensorView*,                                                  \
                         Nvf::TensorView*>(                                                 \
            {fd->recordingState(arg1()),                                                    \
             fd->recordingState(arg2()),                                                    \
             fd->recordingState(arg3())},                                                   \
            {fd->recordingState(output())},                                                 \
            ("ops." op_str),                                                                \
            static_cast<                                                                    \
                Nvf::                                                                       \
                    TensorView* (*)(Nvf::TensorView*, Nvf::TensorView*, Nvf::TensorView*)>( \
                Nvf::op_name)));                                                            \
        return output;                                                                      \
      },                                                                                    \
      py::return_value_policy::reference);                                                  \
  nvf_ops.def(                                                                              \
      op_str,                                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                                        \
         nvfuser::Tensor arg1,                                                              \
         nvfuser::Tensor arg2,                                                              \
         nvfuser::Scalar arg3) -> nvfuser::Tensor {                                         \
        FUSER_PERF_SCOPE("Operators." op_str);                                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                             \
        nvfuser::Tensor output = fd->defineTensor();                                        \
        fd->defineRecord(new nvfuser::OpRecord<                                             \
                         Nvf::TensorView*,                                                  \
                         Nvf::TensorView*,                                                  \
                         Nvf::TensorView*,                                                  \
                         Nvf::Val*>(                                                        \
            {fd->recordingState(arg1()),                                                    \
             fd->recordingState(arg2()),                                                    \
             fd->recordingState(arg3())},                                                   \
            {fd->recordingState(output())},                                                 \
            ("ops." op_str),                                                                \
            static_cast<                                                                    \
                Nvf::                                                                       \
                    TensorView* (*)(Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*)>(        \
                Nvf::op_name)));                                                            \
        return output;                                                                      \
      },                                                                                    \
      py::return_value_policy::reference);                                                  \
  nvf_ops.def(                                                                              \
      op_str,                                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                                        \
         nvfuser::Tensor arg1,                                                              \
         nvfuser::Scalar arg2,                                                              \
         nvfuser::Tensor arg3) -> nvfuser::Tensor {                                         \
        FUSER_PERF_SCOPE("Operators." op_str);                                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                             \
        nvfuser::Tensor output = fd->defineTensor();                                        \
        fd->defineRecord(new nvfuser::OpRecord<                                             \
                         Nvf::TensorView*,                                                  \
                         Nvf::TensorView*,                                                  \
                         Nvf::Val*,                                                         \
                         Nvf::TensorView*>(                                                 \
            {fd->recordingState(arg1()),                                                    \
             fd->recordingState(arg2()),                                                    \
             fd->recordingState(arg3())},                                                   \
            {fd->recordingState(output())},                                                 \
            ("ops." op_str),                                                                \
            static_cast<                                                                    \
                Nvf::                                                                       \
                    TensorView* (*)(Nvf::TensorView*, Nvf::Val*, Nvf::TensorView*)>(        \
                Nvf::op_name)));                                                            \
        return output;                                                                      \
      },                                                                                    \
      py::return_value_policy::reference);                                                  \
  nvf_ops.def(                                                                              \
      op_str,                                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                                        \
         nvfuser::Scalar arg1,                                                              \
         nvfuser::Tensor arg2,                                                              \
         nvfuser::Tensor arg3) -> nvfuser::Tensor {                                         \
        FUSER_PERF_SCOPE("Operators." op_str);                                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                             \
        nvfuser::Tensor output = fd->defineTensor();                                        \
        fd->defineRecord(new nvfuser::OpRecord<                                             \
                         Nvf::TensorView*,                                                  \
                         Nvf::Val*,                                                         \
                         Nvf::TensorView*,                                                  \
                         Nvf::TensorView*>(                                                 \
            {fd->recordingState(arg1()),                                                    \
             fd->recordingState(arg2()),                                                    \
             fd->recordingState(arg3())},                                                   \
            {fd->recordingState(output())},                                                 \
            ("ops." op_str),                                                                \
            static_cast<                                                                    \
                Nvf::                                                                       \
                    TensorView* (*)(Nvf::Val*, Nvf::TensorView*, Nvf::TensorView*)>(        \
                Nvf::op_name)));                                                            \
        return output;                                                                      \
      },                                                                                    \
      py::return_value_policy::reference);                                                  \
  nvf_ops.def(                                                                              \
      op_str,                                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                                        \
         nvfuser::Scalar arg1,                                                              \
         nvfuser::Scalar arg2,                                                              \
         nvfuser::Tensor arg3) -> nvfuser::Tensor {                                         \
        FUSER_PERF_SCOPE("Operators." op_str);                                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                             \
        nvfuser::Tensor output = fd->defineTensor();                                        \
        fd->defineRecord(new nvfuser::OpRecord<                                             \
                         Nvf::TensorView*,                                                  \
                         Nvf::Val*,                                                         \
                         Nvf::Val*,                                                         \
                         Nvf::TensorView*>(                                                 \
            {fd->recordingState(arg1()),                                                    \
             fd->recordingState(arg2()),                                                    \
             fd->recordingState(arg3())},                                                   \
            {fd->recordingState(output())},                                                 \
            ("ops." op_str),                                                                \
            static_cast<                                                                    \
                Nvf::TensorView* (*)(Nvf::Val*, Nvf::Val*, Nvf::TensorView*)>(              \
                Nvf::op_name)));                                                            \
        return output;                                                                      \
      },                                                                                    \
      py::return_value_policy::reference);                                                  \
  nvf_ops.def(                                                                              \
      op_str,                                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                                        \
         nvfuser::Tensor arg1,                                                              \
         nvfuser::Scalar arg2,                                                              \
         nvfuser::Scalar arg3) -> nvfuser::Tensor {                                         \
        FUSER_PERF_SCOPE("Operators." op_str);                                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                             \
        nvfuser::Tensor output = fd->defineTensor();                                        \
        fd->defineRecord(new nvfuser::OpRecord<                                             \
                         Nvf::TensorView*,                                                  \
                         Nvf::TensorView*,                                                  \
                         Nvf::Val*,                                                         \
                         Nvf::Val*>(                                                        \
            {fd->recordingState(arg1()),                                                    \
             fd->recordingState(arg2()),                                                    \
             fd->recordingState(arg3())},                                                   \
            {fd->recordingState(output())},                                                 \
            ("ops." op_str),                                                                \
            static_cast<                                                                    \
                Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::Val*, Nvf::Val*)>(              \
                Nvf::op_name)));                                                            \
        return output;                                                                      \
      },                                                                                    \
      py::return_value_policy::reference);                                                  \
  nvf_ops.def(                                                                              \
      op_str,                                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                                        \
         nvfuser::Scalar arg1,                                                              \
         nvfuser::Tensor arg2,                                                              \
         nvfuser::Scalar arg3) -> nvfuser::Tensor {                                         \
        FUSER_PERF_SCOPE("Operators." op_str);                                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                             \
        nvfuser::Tensor output = fd->defineTensor();                                        \
        fd->defineRecord(new nvfuser::OpRecord<                                             \
                         Nvf::TensorView*,                                                  \
                         Nvf::Val*,                                                         \
                         Nvf::TensorView*,                                                  \
                         Nvf::Val*>(                                                        \
            {fd->recordingState(arg1()),                                                    \
             fd->recordingState(arg2()),                                                    \
             fd->recordingState(arg3())},                                                   \
            {fd->recordingState(output())},                                                 \
            ("ops." op_str),                                                                \
            static_cast<                                                                    \
                Nvf::TensorView* (*)(Nvf::Val*, Nvf::TensorView*, Nvf::Val*)>(              \
                Nvf::op_name)));                                                            \
        return output;                                                                      \
      },                                                                                    \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_OP("lerp", lerp)
  NVFUSER_PYTHON_BINDING_TERNARY_OP("where", where)
#undef NVFUSER_PYTHON_BINDING_TERNARY_OP

#define NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP(op_str, op_name)              \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](nvfuser::FusionDefinition::Operators& self,                           \
         nvfuser::Scalar arg1,                                                 \
         nvfuser::Scalar arg2,                                                 \
         nvfuser::Scalar arg3) -> nvfuser::Scalar {                            \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                \
        nvfuser::Scalar output = fd->defineScalar();                           \
        fd->defineRecord(                                                      \
            new nvfuser::OpRecord<Nvf::Val*, Nvf::Val*, Nvf::Val*, Nvf::Val*>( \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                static_cast<Nvf::Val* (*)(Nvf::Val*, Nvf::Val*, Nvf::Val*)>(   \
                    Nvf::op_name)));                                           \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](nvfuser::FusionDefinition::Operators& self,                           \
         nvfuser::Tensor arg1,                                                 \
         nvfuser::Scalar arg2,                                                 \
         nvfuser::Scalar arg3) -> nvfuser::Tensor {                            \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                \
        nvfuser::Tensor output = fd->defineTensor();                           \
        fd->defineRecord(new nvfuser::OpRecord<                                \
                         Nvf::TensorView*,                                     \
                         Nvf::TensorView*,                                     \
                         Nvf::Val*,                                            \
                         Nvf::Val*>(                                           \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            static_cast<                                                       \
                Nvf::TensorView* (*)(Nvf::TensorView*, Nvf::Val*, Nvf::Val*)>( \
                Nvf::op_name)));                                               \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP("clamp", clamp)
  NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP("threshold", threshold)
#undef NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP(op_str, op_name)                                  \
  nvf_ops.def(                                                                                         \
      op_str,                                                                                          \
      [](nvfuser::FusionDefinition::Operators& self,                                                   \
         nvfuser::Scalar arg1,                                                                         \
         nvfuser::Scalar arg2,                                                                         \
         nvfuser::Scalar arg3,                                                                         \
         nvfuser::Scalar arg4) -> nvfuser::Scalar {                                                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                                         \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                                        \
        nvfuser::Scalar output = fd->defineScalar();                                                   \
        fd->defineRecord(new nvfuser::OpRecord<                                                        \
                         Nvf::Val*,                                                                    \
                         Nvf::Val*,                                                                    \
                         Nvf::Val*,                                                                    \
                         Nvf::Val*,                                                                    \
                         Nvf::Val*>(                                                                   \
            {fd->recordingState(arg1()),                                                               \
             fd->recordingState(arg2()),                                                               \
             fd->recordingState(arg3()),                                                               \
             fd->recordingState(arg4())},                                                              \
            {fd->recordingState(output())},                                                            \
            ("ops." op_str),                                                                           \
            static_cast<                                                                               \
                Nvf::Val* (*)(Nvf::Val*, Nvf::Val*, Nvf::Val*, Nvf::Val*)>(                            \
                Nvf::op_name)));                                                                       \
        return output;                                                                                 \
      },                                                                                               \
      py::return_value_policy::reference);                                                             \
  nvf_ops.def(                                                                                         \
      op_str,                                                                                          \
      [](nvfuser::FusionDefinition::Operators& self,                                                   \
         nvfuser::Tensor arg1,                                                                         \
         nvfuser::Tensor arg2,                                                                         \
         nvfuser::Tensor arg3,                                                                         \
         nvfuser::Scalar arg4) -> nvfuser::Tensor {                                                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                                         \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                                        \
        nvfuser::Tensor output = fd->defineTensor();                                                   \
        fd->defineRecord(new nvfuser::OpRecord<                                                        \
                         Nvf::TensorView*,                                                             \
                         Nvf::TensorView*,                                                             \
                         Nvf::TensorView*,                                                             \
                         Nvf::TensorView*,                                                             \
                         Nvf::TensorView*>(                                                            \
            {fd->recordingState(arg1()),                                                               \
             fd->recordingState(arg2()),                                                               \
             fd->recordingState(arg3()),                                                               \
             fd->recordingState(arg4())},                                                              \
            {fd->recordingState(output())},                                                            \
            ("ops." op_str),                                                                           \
            static_cast<                                                                               \
                Nvf::                                                                                  \
                    TensorView* (*)(Nvf::TensorView*, Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*)>( \
                Nvf::op_name)));                                                                       \
        return output;                                                                                 \
      },                                                                                               \
      py::return_value_policy::reference);                                                             \
  nvf_ops.def(                                                                                         \
      op_str,                                                                                          \
      [](nvfuser::FusionDefinition::Operators& self,                                                   \
         nvfuser::Tensor arg1,                                                                         \
         nvfuser::Tensor arg2,                                                                         \
         nvfuser::Scalar arg3,                                                                         \
         nvfuser::Scalar arg4) -> nvfuser::Tensor {                                                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                                         \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                                        \
        nvfuser::Tensor output = fd->defineTensor();                                                   \
        fd->defineRecord(new nvfuser::OpRecord<                                                        \
                         Nvf::TensorView*,                                                             \
                         Nvf::TensorView*,                                                             \
                         Nvf::TensorView*,                                                             \
                         Nvf::Val*,                                                                    \
                         Nvf::Val*>(                                                                   \
            {fd->recordingState(arg1()),                                                               \
             fd->recordingState(arg2()),                                                               \
             fd->recordingState(arg3()),                                                               \
             fd->recordingState(arg4())},                                                              \
            {fd->recordingState(output())},                                                            \
            ("ops." op_str),                                                                           \
            static_cast<                                                                               \
                Nvf::                                                                                  \
                    TensorView* (*)(Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*, Nvf::Val*)>(        \
                Nvf::op_name)));                                                                       \
        return output;                                                                                 \
      },                                                                                               \
      py::return_value_policy::reference);                                                             \
  nvf_ops.def(                                                                                         \
      op_str,                                                                                          \
      [](nvfuser::FusionDefinition::Operators& self,                                                   \
         nvfuser::Tensor arg1,                                                                         \
         nvfuser::Scalar arg2,                                                                         \
         nvfuser::Tensor arg3,                                                                         \
         nvfuser::Scalar arg4) -> nvfuser::Tensor {                                                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                                         \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                                        \
        nvfuser::Tensor output = fd->defineTensor();                                                   \
        fd->defineRecord(new nvfuser::OpRecord<                                                        \
                         Nvf::TensorView*,                                                             \
                         Nvf::TensorView*,                                                             \
                         Nvf::Val*,                                                                    \
                         Nvf::TensorView*,                                                             \
                         Nvf::Val*>(                                                                   \
            {fd->recordingState(arg1()),                                                               \
             fd->recordingState(arg2()),                                                               \
             fd->recordingState(arg3()),                                                               \
             fd->recordingState(arg4())},                                                              \
            {fd->recordingState(output())},                                                            \
            ("ops." op_str),                                                                           \
            static_cast<                                                                               \
                Nvf::                                                                                  \
                    TensorView* (*)(Nvf::TensorView*, Nvf::Val*, Nvf::TensorView*, Nvf::Val*)>(        \
                Nvf::op_name)));                                                                       \
        return output;                                                                                 \
      },                                                                                               \
      py::return_value_policy::reference);                                                             \
  nvf_ops.def(                                                                                         \
      op_str,                                                                                          \
      [](nvfuser::FusionDefinition::Operators& self,                                                   \
         nvfuser::Scalar arg1,                                                                         \
         nvfuser::Tensor arg2,                                                                         \
         nvfuser::Tensor arg3,                                                                         \
         nvfuser::Scalar arg4) -> nvfuser::Tensor {                                                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                                         \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                                        \
        nvfuser::Tensor output = fd->defineTensor();                                                   \
        fd->defineRecord(new nvfuser::OpRecord<                                                        \
                         Nvf::TensorView*,                                                             \
                         Nvf::Val*,                                                                    \
                         Nvf::TensorView*,                                                             \
                         Nvf::TensorView*,                                                             \
                         Nvf::Val*>(                                                                   \
            {fd->recordingState(arg1()),                                                               \
             fd->recordingState(arg2()),                                                               \
             fd->recordingState(arg3()),                                                               \
             fd->recordingState(arg4())},                                                              \
            {fd->recordingState(output())},                                                            \
            ("ops." op_str),                                                                           \
            static_cast<                                                                               \
                Nvf::                                                                                  \
                    TensorView* (*)(Nvf::Val*, Nvf::TensorView*, Nvf::TensorView*, Nvf::Val*)>(        \
                Nvf::op_name)));                                                                       \
        return output;                                                                                 \
      },                                                                                               \
      py::return_value_policy::reference);                                                             \
  nvf_ops.def(                                                                                         \
      op_str,                                                                                          \
      [](nvfuser::FusionDefinition::Operators& self,                                                   \
         nvfuser::Scalar arg1,                                                                         \
         nvfuser::Scalar arg2,                                                                         \
         nvfuser::Tensor arg3,                                                                         \
         nvfuser::Scalar arg4) -> nvfuser::Tensor {                                                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                                         \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                                        \
        nvfuser::Tensor output = fd->defineTensor();                                                   \
        fd->defineRecord(new nvfuser::OpRecord<                                                        \
                         Nvf::TensorView*,                                                             \
                         Nvf::Val*,                                                                    \
                         Nvf::Val*,                                                                    \
                         Nvf::TensorView*,                                                             \
                         Nvf::Val*>(                                                                   \
            {fd->recordingState(arg1()),                                                               \
             fd->recordingState(arg2()),                                                               \
             fd->recordingState(arg3()),                                                               \
             fd->recordingState(arg4())},                                                              \
            {fd->recordingState(output())},                                                            \
            ("ops." op_str),                                                                           \
            static_cast<                                                                               \
                Nvf::                                                                                  \
                    TensorView* (*)(Nvf::Val*, Nvf::Val*, Nvf::TensorView*, Nvf::Val*)>(               \
                Nvf::op_name)));                                                                       \
        return output;                                                                                 \
      },                                                                                               \
      py::return_value_policy::reference);                                                             \
  nvf_ops.def(                                                                                         \
      op_str,                                                                                          \
      [](nvfuser::FusionDefinition::Operators& self,                                                   \
         nvfuser::Tensor arg1,                                                                         \
         nvfuser::Scalar arg2,                                                                         \
         nvfuser::Scalar arg3,                                                                         \
         nvfuser::Scalar arg4) -> nvfuser::Tensor {                                                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                                         \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                                        \
        nvfuser::Tensor output = fd->defineTensor();                                                   \
        fd->defineRecord(new nvfuser::OpRecord<                                                        \
                         Nvf::TensorView*,                                                             \
                         Nvf::TensorView*,                                                             \
                         Nvf::Val*,                                                                    \
                         Nvf::Val*,                                                                    \
                         Nvf::Val*>(                                                                   \
            {fd->recordingState(arg1()),                                                               \
             fd->recordingState(arg2()),                                                               \
             fd->recordingState(arg3()),                                                               \
             fd->recordingState(arg4())},                                                              \
            {fd->recordingState(output())},                                                            \
            ("ops." op_str),                                                                           \
            static_cast<                                                                               \
                Nvf::                                                                                  \
                    TensorView* (*)(Nvf::TensorView*, Nvf::Val*, Nvf::Val*, Nvf::Val*)>(               \
                Nvf::op_name)));                                                                       \
        return output;                                                                                 \
      },                                                                                               \
      py::return_value_policy::reference);                                                             \
  nvf_ops.def(                                                                                         \
      op_str,                                                                                          \
      [](nvfuser::FusionDefinition::Operators& self,                                                   \
         nvfuser::Scalar arg1,                                                                         \
         nvfuser::Tensor arg2,                                                                         \
         nvfuser::Scalar arg3,                                                                         \
         nvfuser::Scalar arg4) -> nvfuser::Tensor {                                                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                                         \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                                        \
        nvfuser::Tensor output = fd->defineTensor();                                                   \
        fd->defineRecord(new nvfuser::OpRecord<                                                        \
                         Nvf::TensorView*,                                                             \
                         Nvf::Val*,                                                                    \
                         Nvf::TensorView*,                                                             \
                         Nvf::Val*,                                                                    \
                         Nvf::Val*>(                                                                   \
            {fd->recordingState(arg1()),                                                               \
             fd->recordingState(arg2()),                                                               \
             fd->recordingState(arg3()),                                                               \
             fd->recordingState(arg4())},                                                              \
            {fd->recordingState(output())},                                                            \
            ("ops." op_str),                                                                           \
            static_cast<                                                                               \
                Nvf::                                                                                  \
                    TensorView* (*)(Nvf::Val*, Nvf::TensorView*, Nvf::Val*, Nvf::Val*)>(               \
                Nvf::op_name)));                                                                       \
        return output;                                                                                 \
      },                                                                                               \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP("addcmul", addcmul)
#undef NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP

#define NVFUSER_PYTHON_BINDING_REDUCTION_OP(op_str, op_name)                                          \
  nvf_ops.def(                                                                                        \
      op_str,                                                                                         \
      [](nvfuser::FusionDefinition::Operators& self,                                                  \
         nvfuser::Tensor arg,                                                                         \
         const std::vector<int>& axes,                                                                \
         bool keepdim,                                                                                \
         Nvf::DataType dtype) -> nvfuser::Tensor {                                                    \
        FUSER_PERF_SCOPE("Operators." op_str);                                                        \
        nvfuser::FusionDefinition* fd = self.fusion_definition;                                       \
        nvfuser::Tensor output = fd->defineTensor();                                                  \
        fd->defineRecord(new nvfuser::ReductionOpRecord(                                              \
            {fd->recordingState(arg())},                                                              \
            {fd->recordingState(output())},                                                           \
            ("ops." op_str),                                                                          \
            static_cast<                                                                              \
                Nvf::                                                                                 \
                    TensorView* (*)(Nvf::TensorView*, const std::vector<int>&, bool, Nvf::DataType)>( \
                Nvf::op_name),                                                                        \
            axes,                                                                                     \
            keepdim,                                                                                  \
            dtype));                                                                                  \
        return output;                                                                                \
      },                                                                                              \
      py::arg("arg"),                                                                                 \
      py::arg("axes"),                                                                                \
      py::arg("keepdim") = false,                                                                     \
      py::arg("dtype") = Nvf::DataType::Null,                                                         \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_REDUCTION_OP("sum", sum)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP("max", max)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP("min", min)
#undef NVFUSER_PYTHON_BINDING_REDUCTION_OP

#define NVFUSER_PYTHON_BINDING_CAST_OP(op_str, op_name)                     \
  nvf_ops.def(                                                              \
      op_str,                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                        \
         nvfuser::Tensor arg,                                               \
         Nvf::DataType dtype) -> nvfuser::Tensor {                          \
        FUSER_PERF_SCOPE("Operators." op_str);                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;             \
        nvfuser::Tensor output = fd->defineTensor();                        \
        fd->defineRecord(                                                   \
            new nvfuser::CastOpRecord<Nvf::TensorView*, Nvf::TensorView*>(  \
                {fd->recordingState(arg())},                                \
                {fd->recordingState(output())},                             \
                ("ops." op_str),                                            \
                static_cast<                                                \
                    Nvf::TensorView* (*)(Nvf::DataType, Nvf::TensorView*)>( \
                    Nvf::op_name),                                          \
                dtype));                                                    \
        return output;                                                      \
      },                                                                    \
      py::arg("arg"),                                                       \
      py::arg("dtype"),                                                     \
      py::return_value_policy::reference);                                  \
  nvf_ops.def(                                                              \
      op_str,                                                               \
      [](nvfuser::FusionDefinition::Operators& self,                        \
         nvfuser::Scalar arg,                                               \
         Nvf::DataType dtype) -> nvfuser::Scalar {                          \
        FUSER_PERF_SCOPE("Operators." op_str);                              \
        nvfuser::FusionDefinition* fd = self.fusion_definition;             \
        nvfuser::Scalar output = fd->defineScalar();                        \
        fd->defineRecord(new nvfuser::CastOpRecord<Nvf::Val*, Nvf::Val*>(   \
            {fd->recordingState(arg())},                                    \
            {fd->recordingState(output())},                                 \
            ("ops." op_str),                                                \
            static_cast<Nvf::Val* (*)(Nvf::DataType, Nvf::Val*)>(           \
                Nvf::op_name),                                              \
            dtype));                                                        \
        return output;                                                      \
      },                                                                    \
      py::arg("arg"),                                                       \
      py::arg("dtype"),                                                     \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_CAST_OP("cast", castOp)
#undef NVFUSER_PYTHON_BINDING_CAST_OP

  nvf_ops.def(
      "permute",
      [](nvfuser::FusionDefinition::Operators& self,
         nvfuser::Tensor arg,
         std::vector<int64_t>& dims) -> nvfuser::Tensor {
        nvfuser::FusionDefinition* fd = self.fusion_definition;
        nvfuser::Tensor output = fd->defineTensor();
        self.fusion_definition->defineRecord(new nvfuser::PermuteOpRecord(
            {fd->recordingState(arg())}, {fd->recordingState(output())}, dims));
        return output;
      },
      py::arg("arg"),
      py::arg("dims"),
      py::return_value_policy::reference);

  nvf_ops.def(
      "squeeze",
      [](nvfuser::FusionDefinition::Operators& self,
         nvfuser::Tensor arg,
         std::vector<int64_t>& original_shape,
         int64_t dim) -> nvfuser::Tensor {
        FUSER_PERF_SCOPE("Operators.squeeze");
        nvfuser::FusionDefinition* fd = self.fusion_definition;
        nvfuser::Tensor output = fd->defineTensor();
        fd->defineRecord(new nvfuser::SqueezeOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            original_shape,
            dim));
        return output;
      },
      py::arg("arg"),
      py::arg("original_shape"),
      py::arg("dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "view",
      [](nvfuser::FusionDefinition::Operators& self,
         nvfuser::Tensor arg,
         std::vector<int64_t>& original_shape,
         std::vector<int64_t>& new_shape) -> nvfuser::Tensor {
        nvfuser::FusionDefinition* fd = self.fusion_definition;
        nvfuser::Tensor output = fd->defineTensor();
        self.fusion_definition->defineRecord(new nvfuser::ViewOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            original_shape,
            new_shape));
        return output;
      },
      py::arg("arg"),
      py::arg("original_shape"),
      py::arg("new_shape"),
      py::return_value_policy::reference);

  nvf_ops.def(
      "var",
      [](nvfuser::FusionDefinition::Operators& self,
         nvfuser::Tensor arg,
         std::vector<int>& axes,
         int64_t correction,
         bool keepdim) -> nvfuser::Tensor {
        FUSER_PERF_SCOPE("Operators.var");
        nvfuser::FusionDefinition* fd = self.fusion_definition;
        nvfuser::Tensor output = fd->defineTensor();
        fd->defineRecord(new nvfuser::VarianceOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            axes,
            correction,
            keepdim));
        return output;
      },
      py::arg("arg"),
      py::arg("axes"),
      py::arg("correction"),
      py::arg("keepdim") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "var_mean",
      [](nvfuser::FusionDefinition::Operators& self,
         nvfuser::Tensor arg,
         std::vector<int>& axes,
         int64_t correction,
         bool keepdim) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.var_mean");
        nvfuser::FusionDefinition* fd = self.fusion_definition;
        nvfuser::Tensor var = fd->defineTensor();
        nvfuser::Tensor mean = fd->defineTensor();
        fd->defineRecord(new nvfuser::VarianceMeanOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(var()), fd->recordingState(mean())},
            axes,
            correction,
            keepdim));
        return std::make_tuple(var, mean);
      },
      py::arg("arg"),
      py::arg("axes"),
      py::arg("correction"),
      py::arg("keepdim") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "batch_norm",
      [](nvfuser::FusionDefinition::Operators& self,
         nvfuser::Tensor arg,
         c10::optional<nvfuser::Tensor> weight,
         c10::optional<nvfuser::Tensor> bias,
         c10::optional<nvfuser::Tensor> running_mean,
         c10::optional<nvfuser::Tensor> running_var,
         nvfuser::Scalar momentum,
         nvfuser::Scalar eps,
         bool training,
         bool channels_last) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.batch_norm");
        nvfuser::FusionDefinition* fd = self.fusion_definition;
        nvfuser::Tensor output = fd->defineTensor();
        nvfuser::Tensor mean = fd->defineTensor();
        nvfuser::Tensor invstd = fd->defineTensor();
        auto weight_state = weight.has_value()
            ? fd->recordingState(weight.value()())
            : nvfuser::State(0, nvfuser::StateType::None);
        auto bias_state = bias.has_value()
            ? fd->recordingState(bias.value()())
            : nvfuser::State(0, nvfuser::StateType::None);
        auto running_mean_state = running_mean.has_value()
            ? fd->recordingState(running_mean.value()())
            : nvfuser::State(0, nvfuser::StateType::None);
        auto running_var_state = running_var.has_value()
            ? fd->recordingState(running_var.value()())
            : nvfuser::State(0, nvfuser::StateType::None);
        fd->defineRecord(new nvfuser::BatchNormOpRecord(
            {fd->recordingState(arg()),
             weight_state,
             bias_state,
             running_mean_state,
             running_var_state,
             fd->recordingState(momentum()),
             fd->recordingState(eps())},
            {fd->recordingState(output()),
             fd->recordingState(mean()),
             fd->recordingState(invstd())},
            training,
            channels_last));
        return std::make_tuple(output, mean, invstd);
      },
      py::arg("arg"),
      py::arg("weight").none(true),
      py::arg("bias").none(true),
      py::arg("running_mean").none(true),
      py::arg("running_var").none(true),
      py::arg("momentum"),
      py::arg("eps"),
      py::arg("training"),
      py::arg("channels_last") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "broadcast_in_dim",
      [](nvfuser::FusionDefinition::Operators& self,
         nvfuser::Tensor arg,
         std::vector<int64_t>& output_shape,
         std::vector<int64_t>& broadcast_dims) -> nvfuser::Tensor {
        FUSER_PERF_SCOPE("Operators.broadcast_in_dim");
        nvfuser::FusionDefinition* fd = self.fusion_definition;
        TORCH_CHECK(
            output_shape.size() >= broadcast_dims.size(),
            "broadcast_dims vector size is too big for output shape!");
        nvfuser::Tensor output = fd->defineTensor();
        fd->defineRecord(new nvfuser::BroadcastOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            "ops.broadcast_in_dim",
            output_shape,
            broadcast_dims));
        return output;
      },
      py::arg("arg"),
      py::arg("output_shape"),
      py::arg("broadcast_dims"),
      py::return_value_policy::reference);
}

} // namespace jit
} // namespace torch

#else

namespace torch {
namespace jit {

void initNvFuserPythonBindings(PyObject* module) {}

} // namespace jit
} // namespace torch

#endif // USE_CUDA
