#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>

#ifdef USE_CUDA
#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_definition.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <iostream>

namespace torch {
namespace jit {

void initNvFuserPythonBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  //! Top Level nvFuser Python submodule
  auto nvfuser = m.def_submodule("_nvfuser");

  //! DataTypes supported by nvFuser in the FusionDefinition
  py::enum_<NvfDataType>(nvfuser, "DataType")
      .value("Double", NvfDataType::Double)
      .value("Float", NvfDataType::Float)
      .value("Half", NvfDataType::Half)
      .value("Int", NvfDataType::Int)
      .value("Int32", NvfDataType::Int32)
      .value("Bool", NvfDataType::Bool)
      .value("BFloat16", NvfDataType::BFloat16)
      .value("ComplexFloat", NvfDataType::ComplexFloat)
      .value("ComplexDouble", NvfDataType::ComplexDouble)
      .value("Null", NvfDataType::Null);

  //! Binding an object that owns a FusionExecutorCache instance and provides
  //! an interface
  //! \todo This object will be removed when a FusionManager is added
  //! containing a cache.
  py::class_<nvfuser::FusionOwner> fusion(nvfuser, "Fusion");
  fusion.def(py::init<>())
      .def(
          "execute",
          [](nvfuser::FusionOwner& self, const py::iterable& iter) {
            std::vector<IValue> inputs;
            for (py::handle obj : iter) {
              inputs.push_back(toIValue(obj, c10::AnyType::get()));
            }
            return self.execute(inputs);
          },
          py::return_value_policy::reference)
      .def("print_ir", [](nvfuser::FusionOwner& self) { self.printIr(); })
      .def("print_kernel", [](nvfuser::FusionOwner& self) {
        self.printKernel();
      });

  //! These are the FusionDefinition supported object types that are either
  //! defined as inputs or the output of an operation.
  py::class_<nvfuser::Tensor>(nvfuser, "Tensor");
  py::class_<nvfuser::Scalar>(nvfuser, "Scalar");

  //! The FusionDefinition is a context manager in Python where the user will
  //! define the set the operations and connections between operations for
  //! nvFuser to create.
  py::class_<nvfuser::FusionDefinition> fusion_def(nvfuser, "FusionDefinition");
  fusion_def.def(py::init<nvfuser::FusionOwner*>())
      .def_readwrite("ops", &nvfuser::FusionDefinition::ops)
      .def(
          "__enter__",
          [](nvfuser::FusionDefinition& self) -> nvfuser::FusionDefinition* {
            return self.enter();
          })
      .def(
          "__exit__",
          [](nvfuser::FusionDefinition& self,
             void* exc_type,
             void* exc_value,
             void* traceback) { self.exit(); })
      .def(
          "add_output",
          [](nvfuser::FusionDefinition& self, nvfuser::Scalar* output) {
            self.defineRecord(
                new nvfuser::OutputRecord<NvfVal>({output->index}));
          })
      .def(
          "add_output",
          [](nvfuser::FusionDefinition& self, nvfuser::Tensor* output) {
            self.defineRecord(
                new nvfuser::OutputRecord<NvfTensorView>({output->index}));
          })
      .def(
          "define_tensor",
          [](nvfuser::FusionDefinition& self,
             size_t ndims,
             NvfDataType dtype = NvfDataType::Float) -> nvfuser::Tensor* {
            std::vector<int64_t> maybe_symbolic_sizes(ndims, -1);
            ;
            std::vector<bool> contig_info(ndims, false);

            nvfuser::Tensor* out = self.defineTensor();
            self.defineRecord(new nvfuser::InputTensorRecord(
                {out->index},
                std::move(maybe_symbolic_sizes),
                std::move(contig_info),
                dtype));

            return out;
          },
          py::arg("ndims"),
          py::arg("dtype") = torch::jit::fuser::cuda::DataType::Float,
          py::return_value_policy::reference)
      .def(
          "define_tensor",
          [](nvfuser::FusionDefinition& self,
             std::vector<int64_t> sizes,
             std::vector<int64_t> strides,
             NvfDataType dtype = NvfDataType::Float) -> nvfuser::Tensor* {
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

            nvfuser::Tensor* out = self.defineTensor();
            self.defineRecord(new nvfuser::InputTensorRecord(
                {out->index},
                std::move(maybe_symbolic_sizes),
                std::move(contig_info),
                dtype));

            return out;
          },
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("dtype") = NvfDataType::Float,
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](nvfuser::FusionDefinition& self, double val) -> nvfuser::Scalar* {
            nvfuser::Scalar* out = self.defineScalar();
            self.defineRecord(
                new nvfuser::
                    ConstantRecord<torch::jit::fuser::cuda::Double, double>(
                        {out->index}, val));
            return out;
          },
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](nvfuser::FusionDefinition& self,
             c10::complex<double> val) -> nvfuser::Scalar* {
            nvfuser::Scalar* out = self.defineScalar();
            self.defineRecord(new nvfuser::ConstantRecord<
                              torch::jit::fuser::cuda::ComplexDouble,
                              c10::complex<double>>({out->index}, val));
            return out;
          },
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](nvfuser::FusionDefinition& self, bool val) -> nvfuser::Scalar* {
            nvfuser::Scalar* out = self.defineScalar();
            self.defineRecord(
                new nvfuser::
                    ConstantRecord<torch::jit::fuser::cuda::Bool, bool>(
                        {out->index}, val));
            return out;
          },
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](nvfuser::FusionDefinition& self, int64_t val) -> nvfuser::Scalar* {
            nvfuser::Scalar* out = self.defineScalar();
            self.defineRecord(
                new nvfuser::
                    ConstantRecord<torch::jit::fuser::cuda::Int, int64_t>(
                        {out->index}, val));
            return out;
          },
          py::return_value_policy::reference)
      .def(
          "define_scalar",
          [](nvfuser::FusionDefinition& self,
             NvfDataType dtype = torch::jit::fuser::cuda::DataType::Double)
              -> nvfuser::Scalar* {
            nvfuser::Scalar* out = self.defineScalar();
            self.defineRecord(new nvfuser::ScalarRecord({out->index}, dtype));
            return out;
          },
          py::arg("dtype") = torch::jit::fuser::cuda::DataType::Double,
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

#define NVFUSER_PYTHON_BINDING_UNARY_OP(op_str, op_name)                  \
  nvf_ops.def(                                                            \
      op_str,                                                             \
      [](nvfuser::FusionDefinition::Operators& self,                      \
         nvfuser::Tensor* input) -> nvfuser::Tensor* {                    \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor(); \
        self.fusion_definition->defineRecord(                             \
            new nvfuser::OpRecord<NvfTensorView*, NvfTensorView*>(        \
                {input->index},                                           \
                {output->index},                                          \
                static_cast<NvfTensorView* (*)(NvfTensorView*)>(          \
                    torch::jit::fuser::cuda::op_name)));                  \
        return output;                                                    \
      },                                                                  \
      py::return_value_policy::reference);                                \
  nvf_ops.def(                                                            \
      op_str,                                                             \
      [](nvfuser::FusionDefinition::Operators& self,                      \
         nvfuser::Scalar* input) -> nvfuser::Scalar* {                    \
        nvfuser::Scalar* output = self.fusion_definition->defineScalar(); \
        self.fusion_definition->defineRecord(                             \
            new nvfuser::OpRecord<NvfVal*, NvfVal*>(                      \
                {input->index},                                           \
                {output->index},                                          \
                static_cast<NvfVal* (*)(NvfVal*)>(                        \
                    torch::jit::fuser::cuda::op_name)));                  \
        return output;                                                    \
      },                                                                  \
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

#define NVFUSER_PYTHON_BINDING_BINARY_OP(op_str, op_name)                    \
  nvf_ops.def(                                                               \
      op_str,                                                                \
      [](nvfuser::FusionDefinition::Operators& self,                         \
         nvfuser::Tensor* arg1,                                              \
         nvfuser::Tensor* arg2) -> nvfuser::Tensor* {                        \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();    \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<          \
                                             NvfTensorView*,                 \
                                             NvfTensorView*,                 \
                                             NvfTensorView*>(                \
            {arg1->index, arg2->index},                                      \
            {output->index},                                                 \
            static_cast<NvfTensorView* (*)(NvfTensorView*, NvfTensorView*)>( \
                torch::jit::fuser::cuda::op_name)));                         \
        return output;                                                       \
      },                                                                     \
      py::return_value_policy::reference);                                   \
  nvf_ops.def(                                                               \
      op_str,                                                                \
      [](nvfuser::FusionDefinition::Operators& self,                         \
         nvfuser::Tensor* arg1,                                              \
         nvfuser::Scalar* arg2) -> nvfuser::Tensor* {                        \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();    \
        self.fusion_definition->defineRecord(                                \
            new nvfuser::OpRecord<NvfTensorView*, NvfTensorView*, NvfVal*>(  \
                {arg1->index, arg2->index},                                  \
                {output->index},                                             \
                static_cast<NvfTensorView* (*)(NvfTensorView*, NvfVal*)>(    \
                    torch::jit::fuser::cuda::op_name)));                     \
        return output;                                                       \
      },                                                                     \
      py::return_value_policy::reference);                                   \
  nvf_ops.def(                                                               \
      op_str,                                                                \
      [](nvfuser::FusionDefinition::Operators& self,                         \
         nvfuser::Scalar* arg1,                                              \
         nvfuser::Tensor* arg2) -> nvfuser::Tensor* {                        \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();    \
        self.fusion_definition->defineRecord(                                \
            new nvfuser::OpRecord<NvfTensorView*, NvfVal*, NvfTensorView*>(  \
                {arg1->index, arg2->index},                                  \
                {output->index},                                             \
                static_cast<NvfTensorView* (*)(NvfVal*, NvfTensorView*)>(    \
                    torch::jit::fuser::cuda::op_name)));                     \
        return output;                                                       \
      },                                                                     \
      py::return_value_policy::reference);                                   \
  nvf_ops.def(                                                               \
      op_str,                                                                \
      [](nvfuser::FusionDefinition::Operators& self,                         \
         nvfuser::Scalar* arg1,                                              \
         nvfuser::Scalar* arg2) -> nvfuser::Scalar* {                        \
        nvfuser::Scalar* output = self.fusion_definition->defineScalar();    \
        self.fusion_definition->defineRecord(                                \
            new nvfuser::OpRecord<NvfVal*, NvfVal*, NvfVal*>(                \
                {arg1->index, arg2->index},                                  \
                {output->index},                                             \
                static_cast<NvfVal* (*)(NvfVal*, NvfVal*)>(                  \
                    torch::jit::fuser::cuda::op_name)));                     \
        return output;                                                       \
      },                                                                     \
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

#define NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP(op_str, op_name)           \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](nvfuser::FusionDefinition::Operators& self,                           \
         nvfuser::Tensor* arg1,                                                \
         nvfuser::Tensor* arg2,                                                \
         nvfuser::Scalar* arg3) -> nvfuser::Tensor* {                          \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();      \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<            \
                                             NvfTensorView*,                   \
                                             NvfTensorView*,                   \
                                             NvfTensorView*,                   \
                                             NvfVal*>(                         \
            {arg1->index, arg2->index, arg3->index},                           \
            {output->index},                                                   \
            static_cast<                                                       \
                NvfTensorView* (*)(NvfTensorView*, NvfTensorView*, NvfVal*)>(  \
                torch::jit::fuser::cuda::op_name)));                           \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](nvfuser::FusionDefinition::Operators& self,                           \
         nvfuser::Tensor* arg1,                                                \
         nvfuser::Scalar* arg2,                                                \
         nvfuser::Scalar* arg3) -> nvfuser::Tensor* {                          \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();      \
        self.fusion_definition->defineRecord(                                  \
            new nvfuser::                                                      \
                OpRecord<NvfTensorView*, NvfTensorView*, NvfVal*, NvfVal*>(    \
                    {arg1->index, arg2->index, arg3->index},                   \
                    {output->index},                                           \
                    static_cast<                                               \
                        NvfTensorView* (*)(NvfTensorView*, NvfVal*, NvfVal*)>( \
                        torch::jit::fuser::cuda::op_name)));                   \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](nvfuser::FusionDefinition::Operators& self,                           \
         nvfuser::Scalar* arg1,                                                \
         nvfuser::Tensor* arg2,                                                \
         nvfuser::Scalar* arg3) -> nvfuser::Tensor* {                          \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();      \
        self.fusion_definition->defineRecord(                                  \
            new nvfuser::                                                      \
                OpRecord<NvfTensorView*, NvfVal*, NvfTensorView*, NvfVal*>(    \
                    {arg1->index, arg2->index, arg3->index},                   \
                    {output->index},                                           \
                    static_cast<                                               \
                        NvfTensorView* (*)(NvfVal*, NvfTensorView*, NvfVal*)>( \
                        torch::jit::fuser::cuda::op_name)));                   \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](nvfuser::FusionDefinition::Operators& self,                           \
         nvfuser::Scalar* arg1,                                                \
         nvfuser::Scalar* arg2,                                                \
         nvfuser::Scalar* arg3) -> nvfuser::Scalar* {                          \
        nvfuser::Scalar* output = self.fusion_definition->defineScalar();      \
        self.fusion_definition->defineRecord(                                  \
            new nvfuser::OpRecord<NvfVal*, NvfVal*, NvfVal*, NvfVal*>(         \
                {arg1->index, arg2->index, arg3->index},                       \
                {output->index},                                               \
                static_cast<NvfVal* (*)(NvfVal*, NvfVal*, NvfVal*)>(           \
                    torch::jit::fuser::cuda::op_name)));                       \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP("add_alpha", add_alpha)
  NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP("sub_alpha", sub_alpha)
#undef NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_OP(op_str, op_name)                           \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Scalar* arg1,                                                      \
         nvfuser::Scalar* arg2,                                                      \
         nvfuser::Scalar* arg3) -> nvfuser::Scalar* {                                \
        nvfuser::Scalar* output = self.fusion_definition->defineScalar();            \
        self.fusion_definition->defineRecord(                                        \
            new nvfuser::OpRecord<NvfVal*, NvfVal*, NvfVal*, NvfVal*>(               \
                {arg1->index, arg2->index, arg3->index},                             \
                {output->index},                                                     \
                static_cast<NvfVal* (*)(NvfVal*, NvfVal*, NvfVal*)>(                 \
                    torch::jit::fuser::cuda::op_name)));                             \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);                                           \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Tensor* arg1,                                                      \
         nvfuser::Tensor* arg2,                                                      \
         nvfuser::Tensor* arg3) -> nvfuser::Tensor* {                                \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();            \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<                  \
                                             NvfTensorView*,                         \
                                             NvfTensorView*,                         \
                                             NvfTensorView*,                         \
                                             NvfTensorView*>(                        \
            {arg1->index, arg2->index, arg3->index},                                 \
            {output->index},                                                         \
            static_cast<                                                             \
                NvfTensorView* (*)(NvfTensorView*, NvfTensorView*, NvfTensorView*)>( \
                torch::jit::fuser::cuda::op_name)));                                 \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);                                           \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Tensor* arg1,                                                      \
         nvfuser::Tensor* arg2,                                                      \
         nvfuser::Scalar* arg3) -> nvfuser::Tensor* {                                \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();            \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<                  \
                                             NvfTensorView*,                         \
                                             NvfTensorView*,                         \
                                             NvfTensorView*,                         \
                                             NvfVal*>(                               \
            {arg1->index, arg2->index, arg3->index},                                 \
            {output->index},                                                         \
            static_cast<                                                             \
                NvfTensorView* (*)(NvfTensorView*, NvfTensorView*, NvfVal*)>(        \
                torch::jit::fuser::cuda::op_name)));                                 \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);                                           \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Tensor* arg1,                                                      \
         nvfuser::Scalar* arg2,                                                      \
         nvfuser::Tensor* arg3) -> nvfuser::Tensor* {                                \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();            \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<                  \
                                             NvfTensorView*,                         \
                                             NvfTensorView*,                         \
                                             NvfVal*,                                \
                                             NvfTensorView*>(                        \
            {arg1->index, arg2->index, arg3->index},                                 \
            {output->index},                                                         \
            static_cast<                                                             \
                NvfTensorView* (*)(NvfTensorView*, NvfVal*, NvfTensorView*)>(        \
                torch::jit::fuser::cuda::op_name)));                                 \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);                                           \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Scalar* arg1,                                                      \
         nvfuser::Tensor* arg2,                                                      \
         nvfuser::Tensor* arg3) -> nvfuser::Tensor* {                                \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();            \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<                  \
                                             NvfTensorView*,                         \
                                             NvfVal*,                                \
                                             NvfTensorView*,                         \
                                             NvfTensorView*>(                        \
            {arg1->index, arg2->index, arg3->index},                                 \
            {output->index},                                                         \
            static_cast<                                                             \
                NvfTensorView* (*)(NvfVal*, NvfTensorView*, NvfTensorView*)>(        \
                torch::jit::fuser::cuda::op_name)));                                 \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);                                           \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Scalar* arg1,                                                      \
         nvfuser::Scalar* arg2,                                                      \
         nvfuser::Tensor* arg3) -> nvfuser::Tensor* {                                \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();            \
        self.fusion_definition->defineRecord(                                        \
            new nvfuser::                                                            \
                OpRecord<NvfTensorView*, NvfVal*, NvfVal*, NvfTensorView*>(          \
                    {arg1->index, arg2->index, arg3->index},                         \
                    {output->index},                                                 \
                    static_cast<                                                     \
                        NvfTensorView* (*)(NvfVal*, NvfVal*, NvfTensorView*)>(       \
                        torch::jit::fuser::cuda::op_name)));                         \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);                                           \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Tensor* arg1,                                                      \
         nvfuser::Scalar* arg2,                                                      \
         nvfuser::Scalar* arg3) -> nvfuser::Tensor* {                                \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();            \
        self.fusion_definition->defineRecord(                                        \
            new nvfuser::                                                            \
                OpRecord<NvfTensorView*, NvfTensorView*, NvfVal*, NvfVal*>(          \
                    {arg1->index, arg2->index, arg3->index},                         \
                    {output->index},                                                 \
                    static_cast<                                                     \
                        NvfTensorView* (*)(NvfTensorView*, NvfVal*, NvfVal*)>(       \
                        torch::jit::fuser::cuda::op_name)));                         \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);                                           \
  nvf_ops.def(                                                                       \
      op_str,                                                                        \
      [](nvfuser::FusionDefinition::Operators& self,                                 \
         nvfuser::Scalar* arg1,                                                      \
         nvfuser::Tensor* arg2,                                                      \
         nvfuser::Scalar* arg3) -> nvfuser::Tensor* {                                \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();            \
        self.fusion_definition->defineRecord(                                        \
            new nvfuser::                                                            \
                OpRecord<NvfTensorView*, NvfVal*, NvfTensorView*, NvfVal*>(          \
                    {arg1->index, arg2->index, arg3->index},                         \
                    {output->index},                                                 \
                    static_cast<                                                     \
                        NvfTensorView* (*)(NvfVal*, NvfTensorView*, NvfVal*)>(       \
                        torch::jit::fuser::cuda::op_name)));                         \
        return output;                                                               \
      },                                                                             \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_OP("lerp", lerp)
  NVFUSER_PYTHON_BINDING_TERNARY_OP("where", where)
#undef NVFUSER_PYTHON_BINDING_TERNARY_OP

#define NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP(op_str, op_name)              \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](nvfuser::FusionDefinition::Operators& self,                           \
         nvfuser::Scalar* arg1,                                                \
         nvfuser::Scalar* arg2,                                                \
         nvfuser::Scalar* arg3) -> nvfuser::Scalar* {                          \
        nvfuser::Scalar* output = self.fusion_definition->defineScalar();      \
        self.fusion_definition->defineRecord(                                  \
            new nvfuser::OpRecord<NvfVal*, NvfVal*, NvfVal*, NvfVal*>(         \
                {arg1->index, arg2->index, arg3->index},                       \
                {output->index},                                               \
                static_cast<NvfVal* (*)(NvfVal*, NvfVal*, NvfVal*)>(           \
                    torch::jit::fuser::cuda::op_name)));                       \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](nvfuser::FusionDefinition::Operators& self,                           \
         nvfuser::Tensor* arg1,                                                \
         nvfuser::Scalar* arg2,                                                \
         nvfuser::Scalar* arg3) -> nvfuser::Tensor* {                          \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();      \
        self.fusion_definition->defineRecord(                                  \
            new nvfuser::                                                      \
                OpRecord<NvfTensorView*, NvfTensorView*, NvfVal*, NvfVal*>(    \
                    {arg1->index, arg2->index, arg3->index},                   \
                    {output->index},                                           \
                    static_cast<                                               \
                        NvfTensorView* (*)(NvfTensorView*, NvfVal*, NvfVal*)>( \
                        torch::jit::fuser::cuda::op_name)));                   \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP("clamp", clamp)
  NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP("threshold", threshold)
#undef NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP(op_str, op_name)                         \
  nvf_ops.def(                                                                                \
      op_str,                                                                                 \
      [](nvfuser::FusionDefinition::Operators& self,                                          \
         nvfuser::Scalar* arg1,                                                               \
         nvfuser::Scalar* arg2,                                                               \
         nvfuser::Scalar* arg3,                                                               \
         nvfuser::Scalar* arg4) -> nvfuser::Scalar* {                                         \
        nvfuser::Scalar* output = self.fusion_definition->defineScalar();                     \
        self.fusion_definition->defineRecord(                                                 \
            new nvfuser::                                                                     \
                OpRecord<NvfVal*, NvfVal*, NvfVal*, NvfVal*, NvfVal*>(                        \
                    {arg1->index, arg2->index, arg3->index, arg4->index},                     \
                    {output->index},                                                          \
                    static_cast<                                                              \
                        NvfVal* (*)(NvfVal*, NvfVal*, NvfVal*, NvfVal*)>(                     \
                        torch::jit::fuser::cuda::op_name)));                                  \
        return output;                                                                        \
      },                                                                                      \
      py::return_value_policy::reference);                                                    \
  nvf_ops.def(                                                                                \
      op_str,                                                                                 \
      [](nvfuser::FusionDefinition::Operators& self,                                          \
         nvfuser::Tensor* arg1,                                                               \
         nvfuser::Tensor* arg2,                                                               \
         nvfuser::Tensor* arg3,                                                               \
         nvfuser::Scalar* arg4) -> nvfuser::Tensor* {                                         \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();                     \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<                           \
                                             NvfTensorView*,                                  \
                                             NvfTensorView*,                                  \
                                             NvfTensorView*,                                  \
                                             NvfTensorView*,                                  \
                                             NvfTensorView*>(                                 \
            {arg1->index, arg2->index, arg3->index, arg4->index},                             \
            {output->index},                                                                  \
            static_cast<                                                                      \
                NvfTensorView* (*)(NvfTensorView*, NvfTensorView*, NvfTensorView*, NvfVal*)>( \
                torch::jit::fuser::cuda::op_name)));                                          \
        return output;                                                                        \
      },                                                                                      \
      py::return_value_policy::reference);                                                    \
  nvf_ops.def(                                                                                \
      op_str,                                                                                 \
      [](nvfuser::FusionDefinition::Operators& self,                                          \
         nvfuser::Tensor* arg1,                                                               \
         nvfuser::Tensor* arg2,                                                               \
         nvfuser::Scalar* arg3,                                                               \
         nvfuser::Scalar* arg4) -> nvfuser::Tensor* {                                         \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();                     \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<                           \
                                             NvfTensorView*,                                  \
                                             NvfTensorView*,                                  \
                                             NvfTensorView*,                                  \
                                             NvfVal*,                                         \
                                             NvfVal*>(                                        \
            {arg1->index, arg2->index, arg3->index, arg4->index},                             \
            {output->index},                                                                  \
            static_cast<                                                                      \
                NvfTensorView* (*)(NvfTensorView*, NvfTensorView*, NvfVal*, NvfVal*)>(        \
                torch::jit::fuser::cuda::op_name)));                                          \
        return output;                                                                        \
      },                                                                                      \
      py::return_value_policy::reference);                                                    \
  nvf_ops.def(                                                                                \
      op_str,                                                                                 \
      [](nvfuser::FusionDefinition::Operators& self,                                          \
         nvfuser::Tensor* arg1,                                                               \
         nvfuser::Scalar* arg2,                                                               \
         nvfuser::Tensor* arg3,                                                               \
         nvfuser::Scalar* arg4) -> nvfuser::Tensor* {                                         \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();                     \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<                           \
                                             NvfTensorView*,                                  \
                                             NvfTensorView*,                                  \
                                             NvfVal*,                                         \
                                             NvfTensorView*,                                  \
                                             NvfVal*>(                                        \
            {arg1->index, arg2->index, arg3->index, arg4->index},                             \
            {output->index},                                                                  \
            static_cast<                                                                      \
                NvfTensorView* (*)(NvfTensorView*, NvfVal*, NvfTensorView*, NvfVal*)>(        \
                torch::jit::fuser::cuda::op_name)));                                          \
        return output;                                                                        \
      },                                                                                      \
      py::return_value_policy::reference);                                                    \
  nvf_ops.def(                                                                                \
      op_str,                                                                                 \
      [](nvfuser::FusionDefinition::Operators& self,                                          \
         nvfuser::Scalar* arg1,                                                               \
         nvfuser::Tensor* arg2,                                                               \
         nvfuser::Tensor* arg3,                                                               \
         nvfuser::Scalar* arg4) -> nvfuser::Tensor* {                                         \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();                     \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<                           \
                                             NvfTensorView*,                                  \
                                             NvfVal*,                                         \
                                             NvfTensorView*,                                  \
                                             NvfTensorView*,                                  \
                                             NvfVal*>(                                        \
            {arg1->index, arg2->index, arg3->index, arg4->index},                             \
            {output->index},                                                                  \
            static_cast<                                                                      \
                NvfTensorView* (*)(NvfVal*, NvfTensorView*, NvfTensorView*, NvfVal*)>(        \
                torch::jit::fuser::cuda::op_name)));                                          \
        return output;                                                                        \
      },                                                                                      \
      py::return_value_policy::reference);                                                    \
  nvf_ops.def(                                                                                \
      op_str,                                                                                 \
      [](nvfuser::FusionDefinition::Operators& self,                                          \
         nvfuser::Scalar* arg1,                                                               \
         nvfuser::Scalar* arg2,                                                               \
         nvfuser::Tensor* arg3,                                                               \
         nvfuser::Scalar* arg4) -> nvfuser::Tensor* {                                         \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();                     \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<                           \
                                             NvfTensorView*,                                  \
                                             NvfVal*,                                         \
                                             NvfVal*,                                         \
                                             NvfTensorView*,                                  \
                                             NvfVal*>(                                        \
            {arg1->index, arg2->index, arg3->index, arg4->index},                             \
            {output->index},                                                                  \
            static_cast<                                                                      \
                NvfTensorView* (*)(NvfVal*, NvfVal*, NvfTensorView*, NvfVal*)>(               \
                torch::jit::fuser::cuda::op_name)));                                          \
        return output;                                                                        \
      },                                                                                      \
      py::return_value_policy::reference);                                                    \
  nvf_ops.def(                                                                                \
      op_str,                                                                                 \
      [](nvfuser::FusionDefinition::Operators& self,                                          \
         nvfuser::Tensor* arg1,                                                               \
         nvfuser::Scalar* arg2,                                                               \
         nvfuser::Scalar* arg3,                                                               \
         nvfuser::Scalar* arg4) -> nvfuser::Tensor* {                                         \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();                     \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<                           \
                                             NvfTensorView*,                                  \
                                             NvfTensorView*,                                  \
                                             NvfVal*,                                         \
                                             NvfVal*,                                         \
                                             NvfVal*>(                                        \
            {arg1->index, arg2->index, arg3->index, arg4->index},                             \
            {output->index},                                                                  \
            static_cast<                                                                      \
                NvfTensorView* (*)(NvfTensorView*, NvfVal*, NvfVal*, NvfVal*)>(               \
                torch::jit::fuser::cuda::op_name)));                                          \
        return output;                                                                        \
      },                                                                                      \
      py::return_value_policy::reference);                                                    \
  nvf_ops.def(                                                                                \
      op_str,                                                                                 \
      [](nvfuser::FusionDefinition::Operators& self,                                          \
         nvfuser::Scalar* arg1,                                                               \
         nvfuser::Tensor* arg2,                                                               \
         nvfuser::Scalar* arg3,                                                               \
         nvfuser::Scalar* arg4) -> nvfuser::Tensor* {                                         \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();                     \
        self.fusion_definition->defineRecord(new nvfuser::OpRecord<                           \
                                             NvfTensorView*,                                  \
                                             NvfVal*,                                         \
                                             NvfTensorView*,                                  \
                                             NvfVal*,                                         \
                                             NvfVal*>(                                        \
            {arg1->index, arg2->index, arg3->index, arg4->index},                             \
            {output->index},                                                                  \
            static_cast<                                                                      \
                NvfTensorView* (*)(NvfVal*, NvfTensorView*, NvfVal*, NvfVal*)>(               \
                torch::jit::fuser::cuda::op_name)));                                          \
        return output;                                                                        \
      },                                                                                      \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP("addcmul", addcmul)
#undef NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP

#define NVFUSER_PYTHON_BINDING_REDUCTION_OP(op_str, op_name)                 \
  nvf_ops.def(                                                               \
      op_str,                                                                \
      [](nvfuser::FusionDefinition::Operators& self,                         \
         nvfuser::Tensor* arg,                                               \
         const std::vector<int>& axes,                                       \
         bool keep_dim,                                                      \
         NvfDataType dtype) -> nvfuser::Tensor* {                            \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();    \
        self.fusion_definition->defineRecord(new nvfuser::ReductionOpRecord( \
            {arg->index},                                                    \
            {output->index},                                                 \
            torch::jit::fuser::cuda::op_name,                                \
            axes,                                                            \
            keep_dim,                                                        \
            dtype));                                                         \
        return output;                                                       \
      },                                                                     \
      py::arg("arg"),                                                        \
      py::arg("axes"),                                                       \
      py::arg("keep_dim"),                                                   \
      py::arg("dtype") = torch::jit::fuser::cuda::DataType::Null,            \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_REDUCTION_OP("sum", sum)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP("max", max)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP("min", min)
#undef NVFUSER_PYTHON_BINDING_REDUCTION_OP

#define NVFUSER_PYTHON_BINDING_CAST_OP(op_str, op_name)                       \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](nvfuser::FusionDefinition::Operators& self,                          \
         nvfuser::Tensor* arg,                                                \
         NvfDataType dtype) -> nvfuser::Tensor* {                             \
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();     \
        self.fusion_definition->defineRecord(                                 \
            new nvfuser::CastOpRecord<NvfTensorView*, NvfTensorView*>(        \
                {arg->index},                                                 \
                {output->index},                                              \
                static_cast<NvfTensorView* (*)(NvfDataType, NvfTensorView*)>( \
                    torch::jit::fuser::cuda::op_name),                        \
                dtype));                                                      \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](nvfuser::FusionDefinition::Operators& self,                          \
         nvfuser::Scalar* arg,                                                \
         NvfDataType dtype) -> nvfuser::Scalar* {                             \
        nvfuser::Scalar* output = self.fusion_definition->defineScalar();     \
        self.fusion_definition->defineRecord(                                 \
            new nvfuser::CastOpRecord<NvfVal*, NvfVal*>(                      \
                {arg->index},                                                 \
                {output->index},                                              \
                static_cast<NvfVal* (*)(NvfDataType, NvfVal*)>(               \
                    torch::jit::fuser::cuda::op_name),                        \
                dtype));                                                      \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_CAST_OP("cast", castOp)
#undef NVFUSER_PYTHON_BINDING_CAST_OP

  nvf_ops.def(
      "var",
      [](nvfuser::FusionDefinition::Operators& self,
         nvfuser::Tensor* arg,
         std::vector<int>& axes,
         int64_t correction,
         bool keepdim) -> nvfuser::Tensor* {
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();
        self.fusion_definition->defineRecord(new nvfuser::VarianceOpRecord(
            {arg->index}, {output->index}, axes, correction, keepdim));
        return output;
      },
      py::return_value_policy::reference);

  nvf_ops.def(
      "broadcast_in_dim",
      [](nvfuser::FusionDefinition::Operators& self,
         nvfuser::Tensor* arg,
         std::vector<int64_t>& output_shape,
         std::vector<int64_t>& broadcast_dims) -> nvfuser::Tensor* {
        nvfuser::Tensor* output = self.fusion_definition->defineTensor();
        self.fusion_definition->defineRecord(new nvfuser::BroadcastOpRecord(
            {arg->index}, {output->index}, output_shape, broadcast_dims));
        return output;
      },
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
