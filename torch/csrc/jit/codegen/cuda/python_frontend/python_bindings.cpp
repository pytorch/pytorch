#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>

#ifdef USE_CUDA
#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ops/normalization.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_definition.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <iostream>

namespace torch {
namespace jit {

void initNvFuserPythonBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto nvfuser = m.def_submodule("_nvfuser");

  // DataTypes supported by NVFuser in Fusion Definition
  // Types not related to values found in fusion defintions
  // were purposely left out.
  // NOTE: DataType was ambiguous under torch::jit without full qualification.
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

  // Binding an object that owns a FusionExecutorCache instance and provides an
  // interface
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
      .def("print_kernel", [](nvfuser::FusionOwner& self) { self.printKernel(); });

  py::class_<nvfuser::Tensor>(nvfuser, "Tensor");
  py::class_<nvfuser::Scalar>(nvfuser, "Scalar");

  // C++ Side of Context Manager used to mimic the FusionGuard as a way
  // to programatically distinguish code used to define the Fusion instead
  // of having the user mysteriously create an object prior to adding definition
  // code where the object is not used.
  py::class_<nvfuser::FusionDefinition> fusion_def(nvfuser, "FusionDefinition");
  fusion_def.def(py::init<nvfuser::FusionOwner*>())
      .def_readwrite("ops", &nvfuser::FusionDefinition::ops)
      .def(
          "__enter__",
          [](nvfuser::FusionDefinition& self)->nvfuser::FusionDefinition* { return self.enter(); })
      .def(
          "__exit__",
          [](nvfuser::FusionDefinition& self,
             void* exc_type,
             void* exc_value,
             void* traceback) { 
               self.exit();
          })
      .def(
          "add_output",
          [](nvfuser::FusionDefinition& self, nvfuser::Scalar* output) { 
            self.recording.emplace_back(
                new nvfuser::OutputRecord<NvfVal>({output->index}));
          })
      .def(
          "add_output",
          [](nvfuser::FusionDefinition& self, nvfuser::Tensor* output) {
            self.recording.emplace_back(
                new nvfuser::OutputRecord<NvfTensorView>({output->index}));
          })
      .def(
          "define_tensor",
          [](nvfuser::FusionDefinition& self,
             std::vector<int64_t> sizes,
             std::vector<int64_t> strides,
             NvfDataType dtype =
                 NvfDataType::Float) -> nvfuser::Tensor* {
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

            nvfuser::Tensor* out = new nvfuser::Tensor(self.recording_state.size());
            self.recording_state.emplace_back(out);
            self.recording.emplace_back(
                new nvfuser::InputTensorRecord({out->index},
                                               std::move(maybe_symbolic_sizes),
                                               std::move(contig_info),
                                               dtype));

            return out;
          },
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("dtype") = NvfDataType::Float,
          py::return_value_policy::reference);
      /*
      .def(
          "define_constant",
          [](FusionDefinitionContextManager& self,
             double val) -> torch::jit::fuser::cuda::Val* {
            return IrBuilder::create<Double>(val);
          },
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](FusionDefinitionContextManager& self,
             std::complex<double> val) -> torch::jit::fuser::cuda::Val* {
            return IrBuilder::create<ComplexDouble>(c10::complex<double>(val));
          },
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](FusionDefinitionContextManager& self,
             bool val) -> torch::jit::fuser::cuda::Val* {
            return IrBuilder::create<Bool>(val);
          },
          py::return_value_policy::reference)
      .def(
          "define_constant",
          [](FusionDefinitionContextManager& self,
             int64_t val) -> torch::jit::fuser::cuda::Val* {
            return IrBuilder::create<Int>(val);
          },
          py::return_value_policy::reference)
      .def(
          "define_scalar",
          [](FusionDefinitionContextManager& self,
             torch::jit::fuser::cuda::DataType dtype =
                 torch::jit::fuser::cuda::DataType::Double)
              -> torch::jit::fuser::cuda::Val* {
            if (dtype == torch::jit::fuser::cuda::DataType::Double) {
              return IrBuilder::create<Double>();
            } else if (
                dtype == torch::jit::fuser::cuda::DataType::ComplexDouble) {
              return IrBuilder::create<ComplexDouble>();
            } else if (dtype == torch::jit::fuser::cuda::DataType::Bool) {
              return IrBuilder::create<Bool>();
            } else if (dtype == torch::jit::fuser::cuda::DataType::Int) {
              return IrBuilder::create<Int>();
            } else {
              TORCH_CHECK(false, "Dtype is not supported:", dtype);
            }
          },
          py::arg("dtype") = torch::jit::fuser::cuda::DataType::Double,
          py::return_value_policy::reference); */

  py::class_<nvfuser::FusionDefinition::Operators> nvf_ops(fusion_def, "Operators");
  nvf_ops.def(py::init<nvfuser::FusionDefinition*>());

  // ******************** INSERT OP BINDINGS BELOW HERE ********************

#define NVFUSER_PYTHON_BINDING_UNARY_OP(op_str, op_name)                       \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](nvfuser::FusionDefinition::Operators& self, nvfuser::Tensor* input)   \
         -> nvfuser::Tensor* {                                                 \
        nvfuser::Tensor* output =                                              \
          new nvfuser::Tensor(self.fusion_definition->recording_state.size()); \
        self.fusion_definition->recording_state.emplace_back(output);          \
        self.fusion_definition->recording.emplace_back(                        \
          new nvfuser::UnaryOpRecord<NvfTensorView>(                           \
            {input->index}, {output->index},                                   \
            static_cast<NvfTensorView*(*)(NvfTensorView*)>(                    \
                torch::jit::fuser::cuda::op_name)));                           \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](nvfuser::FusionDefinition::Operators& self, nvfuser::Scalar* input)   \
         -> nvfuser::Scalar* {                                                 \
        nvfuser::Scalar* output =                                              \
          new nvfuser::Scalar(self.fusion_definition->recording_state.size()); \
        self.fusion_definition->recording_state.emplace_back(output);          \
        self.fusion_definition->recording.emplace_back(                        \
          new nvfuser::UnaryOpRecord<NvfVal>(                                  \
            {input->index}, {output->index},                                   \
            static_cast<NvfVal*(*)(NvfVal*)>(                                  \
                torch::jit::fuser::cuda::op_name)));                           \
        return output;                                                         \
      },                                                                       \
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
#undef NVFUSER_PYTHON_BINDING_UNARY_OP

/*
#define NVFUSER_PYTHON_BINDING_BINARY_OP(op_str, op_name)                    \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<TensorView*, TensorView*>(                           \
          &torch::jit::fuser::cuda::op_name),                                \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<TensorView*, torch::jit::fuser::cuda::Val*>(         \
          &torch::jit::fuser::cuda::op_name),                                \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<torch::jit::fuser::cuda::Val*, TensorView*>(         \
          &torch::jit::fuser::cuda::op_name),                                \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
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
*/
/*
#define NVFUSER_PYTHON_BINDING_TERNARY_OP(op_str, op_name)                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<TensorView*, TensorView*, TensorView*>(              \
          &torch::jit::fuser::cuda::op_name),                                \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          TensorView*,                                                       \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*,                                     \
          TensorView*>(&torch::jit::fuser::cuda::op_name),                   \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          TensorView*,                                                       \
          TensorView*>(&torch::jit::fuser::cuda::op_name),                   \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          TensorView*>(&torch::jit::fuser::cuda::op_name),                   \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_OP("lerp", lerp)
  NVFUSER_PYTHON_BINDING_TERNARY_OP("where", where)
#undef NVFUSER_PYTHON_BINDING_TERNARY_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_ABRV1_OP(op_str, op_name)             \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_ABRV1_OP("clamp", clamp)
  NVFUSER_PYTHON_BINDING_TERNARY_ABRV1_OP("threshold", threshold)
#undef NVFUSER_PYTHON_BINDING_TERNARY_ABRV1_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_ABRV2_OP(op_str, op_name)             \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          TensorView*,                                                       \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_ABRV2_OP("add_alpha", add_alpha)
  NVFUSER_PYTHON_BINDING_TERNARY_ABRV2_OP("sub_alpha", sub_alpha)
#undef NVFUSER_PYTHON_BINDING_TERNARY_ABRV2_OP

#define NVFUSER_PYTHON_BINDING_QUAD_ABRV3_OP(op_str, op_name)                \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          TensorView*,                                                       \
          TensorView*,                                                       \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          TensorView*,                                                       \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*,                                     \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          TensorView*,                                                       \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          TensorView*,                                                       \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                                   \
  nvf_ops.def_static(                                                        \
      op_str,                                                                \
      py::overload_cast<                                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*,                                     \
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_QUAD_ABRV3_OP("addcmul", addcmul)
#undef NVFUSER_PYTHON_BINDING_QUAD_ABRV3_OP

  // Reduction Operations
  nvf_ops.def_static(
      "max", &torch::jit::fuser::cuda::max, py::return_value_policy::reference);
  nvf_ops.def_static(
      "min", &torch::jit::fuser::cuda::min, py::return_value_policy::reference);
  nvf_ops.def_static(
      "sum", &torch::jit::fuser::cuda::sum, py::return_value_policy::reference);
  nvf_ops.def_static(
      "var",
      [](TensorView* input,
         const std::vector<int>& dims,
         int64_t correction,
         bool keepdim) -> TensorView* {
        return torch::jit::fuser::cuda::variance(
            input, dims, correction, keepdim);
      },
      py::return_value_policy::reference);

  // Broadcast operations
  nvf_ops.def_static(
      "broadcast",
      &torch::jit::fuser::cuda::broadcast,
      py::return_value_policy::reference);
  // TODO: We don't have a way to realize a tensor if the operation creates
  // the output of a fusion.
  nvf_ops.def_static(
      "broadcast_in_dim",
      [](TensorView* input,
         std::vector<int>& output_shape,
         std::vector<int>& broadcast_dims) -> TensorView* {
        const auto input_ndims = input->domain()->noReductions().size();
        TORCH_CHECK(
            output_shape.size() >= input_ndims,
            "The new shape is expected to be greater-then-or-equal to the input",
            output_shape.size(),
            input_ndims);
        TORCH_CHECK(
            input_ndims == broadcast_dims.size(),
            "The broadcast dimensions should match the input dimensions.",
            input_ndims,
            broadcast_dims.size());

        std::vector<bool> is_broadcast_dim(output_shape.size(), true);
        for (const auto idx : c10::irange(broadcast_dims.size())) {
          if (idx > 0) {
            TORCH_CHECK(
                broadcast_dims[idx - 1] < broadcast_dims[idx],
                "Broadcast dimension is not greater than the previous value.");
          }
          TORCH_CHECK(
              broadcast_dims[idx] < static_cast<int>(output_shape.size()),
              "Invalid broadcast_dims value.");
          is_broadcast_dim.at(broadcast_dims[idx]) = false;
        }

        return torch::jit::fuser::cuda::broadcast(input, is_broadcast_dim);
      },
      py::return_value_policy::reference);

  // Cast Operations
  nvf_ops.def_static(
      "cast",
      py::overload_cast<torch::jit::fuser::cuda::DataType, TensorView*>(
          &torch::jit::fuser::cuda::castOp),
      py::return_value_policy::reference);
  nvf_ops.def_static(
      "cast",
      py::overload_cast<
          torch::jit::fuser::cuda::DataType,
          torch::jit::fuser::cuda::Val*>(&torch::jit::fuser::cuda::castOp),
      py::return_value_policy::reference);
*/
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
