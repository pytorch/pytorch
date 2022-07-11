#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>

#ifdef USE_CUDA
#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/ops/normalization.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <iostream>

using namespace torch::jit::fuser::cuda;

namespace {

class PythonFusionOwner {
 public:
  PythonFusionOwner() : executor_cache_(std::make_unique<Fusion>()) {}

  // Non-copyable
  PythonFusionOwner(const PythonFusionOwner&) = delete;
  PythonFusionOwner& operator=(const PythonFusionOwner&) = delete;

  std::vector<at::Tensor> execute(const at::ArrayRef<c10::IValue>& inputs) {
    return executor_cache_.runFusionWithInputs(inputs);
  }
  Fusion* fusionPtr() {
    return executor_cache_.fusion();
  }

  void printIr() {
    executor_cache_.printFusion();
  }
  void printKernel() {
    executor_cache_.fusion()->printKernel();
  }

 private:
  FusionExecutorCache executor_cache_;
};

// Manually applying the fusion guard via a context manager
class FusionDefinitionContextManager {
 public:
  FusionDefinitionContextManager(PythonFusionOwner* fusion_owner)
      : fusion_owner_(fusion_owner), prev_fusion_(nullptr) {}

  // Context Manager Methods
  FusionDefinitionContextManager* enter() {
    prev_fusion_ = FusionGuard::getCurFusion();
    FusionGuard::setCurFusion(fusionPtr());
    return this;
  }

  void exit() {
    FusionGuard::setCurFusion(prev_fusion_);
    prev_fusion_ = nullptr;
  }

  void addInput(torch::jit::fuser::cuda::Val* input) {
    fusionPtr()->addInput(input);
  }
  void addOutput(torch::jit::fuser::cuda::Val* output) {
    fusionPtr()->addOutput(output);
  }

  Fusion* fusionPtr() {
    return fusion_owner_->fusionPtr();
  }

  // An Empty namespace to add arith ops
  struct Ops {};

 private:
  PythonFusionOwner* fusion_owner_;
  Fusion* prev_fusion_;
};

} // namespace

namespace torch {
namespace jit {

void initNvFuserPythonBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto nvfuser = m.def_submodule("_nvfuser");

  // DataTypes supported by NVFuser in Fusion Definition
  // Types not related to values found in fusion defintions
  // were purposely left out.
  // NOTE: DataType was ambiguous under torch::jit without full qualification.
  py::enum_<torch::jit::fuser::cuda::DataType>(nvfuser, "DataType")
      .value("Double", torch::jit::fuser::cuda::DataType::Double)
      .value("Float", torch::jit::fuser::cuda::DataType::Float)
      .value("Half", torch::jit::fuser::cuda::DataType::Half)
      .value("Int", torch::jit::fuser::cuda::DataType::Int)
      .value("Int32", torch::jit::fuser::cuda::DataType::Int32)
      .value("Bool", torch::jit::fuser::cuda::DataType::Bool)
      .value("BFloat16", torch::jit::fuser::cuda::DataType::BFloat16)
      .value("ComplexFloat", torch::jit::fuser::cuda::DataType::ComplexFloat)
      .value("ComplexDouble", torch::jit::fuser::cuda::DataType::ComplexDouble)
      .value("Null", torch::jit::fuser::cuda::DataType::Null);

  // Binding an object that owns a FusionExecutorCache instance and provides an
  // interface
  py::class_<PythonFusionOwner> fusion(nvfuser, "Fusion");
  fusion.def(py::init<>())
      .def(
          "execute",
          [](PythonFusionOwner& self, const py::iterable& iter) {
            std::vector<IValue> inputs;
            for (py::handle obj : iter) {
              inputs.push_back(toIValue(obj, c10::AnyType::get()));
            }
            return self.execute(inputs);
          },
          py::return_value_policy::reference)
      .def("print_ir", [](PythonFusionOwner& self) { self.printIr(); })
      .def("print_kernel", [](PythonFusionOwner& self) { self.printKernel(); });

  // Bindings to Types required for Tensor/Scalar Creation
  py::class_<TensorView>(nvfuser, "TensorView")
      .def(
          "__str__",
          [](TensorView& self) -> std::string {
            std::stringstream ss;
            TORCH_CHECK(
                self.getDataType().has_value(),
                "TensorView does not have DataType?");
            ss << self.getDataType().value();
            return self.toString() + " DataType: " + ss.str() +
                " Contiguity: " + self.domain()->getContiguityString();
          },
          py::return_value_policy::reference);
  py::class_<torch::jit::fuser::cuda::Val>(nvfuser, "Val")
      .def(
          "__str__",
          [](torch::jit::fuser::cuda::Val& self) -> std::string {
            return self.toString();
          },
          py::return_value_policy::reference);

  // C++ Side of Context Manager used to mimic the FusionGuard as a way
  // to programatically distinguish code used to define the Fusion instead
  // of having the user mysteriously create an object prior to adding definition
  // code where the object is not used.
  py::class_<FusionDefinitionContextManager> fusion_def(
      nvfuser, "FusionDefinition");
  fusion_def.def(py::init<PythonFusionOwner*>())
      .def(
          "__enter__",
          [](FusionDefinitionContextManager& self) { return self.enter(); })
      .def(
          "__exit__",
          [](FusionDefinitionContextManager& self,
             void* exc_type,
             void* exc_value,
             void* traceback) { self.exit(); })
      .def(
          "add_input",
          [](FusionDefinitionContextManager& self,
             torch::jit::fuser::cuda::Val* input) { self.addInput(input); })
      .def(
          "add_input",
          [](FusionDefinitionContextManager& self, TensorView* input) {
            self.addInput(input);
          })
      .def(
          "add_output",
          [](FusionDefinitionContextManager& self,
             torch::jit::fuser::cuda::Val* output) { self.addOutput(output); })
      .def(
          "add_output",
          [](FusionDefinitionContextManager& self, TensorView* output) {
            self.addOutput(output);
          })
      .def(
          "define_tensor",
          [](FusionDefinitionContextManager& self,
             size_t ndims,
             torch::jit::fuser::cuda::DataType dtype =
                 torch::jit::fuser::cuda::DataType::Float) -> TensorView* {
            return TensorViewBuilder()
                .ndims(ndims)
                .dtype(dtype)
                .contiguity(std::vector<bool>(ndims, true))
                .build();
          },
          py::arg("ndims"),
          py::arg("dtype") = torch::jit::fuser::cuda::DataType::Float,
          py::return_value_policy::reference)
      .def(
          // TODO: Should the inernals of this function live more explicitly in
          // TensorViewBuilder?
          "define_tensor",
          [](FusionDefinitionContextManager& self,
             // TODO: This should come in as int64_t not int
             std::vector<int> sizes,
             std::vector<int> strides,
             torch::jit::fuser::cuda::DataType dtype =
                 torch::jit::fuser::cuda::DataType::Float) -> TensorView* {
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

            return TensorViewBuilder()
                .ndims(maybe_symbolic_sizes.size())
                .contiguity(contig_info)
                .shape(maybe_symbolic_sizes)
                .dtype(dtype)
                .build();
          },
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("dtype") = torch::jit::fuser::cuda::DataType::Float,
          py::return_value_policy::reference)
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
          py::return_value_policy::reference);

  py::class_<FusionDefinitionContextManager::Ops> nvf_ops(fusion_def, "Ops");

  // ******************** INSERT OP BINDINGS BELOW HERE ********************

#define NVFUSER_PYTHON_BINDING_UNARY_OP(op_str, op_name)                 \
  nvf_ops.def_static(                                                    \
      op_str,                                                            \
      py::overload_cast<TensorView*>(&torch::jit::fuser::cuda::op_name), \
      py::return_value_policy::reference);                               \
  nvf_ops.def_static(                                                    \
      op_str,                                                            \
      py::overload_cast<torch::jit::fuser::cuda::Val*>(                  \
          &torch::jit::fuser::cuda::op_name),                            \
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
