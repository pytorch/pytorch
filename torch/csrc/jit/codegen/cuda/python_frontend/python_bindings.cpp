#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>

#ifdef USE_CUDA
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
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
    FusionGuard::setCurFusion(fusion_owner_->fusionPtr());
    return this;
  }

  void exit() {
    FusionGuard::setCurFusion(prev_fusion_);
    prev_fusion_ = nullptr;
  }

  void addInput(torch::jit::fuser::cuda::Val* input) {
    fusion_owner_->fusionPtr()->addInput(input);
  }
  void addOutput(torch::jit::fuser::cuda::Val* output) {
    fusion_owner_->fusionPtr()->addOutput(output);
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
      .value("ComplexDouble", torch::jit::fuser::cuda::DataType::ComplexDouble);

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
  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<TensorView>(nvfuser, "TensorView");
  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<torch::jit::fuser::cuda::Val>(nvfuser, "Val");

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
      .def_static(
          "define_tensor",
          [](size_t ndims,
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
      .def_static(
          "define_constant",
          [](double val) -> torch::jit::fuser::cuda::Val* {
            return IrBuilder::create<Double>(val);
          },
          py::return_value_policy::reference)
      .def_static(
          "define_constant",
          [](bool val) -> torch::jit::fuser::cuda::Val* {
            return IrBuilder::create<Bool>(val);
          },
          py::return_value_policy::reference)
      .def_static(
          "define_constant",
          [](int64_t val) -> torch::jit::fuser::cuda::Val* {
            return IrBuilder::create<Int>(val);
          },
          py::return_value_policy::reference)
      .def_static(
          "define_scalar",
          [](torch::jit::fuser::cuda::DataType dtype =
                 torch::jit::fuser::cuda::DataType::Double)
              -> torch::jit::fuser::cuda::Val* {
            if (dtype == torch::jit::fuser::cuda::DataType::Double) {
              return IrBuilder::create<Double>();
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
  NVFUSER_PYTHON_BINDING_UNARY_OP("not_op", notOp)
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
  NVFUSER_PYTHON_BINDING_BINARY_OP("and_op", andOp)
  NVFUSER_PYTHON_BINDING_BINARY_OP("div", div)
  NVFUSER_PYTHON_BINDING_BINARY_OP("eq", eq)
  NVFUSER_PYTHON_BINDING_BINARY_OP("fmod", fmod)
  NVFUSER_PYTHON_BINDING_BINARY_OP("ge", ge)
  NVFUSER_PYTHON_BINDING_BINARY_OP("gt", gt)
  NVFUSER_PYTHON_BINDING_BINARY_OP("le", le)
  NVFUSER_PYTHON_BINDING_BINARY_OP("lt", lt)
  NVFUSER_PYTHON_BINDING_BINARY_OP("lshift", lshift)
  NVFUSER_PYTHON_BINDING_BINARY_OP("mod", mod)
  NVFUSER_PYTHON_BINDING_BINARY_OP("mul", mul)
  NVFUSER_PYTHON_BINDING_BINARY_OP("ne", ne)
  NVFUSER_PYTHON_BINDING_BINARY_OP("or_op", orOp)
  NVFUSER_PYTHON_BINDING_BINARY_OP("pow", pow)
  NVFUSER_PYTHON_BINDING_BINARY_OP("remainder", remainder)
  NVFUSER_PYTHON_BINDING_BINARY_OP("rshift", rshift)
  NVFUSER_PYTHON_BINDING_BINARY_OP("sub", sub)
  NVFUSER_PYTHON_BINDING_BINARY_OP("xor_op", xorOp)
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

  // Broadcast operation
  nvf_ops.def_static(
      "broadcast",
      &torch::jit::fuser::cuda::broadcast,
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
