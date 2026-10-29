// Isolated pybind module for the op-option registry and its passes.
// Registers only the new op_options types; core LoopsIR types (KernelProgram,
// ExternCall, Compute) are NOT re-registered: they cast through the core
// extension's registrations, which must be loaded first. Fused regions are
// core Compute bodies built through RegionBuilder (forwarding to the core
// Index/Instruction factories) or taken from a core KernelProgram.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "op_candidates.h"

namespace py = pybind11;
using namespace torch::inductor::loops_ir;
using namespace torch::inductor::loops_ir::op_options;

namespace {

PyObject* g_option_rejected = nullptr;

struct LoopIndex {
  Index index;
  static LoopIndex var(const std::string& n) {
    return {Index::Variable(n)};
  }
  static LoopIndex constant(int64_t v) {
    return {Index::Constant(v)};
  }
  LoopIndex binary(const std::string& op, const LoopIndex& rhs) const {
    return {Index::Binary(op, index, rhs.index)};
  }
};

struct RegionBody {
  Compute compute;
};

class RegionBuilder {
 public:
  RegionBuilder(std::string name,
                std::vector<std::string> inputs,
                std::string result,
                int64_t result_elements,
                const std::vector<std::tuple<std::string, int64_t, int64_t>>& ranges,
                const std::vector<std::tuple<std::string, int64_t, int64_t>>& reduction_ranges,
                const std::vector<std::string>& accumulators)
      : compute_(std::move(name),
                 std::move(inputs),
                 {Buffer(std::move(result), "f32", result_elements)},
                 toRanges(ranges),
                 toRanges(reduction_ranges),
                 toAccumulators(accumulators)) {}

  int constant(double v) {
    compute_.addPostInstruction(Instruction::Constant(next_, v));
    return next_++;
  }
  int load(const std::string& buffer, const LoopIndex& idx) {
    compute_.addPostInstruction(Instruction::Load(next_, buffer, idx.index));
    return next_++;
  }
  int binary(const std::string& op, int lhs, int rhs) {
    compute_.addPostInstruction(Instruction::Binary(next_, op, lhs, rhs));
    return next_++;
  }
  int relu(int arg) {
    compute_.addPostInstruction(Instruction::Relu(next_, arg));
    return next_++;
  }
  void store(const std::string& buffer, const LoopIndex& idx, int value) {
    compute_.addPostInstruction(Instruction::Store(buffer, idx.index, value));
  }
  int reduction_load(const std::string& buffer, const LoopIndex& idx) {
    compute_.addReductionInstruction(Instruction::Load(next_, buffer, idx.index));
    return next_++;
  }
  void accumulate(std::vector<int> args) {
    compute_.addReductionInstruction(Instruction::Accumulate(std::move(args)));
  }
  int read_accumulator(int acc) {
    compute_.addPostInstruction(Instruction::ReadAccumulator(next_, acc));
    return next_++;
  }
  RegionBody build() const {
    return {compute_};
  }

 private:
  static std::vector<Range> toRanges(const std::vector<std::tuple<std::string, int64_t, int64_t>>& rs) {
    std::vector<Range> out;
    for (const auto& r : rs) {
      out.emplace_back(std::get<0>(r), std::get<1>(r), std::get<2>(r));
    }
    return out;
  }
  static std::vector<Accumulator> toAccumulators(const std::vector<std::string>& names) {
    std::vector<Accumulator> out;
    for (const auto& n : names) {
      out.emplace_back(n, "f32", 0.0);
    }
    return out;
  }
  Compute compute_;
  int next_ = 0;
};

void bindEnums(py::module_& m) {
  py::enum_<OpKind>(m, "OpKind")
      .value("MM", OpKind::Mm)
      .value("ADDMM", OpKind::Addmm)
      .value("BMM", OpKind::Bmm)
      .value("SUM_DIM", OpKind::SumDim)
      .value("MEAN_DIM", OpKind::MeanDim)
      .value("VAR_CORRECTION", OpKind::VarCorrection)
      .value("STD_CORRECTION", OpKind::StdCorrection)
      .value("SOFTMAX", OpKind::Softmax)
      .value("LOG_SOFTMAX", OpKind::LogSoftmax);
  py::enum_<StorageDtype>(m, "StorageDtype")
      .value("F16", StorageDtype::F16)
      .value("BF16", StorageDtype::BF16)
      .value("F32", StorageDtype::F32)
      .value("F64", StorageDtype::F64)
      .value("I32", StorageDtype::I32)
      .value("I64", StorageDtype::I64)
      .value("BOOL", StorageDtype::Bool);
  py::enum_<ShapeKind>(m, "ShapeKind").value("STATIC", ShapeKind::Static).value("DYNAMIC", ShapeKind::Dynamic);
  py::enum_<DeviceKind>(m, "DeviceKind").value("CUDA", DeviceKind::Cuda).value("CPU", DeviceKind::Cpu);
  py::enum_<NumericsPolicy>(m, "NumericsPolicy")
      .value("STRICT_EAGER", NumericsPolicy::StrictEager)
      .value("INDUCTOR", NumericsPolicy::Inductor);
  py::enum_<MatmulPrecision>(m, "MatmulPrecision")
      .value("IEEE", MatmulPrecision::Ieee)
      .value("TF32", MatmulPrecision::Tf32);
  py::enum_<RejectReason>(m, "RejectReason")
      .value("UNKNOWN_OPTION", RejectReason::UnknownOption)
      .value("OP_KIND", RejectReason::OpKind)
      .value("DEVICE", RejectReason::Device)
      .value("DTYPE", RejectReason::Dtype)
      .value("DYNAMIC_SHAPE", RejectReason::DynamicShape)
      .value("RANK", RejectReason::Rank)
      .value("LAYOUT", RejectReason::Layout)
      .value("NUMERICS_FENCE", RejectReason::NumericsFence)
      .value("CONFIG", RejectReason::Config)
      .value("FUSION", RejectReason::Fusion)
      .value("CONTEXT", RejectReason::Context)
      .value("SIGNATURE", RejectReason::Signature)
      .value("ALIAS", RejectReason::Alias);
  py::enum_<AddmmLtPolicy>(m, "AddmmLtPolicy")
      .value("CUDA_DEFAULT", AddmmLtPolicy::CudaDefault)
      .value("CUDA_DISABLED", AddmmLtPolicy::CudaDisabled)
      .value("UNKNOWN", AddmmLtPolicy::Unknown);
  py::enum_<AddmmPath>(m, "AddmmPath")
      .value("BETA_ZERO", AddmmPath::BetaZero)
      .value("EMPTY_K", AddmmPath::EmptyK)
      .value("LT_BIAS", AddmmPath::LtBias)
      .value("GEMM", AddmmPath::Gemm);
  py::enum_<AccDtype>(m, "AccDtype").value("F32", AccDtype::F32);
  py::enum_<NativeVendor>(m, "NativeVendor").value("ATEN_EAGER_DEFAULT", NativeVendor::AtenEagerDefault);
  py::enum_<ConfigKind>(m, "ConfigKind")
      .value("MATMUL", ConfigKind::Matmul)
      .value("REDUCTION", ConfigKind::Reduction)
      .value("NATIVE", ConfigKind::Native);
  py::enum_<ParamKind>(m, "ParamKind").value("INT", ParamKind::Int).value("ENUM", ParamKind::Enum);
  py::enum_<RegionOpKind>(m, "RegionOpKind")
      .value("LOAD_SELF", RegionOpKind::LoadSelf)
      .value("LOAD_EXTRA", RegionOpKind::LoadExtra)
      .value("CONSTANT", RegionOpKind::Constant)
      .value("ADD", RegionOpKind::Add)
      .value("MUL", RegionOpKind::Mul)
      .value("RELU", RegionOpKind::Relu);
  py::enum_<CodegenStrategy>(m, "CodegenStrategy")
      .value("NATIVE_ATEN", CodegenStrategy::NativeAten)
      .value("TRITON_GEMM", CodegenStrategy::TritonGemm)
      .value("TRITON_REDUCTION", CodegenStrategy::TritonReduction);
  py::enum_<NumericsFence>(m, "NumericsFence")
      .value("STRICT_EAGER", NumericsFence::StrictEager)
      .value("INDUCTOR_ONLY", NumericsFence::InductorOnly);
  py::enum_<NativeTarget>(m, "NativeTarget")
      .value("ATEN_MM", NativeTarget::AtenMm)
      .value("ATEN_ADDMM", NativeTarget::AtenAddmm)
      .value("ATEN_BMM", NativeTarget::AtenBmm)
      .value("ATEN_SUM_DIM_INTLIST", NativeTarget::AtenSumDimIntList)
      .value("ATEN_MEAN_DIM", NativeTarget::AtenMeanDim);
}

void bindSignature(py::module_& m) {
  py::class_<TensorLayout>(m, "TensorLayout")
      .def(py::init([](StorageDtype dtype, std::vector<int64_t> sizes, std::vector<int64_t> strides, ShapeKind kind) {
             return TensorLayout{dtype, std::move(sizes), std::move(strides), kind};
           }),
           py::arg("dtype"),
           py::arg("sizes"),
           py::arg("strides"),
           py::arg("shape_kind") = ShapeKind::Static)
      .def_readonly("dtype", &TensorLayout::dtype)
      .def_readonly("sizes", &TensorLayout::sizes)
      .def_readonly("strides", &TensorLayout::strides)
      .def_readonly("shape_kind", &TensorLayout::shape_kind)
      .def("numel", &TensorLayout::numel)
      .def("extent", &TensorLayout::extent)
      .def("is_contiguous", &TensorLayout::isContiguous)
      .def("is_non_overlapping", &TensorLayout::isNonOverlapping)
      .def("canonical", &TensorLayout::canonical);
  py::class_<TensorOperand>(m, "TensorOperand")
      .def(py::init([](std::string n, TensorLayout l) { return TensorOperand{std::move(n), std::move(l)}; }),
           py::arg("name"),
           py::arg("layout"))
      .def_readonly("name", &TensorOperand::name)
      .def_readonly("layout", &TensorOperand::layout);
  py::class_<MatmulAttrs>(m, "MatmulAttrs")
      .def_readonly("alpha", &MatmulAttrs::alpha)
      .def_readonly("beta", &MatmulAttrs::beta);
  py::class_<ReductionAttrs>(m, "ReductionAttrs")
      .def_readonly("dims", &ReductionAttrs::dims)
      .def_readonly("keepdim", &ReductionAttrs::keepdim)
      .def_readonly("correction", &ReductionAttrs::correction);
  py::class_<GemmDims>(m, "GemmDims")
      .def_readonly("batch", &GemmDims::batch)
      .def_readonly("m", &GemmDims::m)
      .def_readonly("n", &GemmDims::n)
      .def_readonly("k", &GemmDims::k);
  py::class_<OpSignature>(m, "OpSignature")
      .def_static("mm", &OpSignature::Mm, py::arg("device"), py::arg("a"), py::arg("b"), py::arg("out"))
      .def_static("addmm",
                  &OpSignature::Addmm,
                  py::arg("device"),
                  py::arg("bias"),
                  py::arg("a"),
                  py::arg("b"),
                  py::arg("out"),
                  py::arg("alpha"),
                  py::arg("beta"))
      .def_static("bmm", &OpSignature::Bmm, py::arg("device"), py::arg("a"), py::arg("b"), py::arg("out"))
      .def_static("reduction",
                  &OpSignature::Reduction,
                  py::arg("kind"),
                  py::arg("device"),
                  py::arg("x"),
                  py::arg("dims"),
                  py::arg("keepdim"),
                  py::arg("correction"),
                  py::arg("out"))
      .def_property_readonly("kind", &OpSignature::kind)
      .def_property_readonly("device", &OpSignature::device)
      .def_property_readonly("inputs", &OpSignature::inputs)
      .def_property_readonly("output", &OpSignature::output)
      .def_property_readonly("is_matmul", &OpSignature::isMatmul)
      .def_property_readonly("matmul", &OpSignature::matmul)
      .def_property_readonly("reduction_attrs", &OpSignature::reduction)
      .def_property_readonly("gemm_dims", &OpSignature::gemmDims)
      .def_property_readonly("gemm_operand_a", &OpSignature::gemmOperandA)
      .def_property_readonly("gemm_operand_b", &OpSignature::gemmOperandB)
      .def("canonical", &OpSignature::canonical);
  py::class_<CodegenContext>(m, "CodegenContext")
      .def(py::init([](NumericsPolicy numerics,
                       DeviceKind device,
                       int64_t cc,
                       int64_t smem,
                       MatmulPrecision stock,
                       bool fp_fusion,
                       bool ftz,
                       bool division_rounding,
                       int64_t device_index,
                       AddmmLtPolicy lt) {
             return CodegenContext{
                 numerics, device, cc, smem, stock, fp_fusion, ftz, division_rounding, device_index, lt};
           }),
           py::kw_only(),
           py::arg("numerics"),
           py::arg("device"),
           py::arg("compute_capability"),
           py::arg("max_shared_memory_bytes"),
           py::arg("stock_fp32_matmul_precision"),
           py::arg("enable_fp_fusion"),
           py::arg("disable_ftz"),
           py::arg("division_rounding"),
           py::arg("device_index") = 0,
           py::arg("addmm_lt_policy") = AddmmLtPolicy::CudaDefault)
      .def_readonly("numerics", &CodegenContext::numerics)
      .def_readonly("device", &CodegenContext::device)
      .def_readonly("compute_capability", &CodegenContext::compute_capability)
      .def_readonly("max_shared_memory_bytes", &CodegenContext::max_shared_memory_bytes)
      .def_readonly("stock_fp32_matmul_precision", &CodegenContext::stock_fp32_matmul_precision)
      .def_readonly("enable_fp_fusion", &CodegenContext::enable_fp_fusion)
      .def_readonly("disable_ftz", &CodegenContext::disable_ftz)
      .def_readonly("division_rounding", &CodegenContext::division_rounding)
      .def_readonly("device_index", &CodegenContext::device_index)
      .def_readonly("addmm_lt_policy", &CodegenContext::addmm_lt_policy)
      .def("validate", &CodegenContext::validate)
      .def("canonical", &CodegenContext::canonical);
  m.def("stock_allows_tf32", &stockAllowsTf32, py::arg("signature"), py::arg("context"));
}

void bindConfigs(py::module_& m) {
  py::class_<MatmulConfig>(m, "MatmulConfig")
      .def(py::init([](int64_t bm, int64_t bn, int64_t bk, int64_t gm, int64_t w, int64_t s, int64_t sk,
                       MatmulPrecision p, AccDtype acc) { return MatmulConfig{bm, bn, bk, gm, w, s, sk, p, acc}; }),
           py::kw_only(),
           py::arg("block_m"),
           py::arg("block_n"),
           py::arg("block_k"),
           py::arg("group_m"),
           py::arg("num_warps"),
           py::arg("num_stages"),
           py::arg("split_k") = 1,
           py::arg("matrix_precision") = MatmulPrecision::Ieee,
           py::arg("acc_dtype") = AccDtype::F32)
      .def_readonly("block_m", &MatmulConfig::block_m)
      .def_readonly("block_n", &MatmulConfig::block_n)
      .def_readonly("block_k", &MatmulConfig::block_k)
      .def_readonly("group_m", &MatmulConfig::group_m)
      .def_readonly("num_warps", &MatmulConfig::num_warps)
      .def_readonly("num_stages", &MatmulConfig::num_stages)
      .def_readonly("split_k", &MatmulConfig::split_k)
      .def_readonly("matrix_precision", &MatmulConfig::matrix_precision)
      .def_readonly("acc_dtype", &MatmulConfig::acc_dtype)
      .def("canonical", &MatmulConfig::canonical)
      .def("__eq__", &MatmulConfig::operator==)
      .def("__repr__", &MatmulConfig::canonical);
  py::class_<ReductionConfig>(m, "ReductionConfig")
      .def(py::init([](int64_t bx, int64_t br, int64_t w, int64_t s, AccDtype acc) {
             return ReductionConfig{bx, br, w, s, acc};
           }),
           py::kw_only(),
           py::arg("block_x"),
           py::arg("block_r"),
           py::arg("num_warps"),
           py::arg("num_stages") = 1,
           py::arg("acc_dtype") = AccDtype::F32)
      .def_readonly("block_x", &ReductionConfig::block_x)
      .def_readonly("block_r", &ReductionConfig::block_r)
      .def_readonly("num_warps", &ReductionConfig::num_warps)
      .def_readonly("num_stages", &ReductionConfig::num_stages)
      .def_readonly("acc_dtype", &ReductionConfig::acc_dtype)
      .def("canonical", &ReductionConfig::canonical)
      .def("__eq__", &ReductionConfig::operator==)
      .def("__repr__", &ReductionConfig::canonical);
  py::class_<NativeConfig>(m, "NativeConfig")
      .def(py::init([](NativeVendor v) { return NativeConfig{v}; }),
           py::kw_only(),
           py::arg("vendor") = NativeVendor::AtenEagerDefault)
      .def_readonly("vendor", &NativeConfig::vendor)
      .def("canonical", &NativeConfig::canonical)
      .def("__eq__", &NativeConfig::operator==)
      .def("__repr__", &NativeConfig::canonical);
  py::class_<ParamSpec>(m, "ParamSpec")
      .def_readonly("name", &ParamSpec::name)
      .def_readonly("kind", &ParamSpec::kind)
      .def_readonly("legal_ints", &ParamSpec::legal_ints)
      .def_readonly("legal_enums", &ParamSpec::legal_enums)
      .def_readonly("default_int", &ParamSpec::default_int)
      .def_readonly("default_enum", &ParamSpec::default_enum)
      .def_readonly("tunable", &ParamSpec::tunable)
      .def_readonly("meaning", &ParamSpec::meaning);
  py::class_<ConfigSchema>(m, "ConfigSchema")
      .def_readonly("kind", &ConfigSchema::kind)
      .def_readonly("params", &ConfigSchema::params)
      .def_readonly("constraints", &ConfigSchema::constraints)
      .def("param", &ConfigSchema::param, py::return_value_policy::reference_internal);
  m.def("config_schema", &configSchema, py::return_value_policy::reference);
  m.def("validate_config_domain", &validateDomain);
}

void bindFusion(py::module_& m) {
  py::class_<LoopIndex>(m, "LoopIndex")
      .def_static("var", &LoopIndex::var)
      .def_static("const", &LoopIndex::constant)
      .def("binary", &LoopIndex::binary)
      .def("__add__", [](const LoopIndex& a, const LoopIndex& b) { return a.binary("add", b); })
      .def("__mul__", [](const LoopIndex& a, const LoopIndex& b) { return a.binary("mul", b); })
      .def("__floordiv__", [](const LoopIndex& a, const LoopIndex& b) { return a.binary("floordiv", b); })
      .def("__mod__", [](const LoopIndex& a, const LoopIndex& b) { return a.binary("mod", b); })
      .def("__repr__", [](const LoopIndex& a) { return a.index.toString(); });
  py::class_<RegionBody>(m, "RegionBody").def_property_readonly("name", [](const RegionBody& b) {
    return b.compute.name();
  });
  py::class_<RegionBuilder>(m, "RegionBuilder")
      .def(py::init<std::string,
                    std::vector<std::string>,
                    std::string,
                    int64_t,
                    const std::vector<std::tuple<std::string, int64_t, int64_t>>&,
                    const std::vector<std::tuple<std::string, int64_t, int64_t>>&,
                    const std::vector<std::string>&>(),
           py::kw_only(),
           py::arg("name"),
           py::arg("inputs"),
           py::arg("result"),
           py::arg("result_elements"),
           py::arg("ranges"),
           py::arg("reduction_ranges") = std::vector<std::tuple<std::string, int64_t, int64_t>>{},
           py::arg("accumulators") = std::vector<std::string>{})
      .def("constant", &RegionBuilder::constant)
      .def("load", &RegionBuilder::load)
      .def("binary", &RegionBuilder::binary)
      .def("relu", &RegionBuilder::relu)
      .def("store", &RegionBuilder::store)
      .def("reduction_load", &RegionBuilder::reduction_load)
      .def("accumulate", &RegionBuilder::accumulate)
      .def("read_accumulator", &RegionBuilder::read_accumulator)
      .def("build", &RegionBuilder::build);
  py::class_<ExtraInput>(m, "ExtraInput").def_readonly("name", &ExtraInput::name).def_readonly("layout", &ExtraInput::layout);
  py::class_<FusedLoops>(m, "FusedLoops")
      .def(py::init<>())
      .def(
          "add_prologue",
          [](FusedLoops& f, int64_t operand, const RegionBody& b, int64_t store_offset) {
            f.prologues.push_back(PrologueRegion{operand, b.compute, store_offset});
          },
          py::arg("operand"),
          py::arg("body"),
          py::arg("store_offset") = 0)
      .def(
          "set_epilogue",
          [](FusedLoops& f,
             const std::string& op_result,
             const RegionBody& b,
             std::optional<TensorLayout> op_result_layout,
             int64_t op_result_offset,
             int64_t store_offset) {
            f.epilogue = EpilogueRegion{op_result, b.compute, std::move(op_result_layout), op_result_offset, store_offset};
          },
          py::arg("op_result"),
          py::arg("body"),
          py::arg("op_result_layout") = std::nullopt,
          py::arg("op_result_offset") = 0,
          py::arg("store_offset") = 0)
      .def_property_readonly("prologue_operands",
                             [](const FusedLoops& f) {
                               std::vector<int64_t> out;
                               for (const auto& p : f.prologues) {
                                 out.push_back(p.operand);
                               }
                               return out;
                             })
      .def_property_readonly("prologue_bodies",
                             [](const FusedLoops& f) {
                               std::vector<Compute> out;
                               for (const auto& p : f.prologues) {
                                 out.push_back(p.body);
                               }
                               return out;
                             })
      .def_property_readonly("has_epilogue", [](const FusedLoops& f) { return f.epilogue.has_value(); })
      .def_property_readonly("epilogue_op_result",
                             [](const FusedLoops& f) {
                               return f.epilogue ? std::optional<std::string>(f.epilogue->op_result) : std::nullopt;
                             })
      .def_property_readonly("epilogue_body",
                             [](const FusedLoops& f) {
                               return f.epilogue ? std::optional<Compute>(f.epilogue->body) : std::nullopt;
                             })
      .def("add_extra_input",
           [](FusedLoops& f, const std::string& name, const TensorLayout& l) {
             f.extra_inputs.push_back(ExtraInput{name, l});
           })
      .def_readonly("extra_inputs", &FusedLoops::extra_inputs);
  py::class_<AccessMap>(m, "AccessMap").def_readonly("offset", &AccessMap::offset).def_readonly("coeffs", &AccessMap::coeffs);
  py::class_<RegionOp>(m, "RegionOp")
      .def_readonly("kind", &RegionOp::kind)
      .def_readonly("id", &RegionOp::id)
      .def_readonly("lhs", &RegionOp::lhs)
      .def_readonly("rhs", &RegionOp::rhs)
      .def_readonly("extra", &RegionOp::extra)
      .def_readonly("access", &RegionOp::access)
      .def_readonly("value", &RegionOp::value);
  py::class_<LoweredRegion>(m, "LoweredRegion")
      .def_readonly("domain", &LoweredRegion::domain)
      .def_readonly("ops", &LoweredRegion::ops)
      .def_readonly("store_value", &LoweredRegion::store_value)
      .def_readonly("store_offset", &LoweredRegion::store_offset)
      .def_readonly("reads_self", &LoweredRegion::reads_self)
      .def("canonical", &LoweredRegion::canonical);
  py::class_<LoweredPrologue>(m, "LoweredPrologue")
      .def_readonly("operand", &LoweredPrologue::operand)
      .def_readonly("region", &LoweredPrologue::region);
  py::class_<ValidatedFusion>(m, "ValidatedFusion")
      .def_readonly("prologues", &ValidatedFusion::prologues)
      .def_readonly("epilogue", &ValidatedFusion::epilogue)
      .def_readonly("extra_inputs", &ValidatedFusion::extra_inputs)
      .def("empty", &ValidatedFusion::empty)
      .def("prologue_for", &ValidatedFusion::prologueFor, py::return_value_policy::reference_internal)
      .def("canonical", &ValidatedFusion::canonical);
  py::class_<FusionContract>(m, "FusionContract")
      .def_readonly("prologue_operands", &FusionContract::prologue_operands)
      .def_readonly("epilogue", &FusionContract::epilogue)
      .def_readonly("allowed_ops", &FusionContract::allowed_ops)
      .def_readonly("max_extra_inputs", &FusionContract::max_extra_inputs)
      .def_readonly("allow_broadcast", &FusionContract::allow_broadcast)
      .def("canonical", &FusionContract::canonical);
}

void bindRegistry(py::module_& m) {
  py::class_<NativeBiasView>(m, "NativeBiasView")
      .def_readonly("extra", &NativeBiasView::extra)
      .def_readonly("direct", &NativeBiasView::direct)
      .def_readonly("sizes", &NativeBiasView::sizes)
      .def_readonly("strides", &NativeBiasView::strides)
      .def_readonly("offset", &NativeBiasView::offset);
  py::class_<NativeCallPlan>(m, "NativeCallPlan")
      .def_readonly("target", &NativeCallPlan::target)
      .def_readonly("operand_slots", &NativeCallPlan::operand_slots)
      .def_readonly("bias_view", &NativeCallPlan::bias_view)
      .def_readonly("alpha", &NativeCallPlan::alpha)
      .def_readonly("beta", &NativeCallPlan::beta)
      .def_readonly("dims", &NativeCallPlan::dims)
      .def_readonly("keepdim", &NativeCallPlan::keepdim)
      .def("canonical", &NativeCallPlan::canonical);
  m.def("native_target_name", &nativeTargetName);
  py::class_<AddmmEagerPlan>(m, "AddmmEagerPlan")
      .def_readonly("path", &AddmmEagerPlan::path)
      .def_readonly("read_operands", &AddmmEagerPlan::read_operands)
      .def_readonly("scale_product", &AddmmEagerPlan::scale_product)
      .def_readonly("load_bias", &AddmmEagerPlan::load_bias)
      .def_readonly("scale_bias", &AddmmEagerPlan::scale_bias)
      .def_readonly("add_positive_zero", &AddmmEagerPlan::add_positive_zero)
      .def("canonical", &AddmmEagerPlan::canonical);
  m.def("addmm_eager_plan", &addmmEagerPlan, py::arg("signature"), py::arg("context"));
  py::class_<ValidatedSelection>(m, "ValidatedSelection")
      .def_readonly("option_id", &ValidatedSelection::option_id)
      .def_readonly("option_version", &ValidatedSelection::option_version)
      .def_readonly("strategy", &ValidatedSelection::strategy)
      .def_readonly("fence", &ValidatedSelection::fence)
      .def_readonly("signature", &ValidatedSelection::signature)
      .def_readonly("config", &ValidatedSelection::config)
      .def_readonly("fusion", &ValidatedSelection::fusion)
      .def_readonly("context", &ValidatedSelection::context)
      .def_readonly("native", &ValidatedSelection::native)
      .def_readonly("addmm", &ValidatedSelection::addmm)
      .def_readonly("canonical", &ValidatedSelection::canonical)
      .def_readonly("cache_key", &ValidatedSelection::cache_key);
  py::class_<OptionDescriptor>(m, "OptionDescriptor")
      .def_readonly("id", &OptionDescriptor::id)
      .def_readonly("version", &OptionDescriptor::version)
      .def_readonly("op_kinds", &OptionDescriptor::op_kinds)
      .def_readonly("fence", &OptionDescriptor::fence)
      .def_readonly("strategy", &OptionDescriptor::strategy)
      .def_readonly("config_kind", &OptionDescriptor::config_kind)
      .def_readonly("effects", &OptionDescriptor::effects)
      .def("fusion_for", &OptionDescriptor::fusionFor);
  py::class_<OptionRegistry>(m, "OptionRegistry")
      .def("option_ids", &OptionRegistry::optionIds)
      .def("options_for", &OptionRegistry::optionsFor)
      .def("get", &OptionRegistry::get, py::return_value_policy::reference_internal)
      .def("eligible_options", &OptionRegistry::eligibleOptions)
      .def("candidate_configs", &OptionRegistry::candidateConfigs)
      .def("validate",
           &OptionRegistry::validate,
           py::arg("option_id"),
           py::arg("signature"),
           py::arg("config"),
           py::arg("fused_loops"),
           py::arg("context"));
  m.def("default_registry", &defaultRegistry, py::return_value_policy::reference);
}

void bindRejection(py::module_& m) {
  static py::exception<OptionRejected> exc(m, "OptionRejected", PyExc_ValueError);
  g_option_rejected = exc.ptr();
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const OptionRejected& e) {
      py::object type = py::reinterpret_borrow<py::object>(g_option_rejected);
      py::object inst = type(py::str(e.what()));
      inst.attr("reason") = py::cast(e.reason());
      PyErr_SetObject(g_option_rejected, inst.ptr());
    }
  });
}

void bindPassEnums(py::module_& m) {
  py::enum_<DecodeStatus>(m, "DecodeStatus")
      .value("DECODED", DecodeStatus::Decoded)
      .value("UNKNOWN_TARGET", DecodeStatus::UnknownTarget)
      .value("UNSUPPORTED", DecodeStatus::Unsupported)
      .value("MALFORMED_ABI", DecodeStatus::MalformedAbi);
  py::enum_<FusionVetoKind>(m, "FusionVetoKind")
      .value("NOT_POINTWISE_KERNEL", FusionVetoKind::NotPointwiseKernel)
      .value("MULTIPLE_REGIONS", FusionVetoKind::MultipleRegions)
      .value("REDUCTION_REGION", FusionVetoKind::ReductionRegion)
      .value("VIEW_READ", FusionVetoKind::ViewRead)
      .value("UNUSED_INPUT", FusionVetoKind::UnusedInput)
      .value("MULTIPLE_USES", FusionVetoKind::MultipleUses)
      .value("ESCAPES_OUTPUT", FusionVetoKind::EscapesOutput)
      .value("CROSSES_EXTERN", FusionVetoKind::CrossesExtern)
      .value("OPERAND_REPEATED", FusionVetoKind::OperandRepeated)
      .value("DOMAIN_MISMATCH", FusionVetoKind::DomainMismatch)
      .value("NON_AFFINE_STORE", FusionVetoKind::NonAffineStore)
      .value("NO_OPTION_CONTRACT", FusionVetoKind::NoOptionContract);
  py::enum_<CandidateKind>(m, "CandidateKind")
      .value("ORIGINAL_EXTERN", CandidateKind::OriginalExtern)
      .value("NATIVE_OPTION", CandidateKind::NativeOption)
      .value("GENERATED_OPTION", CandidateKind::GeneratedOption);
}

void bindDecode(py::module_& m) {
  py::class_<TensorBinding>(m, "TensorBinding")
      .def_readonly("ir_name", &TensorBinding::ir_name)
      .def_readonly("root", &TensorBinding::root)
      .def_readonly("root_offset", &TensorBinding::root_offset)
      .def_readonly("root_elements", &TensorBinding::root_elements)
      .def_readonly("node_result", &TensorBinding::node_result);
  py::class_<DecodedOp>(m, "DecodedOp")
      .def_readonly("node_index", &DecodedOp::node_index)
      .def_readonly("status", &DecodedOp::status)
      .def_readonly("reason", &DecodedOp::reason)
      .def_readonly("detail", &DecodedOp::detail)
      .def_readonly("signature", &DecodedOp::signature)
      .def_readonly("input_bindings", &DecodedOp::input_bindings)
      .def_readonly("output_binding", &DecodedOp::output_binding)
      .def_readonly("fallback", &DecodedOp::fallback);
  py::class_<DecodedOps>(m, "DecodedOps")
      .def_readonly("numerics", &DecodedOps::numerics)
      .def_readonly("ops", &DecodedOps::ops);
  m.def("decode_op_signatures", &decodeOpSignatures, py::arg("program"), py::arg("device"));
}

void bindRegions(py::module_& m) {
  py::class_<FusionVeto>(m, "FusionVeto")
      .def_readonly("kind", &FusionVeto::kind)
      .def_readonly("role", &FusionVeto::role)
      .def_readonly("node", &FusionVeto::node)
      .def_readonly("detail", &FusionVeto::detail);
  py::class_<OptionContractResult>(m, "OptionContractResult")
      .def_readonly("option_id", &OptionContractResult::option_id)
      .def_readonly("accepted", &OptionContractResult::accepted)
      .def_readonly("reason", &OptionContractResult::reason)
      .def_readonly("detail", &OptionContractResult::detail);
  py::class_<FusionVariant>(m, "FusionVariant")
      .def_readonly("prologue_operands", &FusionVariant::prologue_operands)
      .def_readonly("epilogue", &FusionVariant::epilogue)
      .def_readonly("signature", &FusionVariant::signature)
      .def_readonly("fused_loops", &FusionVariant::fused_loops)
      .def_readonly("input_bindings", &FusionVariant::input_bindings)
      .def_readonly("extra_bindings", &FusionVariant::extra_bindings)
      .def_readonly("output_binding", &FusionVariant::output_binding)
      .def_readonly("consumed_nodes", &FusionVariant::consumed_nodes)
      .def_readonly("placement", &FusionVariant::placement)
      .def_readonly("produces", &FusionVariant::produces)
      .def_readonly("eliminated_buffers", &FusionVariant::eliminated_buffers)
      .def_readonly("contracts", &FusionVariant::contracts)
      .def("accepted_by", &FusionVariant::acceptedBy);
  py::class_<OpRegionSet>(m, "OpRegionSet")
      .def_readonly("decoded_index", &OpRegionSet::decoded_index)
      .def_readonly("node_index", &OpRegionSet::node_index)
      .def_readonly("variants", &OpRegionSet::variants)
      .def_readonly("vetoes", &OpRegionSet::vetoes);
  py::class_<FusedOpRegions>(m, "FusedOpRegions")
      .def_readonly("ops", &FusedOpRegions::ops)
      .def_readonly("shared_nodes", &FusedOpRegions::shared_nodes);
  m.def("form_fused_op_regions",
        &formFusedOpRegions,
        py::arg("program"),
        py::arg("decoded"),
        py::arg("registry"));
}

void bindCandidates(py::module_& m) {
  py::class_<Candidate>(m, "Candidate")
      .def_readonly("kind", &Candidate::kind)
      .def_readonly("variant", &Candidate::variant)
      .def_readonly("option_id", &Candidate::option_id)
      .def_readonly("option_version", &Candidate::option_version)
      .def_readonly("config", &Candidate::config)
      .def_readonly("cache_key", &Candidate::cache_key)
      .def_readonly("baseline", &Candidate::baseline);
  py::class_<CandidateRejection>(m, "CandidateRejection")
      .def_readonly("variant", &CandidateRejection::variant)
      .def_readonly("option_id", &CandidateRejection::option_id)
      .def_readonly("reason", &CandidateRejection::reason)
      .def_readonly("detail", &CandidateRejection::detail);
  py::class_<OpCandidates>(m, "OpCandidates")
      .def_readonly("decoded_index", &OpCandidates::decoded_index)
      .def_readonly("node_index", &OpCandidates::node_index)
      .def_readonly("fallback", &OpCandidates::fallback)
      .def_readonly("variants", &OpCandidates::variants)
      .def_readonly("candidates", &OpCandidates::candidates)
      .def_readonly("rejections", &OpCandidates::rejections);
  py::class_<CandidatePlan>(m, "CandidatePlan")
      .def_readonly("numerics", &CandidatePlan::numerics)
      .def_readonly("context", &CandidatePlan::context)
      .def_readonly("ops", &CandidatePlan::ops)
      .def_readonly("shared_nodes", &CandidatePlan::shared_nodes);
  m.def("enumerate_op_candidates",
        &enumerateOpCandidates,
        py::arg("regions"),
        py::arg("decoded"),
        py::arg("registry"),
        py::arg("context"));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "LoopsIR per-operation implementation-option registry (isolated module)";
  bindEnums(m);
  bindSignature(m);
  bindConfigs(m);
  bindFusion(m);
  bindRegistry(m);
  bindPassEnums(m);
  bindDecode(m);
  bindRegions(m);
  bindCandidates(m);
  bindRejection(m);
}
