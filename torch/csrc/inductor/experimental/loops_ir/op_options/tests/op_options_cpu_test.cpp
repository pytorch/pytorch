// Standalone CPU test for the op-option registry (no torch/pybind). Built
// from the op_options directory with g++ -std=c++17 over ../ir.cpp, the six
// option units and this file; see test/inductor/test_loops_ir_op_options.py.

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <set>
#include <string>

#include "../option_registry.h"

using namespace torch::inductor::loops_ir;
using namespace torch::inductor::loops_ir::op_options;

namespace {

int g_checks = 0;

void check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    std::fprintf(stderr, "FAILED: %s\n", what.c_str());
    std::exit(1);
  }
}

void expectReject(RejectReason reason, const std::function<void()>& fn, const std::string& what) {
  ++g_checks;
  try {
    fn();
  } catch (const OptionRejected& e) {
    if (e.reason() != reason) {
      std::fprintf(stderr, "FAILED: %s: wrong reason: %s\n", what.c_str(), e.what());
      std::exit(1);
    }
    return;
  }
  std::fprintf(stderr, "FAILED: %s: not rejected\n", what.c_str());
  std::exit(1);
}

TensorOperand rowMajor(const std::string& name, std::vector<int64_t> sizes, StorageDtype dt = StorageDtype::F32) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (size_t d = sizes.size(); d-- > 1;) {
    strides[d - 1] = strides[d] * std::max<int64_t>(sizes[d], 1);
  }
  return TensorOperand{name, TensorLayout{dt, sizes, strides, ShapeKind::Static}};
}

CodegenContext cudaCtx(NumericsPolicy p) {
  CodegenContext c;
  c.numerics = p;
  c.device = DeviceKind::Cuda;
  c.compute_capability = 90;
  c.max_shared_memory_bytes = 232448;
  c.enable_fp_fusion = p == NumericsPolicy::Inductor;
  c.disable_ftz = p == NumericsPolicy::StrictEager;
  return c;
}

OpSignature mmSig(int64_t m, int64_t n, int64_t k, StorageDtype dt = StorageDtype::F32) {
  return OpSignature::Mm(
      DeviceKind::Cuda, rowMajor("a", {m, k}, dt), rowMajor("b", {k, n}, dt), rowMajor("out", {m, n}, dt));
}

Index iv(const std::string& n) {
  return Index::Variable(n);
}
Index ic(int64_t v) {
  return Index::Constant(v);
}
Index add(const Index& a, const Index& b) {
  return Index::Binary("add", a, b);
}
Index mul(const Index& a, const Index& b) {
  return Index::Binary("mul", a, b);
}

// out[i,j] = relu(acc[i,j] + bias[j]) * 2
Compute biasReluEpilogue(int64_t m, int64_t n, Index accIndex, bool withRelu = true) {
  Compute c("epi", {"acc", "bias"}, {Buffer("fused", "f32", m * n)}, {Range("i", 0, m), Range("j", 0, n)}, {}, {});
  const Index outIdx = add(mul(iv("i"), ic(n)), iv("j"));
  c.addPostInstruction(Instruction::Load(0, "acc", accIndex));
  c.addPostInstruction(Instruction::Load(1, "bias", iv("j")));
  c.addPostInstruction(Instruction::Binary(2, "add", 0, 1));
  int last = 2;
  if (withRelu) {
    c.addPostInstruction(Instruction::Relu(3, 2));
    c.addPostInstruction(Instruction::Constant(4, 2.0));
    c.addPostInstruction(Instruction::Binary(5, "mul", 3, 4));
    last = 5;
  }
  c.addPostInstruction(Instruction::Store("fused", outIdx, last));
  return c;
}

FusedLoops epilogueLoops(Compute body, int64_t n) {
  FusedLoops f;
  f.epilogue = EpilogueRegion{"acc", std::move(body), std::nullopt, 0, 0};
  f.extra_inputs.push_back(ExtraInput{"bias", rowMajor("bias", {n}).layout});
  return f;
}

void testRegistryListing() {
  const auto& reg = defaultRegistry();
  const std::vector<std::string> expected{"native_aten_gemm",
                                          "native_aten_mm_bias_epilogue",
                                          "triton_tiled_gemm",
                                          "native_aten_reduction",
                                          "triton_row_reduction"};
  check(reg.optionIds() == expected, "registry lists exactly the implemented options");
  for (OpKind k : {OpKind::VarCorrection, OpKind::StdCorrection, OpKind::Softmax, OpKind::LogSoftmax}) {
    check(reg.optionsFor(k).empty(), "reserved op kinds have no option");
  }
  expectReject(RejectReason::UnknownOption, [&] { reg.get("triton_template_mm"); }, "unknown option id");
  const auto sig = mmSig(64, 64, 64);
  check(reg.eligibleOptions(sig, cudaCtx(NumericsPolicy::StrictEager)) ==
            std::vector<std::string>{"native_aten_gemm"},
        "strict mm: native only");
  check(reg.eligibleOptions(sig, cudaCtx(NumericsPolicy::Inductor)).size() == 3, "inductor mm: three options");
}

void testSignatures() {
  expectReject(
      RejectReason::Signature,
      [] {
        OpSignature::Mm(DeviceKind::Cuda, rowMajor("a", {4, 5}), rowMajor("b", {6, 7}), rowMajor("out", {4, 7}));
      },
      "mm K mismatch");
  expectReject(
      RejectReason::Rank,
      [] { OpSignature::Mm(DeviceKind::Cuda, rowMajor("a", {2, 4, 5}), rowMajor("b", {5, 7}), rowMajor("o", {4, 7})); },
      "mm rank");
  expectReject(
      RejectReason::Signature,
      [] {
        OpSignature::Addmm(DeviceKind::Cuda,
                           rowMajor("bias", {3}),
                           rowMajor("a", {4, 5}),
                           rowMajor("b", {5, 7}),
                           rowMajor("out", {4, 7}),
                           1.0,
                           1.0);
      },
      "addmm bias not broadcastable");
  const auto s = OpSignature::Reduction(
      OpKind::SumDim, DeviceKind::Cuda, rowMajor("x", {3, 5}), {-1}, true, std::nullopt, rowMajor("out", {3, 1}));
  check(s.reduction().dims == std::vector<int64_t>{1}, "reduction dims normalized");
  expectReject(
      RejectReason::Signature,
      [] {
        OpSignature::Reduction(
            OpKind::VarCorrection, DeviceKind::Cuda, rowMajor("x", {3, 5}), {1}, false, std::nullopt, rowMajor("o", {3}));
      },
      "var requires correction");
  const auto addmm = OpSignature::Addmm(DeviceKind::Cuda,
                                        rowMajor("bias", {7}),
                                        rowMajor("a", {4, 5}),
                                        rowMajor("b", {5, 7}),
                                        rowMajor("out", {4, 7}),
                                        0.1,
                                        -0.0);
  const auto addmm2 = OpSignature::Addmm(DeviceKind::Cuda,
                                         rowMajor("bias", {7}),
                                         rowMajor("a", {4, 5}),
                                         rowMajor("b", {5, 7}),
                                         rowMajor("out", {4, 7}),
                                         0.1,
                                         0.0);
  check(addmm.canonical() != addmm2.canonical(), "beta -0.0 vs 0.0 bits preserved");
}

void testConfigValidation() {
  const auto& reg = defaultRegistry();
  const auto sig = mmSig(128, 128, 128);
  const auto ctx = cudaCtx(NumericsPolicy::Inductor);
  MatmulConfig ok;
  reg.validate("triton_tiled_gemm", sig, ok, FusedLoops{}, ctx);
  auto bad = [&](const std::function<void(MatmulConfig&)>& edit, RejectReason why, const std::string& what) {
    MatmulConfig c;
    edit(c);
    expectReject(why, [&] { reg.validate("triton_tiled_gemm", sig, c, FusedLoops{}, ctx); }, what);
  };
  bad([](MatmulConfig& c) { c.block_m = 24; }, RejectReason::Config, "block_m outside domain");
  bad([](MatmulConfig& c) { c.num_warps = 3; }, RejectReason::Config, "num_warps outside domain");
  bad([](MatmulConfig& c) { c.split_k = 2; }, RejectReason::Config, "split_k > 1");
  bad(
      [](MatmulConfig& c) {
        c.block_m = c.block_n = c.block_k = 256;
        c.num_stages = 5;
      },
      RejectReason::Config,
      "shared memory overflow");
  bad([](MatmulConfig& c) { c.matrix_precision = MatmulPrecision::Tf32; }, RejectReason::Config, "tf32 not allowed");
  expectReject(
      RejectReason::Config,
      [&] { reg.validate("triton_tiled_gemm", sig, ReductionConfig{}, FusedLoops{}, ctx); },
      "wrong config kind");
  expectReject(
      RejectReason::NumericsFence,
      [&] { reg.validate("triton_tiled_gemm", sig, ok, FusedLoops{}, cudaCtx(NumericsPolicy::StrictEager)); },
      "strict fence on generated gemm");
  expectReject(
      RejectReason::Dtype,
      [&] { reg.validate("triton_tiled_gemm", mmSig(8, 8, 8, StorageDtype::BF16), ok, FusedLoops{}, ctx); },
      "bf16 rejected");
  auto dyn = rowMajor("a", {8, 8});
  dyn.layout.shape_kind = ShapeKind::Dynamic;
  const auto dynSig = OpSignature::Mm(DeviceKind::Cuda, dyn, rowMajor("b", {8, 8}), rowMajor("out", {8, 8}));
  expectReject(
      RejectReason::DynamicShape,
      [&] { reg.validate("triton_tiled_gemm", dynSig, ok, FusedLoops{}, ctx); },
      "dynamic rejected");
  TensorOperand overlapped{"out", TensorLayout{StorageDtype::F32, {8, 8}, {1, 1}, ShapeKind::Static}};
  const auto ovSig = OpSignature::Mm(DeviceKind::Cuda, rowMajor("a", {8, 8}), rowMajor("b", {8, 8}), overlapped);
  expectReject(
      RejectReason::Layout,
      [&] { reg.validate("triton_tiled_gemm", ovSig, ok, FusedLoops{}, ctx); },
      "overlapping output rejected");
  auto tf32Ctx = ctx;
  tf32Ctx.stock_fp32_matmul_precision = MatmulPrecision::Tf32;
  MatmulConfig tf32;
  tf32.matrix_precision = MatmulPrecision::Tf32;
  reg.validate("triton_tiled_gemm", mmSig(64, 512, 512), tf32, FusedLoops{}, tf32Ctx);
  expectReject(
      RejectReason::Config,
      [&] { reg.validate("triton_tiled_gemm", mmSig(64, 512, 256), tf32, FusedLoops{}, tf32Ctx); },
      "tf32 below stock size threshold");
}

void testCandidates() {
  const auto& reg = defaultRegistry();
  const auto ctx = cudaCtx(NumericsPolicy::Inductor);
  const auto a = reg.candidateConfigs("triton_tiled_gemm", mmSig(1000, 1000, 1000), ctx);
  const auto b = reg.candidateConfigs("triton_tiled_gemm", mmSig(1000, 1000, 1000), ctx);
  check(!a.empty() && a.size() <= 21, "gemm candidates bounded");
  bool same = a.size() == b.size();
  for (size_t i = 0; same && i < a.size(); ++i) {
    same = canonicalConfig(a[i]) == canonicalConfig(b[i]);
  }
  check(same, "gemm candidates deterministic");
  const auto small = reg.candidateConfigs("triton_tiled_gemm", mmSig(5, 7, 3), ctx);
  for (const auto& c : small) {
    const auto& m = std::get<MatmulConfig>(c);
    check(m.block_m == 16 && m.block_n == 16 && m.block_k == 16, "small shape clamps blocks to 16");
  }
  const auto rsig = OpSignature::Reduction(
      OpKind::SumDim, DeviceKind::Cuda, rowMajor("x", {64, 5000}), {1}, false, std::nullopt, rowMajor("out", {64}));
  const auto rc = reg.candidateConfigs("triton_row_reduction", rsig, ctx);
  check(!rc.empty() && rc.size() <= 20, "reduction candidates bounded");
  for (const auto& c : rc) {
    check(std::get<ReductionConfig>(c).block_r < 5000, "R=5000 has only looped candidates");
  }
  expectReject(
      RejectReason::NumericsFence,
      [&] { reg.candidateConfigs("triton_row_reduction", rsig, cudaCtx(NumericsPolicy::StrictEager)); },
      "no generated candidates under strict");
  check(reg.candidateConfigs("native_aten_reduction", rsig, cudaCtx(NumericsPolicy::StrictEager)).size() == 1,
        "native reduction has one config under strict");
}

void testFusion() {
  const auto& reg = defaultRegistry();
  const auto ctx = cudaCtx(NumericsPolicy::Inductor);
  const int64_t m = 8, n = 16;
  const auto sig = mmSig(m, n, 4);
  const Index canon = add(mul(iv("i"), ic(n)), iv("j"));
  const auto sel = reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, epilogueLoops(biasReluEpilogue(m, n, canon), n), ctx);
  check(sel.fusion.epilogue && sel.fusion.epilogue->ops.size() == 6, "epilogue lowered");
  const auto& biasLoad = sel.fusion.epilogue->ops[1];
  check(biasLoad.kind == RegionOpKind::LoadExtra && biasLoad.access.coeffs == std::vector<int64_t>{0, 1},
        "bias broadcast map");

  expectReject(
      RejectReason::Fusion,
      [&] {
        const Index transposed = add(mul(iv("j"), ic(m)), iv("i"));
        reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, epilogueLoops(biasReluEpilogue(m, n, transposed), n), ctx);
      },
      "acc at non-identity map");
  expectReject(
      RejectReason::Fusion,
      [&] {
        const Index nonAffine = Index::Binary("add", Index::Binary("floordiv", canon, ic(1)), ic(0));
        reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, epilogueLoops(biasReluEpilogue(m, n, nonAffine), n), ctx);
      },
      "floordiv index rejected");
  expectReject(
      RejectReason::Fusion,
      [&] {
        reg.validate("native_aten_gemm", sig, NativeConfig{}, epilogueLoops(biasReluEpilogue(m, n, canon), n), ctx);
      },
      "native gemm has no epilogue contract");
  expectReject(
      RejectReason::Fusion,
      [&] {
        reg.validate(
            "native_aten_mm_bias_epilogue", sig, NativeConfig{}, epilogueLoops(biasReluEpilogue(m, n, canon), n), ctx);
      },
      "native addmm epilogue rejects relu");
  const auto nat = reg.validate(
      "native_aten_mm_bias_epilogue", sig, NativeConfig{}, epilogueLoops(biasReluEpilogue(m, n, canon, false), n), ctx);
  check(nat.native && nat.native->target == NativeTarget::AtenAddmm && nat.native->bias_view &&
            nat.native->bias_view->direct && nat.native->bias_view->strides == std::vector<int64_t>{0, 1},
        "native bias epilogue -> addmm bias view");
  expectReject(
      RejectReason::Fusion,
      [&] {
        FusedLoops f = epilogueLoops(biasReluEpilogue(m, n, canon), n);
        f.extra_inputs[0].layout = rowMajor("bias", {n - 1}).layout;
        reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, f, ctx);
      },
      "out-of-bounds extra load rejected by core verifier");
  expectReject(
      RejectReason::Dtype,
      [&] {
        FusedLoops f = epilogueLoops(biasReluEpilogue(m, n, canon), n);
        f.extra_inputs[0].layout.dtype = StorageDtype::BF16;
        reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, f, ctx);
      },
      "bf16 extra input rejected");
  expectReject(
      RejectReason::Fusion,
      [&] {
        Compute red("epi", {"acc"}, {Buffer("fused", "f32", m * n)}, {Range("i", 0, m), Range("j", 0, n)},
                    {Range("r", 0, 2)}, {Accumulator("s", "f32", 0.0)});
        red.addReductionInstruction(Instruction::Load(0, "acc", canon));
        red.addReductionInstruction(Instruction::Accumulate({0}));
        red.addPostInstruction(Instruction::ReadAccumulator(1, 0));
        red.addPostInstruction(Instruction::Store("fused", canon, 1));
        FusedLoops f;
        f.epilogue = EpilogueRegion{"acc", red, std::nullopt, 0, 0};
        reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, f, ctx);
      },
      "reduction inside fused region rejected");
  expectReject(
      RejectReason::Fusion,
      [&] {
        FusedLoops f;
        Compute pro("pro", {"bias"}, {Buffer("p", "f32", n)}, {Range("j", 0, n)}, {}, {});
        pro.addPostInstruction(Instruction::Load(0, "bias", iv("j")));
        pro.addPostInstruction(Instruction::Store("p", iv("j"), 0));
        f.prologues.push_back(PrologueRegion{2, pro});
        reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, f, ctx);
      },
      "prologue on a non-operand slot");
}

void testCacheKeys() {
  const auto& reg = defaultRegistry();
  const auto ctx = cudaCtx(NumericsPolicy::Inductor);
  const auto sig = mmSig(8, 16, 4);
  const Index canon = add(mul(iv("i"), ic(16)), iv("j"));
  MatmulConfig c1;
  MatmulConfig c2;
  c2.block_n = 32;
  const auto k1 = reg.validate("triton_tiled_gemm", sig, c1, FusedLoops{}, ctx).cache_key;
  check(k1 == reg.validate("triton_tiled_gemm", sig, c1, FusedLoops{}, ctx).cache_key, "cache key deterministic");
  check(k1 != reg.validate("triton_tiled_gemm", sig, c2, FusedLoops{}, ctx).cache_key, "cache key tracks config");
  const auto kf = reg.validate("triton_tiled_gemm", sig, c1, epilogueLoops(biasReluEpilogue(8, 16, canon), 16), ctx);
  const auto kg =
      reg.validate("triton_tiled_gemm", sig, c1, epilogueLoops(biasReluEpilogue(8, 16, canon, false), 16), ctx);
  check(k1 != kf.cache_key && kf.cache_key != kg.cache_key, "cache key tracks fused loops");
}

void testReductionEligibility() {
  const auto& reg = defaultRegistry();
  const auto ctx = cudaCtx(NumericsPolicy::Inductor);
  const auto multi = OpSignature::Reduction(
      OpKind::MeanDim, DeviceKind::Cuda, rowMajor("x", {3, 4, 5}), {0, 2}, false, std::nullopt, rowMajor("out", {4}));
  expectReject(
      RejectReason::Rank,
      [&] { reg.validate("triton_row_reduction", multi, ReductionConfig{}, FusedLoops{}, ctx); },
      "multi-dim generated reduction");
  reg.validate("native_aten_reduction", multi, NativeConfig{}, FusedLoops{}, cudaCtx(NumericsPolicy::StrictEager));
  TensorOperand perm{"x", TensorLayout{StorageDtype::F32, {3, 4, 5}, {1, 3, 12}, ShapeKind::Static}};
  const auto permSig = OpSignature::Reduction(
      OpKind::SumDim, DeviceKind::Cuda, perm, {2}, false, std::nullopt, rowMajor("out", {3, 4}));
  expectReject(
      RejectReason::Layout,
      [&] { reg.validate("native_aten_reduction", permSig, NativeConfig{}, FusedLoops{}, ctx); },
      "native reduction with unpredictable eager layout");
  ReductionConfig big;
  big.block_x = 128;
  big.block_r = 4096;
  expectReject(
      RejectReason::Config,
      [&] { reg.validate("triton_row_reduction", permSig, big, FusedLoops{}, ctx); },
      "reduction tile budget");
}

void testEnumAndContextValidation() {
  const auto& reg = defaultRegistry();
  const auto sig = mmSig(64, 64, 64);
  const auto ctx = cudaCtx(NumericsPolicy::Inductor);
  MatmulConfig badPrec;
  badPrec.matrix_precision = static_cast<MatmulPrecision>(7);
  expectReject(RejectReason::Config, [&] { reg.validate("triton_tiled_gemm", sig, badPrec, FusedLoops{}, ctx); },
               "invalid matrix_precision enum");
  MatmulConfig badAcc;
  badAcc.acc_dtype = static_cast<AccDtype>(3);
  expectReject(RejectReason::Config, [&] { validateDomain(badAcc); }, "invalid matmul acc_dtype enum");
  ReductionConfig badRed;
  badRed.acc_dtype = static_cast<AccDtype>(2);
  expectReject(RejectReason::Config, [&] { validateDomain(badRed); }, "invalid reduction acc_dtype enum");
  NativeConfig badVendor;
  badVendor.vendor = static_cast<NativeVendor>(5);
  expectReject(RejectReason::Config, [&] { validateDomain(badVendor); }, "invalid native vendor enum");
  check(MatmulConfig{}.canonical().find("acc_dtype=f32") != std::string::npos, "canonical uses enum names");
  auto badDev = ctx;
  badDev.device = static_cast<DeviceKind>(9);
  expectReject(RejectReason::Context, [&] { badDev.validate(); }, "invalid context device enum");
  auto badNum = ctx;
  badNum.numerics = static_cast<NumericsPolicy>(4);
  expectReject(RejectReason::Context, [&] { badNum.validate(); }, "invalid context numerics enum");
  auto badIdx = ctx;
  badIdx.device_index = -1;
  expectReject(RejectReason::Context, [&] { badIdx.validate(); }, "negative device index");
  expectReject(
      RejectReason::Device,
      [] { OpSignature::Mm(static_cast<DeviceKind>(3), rowMajor("a", {2, 2}), rowMajor("b", {2, 2}), rowMajor("o", {2, 2})); },
      "invalid signature device enum");
  expectReject(
      RejectReason::Signature,
      [] { mmSig(2, 2, 2, static_cast<StorageDtype>(42)); },
      "invalid dtype enum");
  auto dev1 = ctx;
  dev1.device_index = 1;
  auto noLt = ctx;
  noLt.addmm_lt_policy = AddmmLtPolicy::CudaDisabled;
  const auto k0 = reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, FusedLoops{}, ctx).cache_key;
  check(k0 != reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, FusedLoops{}, dev1).cache_key,
        "cache key tracks device_index");
  check(k0 != reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, FusedLoops{}, noLt).cache_key,
        "cache key tracks the addmm vendor policy");
}

OpSignature addmmSig(TensorOperand bias, int64_t m, int64_t n, int64_t k, double alpha, double beta) {
  return OpSignature::Addmm(DeviceKind::Cuda, bias, rowMajor("a", {m, k}), rowMajor("b", {k, n}),
                            rowMajor("out", {m, n}), alpha, beta);
}

void testAddmmEagerPlan() {
  const auto ctx = cudaCtx(NumericsPolicy::Inductor);
  auto plan = [&](TensorOperand bias, int64_t k, double alpha, double beta, CodegenContext c) {
    return addmmEagerPlan(addmmSig(bias, 33, 70, k, alpha, beta), c);
  };
  const auto row = rowMajor("bias", {70});
  const auto full = rowMajor("bias", {33, 70});
  auto p = plan(row, 45, 0.0, 1.0, ctx);
  check(p.path == AddmmPath::LtBias && p.read_operands && p.scale_product && p.load_bias && !p.scale_bias,
        "1-D bias beta=1: Lt bias epilogue reads A/B even at alpha=0");
  p = plan(rowMajor("bias", {1, 70}), 45, -0.0, 1.0, ctx);
  check(p.path == AddmmPath::LtBias && p.read_operands && p.scale_product, "[1,N] bias squeezes to Lt");
  p = plan(full, 45, 0.0, 1.0, ctx);
  check(p.path == AddmmPath::Gemm && !p.read_operands && !p.scale_bias && !p.add_positive_zero,
        "2-D bias alpha=0 beta=1: exact bias, A/B unread");
  p = plan(full, 45, 0.0, 2.5, ctx);
  check(p.path == AddmmPath::Gemm && !p.read_operands && p.scale_bias && p.add_positive_zero,
        "alpha=0 beta=2.5: beta*bias + 0");
  p = plan(rowMajor("bias", {33, 1}), 45, 0.0, 1.0, ctx);
  check(p.path == AddmmPath::Gemm && !p.read_operands, "column bias is not Lt");
  TensorOperand strided{"bias", TensorLayout{StorageDtype::F32, {70}, {2}, ShapeKind::Static}};
  check(plan(strided, 45, 0.0, 1.0, ctx).path == AddmmPath::Gemm, "noncontiguous 1-D bias is not Lt");
  check(plan(row, 1, 0.0, 1.0, ctx).path == AddmmPath::Gemm, "K=1 excluded from Lt");
  auto noLt = ctx;
  noLt.addmm_lt_policy = AddmmLtPolicy::CudaDisabled;
  check(!plan(row, 45, 0.0, 1.0, noLt).read_operands, "DISABLE_ADDMM_CUDA_LT: gemm path skips A/B");
  for (double beta : {0.0, -0.0}) {
    p = plan(row, 45, 2.0, beta, ctx);
    check(p.path == AddmmPath::BetaZero && p.read_operands && p.scale_product && !p.load_bias, "beta=+-0 ignores bias");
    check(!plan(row, 45, 0.0, beta, ctx).read_operands, "alpha=0 beta=0: zeros");
  }
  p = plan(full, 0, 1.0, 3.0, ctx);
  check(p.path == AddmmPath::EmptyK && !p.read_operands && p.scale_bias, "K=0: beta*bias");
  auto unknown = ctx;
  unknown.addmm_lt_policy = AddmmLtPolicy::Unknown;
  expectReject(RejectReason::Context,
               [&] { defaultRegistry().validate("triton_tiled_gemm", addmmSig(rowMajor("bias", {8}), 8, 8, 8, 1.0, 1.0), MatmulConfig{},
                                                 FusedLoops{}, unknown); },
               "generated addmm needs a modelled vendor policy");
  defaultRegistry().validate("triton_tiled_gemm", mmSig(8, 8, 8), MatmulConfig{}, FusedLoops{}, unknown);
  const auto sel = defaultRegistry().validate("triton_tiled_gemm", addmmSig(rowMajor("bias", {8, 8}), 8, 8, 8, 0.0, 1.0), MatmulConfig{},
                                              FusedLoops{}, ctx);
  check(sel.addmm && !sel.addmm->read_operands && sel.canonical.find("addmm=gemm") != std::string::npos,
        "plan is part of the validated selection key");
}

void testOffsetsAndTunables() {
  const auto& reg = defaultRegistry();
  const auto ctx = cudaCtx(NumericsPolicy::Inductor);
  const int64_t m = 8, n = 16;
  const auto sig = mmSig(m, n, 4);
  FusedLoops f = epilogueLoops(biasReluEpilogue(m, n, add(mul(iv("i"), ic(n)), iv("j"))), n);
  f.epilogue->store_offset = -1;
  expectReject(RejectReason::Fusion, [&] { reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, f, ctx); },
               "negative store offset");
  f.epilogue->store_offset = 0;
  f.epilogue->op_result_offset = -2;
  expectReject(RejectReason::Fusion, [&] { reg.validate("triton_tiled_gemm", sig, MatmulConfig{}, f, ctx); },
               "negative op_result offset");
  const ConfigSchema& schema = configSchema(ConfigKind::Matmul);
  const auto cands = reg.candidateConfigs("triton_tiled_gemm", mmSig(2048, 2048, 2048), ctx);
  auto distinct = [&](int64_t MatmulConfig::*member) {
    std::set<int64_t> vals;
    for (const auto& c : cands) {
      vals.insert(std::get<MatmulConfig>(c).*member);
    }
    return vals.size();
  };
  const std::vector<std::pair<const char*, int64_t MatmulConfig::*>> fields{
      {"block_m", &MatmulConfig::block_m}, {"block_n", &MatmulConfig::block_n},
      {"block_k", &MatmulConfig::block_k}, {"group_m", &MatmulConfig::group_m},
      {"num_warps", &MatmulConfig::num_warps}, {"num_stages", &MatmulConfig::num_stages},
      {"split_k", &MatmulConfig::split_k}};
  for (const auto& [name, member] : fields) {
    check(schema.param(name).tunable == (distinct(member) > 1),
          std::string("schema tunable flag is faithful to the manifest: ") + name);
  }
}

} // namespace

int main() {
  testRegistryListing();
  testSignatures();
  testConfigValidation();
  testCandidates();
  testFusion();
  testCacheKeys();
  testReductionEligibility();
  testEnumAndContextValidation();
  testAddmmEagerPlan();
  testOffsetsAndTunables();
  std::printf("op_options_cpu_test: %d checks passed\n", g_checks);
  return 0;
}
