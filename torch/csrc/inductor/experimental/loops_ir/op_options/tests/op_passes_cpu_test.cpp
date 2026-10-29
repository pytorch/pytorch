// Standalone CPU test for the decode / fused-region / candidate passes.
// Builds KernelPrograms through inline core constructors only; links
// ../ir.cpp plus the op_options units (see test_loops_ir_op_options.py).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

#include "../op_candidates.h"

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

std::vector<int64_t> contiguous(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> st(sizes.size(), 1);
  for (size_t d = sizes.size(); d-- > 1;) {
    st[d - 1] = st[d] * std::max<int64_t>(sizes[d], 1);
  }
  return st;
}

int64_t numel(const std::vector<int64_t>& sizes) {
  int64_t n = 1;
  for (int64_t s : sizes) {
    n *= s;
  }
  return n;
}

CallArg T(const std::string& n) {
  return CallArg::MakeTensor(n);
}

struct Builder {
  KernelProgram p{"prog"};

  void input(const std::string& n, int64_t elems, const std::string& dt = "f32") {
    p.addInput(Buffer(n, dt, elems));
  }
  // Structured extern with one result; descriptors argK for each distinct
  // tensor argument (given explicitly) plus "result".
  void extern_(const std::string& name,
               const std::string& target,
               std::vector<CallArg> args,
               std::vector<CallKeyword> kwargs,
               const std::vector<Descriptor>& argDescs,
               const std::vector<int64_t>& outSizes,
               const std::string& dt = "f32") {
    std::vector<Descriptor> descs = argDescs;
    descs.emplace_back("result", name, outSizes, contiguous(outSizes), 0);
    p.addExternCall(ExternCall::Structured(
        name, target, std::move(args), std::move(kwargs), {Buffer(name, dt, numel(outSizes))}, descs));
  }
  void kernel(const Compute& c) {
    p.addFunction(KernelFunction("k_" + c.name(), {c}));
  }
  void output(const std::string& buffer, const std::vector<int64_t>& sizes) {
    p.addOutput(Descriptor("out" + std::to_string(p.outputs().size()), buffer, sizes, contiguous(sizes), 0));
  }
};

Descriptor D(const std::string& name, const std::string& buf, const std::vector<int64_t>& sizes,
             std::vector<int64_t> strides = {}, int64_t off = 0) {
  return Descriptor(name, buf, sizes, strides.empty() ? contiguous(sizes) : strides, off);
}

Index V(const std::string& n) {
  return Index::Variable(n);
}
Index C(int64_t v) {
  return Index::Constant(v);
}
Index lin(const std::vector<std::string>& vars, const std::vector<int64_t>& coeffs, int64_t off = 0) {
  Index e = C(off);
  for (size_t i = 0; i < vars.size(); ++i) {
    e = Index::Binary("add", e, Index::Binary("mul", V(vars[i]), C(coeffs[i])));
  }
  return e;
}

// out[d...] = relu(src[map] (+ extra[extraMap])) ; contiguous store.
Compute pointwise(const std::string& name,
                  const std::string& src,
                  const std::vector<int64_t>& sizes,
                  const std::vector<int64_t>& srcCoeffs,
                  const std::string& extra = "",
                  const std::vector<int64_t>& extraCoeffs = {},
                  bool relu = true) {
  std::vector<std::string> vars;
  std::vector<Range> ranges;
  for (size_t d = 0; d < sizes.size(); ++d) {
    vars.push_back("d" + std::to_string(d));
    ranges.emplace_back(vars.back(), 0, sizes[d]);
  }
  std::vector<std::string> ins{src};
  if (!extra.empty()) {
    ins.push_back(extra);
  }
  Compute c(name, ins, {Buffer(name, "f32", numel(sizes))}, ranges, {}, {});
  c.addPostInstruction(Instruction::Load(0, src, lin(vars, srcCoeffs)));
  int v = 0;
  if (!extra.empty()) {
    c.addPostInstruction(Instruction::Load(1, extra, lin(vars, extraCoeffs)));
    c.addPostInstruction(Instruction::Binary(2, "add", 0, 1));
    v = 2;
  }
  if (relu) {
    c.addPostInstruction(Instruction::Relu(3, v));
    v = 3;
  }
  c.addPostInstruction(Instruction::Store(name, lin(vars, contiguous(sizes)), v));
  return c;
}

CodegenContext ctx(NumericsPolicy p) {
  CodegenContext c;
  c.numerics = p;
  c.compute_capability = 90;
  c.max_shared_memory_bytes = 232448;
  c.enable_fp_fusion = p == NumericsPolicy::Inductor;
  c.disable_ftz = p == NumericsPolicy::StrictEager;
  return c;
}

std::string bits(double v) {
  return f64Bits(v);
}

template <typename Plan>
const auto& first(const Plan& p, int line) {
  check(!p.ops.empty(), "line " + std::to_string(line) + ": no fused/candidate op entries");
  return p.ops.front();
}
#define FIRST(x) first((x), __LINE__)

DecodedOp only(const DecodedOps& d) {
  check(d.ops.size() == 1, "exactly one extern decoded");
  return d.ops.front();
}

DecodedOps decode(const Builder& b) {
  return decodeOpSignatures(b.p, DeviceKind::Cuda);
}

Builder mmProgram(bool kwargs) {
  Builder b;
  b.input("a", 20);
  b.input("bb", 35);
  std::vector<CallArg> args;
  std::vector<CallKeyword> kw;
  if (kwargs) {
    kw = {CallKeyword("mat2", T("bb")), CallKeyword("self", T("a"))};
  } else {
    args = {T("a"), T("bb")};
  }
  b.extern_("mm", "aten.mm.default", args, kw, {D("arg0", "a", {4, 5}), D("arg1", "bb", {5, 7})}, {4, 7});
  return b;
}

void testDecodeArgsKwargs() {
  const DecodedOps pa = decode(mmProgram(false));
  const DecodedOps pk = decode(mmProgram(true));
  check(only(pa).status == DecodeStatus::Decoded && only(pk).status == DecodeStatus::Decoded, "mm args/kwargs decode");
  check(only(pa).signature->canonical() == only(pk).signature->canonical(), "args and kwargs give the same signature");
  check(only(pa).input_bindings[1].root == "bb" && only(pa).output_binding->root == "mm", "bindings");
}

void testDecodeAddmmScalars() {
  Builder b;
  b.input("c", 7);
  b.input("a", 20);
  b.input("w", 35);
  b.extern_("addmm",
            "aten.addmm.default",
            {T("c"), T("a"), T("w")},
            {CallKeyword("beta", CallArg::MakeFloating(-0.0)), CallKeyword("alpha", CallArg::MakeInteger(3))},
            {D("arg0", "c", {7}), D("arg1", "a", {4, 5}), D("arg2", "w", {5, 7})},
            {4, 7});
  const DecodedOp& op = only(decode(b));
  check(op.status == DecodeStatus::Decoded, "addmm decodes: " + op.detail);
  check(bits(op.signature->matmul().beta) == bits(-0.0) && op.signature->matmul().alpha == 3.0,
        "beta -0.0 bits and integer alpha preserved");
  Builder bad;
  bad.input("c", 7);
  bad.input("a", 20);
  bad.input("w", 35);
  bad.extern_("addmm",
              "aten.addmm.default",
              {T("c"), T("a"), T("w"), CallArg::MakeFloating(2.0)},
              {},
              {D("arg0", "c", {7}), D("arg1", "a", {4, 5}), D("arg2", "w", {5, 7})},
              {4, 7});
  check(only(decode(bad)).status == DecodeStatus::MalformedAbi, "keyword-only beta given positionally is malformed");
}

Builder reduction(const std::string& target, std::vector<CallArg> args, std::vector<CallKeyword> kw,
                  const std::vector<int64_t>& outSizes) {
  Builder b;
  b.input("x", 60);
  b.extern_("red", target, std::move(args), std::move(kw), {D("arg0", "x", {3, 4, 5})}, outSizes);
  return b;
}

void testDecodeReductions() {
  auto list = [](std::vector<int64_t> v) {
    std::vector<CallArg> items;
    for (int64_t x : v) {
      items.push_back(CallArg::MakeInteger(x));
    }
    return CallArg::MakeList(items);
  };
  const DecodedOp a = only(decode(reduction(
      "aten.sum.dim_IntList", {T("x"), list({-1}), CallArg::MakeBoolean(true)},
      {CallKeyword("dtype", CallArg::MakeNone())}, {3, 4, 1})));
  check(a.status == DecodeStatus::Decoded && a.signature->reduction().dims == std::vector<int64_t>{2} &&
            a.signature->reduction().keepdim,
        "sum dim=-1 keepdim dtype=None");
  const DecodedOp m = only(decode(reduction("aten.mean.dim", {T("x"), CallArg::MakeInteger(1)}, {}, {3, 5})));
  check(m.status == DecodeStatus::Decoded && m.signature->kind() == OpKind::MeanDim, "mean with int dim");
  const DecodedOp all = only(decode(reduction("aten.sum.dim_IntList", {T("x"), CallArg::MakeNone()}, {}, {})));
  check(all.status == DecodeStatus::Decoded && all.signature->reduction().dims.size() == 3, "dim=None reduces all");
  const DecodedOp empty = only(decode(reduction("aten.sum.dim_IntList", {T("x"), list({})}, {}, {})));
  check(empty.status == DecodeStatus::Unsupported, "dim=[] unsupported");
  const DecodedOp dt = only(decode(reduction(
      "aten.sum.dim_IntList", {T("x"), list({0})}, {CallKeyword("dtype", CallArg::MakeDtype("float32"))}, {4, 5})));
  check(dt.status == DecodeStatus::Unsupported && dt.reason == RejectReason::Dtype, "explicit dtype unsupported");
  const DecodedOp wrongOut = only(decode(reduction("aten.sum.dim_IntList", {T("x"), list({0})}, {}, {4, 4})));
  check(wrongOut.status == DecodeStatus::MalformedAbi, "result descriptor disagreeing with the op is malformed");
}

void testDecodeLayoutsAndBatches() {
  Builder b;
  b.input("base", 64);
  b.input("w", 35);
  b.input("c", 7);
  // v2 = view(view(base, off 3), off 2): transposed [4,5] operand at root offset 5 (+1 descriptor offset).
  b.p.addView(View("v1", "base", {60}, {1}, 3));
  b.p.addView(View("v2", "v1", {4, 5}, {1, 4}, 2));
  b.extern_("addmm", "aten.addmm.default", {T("c"), T("v2"), T("w")}, {},
            {D("arg0", "c", {4, 7}, {0, 1}), D("arg1", "v2", {4, 5}, {1, 4}, 1), D("arg2", "w", {5, 7})}, {4, 7});
  const DecodedOp op = only(decode(b));
  check(op.status == DecodeStatus::Decoded, "broadcast/view addmm decodes");
  check(op.input_bindings[1].root == "base" && op.input_bindings[1].root_offset == 6, "view chain offsets compose");
  check(op.signature->inputs()[0].layout.strides == std::vector<int64_t>{0, 1}, "broadcast bias strides kept");
  Builder bm;
  bm.input("a", 3 * 4 * 5);
  bm.input("w", 3 * 5 * 2);
  bm.extern_("bmm", "aten.bmm.default", {T("a"), T("w")}, {},
             {D("arg0", "a", {3, 4, 5}), D("arg1", "w", {3, 5, 2}, {10, 1, 5})}, {3, 4, 2});
  check(only(decode(bm)).signature->gemmDims().batch == 3, "bmm batch decoded");
  Builder z;
  z.input("a", 0);
  z.input("w", 0);
  z.extern_("mm", "aten.mm.default", {T("a"), T("w")}, {}, {D("arg0", "a", {4, 0}), D("arg1", "w", {0, 7})}, {4, 7});
  check(only(decode(z)).status == DecodeStatus::Decoded, "K=0 mm decodes");
}

void testDecodeRejections() {
  Builder u;
  u.input("a", 4);
  u.extern_("r", "aten.relu.default", {T("a")}, {}, {D("arg0", "a", {4})}, {4});
  check(only(decode(u)).status == DecodeStatus::UnknownTarget && !only(decode(u)).signature, "unknown target untouched");
  Builder bf;
  bf.input("a", 20, "bf16");
  bf.input("w", 35, "bf16");
  bf.extern_("mm", "aten.mm.default", {T("a"), T("w")}, {}, {D("arg0", "a", {4, 5}), D("arg1", "w", {5, 7})}, {4, 7}, "bf16");
  const DecodedOp bfOp = only(decode(bf));
  check(bfOp.status == DecodeStatus::Unsupported && bfOp.reason == RejectReason::Dtype && bfOp.signature,
        "bf16 mm: typed unsupported");
  Builder missing;
  missing.input("a", 20);
  missing.input("w", 35);
  missing.extern_("mm", "aten.mm.default", {T("a"), T("w")}, {}, {D("arg0", "a", {4, 5})}, {4, 7});
  check(only(decode(missing)).status == DecodeStatus::MalformedAbi, "missing descriptor is malformed");
  Builder bogus;
  bogus.input("a", 20);
  bogus.input("w", 35);
  bogus.extern_("mm", "aten.mm.default", {T("a")}, {CallKeyword("other", T("w"))},
                {D("arg0", "a", {4, 5}), D("arg1", "w", {5, 7})}, {4, 7});
  check(only(decode(bogus)).status == DecodeStatus::MalformedAbi, "unknown kwarg is malformed");
  auto legacy = [](std::vector<double> scalars) {
    Builder l;
    l.input("a", 20);
    l.input("w", 35);
    l.p.addExternCall(ExternCall("mm", "aten.mm.default", {"a", "w"}, {Buffer("mm", "f32", 28)}, scalars, "unknown",
                                 false, {D("lhs", "a", {4, 5}), D("rhs", "w", {5, 7}), D("result", "mm", {4, 7})}));
    return only(decode(l)).status;
  };
  check(legacy({}) == DecodeStatus::Decoded, "legacy typed positional mm decodes (descriptors matched by buffer)");
  check(legacy({1.0}) == DecodeStatus::MalformedAbi, "stray scalar argument fails closed");
}

// mm(a, w) -> epi(relu(mm + bias[j])) -> output
Builder mmEpilogue(bool relu, bool outputMm = false) {
  Builder b;
  b.input("a", 20);
  b.input("w", 35);
  b.input("bias", 7);
  b.extern_("mm", "aten.mm.default", {T("a"), T("w")}, {}, {D("arg0", "a", {4, 5}), D("arg1", "w", {5, 7})}, {4, 7});
  b.kernel(pointwise("epi", "mm", {4, 7}, {7, 1}, "bias", {0, 1}, relu));
  b.output("epi", {4, 7});
  if (outputMm) {
    b.p.addView(View("mm_t", "mm", {7, 4}, {1, 7}, 0));
    b.output("mm_t", {7, 4});
  }
  return b;
}

FusedOpRegions regions(const Builder& b, const DecodedOps& d) {
  return formFusedOpRegions(b.p, d, defaultRegistry());
}

bool hasVeto(const OpRegionSet& s, FusionVetoKind k) {
  for (const auto& v : s.vetoes) {
    if (v.kind == k) {
      return true;
    }
  }
  return false;
}

void testEpilogueRegions() {
  Builder b = mmEpilogue(true);
  const auto d = decode(b);
  const auto r = regions(b, d);
  check(r.ops.size() == 1 && FIRST(r).variants.size() == 2, "base + epilogue variants");
  const FusionVariant& v = FIRST(r).variants[1];
  check(v.epilogue && v.produces == "epi" && v.consumed_nodes == std::vector<size_t>{0, 1} && v.placement == 1,
        "epilogue consumes mm and epi, placed at epi");
  check(v.eliminated_buffers == std::vector<std::string>{"mm"} && v.signature.output().name == "epi",
        "mm result eliminated, fused output replaces epi");
  check(v.acceptedBy("triton_tiled_gemm") && !v.acceptedBy("native_aten_mm_bias_epilogue") &&
            !v.acceptedBy("native_aten_gemm"),
        "relu epilogue: only generated gemm contract");
  check(v.extra_bindings.size() == 1 && v.extra_bindings[0].root == "bias", "bias extra binding");
  Builder plain = mmEpilogue(false);
  const auto rp = regions(plain, decode(plain));
  check(FIRST(rp).variants[1].acceptedBy("native_aten_mm_bias_epilogue"), "acc + bias accepted by native addmm");
  Builder esc = mmEpilogue(true, true);
  const auto re = regions(esc, decode(esc));
  check(FIRST(re).variants.size() == 1 && hasVeto(FIRST(re), FusionVetoKind::EscapesOutput),
        "result aliased by an output view vetoes the epilogue");
  Builder multi = mmEpilogue(true);
  multi.kernel(pointwise("other", "mm", {4, 7}, {7, 1}));
  multi.output("other", {4, 7});
  const auto rm = regions(multi, decode(multi));
  check(FIRST(rm).variants.size() == 1 && hasVeto(FIRST(rm), FusionVetoKind::MultipleUses), "second user vetoes");
  Builder viewUse = mmEpilogue(true);
  viewUse.p.addView(View("mm_v", "mm", {28}, {1}, 0));
  viewUse.extern_("cl", "aten.clone.default", {T("mm_v")}, {}, {D("arg0", "mm_v", {28})}, {28});
  viewUse.output("cl", {28});
  check(hasVeto(FIRST(regions(viewUse, decode(viewUse))), FusionVetoKind::MultipleUses), "user through any alias counts");
  Builder transposed;
  transposed.input("a", 20);
  transposed.input("w", 35);
  transposed.extern_("mm", "aten.mm.default", {T("a"), T("w")}, {},
                     {D("arg0", "a", {4, 5}), D("arg1", "w", {5, 7})}, {4, 7});
  transposed.kernel(pointwise("epi", "mm", {7, 4}, {1, 7}));
  transposed.output("epi", {7, 4});
  check(hasVeto(FIRST(regions(transposed, decode(transposed))), FusionVetoKind::DomainMismatch),
        "permuted consumer domain vetoed");
}

// pro = relu(x) ; [barrier] ; mm(pro, w)
Builder mmPrologue(bool barrier, bool proIsOutput = false) {
  Builder b;
  b.input("x", 20);
  b.input("w", 35);
  b.kernel(pointwise("pro", "x", {4, 5}, {5, 1}));
  if (barrier) {
    b.input("s", 4);
    b.extern_("rng", "aten.rand_like.default", {T("s")}, {}, {D("arg0", "s", {4})}, {4});
    b.output("rng", {4});
  }
  b.extern_("mm", "aten.mm.default", {T("pro"), T("w")}, {}, {D("arg0", "pro", {4, 5}), D("arg1", "w", {5, 7})},
            {4, 7});
  b.output("mm", {4, 7});
  if (proIsOutput) {
    b.output("pro", {4, 5});
  }
  return b;
}

void testPrologueRegions() {
  Builder b = mmPrologue(false);
  const auto r = regions(b, decode(b));
  check(FIRST(r).variants.size() == 2, "base + prologue variants");
  const FusionVariant& v = FIRST(r).variants[1];
  check(v.prologue_operands == std::vector<int64_t>{0} && !v.input_bindings[0] && v.input_bindings[1],
        "prologue operand has no kernel binding");
  check(v.eliminated_buffers == std::vector<std::string>{"pro"} && v.extra_bindings[0].root == "x",
        "prologue eliminates pro, loads x");
  check(v.acceptedBy("triton_tiled_gemm") && !v.acceptedBy("native_aten_gemm"), "only generated gemm prologue");
  Builder bar = mmPrologue(true);
  const auto rb = regions(bar, decode(bar));
  bool barrierVeto = false;
  for (const auto& op : rb.ops) {
    barrierVeto = barrierVeto || hasVeto(op, FusionVetoKind::CrossesExtern);
  }
  check(barrierVeto, "RNG/unknown extern between producer and op is a barrier");
  Builder out = mmPrologue(false, true);
  check(hasVeto(FIRST(regions(out, decode(out))), FusionVetoKind::EscapesOutput), "producer that is an output vetoed");
  Builder twice;
  twice.input("x", 25);
  twice.kernel(pointwise("pro", "x", {5, 5}, {5, 1}));
  twice.extern_("mm", "aten.mm.default", {T("pro"), T("pro")}, {}, {D("arg0", "pro", {5, 5})}, {5, 5});
  twice.output("mm", {5, 5});
  check(hasVeto(FIRST(regions(twice, decode(twice))), FusionVetoKind::OperandRepeated), "x@x operand repeated");
}

void testReductionRegionsAndSharedNodes() {
  Builder b;
  b.input("x", 60);
  b.input("s", 1);
  b.kernel(pointwise("pro", "x", {3, 4, 5}, {20, 5, 1}));
  b.extern_("red", "aten.mean.dim", {T("pro"), CallArg::MakeList({CallArg::MakeInteger(2)})}, {},
            {D("arg0", "pro", {3, 4, 5})}, {3, 4});
  b.kernel(pointwise("epi", "red", {3, 4}, {4, 1}, "s", {0, 0}));
  b.output("epi", {3, 4});
  const auto r = regions(b, decode(b));
  check(FIRST(r).variants.size() == 4, "reduction: base, prologue, epilogue, both");
  check(FIRST(r).variants[3].acceptedBy("triton_row_reduction") && FIRST(r).variants[3].consumed_nodes.size() == 3,
        "reduction prologue+epilogue accepted by generated reduction");
  Builder chain;
  chain.input("a", 16);
  chain.input("w", 16);
  chain.extern_("mm1", "aten.mm.default", {T("a"), T("w")}, {}, {D("arg0", "a", {4, 4}), D("arg1", "w", {4, 4})},
                {4, 4});
  chain.kernel(pointwise("mid", "mm1", {4, 4}, {4, 1}));
  chain.extern_("mm2", "aten.mm.default", {T("mid"), T("w")}, {}, {D("arg0", "mid", {4, 4}), D("arg1", "w", {4, 4})},
                {4, 4});
  chain.output("mm2", {4, 4});
  const auto rc = regions(chain, decode(chain));
  check(rc.shared_nodes == std::vector<size_t>{1}, "mm -> relu -> mm: relu claimed by both ops");
}

void testCandidates() {
  Builder b = mmEpilogue(false);
  const auto d = decode(b);
  const auto r = regions(b, d);
  const CandidatePlan strict = enumerateOpCandidates(r, d, defaultRegistry(), ctx(NumericsPolicy::StrictEager));
  const OpCandidates& s = FIRST(strict);
  check(s.candidates.size() == 1 && s.candidates[0].baseline && s.candidates[0].kind == CandidateKind::NativeOption &&
            s.candidates[0].option_id == "native_aten_gemm" && s.candidates[0].variant == 0,
        "strict: only the native baseline");
  bool fenced = false;
  for (const auto& rej : s.rejections) {
    fenced = fenced || rej.reason == RejectReason::NumericsFence;
  }
  check(fenced, "strict: generated/native-fused rejections are typed fences");
  bool threw = false;
  try {
    enumerateOpCandidates(r, d, defaultRegistry(), ctx(NumericsPolicy::Inductor));
  } catch (const OptionRejected& e) {
    threw = e.reason() == RejectReason::Context;
  }
  check(threw, "context numerics must match the program numerics");
  DecodedOps di = d;
  di.numerics = "inductor";
  const CandidatePlan ind = enumerateOpCandidates(r, di, defaultRegistry(), ctx(NumericsPolicy::Inductor));
  const OpCandidates& c = FIRST(ind);
  check(c.candidates[0].baseline && c.candidates[0].option_id == "native_aten_gemm", "inductor: native baseline first");
  size_t gen0 = 0, gen1 = 0, nativeEpi = 0;
  for (const auto& cand : c.candidates) {
    gen0 += cand.kind == CandidateKind::GeneratedOption && cand.variant == 0;
    gen1 += cand.kind == CandidateKind::GeneratedOption && cand.variant == 1;
    nativeEpi += cand.option_id == "native_aten_mm_bias_epilogue";
    check(cand.baseline == (&cand == &c.candidates[0]), "exactly one baseline");
  }
  check(gen0 > 0 && gen1 == gen0 && nativeEpi == 1, "generated configs for both variants + native addmm epilogue");
  Builder perm;
  perm.p = KernelProgram("prog", "inductor");
  perm.input("x", 60);
  perm.extern_("red", "aten.sum.dim_IntList", {T("x"), CallArg::MakeList({CallArg::MakeInteger(2)})}, {},
               {D("arg0", "x", {3, 4, 5}, {1, 3, 12})}, {3, 4});
  perm.output("red", {3, 4});
  const auto pd = decode(perm);
  const auto pr = enumerateOpCandidates(regions(perm, pd), pd, defaultRegistry(), ctx(NumericsPolicy::Inductor));
  check(FIRST(pr).candidates[0].kind == CandidateKind::OriginalExtern && FIRST(pr).candidates[0].baseline,
        "ineligible native reduction: original extern is the baseline");
}

void testGeneratedOutputAliasVeto() {
  Builder b = mmEpilogue(false);
  b.p = KernelProgram("prog", "inductor");
  b.input("a", 20);
  b.input("w", 35);
  b.input("bias", 7);
  b.extern_("mm", "aten.mm.default", {T("a"), T("w")}, {}, {D("arg0", "a", {4, 5}), D("arg1", "w", {5, 7})}, {4, 7});
  b.output("mm", {4, 7});
  const auto d = decode(b);
  check(FIRST(d).output_binding->node_result && !FIRST(d).input_bindings[0].node_result,
        "node_result separates node results from program inputs");
  auto base = regions(b, d);
  auto run = [&](const std::function<void(FusionVariant&)>& edit) {
    FusedOpRegions r = base;
    edit(r.ops[0].variants[0]);
    return enumerateOpCandidates(r, d, defaultRegistry(), ctx(NumericsPolicy::Inductor));
  };
  auto onlyNative = [](const CandidatePlan& p) {
    bool alias = false;
    for (const auto& rj : FIRST(p).rejections) {
      alias = alias || rj.reason == RejectReason::Alias;
    }
    return FIRST(p).candidates.size() == 1 && FIRST(p).candidates[0].kind == CandidateKind::NativeOption && alias;
  };
  check(FIRST(run([](FusionVariant&) {})).candidates.size() > 1, "fresh disjoint output: generated candidates kept");
  check(onlyNative(run([](FusionVariant& v) { v.output_binding.node_result = false; })),
        "program-input output root: generated vetoed (Alias)");
  check(onlyNative(run([](FusionVariant& v) { v.output_binding.root = v.input_bindings[0]->root; })),
        "output root shared with an operand: generated vetoed (Alias)");
  check(onlyNative(run([](FusionVariant& v) { v.output_binding.root_elements = 10; })),
        "store range beyond the root allocation: generated vetoed");
}

void testDecodeBounds() {
  Builder over;
  over.input("a", 19);
  over.input("w", 35);
  over.extern_("mm", "aten.mm.default", {T("a"), T("w")}, {}, {D("arg0", "a", {4, 5}), D("arg1", "w", {5, 7})}, {4, 7});
  check(only(decode(over)).status == DecodeStatus::MalformedAbi, "descriptor beyond its root is malformed");
  Builder neg;
  neg.input("a", 20);
  neg.input("w", 35);
  neg.extern_("mm", "aten.mm.default", {T("a"), T("w")}, {}, {D("arg0", "a", {4, 5}, {-5, 1}, 15),
              D("arg1", "w", {5, 7})}, {4, 7});
  const DecodedOp n = only(decode(neg));
  check(n.status == DecodeStatus::Unsupported && n.reason == RejectReason::Layout, "negative stride: typed unsupported");
}

} // namespace

int main() {
  testDecodeArgsKwargs();
  testDecodeAddmmScalars();
  testDecodeReductions();
  testDecodeLayoutsAndBatches();
  testDecodeRejections();
  testEpilogueRegions();
  testPrologueRegions();
  testReductionRegionsAndSharedNodes();
  testCandidates();
  testGeneratedOutputAliasVeto();
  testDecodeBounds();
  std::printf("op_passes_cpu_test: %d checks passed\n", g_checks);
  return 0;
}
