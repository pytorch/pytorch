// Standalone CPU test for the auxiliary autotune region extraction helper
// (op_autotune_region.h, header-only). Links ../ir.cpp + ../kernel_ir.cpp
// (for baseline.buildDependencies()/verify()) plus the op_options units.
// Build from the op_options directory:
//   g++ -std=c++17 -O0 -Wall ../ir.cpp ../kernel_ir.cpp ../analyses.cpp \
//     ../passes.cpp ../pass_manager.cpp ../printer.cpp ../tiled_ir.cpp \
//     op_types.cpp op_config.cpp fused_region.cpp option_registry.cpp \
//     gemm_options.cpp reduction_options.cpp program_facts.cpp op_decode.cpp \
//     op_fusion_regions.cpp op_candidates.cpp op_autotune_region.cpp \
//     tests/op_autotune_region_cpu_test.cpp -o /tmp/op_autotune_region_cpu_test
// No GPU, no torch/pybind.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "../op_autotune_region.h"

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
      std::fprintf(stderr, "FAILED: %s: wrong reason %s (detail: %s)\n", what.c_str(), e.what(), e.what());
      std::exit(1);
    }
    return;
  }
  std::fprintf(stderr, "FAILED: %s: not rejected\n", what.c_str());
  std::exit(1);
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

// mm(a, w) -> epi(relu(mm + bias[j])) -> output. Nodes: 0=mm, 1=epi.
Builder mmEpilogue() {
  Builder b;
  b.input("a", 20);
  b.input("w", 35);
  b.input("bias", 7);
  b.extern_("mm", "aten.mm.default", {T("a"), T("w")}, {}, {D("arg0", "a", {4, 5}), D("arg1", "w", {5, 7})}, {4, 7});
  b.kernel(pointwise("epi", "mm", {4, 7}, {7, 1}, "bias", {0, 1}, true));
  b.output("epi", {4, 7});
  return b;
}

// pro = relu(x); mm(pro, w). Nodes: 0=pro, 1=mm.
Builder mmPrologue() {
  Builder b;
  b.input("x", 20);
  b.input("w", 35);
  b.kernel(pointwise("pro", "x", {4, 5}, {5, 1}));
  b.extern_("mm", "aten.mm.default", {T("pro"), T("w")}, {}, {D("arg0", "pro", {4, 5}), D("arg1", "w", {5, 7})},
            {4, 7});
  b.output("mm", {4, 7});
  return b;
}

// pro = relu(x); red = mean(pro, dim=2); epi = red + s. Nodes: 0,1,2.
Builder reductionBoth() {
  Builder b;
  b.input("x", 60);
  b.input("s", 1);
  b.kernel(pointwise("pro", "x", {3, 4, 5}, {20, 5, 1}));
  b.extern_("red", "aten.mean.dim", {T("pro"), CallArg::MakeList({CallArg::MakeInteger(2)})}, {},
            {D("arg0", "pro", {3, 4, 5})}, {3, 4});
  b.kernel(pointwise("epi", "red", {3, 4}, {4, 1}, "s", {0, 0}, false));
  b.output("epi", {3, 4});
  return b;
}

DecodedOps decode(const Builder& b) {
  return decodeOpSignatures(b.p, DeviceKind::Cuda);
}

FusedOpRegions regions(const Builder& b, const DecodedOps& d) {
  return formFusedOpRegions(b.p, d, defaultRegistry());
}

const FusionVariant& fusedVariant(const FusedOpRegions& r, size_t vi) {
  check(!r.ops.empty(), "region set non-empty");
  check(vi < r.ops.front().variants.size(), "variant index in range");
  return r.ops.front().variants[vi];
}

// The baseline must verify + build deps through the real base APIs.
void checkBaselineExecutable(AutotuneRegion& region, const std::string& what) {
  region.baseline.buildDependencies();
  try {
    region.baseline.verify();
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "FAILED: %s: baseline.verify() threw: %s\n", what.c_str(), e.what());
    std::exit(1);
  }
  ++g_checks;
}

void testMmReluFullBaseline() {
  Builder b = mmEpilogue();
  const DecodedOps d = decode(b);
  const FusedOpRegions r = regions(b, d);
  const FusionVariant& v = fusedVariant(r, 1); // epilogue variant
  check(v.consumed_nodes == std::vector<size_t>{0, 1}, "mm+relu variant consumes 2 nodes");
  AutotuneRegion region = extractAutotuneRegion(b.p, v);
  check(region.consumed_nodes == std::vector<size_t>{0, 1}, "region consumed {0,1}");
  check(region.placement == 1, "placement is max consumed index");
  check(region.frontier.size() == 3, "frontier is {a, w, bias}");
  check(region.frontier[0].root == "a" && region.frontier[1].root == "w" && region.frontier[2].root == "bias",
        "frontier first-use order a, w, bias");
  check(region.frontier[0].elements == 20 && region.frontier[2].elements == 7, "frontier flat geometry");
  check(region.input_frontier == std::vector<int64_t>{0, 1}, "sig slots map to frontier roots");
  check(region.extra_frontier == std::vector<int64_t>{2}, "bias extra maps to frontier");
  check(region.output_binding.root == "epi", "output is the FINAL variant output, not the anchor");
  check(region.produces == "epi", "produces is the final output");
  check(region.eliminated_buffers == std::vector<std::string>{"mm"}, "mm eliminated");
  check(region.cuda_generated_safe && !region.has_cpu_frontier, "all-CUDA region is generated-safe");
  // Baseline: 3 frontier inputs, exactly the 2 consumed nodes in order,
  // anchor extern intact, epilogue kernel verbatim, final output.
  check(region.baseline.inputs().size() == 3, "baseline has 3 frontier inputs");
  check(region.baseline.nodes().size() == 2, "baseline has exactly the 2 consumed nodes");
  check(std::holds_alternative<ExternCall>(region.baseline.nodes()[0]), "baseline node 0 is the anchor extern");
  check(std::get<ExternCall>(region.baseline.nodes()[0]).target() == "aten.mm.default", "anchor target intact");
  check(std::holds_alternative<KernelFunction>(region.baseline.nodes()[1]), "baseline node 1 is the epilogue kernel");
  check(region.baseline.outputs().size() == 1 && region.baseline.outputs()[0].buffer == "epi",
        "baseline output is the final variant output");
  checkBaselineExecutable(region, "mm+relu baseline verifies");
}

void testGemmPrologue() {
  Builder b = mmPrologue();
  const DecodedOps d = decode(b);
  const FusedOpRegions r = regions(b, d);
  check(r.ops.front().variants.size() == 2, "base + prologue variants");
  const FusionVariant& v = fusedVariant(r, 1);
  check(v.prologue_operands == std::vector<int64_t>{0}, "prologue on operand 0");
  AutotuneRegion region = extractAutotuneRegion(b.p, v);
  check(region.consumed_nodes == std::vector<size_t>{0, 1}, "prologue region consumes pro + anchor");
  check(region.input_frontier[0] == -1, "prologue-computed slot maps to -1");
  check(region.input_frontier[1] >= 0, "operand 1 maps to a frontier root");
  check(region.frontier.size() == 2, "frontier is {x, w}");
  check(region.output_binding.root == "mm", "output is the anchor result (no epilogue)");
  check(region.eliminated_buffers == std::vector<std::string>{"pro"}, "pro eliminated");
  check(region.baseline.nodes().size() == 2, "baseline keeps prologue Compute + anchor");
  check(std::holds_alternative<KernelFunction>(region.baseline.nodes()[0]), "prologue Compute first");
  check(std::holds_alternative<ExternCall>(region.baseline.nodes()[1]), "anchor second");
  checkBaselineExecutable(region, "prologue baseline verifies");
}

void testPrologueEpilogueOrdering() {
  Builder b = reductionBoth();
  const DecodedOps d = decode(b);
  const FusedOpRegions r = regions(b, d);
  check(r.ops.front().variants.size() == 4, "reduction: base, pro, epi, both");
  const FusionVariant& v = fusedVariant(r, 3);
  check(v.consumed_nodes == std::vector<size_t>{0, 1, 2}, "3-node exact ordering");
  AutotuneRegion region = extractAutotuneRegion(b.p, v);
  check(region.placement == 2, "placement is max consumed index");
  check(region.input_frontier == std::vector<int64_t>{-1}, "reduction input is prologue-computed");
  check(region.extra_frontier.size() == 2, "two extras (x from prologue, s from epilogue)");
  check(region.output_binding.root == "epi", "final output is the epilogue result");
  check(region.baseline.nodes().size() == 3, "baseline has all 3 nodes in order");
  check(std::holds_alternative<KernelFunction>(region.baseline.nodes()[0]), "prologue kernel first");
  check(std::holds_alternative<ExternCall>(region.baseline.nodes()[1]), "anchor second");
  check(std::holds_alternative<KernelFunction>(region.baseline.nodes()[2]), "epilogue kernel third");
  checkBaselineExecutable(region, "3-node baseline verifies");
}

void testReductionEpilogue() {
  Builder b = reductionBoth();
  const DecodedOps d = decode(b);
  const FusedOpRegions r = regions(b, d);
  const FusionVariant& v = fusedVariant(r, 1); // epilogue only (mask=0, withEpi=true)
  check(v.epilogue && v.prologue_operands.empty(), "epilogue-only variant");
  AutotuneRegion region = extractAutotuneRegion(b.p, v);
  check(region.consumed_nodes == std::vector<size_t>{1, 2}, "epilogue region consumes anchor + epi");
  check(region.input_frontier.size() == 1 && region.input_frontier[0] >= 0, "reduction input is a frontier root");
  check(region.frontier.size() == 2, "frontier is {pro, s}");
  check(region.output_binding.root == "epi", "final output, not the anchor output");
  checkBaselineExecutable(region, "reduction-epilogue baseline verifies");
}

void testSameRootMultiViewDedup() {
  Builder b;
  b.input("base", 64);
  b.input("w", 35);
  b.p.addView(View("v1", "base", {60}, {1}, 3));
  b.p.addView(View("v2", "v1", {4, 5}, {1, 4}, 2));
  // extra kernel reads v1 while mm reads v2: same root, different offsets.
  b.kernel(pointwise("pre", "v1", {60}, {1}, "", {}, false));
  b.extern_("mm", "aten.mm.default", {T("v2"), T("w")}, {},
            {D("arg0", "v2", {4, 5}, {1, 4}, 1), D("arg1", "w", {5, 7})}, {4, 7});
  b.output("mm", {4, 7});
  const DecodedOps d = decode(b);
  check(d.ops.size() == 1 && d.ops.front().status == DecodeStatus::Decoded, "view-chain mm decodes");
  const FusedOpRegions r = regions(b, d);
  const FusionVariant& v = fusedVariant(r, 0); // unfused baseline variant
  AutotuneRegion region = extractAutotuneRegion(b.p, v);
  check(region.frontier.size() == 2, "frontier dedups to {base, w}");
  check(region.frontier[0].root == "base", "shared root appears once");
  check(region.input_frontier[0] == 0 && region.input_frontier[1] == 1, "both slots map to deduped frontier");
  // Offsets: view chain 3 + 2 + descriptor 1 = 6.
  check(region.input_bindings[0]->root_offset == 6, "composed view/descriptor offsets preserved");
  check(region.baseline.inputs().size() == 2, "baseline allocates one backing tensor per root");
  checkBaselineExecutable(region, "dedup baseline verifies");
}

void testSharedClaimsFailClosed() {
  Builder b = mmEpilogue();
  const DecodedOps d = decode(b);
  const FusedOpRegions r = regions(b, d);
  const FusionVariant& v = fusedVariant(r, 1);
  // Node 1 claimed by another op: selection must keep consumed sets disjoint.
  expectReject(RejectReason::Alias,
               [&] { extractAutotuneRegion(b.p, v, {}, {1}); },
               "shared consumed node fails closed");
  expectReject(RejectReason::Alias,
               [&] { extractAutotuneRegion(b.p, v, {}, {0}); },
               "shared anchor node fails closed");
}

void testCpuVeto() {
  Builder b = mmEpilogue();
  const DecodedOps d = decode(b);
  const FusedOpRegions r = regions(b, d);
  const FusionVariant& v = fusedVariant(r, 1);
  // Frozen constant metadata says the bias root is CPU.
  AutotuneRegion region = extractAutotuneRegion(b.p, v, {"bias"});
  check(region.has_cpu_frontier, "CPU extra root recorded");
  check(!region.cuda_generated_safe, "generated CUDA vetoed");
  check(!autotuneRegionCudaSafe(region), "helper reports unsafe");
  expectReject(RejectReason::Context,
               [&] { requireAutotuneRegionCudaSafe(region); },
               "generated CUDA use with CPU extra is vetoed");
  // Aliased CPU root (via a view name) taints the storage root too.
  Builder b2 = mmEpilogue();
  const DecodedOps d2 = decode(b2);
  const FusedOpRegions r2 = regions(b2, d2);
  AutotuneRegion region2 = extractAutotuneRegion(b2.p, fusedVariant(r2, 1), {"a"});
  check(region2.has_cpu_frontier && !autotuneRegionCudaSafe(region2), "CPU sig root vetoed");
  // All-CUDA passes the gate: native fallback stays available either way.
  AutotuneRegion ok = extractAutotuneRegion(b.p, v);
  requireAutotuneRegionCudaSafe(ok);
  ++g_checks;
}

void testMalformedRejects() {
  Builder b = mmEpilogue();
  const DecodedOps d = decode(b);
  const FusedOpRegions r = regions(b, d);
  // Dangling ir_name.
  {
    FusedOpRegions rr = r;
    rr.ops[0].variants[1].extra_bindings[0].ir_name = "no_such_buffer";
    expectReject(RejectReason::Signature,
                 [&] { extractAutotuneRegion(b.p, rr.ops[0].variants[1]); },
                 "dangling ir_name rejected");
  }
  // Out-of-bounds root_offset.
  {
    FusedOpRegions rr = r;
    rr.ops[0].variants[1].extra_bindings[0].root_offset = 7; // bias has 7 elements, extent 28
    expectReject(RejectReason::Signature,
                 [&] { extractAutotuneRegion(b.p, rr.ops[0].variants[1]); },
                 "out-of-bounds binding rejected");
  }
  // root_elements disagreeing with the program.
  {
    FusedOpRegions rr = r;
    rr.ops[0].variants[1].output_binding.root_elements = 3;
    expectReject(RejectReason::Signature,
                 [&] { extractAutotuneRegion(b.p, rr.ops[0].variants[1]); },
                 "root_elements mismatch rejected");
  }
  // Empty consumed set.
  {
    FusedOpRegions rr = r;
    rr.ops[0].variants[1].consumed_nodes.clear();
    expectReject(RejectReason::Signature,
                 [&] { extractAutotuneRegion(b.p, rr.ops[0].variants[1]); },
                 "empty consumed set rejected");
  }
  // Unordered consumed set.
  {
    FusedOpRegions rr = r;
    rr.ops[0].variants[1].consumed_nodes = {1, 0};
    expectReject(RejectReason::Signature,
                 [&] { extractAutotuneRegion(b.p, rr.ops[0].variants[1]); },
                 "unordered consumed set rejected");
  }
  // node_result flag disagreeing with the program.
  {
    FusedOpRegions rr = r;
    rr.ops[0].variants[1].output_binding.node_result = false;
    expectReject(RejectReason::Signature,
                 [&] { extractAutotuneRegion(b.p, rr.ops[0].variants[1]); },
                 "node_result flag mismatch rejected");
  }
  // Output root not produced by the consumed set: use the prologue-only
  // variant's output binding shape but point it at an external root whose
  // geometry matches, so the ownership (Alias) check is what fires.
  // Simplest deterministic shape: take the unfused mm-only variant (output
  // mm, extent 28) and claim its output is the program input "bias"... that
  // mismatches bounds. Instead build a tiny dedicated program below.
  {
    Builder s;
    s.input("a", 20);
    s.input("w", 35);
    s.input("ext", 28);
    s.extern_("mm", "aten.mm.default", {T("a"), T("w")}, {}, {D("arg0", "a", {4, 5}), D("arg1", "w", {5, 7})}, {4, 7});
    s.output("mm", {4, 7});
    const DecodedOps sd = decode(s);
    const FusedOpRegions sr = regions(s, sd);
    FusedOpRegions rr = sr;
    auto& ob = rr.ops[0].variants[0].output_binding;
    ob.ir_name = "ext";
    ob.root = "ext";
    ob.root_elements = 28;
    ob.node_result = false;
    rr.ops[0].variants[0].produces = "ext";
    expectReject(RejectReason::Alias,
                 [&] { extractAutotuneRegion(s.p, rr.ops[0].variants[0]); },
                 "external output root rejected (not produced by region)");
  }
}

} // namespace

int main() {
  check(autotuneRegionVersion() == "autotune_region_v1", "helper version pinned");
  testMmReluFullBaseline();
  testGemmPrologue();
  testPrologueEpilogueOrdering();
  testReductionEpilogue();
  testSameRootMultiViewDedup();
  testSharedClaimsFailClosed();
  testCpuVeto();
  testMalformedRejects();
  std::printf("op_autotune_region_cpu_test: %d checks passed\n", g_checks);
  return 0;
}
