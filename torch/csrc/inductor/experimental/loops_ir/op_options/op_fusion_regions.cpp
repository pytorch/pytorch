#include "op_fusion_regions.h"

#include <algorithm>
#include <map>
#include <set>
#include <variant>

#include "program_facts.h"

namespace torch::inductor::loops_ir::op_options {

namespace {

struct VetoError {
  FusionVeto veto;
};

[[noreturn]] void veto(FusionVetoKind kind, const std::string& role, size_t node, const std::string& detail) {
  throw VetoError{FusionVeto{kind, role, node, detail}};
}

// One structurally proven neighbour (prologue producer or epilogue consumer).
struct Piece {
  size_t node = 0;
  const Compute* body = nullptr;
  std::vector<std::string> extras;
  int64_t operand = -1; // prologue slot
  int64_t store_offset = 0;
  std::string eliminated; // prologue: operand root
  std::optional<TensorOperand> fused_out; // epilogue
  std::optional<TensorBinding> out_binding; // epilogue
};

const Compute& pointwiseRegion(const KernelProgram& p, size_t node, const std::string& role) {
  const auto* kf = std::get_if<KernelFunction>(&p.nodes().at(node));
  if (!kf) {
    veto(FusionVetoKind::NotPointwiseKernel, role, node, "neighbour is an extern call");
  }
  if (kf->regions().size() != 1) {
    veto(FusionVetoKind::MultipleRegions, role, node, "kernel function has multiple regions");
  }
  const Compute& c = kf->regions().front();
  if (!c.reductionRanges().empty() || !c.accumulators().empty() || !c.reductionBody().empty()) {
    veto(FusionVetoKind::ReductionRegion, role, node, "neighbour is a reduction");
  }
  if (c.results().size() != 1) {
    veto(FusionVetoKind::NotPointwiseKernel, role, node, "neighbour defines more than one result");
  }
  std::set<std::string> loaded;
  for (const Instruction& ins : c.postBody()) {
    if (ins.kind() == Instruction::Kind::ViewRead) {
      veto(FusionVetoKind::ViewRead, role, node, "view_read in neighbour (run view_inline first)");
    }
    if (ins.kind() == Instruction::Kind::Load) {
      loaded.insert(ins.buffer());
    }
  }
  for (const std::string& in : c.inputs()) {
    if (!loaded.count(in)) {
      veto(FusionVetoKind::UnusedInput, role, node, "declared input '" + in + "' is never loaded");
    }
  }
  return c;
}

void requireOnlyKernelsBetween(const KernelProgram& p, size_t lo, size_t hi, const std::string& role) {
  for (size_t n = lo + 1; n < hi; ++n) {
    if (!std::holds_alternative<KernelFunction>(p.nodes()[n])) {
      veto(FusionVetoKind::CrossesExtern, role, n, "fusion would move work across an extern barrier");
    }
  }
}

void requireSingleUse(const ProgramFacts& f, const std::string& root, size_t user, const std::string& role, size_t node) {
  if (f.escaping_roots.count(root)) {
    veto(FusionVetoKind::EscapesOutput, role, node, "'" + root + "' (or an alias) is a program output");
  }
  auto it = f.users.find(root);
  if (it == f.users.end() || it->second != std::vector<size_t>{user}) {
    veto(FusionVetoKind::MultipleUses, role, node, "'" + root + "' has users other than the fused node");
  }
}

void requireDomain(const Compute& c, const std::vector<int64_t>& sizes, const std::string& role, size_t node) {
  const auto& r = c.pointwiseRanges();
  bool ok = r.size() == sizes.size();
  for (size_t d = 0; ok && d < r.size(); ++d) {
    ok = r[d].start == 0 && r[d].stop == sizes[d];
  }
  if (!ok) {
    veto(FusionVetoKind::DomainMismatch, role, node, "region domain differs from the fused logical shape");
  }
}

std::vector<std::string> extrasOf(const Compute& c, const std::string& self) {
  std::vector<std::string> out;
  for (const std::string& in : c.inputs()) {
    if (in != self) {
      out.push_back(in);
    }
  }
  return out;
}

std::optional<Piece> prologueFor(const KernelProgram& p, const ProgramFacts& f, const DecodedOp& op, size_t k) {
  const std::string role = "prologue[" + std::to_string(k) + "]";
  const TensorBinding& b = op.input_bindings[k];
  auto prod = f.producer.find(b.root);
  if (prod == f.producer.end()) {
    return std::nullopt; // program input
  }
  const size_t node = prod->second;
  const Compute& c = pointwiseRegion(p, node, role);
  requireSingleUse(f, b.root, op.node_index, role, node);
  size_t reads = 0;
  for (const std::string& in : op.fallback.inputs()) {
    reads += f.resolve(in).root == b.root ? 1 : 0;
  }
  if (reads != 1) {
    veto(FusionVetoKind::OperandRepeated, role, node, "operand storage is read through more than one argument");
  }
  requireOnlyKernelsBetween(p, node, op.node_index, role);
  requireDomain(c, op.signature->inputs()[k].layout.sizes, role, node);
  Piece piece;
  piece.node = node;
  piece.body = &c;
  piece.extras = extrasOf(c, "");
  piece.operand = static_cast<int64_t>(k);
  piece.store_offset = b.root_offset;
  piece.eliminated = b.root;
  return piece;
}

std::optional<Piece> epilogueFor(const KernelProgram& p, const ProgramFacts& f, const DecodedOp& op) {
  const std::string role = "epilogue";
  const std::string& r = op.output_binding->root;
  auto users = f.users.find(r);
  if (users == f.users.end() || users->second.empty()) {
    return std::nullopt;
  }
  const size_t node = users->second.front();
  requireSingleUse(f, r, node, role, node);
  const Compute& c = pointwiseRegion(p, node, role);
  requireOnlyKernelsBetween(p, op.node_index, node, role);
  const auto& sizes = op.signature->output().layout.sizes;
  requireDomain(c, sizes, role, node);
  const Instruction* store = nullptr;
  for (const Instruction& ins : c.postBody()) {
    if (ins.kind() == Instruction::Kind::Store) {
      if (store) {
        veto(FusionVetoKind::NonAffineStore, role, node, "multiple stores");
      }
      store = &ins;
    }
  }
  if (!store) {
    veto(FusionVetoKind::NonAffineStore, role, node, "no store");
  }
  AccessMap map;
  try {
    map = affineAccessMap(store->offset(), c.pointwiseRanges(), sizes, role);
  } catch (const OptionRejected& e) {
    veto(FusionVetoKind::NonAffineStore, role, node, e.what());
  }
  const Buffer& y = c.results().front();
  TensorLayout layout{StorageDtype::F32, sizes, map.coeffs, ShapeKind::Static};
  if (std::any_of(map.coeffs.begin(), map.coeffs.end(), [](int64_t s) { return s < 0; }) || map.offset < 0 ||
      !layout.isNonOverlapping()) {
    veto(FusionVetoKind::NonAffineStore, role, node, "store map is negative-strided or not injective");
  }
  Piece piece;
  piece.node = node;
  piece.body = &c;
  piece.extras = extrasOf(c, r);
  piece.store_offset = map.offset;
  piece.fused_out = TensorOperand{y.name, layout};
  piece.out_binding = TensorBinding{y.name, y.name, 0, y.elements, true};
  return piece;
}

void addExtras(const ProgramFacts& f, const Piece& piece, FusionVariant& v) {
  for (const std::string& name : piece.extras) {
    bool seen = false;
    for (const auto& e : v.fused_loops.extra_inputs) {
      seen = seen || e.name == name;
    }
    if (seen) {
      continue;
    }
    const RootRef& r = f.resolve(name);
    const int64_t elems = f.root_elements.at(r.root);
    if (f.root_dtype.at(r.root) != "f32") {
      reject(RejectReason::Dtype, "extra input '" + name + "' is not f32");
    }
    v.fused_loops.extra_inputs.push_back(
        ExtraInput{name, TensorLayout{StorageDtype::F32, {elems}, {1}, ShapeKind::Static}});
    v.extra_bindings.push_back(TensorBinding{name, r.root, r.offset, elems, f.producer.count(r.root) > 0});
  }
}

void checkContracts(const OptionRegistry& reg, FusionVariant& v) {
  for (const std::string& id : reg.optionsFor(v.signature.kind())) {
    const OptionDescriptor& opt = reg.get(id);
    OptionContractResult res;
    res.option_id = id;
    try {
      const ValidatedFusion vf = validateFusion(opt.fusionFor(v.signature), v.signature, v.fused_loops);
      if (opt.native_plan) {
        opt.native_plan(v.signature, vf);
      }
      res.accepted = true;
    } catch (const OptionRejected& e) {
      res.reason = e.reason();
      res.detail = e.what();
    }
    v.contracts.push_back(std::move(res));
  }
}

FusionVariant buildVariant(const ProgramFacts& f,
                           const DecodedOp& op,
                           const std::vector<const Piece*>& prologues,
                           const Piece* epilogue) {
  FusionVariant v(epilogue ? op.signature->withOutput(*epilogue->fused_out) : *op.signature);
  for (const auto& b : op.input_bindings) {
    v.input_bindings.emplace_back(b);
  }
  std::set<size_t> nodes{op.node_index};
  for (const Piece* pr : prologues) {
    v.prologue_operands.push_back(pr->operand);
    v.fused_loops.prologues.push_back(PrologueRegion{pr->operand, *pr->body, pr->store_offset});
    v.input_bindings[static_cast<size_t>(pr->operand)].reset();
    v.eliminated_buffers.push_back(pr->eliminated);
    nodes.insert(pr->node);
    addExtras(f, *pr, v);
  }
  v.output_binding = *op.output_binding;
  v.produces = op.output_binding->root;
  if (epilogue) {
    v.epilogue = true;
    EpilogueRegion e;
    e.op_result = op.output_binding->root;
    e.body = *epilogue->body;
    e.op_result_layout = op.signature->output().layout;
    e.op_result_offset = op.output_binding->root_offset;
    e.store_offset = epilogue->store_offset;
    v.fused_loops.epilogue = std::move(e);
    v.eliminated_buffers.push_back(op.output_binding->root);
    v.output_binding = *epilogue->out_binding;
    v.produces = epilogue->fused_out->name;
    nodes.insert(epilogue->node);
    addExtras(f, *epilogue, v);
  }
  v.consumed_nodes.assign(nodes.begin(), nodes.end());
  v.placement = v.consumed_nodes.back();
  return v;
}

template <typename Fn>
auto tryPiece(std::vector<FusionVeto>& vetoes, Fn&& fn) -> decltype(fn()) {
  try {
    return fn();
  } catch (const VetoError& e) {
    vetoes.push_back(e.veto);
    return std::nullopt;
  }
}

OpRegionSet formForOp(const KernelProgram& p,
                      const ProgramFacts& f,
                      const DecodedOps& decoded,
                      size_t index,
                      const OptionRegistry& reg) {
  const DecodedOp& op = decoded.ops[index];
  OpRegionSet set;
  set.decoded_index = index;
  set.node_index = op.node_index;
  std::vector<Piece> prologues;
  for (size_t k = 0; k < op.input_bindings.size(); ++k) {
    if (auto pr = tryPiece(set.vetoes, [&] { return prologueFor(p, f, op, k); })) {
      prologues.push_back(std::move(*pr));
    }
  }
  const std::optional<Piece> epi = tryPiece(set.vetoes, [&] { return epilogueFor(p, f, op); });
  for (size_t mask = 0; mask < (size_t{1} << prologues.size()); ++mask) {
    for (bool withEpi : {false, true}) {
      if (withEpi && !epi) {
        continue;
      }
      std::vector<const Piece*> chosen;
      for (size_t b = 0; b < prologues.size(); ++b) {
        if (mask & (size_t{1} << b)) {
          chosen.push_back(&prologues[b]);
        }
      }
      try {
        FusionVariant v = buildVariant(f, op, chosen, withEpi ? &*epi : nullptr);
        checkContracts(reg, v);
        const bool accepted =
            std::any_of(v.contracts.begin(), v.contracts.end(), [](const auto& c) { return c.accepted; });
        if (accepted || (mask == 0 && !withEpi)) {
          set.variants.push_back(std::move(v));
        } else {
          set.vetoes.push_back(FusionVeto{
              FusionVetoKind::NoOptionContract, "variant", std::nullopt, "no option contract accepts this region"});
        }
      } catch (const OptionRejected& e) {
        set.vetoes.push_back(FusionVeto{FusionVetoKind::NoOptionContract, "variant", std::nullopt, e.what()});
      }
    }
  }
  return set;
}

} // namespace

const char* fusionVetoKindName(FusionVetoKind kind) {
  switch (kind) {
    case FusionVetoKind::NotPointwiseKernel:
      return "not_pointwise_kernel";
    case FusionVetoKind::MultipleRegions:
      return "multiple_regions";
    case FusionVetoKind::ReductionRegion:
      return "reduction_region";
    case FusionVetoKind::ViewRead:
      return "view_read";
    case FusionVetoKind::UnusedInput:
      return "unused_input";
    case FusionVetoKind::MultipleUses:
      return "multiple_uses";
    case FusionVetoKind::EscapesOutput:
      return "escapes_output";
    case FusionVetoKind::CrossesExtern:
      return "crosses_extern";
    case FusionVetoKind::OperandRepeated:
      return "operand_repeated";
    case FusionVetoKind::DomainMismatch:
      return "domain_mismatch";
    case FusionVetoKind::NonAffineStore:
      return "non_affine_store";
    case FusionVetoKind::NoOptionContract:
      return "no_option_contract";
  }
  return "?";
}

bool FusionVariant::acceptedBy(const std::string& option_id) const {
  for (const auto& c : contracts) {
    if (c.option_id == option_id) {
      return c.accepted;
    }
  }
  return false;
}

FusedOpRegions formFusedOpRegions(const KernelProgram& program,
                                  const DecodedOps& decoded,
                                  const OptionRegistry& registry) {
  const ProgramFacts facts = analyzeProgram(program);
  FusedOpRegions out;
  std::map<size_t, std::set<size_t>> claims;
  for (size_t i = 0; i < decoded.ops.size(); ++i) {
    if (decoded.ops[i].status != DecodeStatus::Decoded) {
      continue;
    }
    OpRegionSet set = formForOp(program, facts, decoded, i, registry);
    for (const auto& v : set.variants) {
      for (size_t n : v.consumed_nodes) {
        claims[n].insert(i);
      }
    }
    out.ops.push_back(std::move(set));
  }
  for (const auto& kv : claims) {
    if (kv.second.size() > 1) {
      out.shared_nodes.push_back(kv.first);
    }
  }
  return out;
}

} // namespace torch::inductor::loops_ir::op_options
