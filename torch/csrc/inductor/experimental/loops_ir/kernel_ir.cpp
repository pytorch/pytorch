// KernelIR: program-level DAG of kernel functions and extern calls.
// Vertical pointwise fusion only; no horizontal/reduction fusion, no fusion
// across extern barriers. Strict numerics: rename SSA only, preserve every
// FP op and order. Conservative: any missing proof leaves nodes unfused.

#include "kernel_ir.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "passes.h"

namespace torch::inductor::loops_ir {
namespace {

[[noreturn]] void kfail(const std::string& ctx, const std::string& msg) {
  throw std::invalid_argument(ctx + ": " + msg);
}

std::string vecStrK(const std::vector<int64_t>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += std::to_string(v[i]);
  }
  return s + "]";
}

std::string buffersStrK(const std::vector<Buffer>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += "(" + v[i].name + " " + v[i].dtype + " " +
        std::to_string(v[i].elements) + ")";
  }
  return s + "]";
}

// True writer of a buffer: the single KernelFunction producing it, or -1.
int writerOf(
    const std::vector<KernelProgram::Node>& nodes,
    const std::string& buffer,
    const std::vector<Buffer>& inputs) {
  for (const auto& b : inputs) {
    if (b.name == buffer) {
      return -2; // program input (no writer node)
    }
  }
  int found = -1;
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto produces = [&](const std::string& r) { return r == buffer; };
    if (const auto* f = std::get_if<KernelFunction>(&nodes[i])) {
      for (const auto& reg : f->regions()) {
        for (const auto& b : reg.results()) {
          if (produces(b.name)) {
            if (found >= 0) {
              return -3; // multiply defined
            }
            found = (int)i;
          }
        }
      }
    } else {
      for (const auto& b : std::get_if<ExternCall>(&nodes[i])->results()) {
        if (produces(b.name)) {
          if (found >= 0) {
            return -3;
          }
          found = (int)i;
        }
      }
    }
  }
  return found;
}

// All buffer/view names a node reads (for dependency edges).
std::set<std::string> nodeReads(const KernelProgram::Node& n) {
  std::set<std::string> out;
  if (const auto* f = std::get_if<KernelFunction>(&n)) {
    for (const auto& reg : f->regions()) {
      for (const auto& in : reg.inputs()) {
        out.insert(in);
      }
    }
  } else {
    for (const auto& in : std::get_if<ExternCall>(&n)->inputs()) {
      out.insert(in);
    }
  }
  return out;
}

// Verify one Compute region as a kernel region: pointwise-only, f32 data.
// Reductions must already be eager externs (no generated reductions).
void checkKernelRegion(const Compute& c, const std::string& ctx) {
  if (!c.reductionRanges().empty() || !c.accumulators().empty() ||
      !c.reductionBody().empty()) {
    kfail(
        ctx,
        "kernel region '" + c.name() +
            "' has reductions (generated reductions unsupported; use eager extern)");
  }
  if (c.results().empty()) {
    kfail(ctx, "kernel region '" + c.name() + "' has no results");
  }
  for (const auto& b : c.results()) {
    if (!isComputeDataDtype(b.dtype)) {
      kfail(
          ctx,
          "kernel region '" + c.name() + "' result '" + b.name +
              "' has dtype '" + b.dtype + "' (must be f32)");
    }
  }
  for (const auto& ins : c.postBody()) {
    if (ins.kind() == Instruction::Kind::ViewRead) {
      kfail(
          ctx,
          "kernel region '" + c.name() +
              "' has view_read (run view_inline first)");
    }
    if (ins.kind() == Instruction::Kind::Accumulate ||
        ins.kind() == Instruction::Kind::ReadAccumulator) {
      kfail(ctx, "kernel region '" + c.name() + "' has reduction ops");
    }
  }
}

// Resolve a view/alias chain to its storage root. Returns "" on cycles.
std::string viewRoot(
    const std::string& name,
    const std::map<std::string, View>& viewMap) {
  std::string cur = name;
  std::set<std::string> seen;
  while (viewMap.count(cur)) {
    if (!seen.insert(cur).second) {
      return "";
    }
    cur = viewMap.at(cur).source;
  }
  return cur;
}

// Alias closure of a buffer: itself plus every view whose transitive root
// is that buffer. Views are element aliases of storage, so reads/outputs
// through views observe the underlying buffer.
std::set<std::string> aliasClosure(
    const std::string& buffer,
    const std::map<std::string, View>& viewMap) {
  std::set<std::string> out{buffer};
  for (const auto& kv : viewMap) {
    if (viewRoot(kv.first, viewMap) == buffer) {
      out.insert(kv.first);
    }
  }
  return out;
}

// Fusion legality for producer P -> consumer C (both single-region pointwise):
//  - P has exactly one result (the fused temp), C reads it.
//  - P's result store proven dense/injective.
//  - identical pointwise ranges (names + extents).
//  - P's store offset structurally equals C's load offset of the temp
//    (same iteration map; indexEquals, never pointer equality).
//  - no other node reads the temp (single use), temp not a program output.
// `nodes` is the current working node list (fixed-point rounds mutate it);
// `outputs` are the (immutable) program outputs.
bool fusiblePair(
    const Compute& prod,
    const Compute& cons,
    const std::vector<KernelProgram::Node>& nodes,
    const std::vector<Descriptor>& outputs,
    const std::map<std::string, View>& viewMap,
    size_t prodIdx,
    size_t consIdx,
    std::string& temp) {
  if (prod.results().size() != 1) {
    return false;
  }
  temp = prod.results()[0].name;
  if (prod.pointwiseRanges().size() != cons.pointwiseRanges().size()) {
    return false;
  }
  for (size_t i = 0; i < prod.pointwiseRanges().size(); ++i) {
    const auto& a = prod.pointwiseRanges()[i];
    const auto& b = cons.pointwiseRanges()[i];
    if (a.name != b.name || a.start != b.start || a.stop != b.stop) {
      return false;
    }
  }
  if (!prove_dense_injective_store(prod, temp)) {
    return false;
  }
  // Consumer must load the temp exactly once-shape; find its load offset.
  const Index* prodStore = nullptr;
  for (const auto& ins : prod.postBody()) {
    if (ins.kind() == Instruction::Kind::Store && ins.buffer() == temp) {
      prodStore = &ins.offset();
      break;
    }
  }
  // Consumer loads of temp: collect offsets; all must equal the store map.
  std::vector<const Index*> consLoads;
  for (const auto& ins : cons.postBody()) {
    if (ins.kind() == Instruction::Kind::Load && ins.buffer() == temp) {
      consLoads.push_back(&ins.offset());
    }
  }
  if (consLoads.empty() || prodStore == nullptr) {
    return false;
  }
  for (const auto* li : consLoads) {
    if (!indexEquals(*prodStore, *li)) {
      return false;
    }
  }
  // Single use: no other node reads the temp, directly or through a view
  // alias of it. Reads through views observe the same storage, so they
  // keep the temp alive exactly like direct reads.
  std::set<std::string> tempAliases = aliasClosure(temp, viewMap);
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (i == prodIdx || i == consIdx) {
      continue;
    }
    std::set<std::string> reads = nodeReads(nodes[i]);
    for (const auto& a : tempAliases) {
      if (reads.count(a)) {
        return false;
      }
    }
  }
  // Temp must not escape through outputs either: an output descriptor may
  // reference the temp directly or any view whose transitive root is it.
  for (const auto& o : outputs) {
    if (tempAliases.count(o.buffer)) {
      return false;
    }
  }
  // Consumer reads must be declared; producer's temp must be its input.
  if (std::find(cons.inputs().begin(), cons.inputs().end(), temp) ==
      cons.inputs().end()) {
    return false;
  }
  return true;
}

// Fuse prod into cons: inline prod's post-body (minus its store) into cons,
// replacing cons loads of temp with prod's value SSA (renamed). Store-offsets
// preserved; FP ops copied verbatim in order. Numerics carried explicitly so
// strict/inductor policies survive fusion.
KernelFunction fusePair(
    const Compute& prod,
    const Compute& cons,
    const std::string& temp,
    const std::string& numerics) {
  // Map prod SSA -> fresh ids in merged body.
  int maxId = -1;
  for (const auto* body : {&prod.postBody(), &cons.postBody()}) {
    for (const auto& ins : *body) {
      if (ins.definesValue()) {
        maxId = std::max(maxId, ins.id());
      }
    }
  }
  int next = maxId + 1;
  std::map<int, int> ren;
  // prod's stored value id.
  int prodVal = -1;
  for (const auto& ins : prod.postBody()) {
    if (ins.kind() == Instruction::Kind::Store && ins.buffer() == temp) {
      prodVal = ins.arg();
      break;
    }
  }
  Compute merged(
      cons.name() + "__fused_" + prod.name(),
      {},
      cons.results(),
      cons.pointwiseRanges(),
      {},
      {});
  // Merged inputs: prod inputs + cons inputs minus temp (order-preserving).
  std::vector<std::string> inputs = prod.inputs();
  for (const auto& in : cons.inputs()) {
    if (in != temp &&
        std::find(inputs.begin(), inputs.end(), in) == inputs.end()) {
      inputs.push_back(in);
    }
  }
  merged = Compute(
      cons.name() + "__fused_" + prod.name(),
      inputs,
      cons.results(),
      cons.pointwiseRanges(),
      {},
      {});
  auto emitProd = [&](const Instruction& ins) {
    switch (ins.kind()) {
      case Instruction::Kind::Constant: {
        int nid = next++;
        ren[ins.id()] = nid;
        merged.addPostInstruction(Instruction::Constant(nid, ins.value()));
        break;
      }
      case Instruction::Kind::Load: {
        int nid = next++;
        ren[ins.id()] = nid;
        merged.addPostInstruction(
            Instruction::Load(nid, ins.buffer(), ins.offset()));
        break;
      }
      case Instruction::Kind::Binary: {
        int nid = next++;
        ren[ins.id()] = nid;
        auto lit = ren.find(ins.lhs());
        auto rit = ren.find(ins.rhs());
        if (lit == ren.end() || rit == ren.end()) {
          kfail("fuse_kernels", "broken SSA chain in producer body");
        }
        merged.addPostInstruction(
            Instruction::Binary(nid, ins.op(), lit->second, rit->second));
        break;
      }
      case Instruction::Kind::Relu: {
        int nid = next++;
        ren[ins.id()] = nid;
        auto ait = ren.find(ins.arg());
        if (ait == ren.end()) {
          kfail("fuse_kernels", "broken SSA chain in producer body");
        }
        merged.addPostInstruction(Instruction::Relu(nid, ait->second));
        break;
      }
      case Instruction::Kind::Store:
        break; // drop producer store of temp
      default:
        kfail("fuse_kernels", "unexpected op in pointwise producer body");
    }
  };
  for (const auto& ins : prod.postBody()) {
    emitProd(ins);
  }
  auto prodRen = ren.find(prodVal);
  if (prodRen == ren.end()) {
    kfail("fuse_kernels", "producer value missing after rename");
  }
  int fusedVal = prodRen->second;
  // Consumer body: loads of temp become the fused value; other ids renamed
  // densely (shift by next offset to avoid collisions).
  std::map<int, int> cren;
  int shift = next;
  for (const auto& ins : cons.postBody()) {
    if (ins.kind() == Instruction::Kind::Load && ins.buffer() == temp) {
      cren[ins.id()] = fusedVal;
      continue;
    }
    switch (ins.kind()) {
      case Instruction::Kind::Constant: {
        int nid = ins.id() + shift;
        cren[ins.id()] = nid;
        merged.addPostInstruction(Instruction::Constant(nid, ins.value()));
        break;
      }
      case Instruction::Kind::Load: {
        int nid = ins.id() + shift;
        cren[ins.id()] = nid;
        merged.addPostInstruction(
            Instruction::Load(nid, ins.buffer(), ins.offset()));
        break;
      }
      case Instruction::Kind::Binary: {
        auto lit = cren.find(ins.lhs());
        auto rit = cren.find(ins.rhs());
        if (lit == cren.end() || rit == cren.end()) {
          kfail("fuse_kernels", "broken SSA chain in consumer body");
        }
        int nid = ins.id() + shift;
        cren[ins.id()] = nid;
        merged.addPostInstruction(
            Instruction::Binary(nid, ins.op(), lit->second, rit->second));
        break;
      }
      case Instruction::Kind::Relu: {
        auto ait = cren.find(ins.arg());
        if (ait == cren.end()) {
          kfail("fuse_kernels", "broken SSA chain in consumer body");
        }
        int nid = ins.id() + shift;
        cren[ins.id()] = nid;
        merged.addPostInstruction(Instruction::Relu(nid, ait->second));
        break;
      }
      case Instruction::Kind::Store: {
        auto ait = cren.find(ins.arg());
        if (ait == cren.end()) {
          kfail("fuse_kernels", "store of undefined value after fusion");
        }
        merged.addPostInstruction(
            Instruction::Store(ins.buffer(), ins.offset(), ait->second));
        break;
      }
      default:
        kfail("fuse_kernels", "unexpected op in pointwise consumer body");
    }
  }
  return KernelFunction(merged.name(), {merged}, numerics);
}

} // namespace

std::vector<Buffer> KernelProgram::buffers() const {
  std::vector<Buffer> out = inputs_;
  for (const auto& n : nodes_) {
    if (const auto* f = std::get_if<KernelFunction>(&n)) {
      for (const auto& reg : f->regions()) {
        for (const auto& b : reg.results()) {
          out.push_back(b);
        }
      }
    } else {
      for (const auto& b : std::get_if<ExternCall>(&n)->results()) {
        out.push_back(b);
      }
    }
  }
  return out;
}

void KernelProgram::buildDependencies() {
  dependencies_.clear();
  dependencies_.resize(nodes_.size());
  std::map<std::string, View> viewMap;
  for (const auto& v : views_) {
    viewMap[v.name] = v;
  }
  auto rootOf = [&](const std::string& name) {
    std::string cur = name;
    std::set<std::string> seen;
    while (viewMap.count(cur)) {
      if (!seen.insert(cur).second) {
        break;
      }
      cur = viewMap[cur].source;
    }
    return cur;
  };
  for (size_t i = 0; i < nodes_.size(); ++i) {
    std::set<size_t> deps;
    for (const auto& r : nodeReads(nodes_[i])) {
      int w = writerOf(nodes_, rootOf(r), inputs_);
      if (w >= 0 && (size_t)w != i) {
        deps.insert((size_t)w);
      }
    }
    // Conservative extern barriers: extern effects are unknown and
    // may_alias_inputs defaults true, so an extern may observe or mutate
    // any live buffer. Every extern depends on ALL prior nodes (not just
    // prior externs), and every node after an extern depends on that
    // extern. This is the DAG contract, not runtime launch order: without
    // it a data-independent kernel could legally reorder across an extern
    // that aliases its buffers. (Optimizations that partition disjoint
    // buffers come later; they must prove non-aliasing first.)
    if (std::get_if<ExternCall>(&nodes_[i])) {
      for (size_t j = 0; j < i; ++j) {
        deps.insert(j);
      }
    } else {
      for (size_t j = 0; j < i; ++j) {
        if (std::get_if<ExternCall>(&nodes_[j])) {
          deps.insert(j);
        }
      }
    }
    dependencies_[i] = std::vector<size_t>(deps.begin(), deps.end());
    std::sort(dependencies_[i].begin(), dependencies_[i].end());
  }
}

void KernelProgram::verify() const {
  std::string pctx = "kernel_program '" + name_ + "'";
  if (name_.empty()) {
    kfail(pctx, "empty program name");
  }
  if (!isValidNumerics(numerics_)) {
    kfail(
        pctx,
        "unsupported numerics '" + numerics_ +
            "' (allowed: 'strict_eager', 'inductor')");
  }
  for (const auto& n : nodes_) {
    if (const auto* f = std::get_if<KernelFunction>(&n)) {
      if (f->numerics() != numerics_) {
        kfail(
            pctx,
            "kernel function '" + f->name() + "' numerics '" + f->numerics() +
                "' does not match program numerics '" + numerics_ + "'");
      }
    }
  }
  // Rebuild a logical Program for structural verification, then check
  // kernel-level invariants (single region, pointwise-only, deps shape).
  // Views are declarative metadata in KernelIR (insertion order is NOT
  // execution order), so each view is emitted when its source becomes
  // available in node order: program inputs first, then computed/extern
  // results as nodes are visited, then remaining views whose sources are
  // earlier views (cycle/unavailable source is an error). A view of a
  // computed result is valid and must verify.
  Program logical(name_, numerics_);
  for (const auto& b : inputs_) {
    logical.addInput(b);
  }
  for (const auto& d : input_descriptors_) {
    logical.addInputDescriptor(d);
  }
  std::set<std::string> available;
  for (const auto& b : inputs_) {
    available.insert(b.name);
  }
  std::vector<View> pending = views_;
  auto emitReadyViews = [&](const std::string& ctx) {
    bool progressed = true;
    while (progressed) {
      progressed = false;
      for (auto it = pending.begin(); it != pending.end();) {
        if (available.count(it->source)) {
          logical.addView(*it);
          available.insert(it->name);
          it = pending.erase(it);
          progressed = true;
        } else {
          ++it;
        }
      }
    }
    (void)ctx;
  };
  emitReadyViews(pctx);
  for (const auto& n : nodes_) {
    if (const auto* f = std::get_if<KernelFunction>(&n)) {
      if (f->name().empty()) {
        kfail(pctx, "kernel function with empty name");
      }
      if (f->regions().size() != 1) {
        kfail(
            pctx,
            "kernel function '" + f->name() + "' has " +
                std::to_string(f->regions().size()) +
                " regions (exactly 1 required; parallel regions not implemented)");
      }
      checkKernelRegion(f->regions()[0], pctx);
      logical.addCompute(f->regions()[0]);
      for (const auto& b : f->regions()[0].results()) {
        available.insert(b.name);
      }
    } else {
      logical.addExternCall(*std::get_if<ExternCall>(&n));
      for (const auto& b : std::get_if<ExternCall>(&n)->results()) {
        available.insert(b.name);
      }
    }
    emitReadyViews(pctx);
  }
  if (!pending.empty()) {
    kfail(
        pctx,
        "view '" + pending.front().name + "' source '" +
            pending.front().source + "' never becomes available");
  }
  for (const auto& d : outputs_) {
    logical.addOutput(d);
  }
  logical.verify();
  if (dependencies_.size() != nodes_.size()) {
    kfail(pctx, "dependencies size mismatch (call buildDependencies)");
  }
  for (size_t i = 0; i < dependencies_.size(); ++i) {
    for (size_t d : dependencies_[i]) {
      if (d >= nodes_.size()) {
        kfail(pctx, "dependency out of range");
      }
      if (d == i) {
        kfail(pctx, "self dependency");
      }
    }
  }
}

std::string KernelProgram::dump() const {
  verify();
  std::ostringstream os;
  os << "kernel_program " << name_ << " numerics=" << numerics_ << "\n";
  os << "inputs\n";
  for (const auto& b : inputs_) {
    os << "  buffer " << b.name << " " << b.dtype << " " << b.elements << "\n";
  }
  for (const auto& n : nodes_) {
    if (const auto* f = std::get_if<KernelFunction>(&n)) {
      os << "kernel_function " << f->name()
         << " regions=" << f->regions().size() << "\n";
      for (const auto& reg : f->regions()) {
        os << "  region " << reg.name()
           << " results=" << buffersStrK(reg.results()) << "\n";
        for (const auto& ins : reg.postBody()) {
          os << "    ";
          switch (ins.kind()) {
            case Instruction::Kind::Constant:
              os << "v" << ins.id() << " = constant "
                 << formatFloat(ins.value());
              break;
            case Instruction::Kind::Load:
              os << "v" << ins.id() << " = load " << ins.buffer() << "["
                 << ins.offset().toString() << "]";
              break;
            case Instruction::Kind::Binary:
              os << "v" << ins.id() << " = " << ins.op() << " v" << ins.lhs()
                 << " v" << ins.rhs();
              break;
            case Instruction::Kind::Relu:
              os << "v" << ins.id() << " = relu v" << ins.arg();
              break;
            case Instruction::Kind::Store:
              os << "store " << ins.buffer() << "[" << ins.offset().toString()
                 << "] v" << ins.arg();
              break;
            default:
              os << ins.kindName();
              break;
          }
          os << "\n";
        }
      }
    } else {
      const auto* e = std::get_if<ExternCall>(&n);
      os << "extern " << e->name() << " target=" << e->target()
         << " results=" << buffersStrK(e->results())
         << " effects=" << e->effects() << "\n";
    }
  }
  os << "dependencies\n";
  for (size_t i = 0; i < dependencies_.size(); ++i) {
    os << "  node" << i << " <-"
       << vecStrK(
              std::vector<int64_t>(
                  dependencies_[i].begin(), dependencies_[i].end()))
       << "\n";
  }
  os << "outputs\n";
  for (const auto& d : outputs_) {
    os << "  descriptor " << d.name << " -> " << d.buffer
       << " sizes=" << vecStrK(d.sizes) << " strides=" << vecStrK(d.strides)
       << " offset=" << d.offset << "\n";
  }
  return os.str();
}

KernelProgram lower_to_kernels(const Program& program) {
  program.verify();
  KernelProgram out(program.name(), program.numerics());
  for (const auto& b : program.inputs()) {
    out.addInput(b);
  }
  for (const auto& d : program.inputDescriptors()) {
    out.addInputDescriptor(d);
  }
  for (const auto& n : program.nodes()) {
    if (const auto* v = std::get_if<View>(&n)) {
      out.addView(*v);
    } else if (const auto* c = std::get_if<Compute>(&n)) {
      if (!c->reductionRanges().empty() || !c->accumulators().empty() ||
          !c->reductionBody().empty()) {
        kfail(
            "lower_to_kernels",
            "compute '" + c->name() +
                "' has reductions (executable lowering requires eager extern)");
      }
      checkKernelRegion(*c, "lower_to_kernels");
      out.addFunction(KernelFunction(c->name(), {*c}, program.numerics()));
    } else {
      out.addExternCall(*std::get_if<ExternCall>(&n));
    }
  }
  for (const auto& d : program.outputs()) {
    out.addOutput(d);
  }
  out.buildDependencies();
  out.verify();
  return out;
}

KernelProgram fuse_kernels(const KernelProgram& program) {
  program.verify();
  // Greedy single-use adjacent pointwise chain fusion, iterated to a stable
  // fixed point in ONE call: each round scans left to right fusing disjoint
  // adjacent pairs (same conservative legality as a single round), then
  // re-scans the shortened list. Rounds stop when a full round fuses
  // nothing, so the result is idempotent: fuse(fuse(p)) == fuse(p).
  // Never crosses extern barriers (externs split the fusion window).
  // Outputs and dependency DAG are rebuilt deterministically at the end.
  std::map<std::string, View> viewMap;
  for (const auto& v : program.views()) {
    viewMap[v.name] = v;
  }
  std::vector<KernelProgram::Node> nodes = program.nodes();
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<KernelProgram::Node> next;
    size_t i = 0;
    while (i < nodes.size()) {
      const auto* fi = std::get_if<KernelFunction>(&nodes[i]);
      const KernelFunction* fj = nullptr;
      size_t j = i + 1;
      if (fi && j < nodes.size()) {
        fj = std::get_if<KernelFunction>(&nodes[j]);
      }
      if (fi && fj && fi->regions().size() == 1 && fj->regions().size() == 1) {
        std::string temp;
        if (fusiblePair(
                fi->regions()[0],
                fj->regions()[0],
                nodes,
                program.outputs(),
                viewMap,
                i,
                j,
                temp)) {
          next.push_back(fusePair(
              fi->regions()[0], fj->regions()[0], temp, program.numerics()));
          changed = true;
          i = j + 1;
          continue;
        }
      }
      next.push_back(nodes[i]);
      ++i;
    }
    nodes = std::move(next);
  }
  KernelProgram out(program.name(), program.numerics());
  for (const auto& b : program.inputs()) {
    out.addInput(b);
  }
  for (const auto& d : program.inputDescriptors()) {
    out.addInputDescriptor(d);
  }
  for (const auto& v : program.views()) {
    out.addView(v);
  }
  for (const auto& n : nodes) {
    if (const auto* f = std::get_if<KernelFunction>(&n)) {
      out.addFunction(*f);
    } else {
      out.addExternCall(*std::get_if<ExternCall>(&n));
    }
  }
  for (const auto& d : program.outputs()) {
    out.addOutput(d);
  }
  out.buildDependencies();
  // Preserve original output descriptors verbatim; removal of a temp is
  // only valid when it escapes nowhere (checked in fusiblePair).
  out.verify();
  return out;
}

} // namespace torch::inductor::loops_ir
