#include "fused_region.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace torch::inductor::loops_ir::op_options {

namespace {

struct Affine {
  int64_t constant = 0;
  std::map<std::string, int64_t> coeffs;
  bool hasVariables() const {
    for (const auto& kv : coeffs) {
      if (kv.second != 0) {
        return true;
      }
    }
    return false;
  }
};

Affine affineOf(const IndexExpr& e, const std::string& ctx) {
  Affine out;
  switch (e.kind) {
    case IndexExpr::Kind::Constant:
      out.constant = e.value;
      return out;
    case IndexExpr::Kind::Variable:
      out.coeffs[e.name] = 1;
      return out;
    case IndexExpr::Kind::Binary:
      break;
  }
  if (e.op != "add" && e.op != "mul") {
    reject(RejectReason::Fusion, ctx + ": non-affine index op '" + e.op + "' not supported by the fusion contract");
  }
  const Affine l = affineOf(*e.lhs, ctx);
  const Affine r = affineOf(*e.rhs, ctx);
  if (e.op == "add") {
    out = l;
    out.constant = checkedAdd64(out.constant, r.constant, ctx);
    for (const auto& kv : r.coeffs) {
      out.coeffs[kv.first] = checkedAdd64(out.coeffs[kv.first], kv.second, ctx);
    }
    return out;
  }
  if (l.hasVariables() && r.hasVariables()) {
    reject(RejectReason::Fusion, ctx + ": product of index variables is not affine");
  }
  const Affine& var = l.hasVariables() ? l : r;
  const int64_t scale = l.hasVariables() ? r.constant : l.constant;
  out.constant = checkedMul64(var.constant, scale, ctx);
  for (const auto& kv : var.coeffs) {
    out.coeffs[kv.first] = checkedMul64(kv.second, scale, ctx);
  }
  return out;
}

AccessMap toAccessMap(const Index& idx,
                      const std::vector<Range>& ranges,
                      const std::vector<int64_t>& domain,
                      const std::string& ctx) {
  Affine a = affineOf(idx.expr(), ctx);
  AccessMap m;
  m.offset = a.constant;
  m.coeffs.assign(domain.size(), 0);
  for (size_t d = 0; d < ranges.size(); ++d) {
    auto it = a.coeffs.find(ranges[d].name);
    if (it != a.coeffs.end()) {
      m.coeffs[d] = domain[d] > 1 ? it->second : 0;
      a.coeffs.erase(it);
    }
  }
  if (a.hasVariables()) {
    reject(RejectReason::Fusion, ctx + ": index uses a variable outside the region domain");
  }
  return m;
}

bool isBroadcast(const AccessMap& m, const std::vector<int64_t>& domain) {
  for (size_t d = 0; d < domain.size(); ++d) {
    if (domain[d] > 1 && m.coeffs[d] == 0) {
      return true;
    }
  }
  return false;
}

struct RegionScope {
  std::string self_name;
  AccessMap self_map;
  int64_t self_elements = 0;
  AccessMap store_map;
  std::vector<int64_t> domain;
  const std::vector<ExtraInput>* extras = nullptr;
  std::string ctx;
};

AccessMap withOffset(AccessMap m, int64_t offset) {
  m.offset = offset;
  return m;
}

int64_t extraSlot(const RegionScope& scope, const std::string& name) {
  for (size_t i = 0; i < scope.extras->size(); ++i) {
    if ((*scope.extras)[i].name == name) {
      return static_cast<int64_t>(i);
    }
  }
  return -1;
}

void verifyWithCore(const RegionScope& scope, const Compute& body) {
  Program p("fusion_check");
  for (const std::string& in : body.inputs()) {
    const int64_t slot = extraSlot(scope, in);
    if (in == scope.self_name) {
      p.addInput(Buffer(in, "f32", scope.self_elements));
    } else if (slot >= 0) {
      const ExtraInput& e = (*scope.extras)[static_cast<size_t>(slot)];
      p.addInput(Buffer(e.name, coreDtypeName(e.layout.dtype), e.layout.extent()));
    } else {
      reject(RejectReason::Fusion, scope.ctx + ": region input '" + in + "' is neither the fused value nor an extra input");
    }
  }
  p.addCompute(body);
  try {
    p.verify();
  } catch (const OptionRejected&) {
    throw;
  } catch (const std::invalid_argument& err) {
    reject(RejectReason::Fusion, scope.ctx + ": core LoopsIR verification failed: " + err.what());
  }
}

void checkRegionShape(const RegionScope& scope, const Compute& body) {
  if (!body.reductionRanges().empty() || !body.accumulators().empty() || !body.reductionBody().empty()) {
    reject(RejectReason::Fusion, scope.ctx + ": reduction inside a fused pointwise region is not supported");
  }
  const auto& ranges = body.pointwiseRanges();
  if (ranges.size() != scope.domain.size()) {
    reject(RejectReason::Fusion, scope.ctx + ": region rank does not match the fused domain");
  }
  for (size_t d = 0; d < ranges.size(); ++d) {
    if (ranges[d].start != 0 || ranges[d].stop != scope.domain[d]) {
      reject(RejectReason::Fusion, scope.ctx + ": region range '" + ranges[d].name + "' does not cover the fused domain");
    }
  }
  if (body.results().size() != 1) {
    reject(RejectReason::Fusion, scope.ctx + ": region must define exactly one result");
  }
}

RegionOp lowerLoad(const RegionScope& scope, const Compute& body, const Instruction& ins, const std::string& ictx) {
  RegionOp op;
  op.id = ins.id();
  const AccessMap map = toAccessMap(ins.offset(), body.pointwiseRanges(), scope.domain, ictx);
  if (ins.buffer() == scope.self_name) {
    if (!(map == scope.self_map)) {
      reject(RejectReason::Fusion, ictx + ": fused value read at a non-identity access map (not available in the tile)");
    }
    op.kind = RegionOpKind::LoadSelf;
    return op;
  }
  op.kind = RegionOpKind::LoadExtra;
  op.extra = extraSlot(scope, ins.buffer());
  if (op.extra < 0) {
    reject(RejectReason::Fusion, ictx + ": load of undeclared buffer '" + ins.buffer() + "'");
  }
  op.access = map;
  return op;
}

LoweredRegion lowerRegion(const RegionScope& scope, const Compute& body) {
  checkRegionShape(scope, body);
  verifyWithCore(scope, body);
  LoweredRegion out;
  out.domain = scope.domain;
  const std::string& result = body.results().front().name;
  int stores = 0;
  for (size_t k = 0; k < body.postBody().size(); ++k) {
    const Instruction& ins = body.postBody()[k];
    const std::string ictx = scope.ctx + " instruction[" + std::to_string(k) + "] " + ins.kindName();
    RegionOp op;
    op.id = ins.id();
    switch (ins.kind()) {
      case Instruction::Kind::Constant:
        op.kind = RegionOpKind::Constant;
        op.value = ins.value();
        break;
      case Instruction::Kind::Load:
        op = lowerLoad(scope, body, ins, ictx);
        break;
      case Instruction::Kind::Binary:
        op.kind = ins.op() == "add" ? RegionOpKind::Add : RegionOpKind::Mul;
        op.lhs = ins.lhs();
        op.rhs = ins.rhs();
        break;
      case Instruction::Kind::Relu:
        op.kind = RegionOpKind::Relu;
        op.lhs = ins.arg();
        break;
      case Instruction::Kind::Store: {
        const AccessMap map = toAccessMap(ins.offset(), body.pointwiseRanges(), scope.domain, ictx);
        if (ins.buffer() != result || !(map == scope.store_map)) {
          reject(RejectReason::Fusion, ictx + ": store must write the region result at the fused layout map");
        }
        out.store_value = ins.arg();
        ++stores;
        continue;
      }
      case Instruction::Kind::ViewRead:
      case Instruction::Kind::Accumulate:
      case Instruction::Kind::ReadAccumulator:
        reject(RejectReason::Fusion, ictx + ": instruction kind not fusible into a tile region");
    }
    out.ops.push_back(op);
  }
  if (stores != 1) {
    reject(RejectReason::Fusion, scope.ctx + ": region must contain exactly one store");
  }
  out.store_offset = scope.store_map.offset;
  out.reads_self = std::any_of(out.ops.begin(), out.ops.end(), [](const RegionOp& op) {
    return op.kind == RegionOpKind::LoadSelf;
  });
  return out;
}

void checkContractOps(const FusionContract& contract, const LoweredRegion& region, const std::string& ctx) {
  for (const RegionOp& op : region.ops) {
    if (std::find(contract.allowed_ops.begin(), contract.allowed_ops.end(), op.kind) == contract.allowed_ops.end()) {
      reject(RejectReason::Fusion, ctx + ": op '" + regionOpKindName(op.kind) + "' not allowed by the option contract");
    }
    if (op.kind == RegionOpKind::LoadExtra && !contract.allow_broadcast && isBroadcast(op.access, region.domain)) {
      reject(RejectReason::Fusion, ctx + ": broadcast extra-input access not allowed by the option contract");
    }
  }
}

void checkExtraInputs(const FusionContract& contract, const OpSignature& sig, const FusedLoops& fused) {
  if (static_cast<int64_t>(fused.extra_inputs.size()) > contract.max_extra_inputs) {
    reject(RejectReason::Fusion, "too many extra inputs for the option contract");
  }
  std::set<std::string> names;
  for (const auto& in : sig.inputs()) {
    names.insert(in.name);
  }
  names.insert(sig.output().name);
  if (fused.epilogue) {
    if (!names.insert(fused.epilogue->op_result).second) {
      reject(RejectReason::Fusion, "epilogue op_result name collides with an operand");
    }
  }
  for (const auto& e : fused.extra_inputs) {
    e.layout.validate("extra input '" + e.name + "'");
    if (e.name.empty() || !names.insert(e.name).second) {
      reject(RejectReason::Fusion, "extra input '" + e.name + "' has an empty or colliding name");
    }
    if (e.layout.shape_kind != ShapeKind::Static) {
      reject(RejectReason::DynamicShape, "extra input '" + e.name + "' has a dynamic shape");
    }
    if (e.layout.dtype != StorageDtype::F32) {
      reject(RejectReason::Dtype, "extra input '" + e.name + "' must be f32");
    }
  }
}

void checkExtrasUsed(const ValidatedFusion& out) {
  std::vector<bool> used(out.extra_inputs.size(), false);
  auto mark = [&](const LoweredRegion& r) {
    for (const auto& op : r.ops) {
      if (op.kind == RegionOpKind::LoadExtra) {
        used[op.extra] = true;
      }
    }
  };
  for (const auto& p : out.prologues) {
    mark(p.region);
  }
  if (out.epilogue) {
    mark(*out.epilogue);
  }
  for (size_t i = 0; i < used.size(); ++i) {
    if (!used[i]) {
      reject(RejectReason::Fusion, "extra input '" + out.extra_inputs[i].name + "' is never loaded");
    }
  }
}

std::string accessText(const AccessMap& m) {
  std::ostringstream os;
  os << m.offset;
  for (int64_t c : m.coeffs) {
    os << "," << c;
  }
  return os.str();
}

} // namespace

const char* regionOpKindName(RegionOpKind kind) {
  switch (kind) {
    case RegionOpKind::LoadSelf:
      return "load_self";
    case RegionOpKind::LoadExtra:
      return "load_extra";
    case RegionOpKind::Constant:
      return "constant";
    case RegionOpKind::Add:
      return "add";
    case RegionOpKind::Mul:
      return "mul";
    case RegionOpKind::Relu:
      return "relu";
  }
  return "?";
}

AccessMap affineAccessMap(const Index& idx,
                          const std::vector<Range>& ranges,
                          const std::vector<int64_t>& domain,
                          const std::string& ctx) {
  return toAccessMap(idx, ranges, domain, ctx);
}

AccessMap selfAccessMap(const TensorLayout& layout) {
  AccessMap m;
  for (size_t d = 0; d < layout.rank(); ++d) {
    m.coeffs.push_back(layout.sizes[d] > 1 ? layout.strides[d] : 0);
  }
  return m;
}

std::string LoweredRegion::canonical() const {
  std::ostringstream os;
  os << "domain[";
  for (int64_t d : domain) {
    os << d << ",";
  }
  os << "]";
  for (const auto& op : ops) {
    os << " v" << op.id << "=" << regionOpKindName(op.kind) << "(" << op.lhs << "," << op.rhs << ",x" << op.extra
       << ",m" << accessText(op.access) << ",c0x" << f64Bits(op.value) << ")";
  }
  os << " store v" << store_value << "@" << store_offset << " reads_self=" << reads_self;
  return os.str();
}

const LoweredRegion* ValidatedFusion::prologueFor(int64_t operand) const {
  for (const auto& p : prologues) {
    if (p.operand == operand) {
      return &p.region;
    }
  }
  return nullptr;
}

std::string ValidatedFusion::canonical() const {
  std::ostringstream os;
  os << "extras{";
  for (const auto& e : extra_inputs) {
    os << e.name << ":" << e.layout.canonical() << ";";
  }
  os << "}";
  for (const auto& p : prologues) {
    os << " prologue[" << p.operand << "]{" << p.region.canonical() << "}";
  }
  if (epilogue) {
    os << " epilogue{" << epilogue->canonical() << "}";
  }
  return os.str();
}

std::string FusionContract::canonical() const {
  std::ostringstream os;
  os << "prologue_operands[";
  for (int64_t o : prologue_operands) {
    os << o << ",";
  }
  os << "] epilogue=" << epilogue << " ops[";
  for (auto k : allowed_ops) {
    os << regionOpKindName(k) << ",";
  }
  os << "] max_extra=" << max_extra_inputs << " broadcast=" << allow_broadcast;
  return os.str();
}

namespace {

void checkOffsets(const FusedLoops& fused) {
  for (const auto& p : fused.prologues) {
    if (p.store_offset < 0) {
      reject(RejectReason::Fusion, "prologue store_offset must be >= 0");
    }
  }
  if (fused.epilogue && (fused.epilogue->op_result_offset < 0 || fused.epilogue->store_offset < 0)) {
    reject(RejectReason::Fusion, "epilogue offsets must be >= 0");
  }
}

} // namespace

ValidatedFusion validateFusion(const FusionContract& contract, const OpSignature& sig, const FusedLoops& fused) {
  checkOffsets(fused);
  checkExtraInputs(contract, sig, fused);
  ValidatedFusion out;
  out.extra_inputs = fused.extra_inputs;
  std::set<int64_t> seen;
  for (const auto& p : fused.prologues) {
    const std::string ctx = "prologue[" + std::to_string(p.operand) + "]";
    if (std::find(contract.prologue_operands.begin(), contract.prologue_operands.end(), p.operand) ==
        contract.prologue_operands.end()) {
      reject(RejectReason::Fusion, ctx + ": operand does not accept a prologue under this option");
    }
    if (!seen.insert(p.operand).second) {
      reject(RejectReason::Fusion, ctx + ": duplicate prologue");
    }
    const TensorOperand& operand = sig.inputs().at(static_cast<size_t>(p.operand));
    if (!operand.layout.isNonOverlapping()) {
      reject(RejectReason::Fusion, ctx + ": operand layout overlaps; the prologue store would not be injective");
    }
    const AccessMap self = selfAccessMap(operand.layout);
    RegionScope scope{
        operand.name, self, operand.layout.extent(), withOffset(self, p.store_offset), operand.layout.sizes,
        &fused.extra_inputs, ctx};
    LoweredRegion region = lowerRegion(scope, p.body);
    checkContractOps(contract, region, ctx);
    out.prologues.push_back(LoweredPrologue{p.operand, std::move(region)});
  }
  std::sort(out.prologues.begin(), out.prologues.end(), [](const auto& a, const auto& b) {
    return a.operand < b.operand;
  });
  if (fused.epilogue) {
    if (!contract.epilogue) {
      reject(RejectReason::Fusion, "epilogue: option does not accept epilogue fusion");
    }
    const EpilogueRegion& epi = *fused.epilogue;
    const TensorLayout& outLayout = sig.output().layout;
    const TensorLayout& valueLayout = epi.op_result_layout ? *epi.op_result_layout : outLayout;
    if (valueLayout.sizes != outLayout.sizes) {
      reject(RejectReason::Fusion, "epilogue: op_result_layout sizes differ from the output");
    }
    RegionScope scope{epi.op_result,
                      withOffset(selfAccessMap(valueLayout), epi.op_result_offset),
                      checkedAdd64(epi.op_result_offset, valueLayout.extent(), "epilogue"),
                      withOffset(selfAccessMap(outLayout), epi.store_offset),
                      outLayout.sizes,
                      &fused.extra_inputs,
                      "epilogue"};
    LoweredRegion region = lowerRegion(scope, epi.body);
    checkContractOps(contract, region, "epilogue");
    out.epilogue = std::move(region);
  }
  checkExtrasUsed(out);
  return out;
}

} // namespace torch::inductor::loops_ir::op_options
