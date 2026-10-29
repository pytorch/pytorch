// Typed-analysis module implementation.
//
// Carries the canonical semantic key serializer (declarations in
// analyses.h) plus the build anchor. Includes tiled_ir.h for
// TiledProgram keys.

#include "analyses.h"

#include <cstring>

#include "tiled_ir.h"

namespace torch::inductor::loops_ir::analyses {

// Build anchor: ensures this TU compiles standalone once the loader lists
// it. Returns the number of typed analysis kinds (6); never used at
// runtime by passes (they call the inline helpers directly).
int analyses_build_anchor() {
  return 6;
}

} // namespace torch::inductor::loops_ir::analyses

// --- Canonical semantic keys (review F1) ---
//
// Moved here from canonical_keys.cpp so the implementation ships in
// the loader-listed analyses.cpp TU. Declarations live in analyses.h.

namespace torch::inductor::loops_ir {
namespace analyses {
namespace {

class KeyWriter {
 public:
  void tag(const char* t) {
    out_ += t;
    out_ += '\x01';
  }
  void str(const std::string& s) {
    out_ += "s" + std::to_string(s.size()) + ":";
    out_ += s;
    out_ += ";";
  }
  void i64(int64_t v) {
    out_ += "i" + std::to_string(v) + ";";
  }
  void boolean(bool v) {
    out_ += v ? "t;" : "f;";
  }
  void bitExactDouble(double v) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(v), "double size");
    std::memcpy(&bits, &v, sizeof(bits));
    out_ += "d" + toHex16(bits >> 32) + toHex16(bits & 0xFFFFFFFFULL) + ";";
  }
  void vecI64(const std::vector<int64_t>& v) {
    out_ += "v" + std::to_string(v.size()) + ":";
    for (int64_t x : v) {
      out_ += std::to_string(x) + ",";
    }
    out_ += ";";
  }
  const std::string& str() const {
    return out_;
  }

 private:
  std::string out_;
};

void writeIndex(KeyWriter& w, const Index& idx) {
  if (idx.empty()) {
    w.tag("index.empty");
    return;
  }
  const IndexExpr& e = idx.expr();
  switch (e.kind) {
    case IndexExpr::Kind::Constant:
      w.tag("index.const");
      w.i64(e.value);
      return;
    case IndexExpr::Kind::Variable:
      w.tag("index.var");
      w.str(e.name);
      return;
    case IndexExpr::Kind::Binary:
      w.tag("index.bin");
      w.str(e.op);
      writeIndex(w, Index(e.lhs));
      writeIndex(w, Index(e.rhs));
      return;
  }
}

void writeBuffer(KeyWriter& w, const Buffer& b) {
  w.tag("buffer");
  w.str(b.name);
  w.str(b.dtype);
  w.i64(b.elements);
}

void writeDescriptor(KeyWriter& w, const Descriptor& d) {
  w.tag("descriptor");
  w.str(d.name);
  w.str(d.buffer);
  w.vecI64(d.sizes);
  w.vecI64(d.strides);
  w.i64(d.offset);
}

void writeView(KeyWriter& w, const View& v) {
  w.tag("view");
  w.str(v.name);
  w.str(v.source);
  w.vecI64(v.sizes);
  w.vecI64(v.strides);
  w.i64(v.offset);
}

void writeRange(KeyWriter& w, const Range& r) {
  w.tag("range");
  w.str(r.name);
  w.i64(r.start);
  w.i64(r.stop);
}

void writeAccumulator(KeyWriter& w, const Accumulator& a) {
  w.tag("accumulator");
  w.str(a.name);
  w.str(a.dtype);
  w.bitExactDouble(a.identity);
  w.str(a.combine);
}

void writeCallArg(KeyWriter& w, const CallArg& a) {
  w.tag("callarg");
  w.str(a.kindName());
  switch (a.kind()) {
    case CallArg::Kind::None:
      return;
    case CallArg::Kind::Tensor:
      w.str(a.tensorName());
      return;
    case CallArg::Kind::Int:
      w.i64(a.intValue());
      return;
    case CallArg::Kind::Float:
      w.bitExactDouble(a.floatValue());
      return;
    case CallArg::Kind::Bool:
      w.boolean(a.boolValue());
      return;
    case CallArg::Kind::String:
    case CallArg::Kind::Dtype:
    case CallArg::Kind::Layout:
    case CallArg::Kind::MemoryFormat:
      w.str(a.text());
      return;
    case CallArg::Kind::List:
    case CallArg::Kind::Tuple:
      w.i64((int64_t)a.items().size());
      for (const auto& it : a.items()) {
        writeCallArg(w, it);
      }
      return;
  }
}

void writeInstruction(KeyWriter& w, const Instruction& ins) {
  w.tag("insn");
  w.str(ins.kindName());
  w.i64(ins.id());
  switch (ins.kind()) {
    case Instruction::Kind::Constant:
      w.bitExactDouble(ins.value());
      return;
    case Instruction::Kind::Load:
      w.str(ins.buffer());
      writeIndex(w, ins.offset());
      return;
    case Instruction::Kind::ViewRead:
      w.str(ins.view());
      w.i64((int64_t)ins.coordinates().size());
      for (const auto& c : ins.coordinates()) {
        writeIndex(w, c);
      }
      return;
    case Instruction::Kind::Binary:
      w.str(ins.op());
      w.i64(ins.lhs());
      w.i64(ins.rhs());
      return;
    case Instruction::Kind::Relu:
      w.i64(ins.arg());
      return;
    case Instruction::Kind::Accumulate:
      w.i64((int64_t)ins.args().size());
      for (int a : ins.args()) {
        w.i64(a);
      }
      return;
    case Instruction::Kind::ReadAccumulator:
      w.i64(ins.accumulator());
      return;
    case Instruction::Kind::Store:
      w.str(ins.buffer());
      writeIndex(w, ins.offset());
      w.i64(ins.arg());
      return;
  }
}

void writeCompute(KeyWriter& w, const Compute& c) {
  w.tag("compute");
  w.str(c.name());
  w.i64((int64_t)c.inputs().size());
  for (const auto& in : c.inputs()) {
    w.str(in);
  }
  w.i64((int64_t)c.results().size());
  for (const auto& b : c.results()) {
    writeBuffer(w, b);
  }
  w.i64((int64_t)c.pointwiseRanges().size());
  for (const auto& r : c.pointwiseRanges()) {
    writeRange(w, r);
  }
  w.i64((int64_t)c.reductionRanges().size());
  for (const auto& r : c.reductionRanges()) {
    writeRange(w, r);
  }
  w.i64((int64_t)c.accumulators().size());
  for (const auto& a : c.accumulators()) {
    writeAccumulator(w, a);
  }
  w.i64((int64_t)c.reductionBody().size());
  for (const auto& ins : c.reductionBody()) {
    writeInstruction(w, ins);
  }
  w.i64((int64_t)c.postBody().size());
  for (const auto& ins : c.postBody()) {
    writeInstruction(w, ins);
  }
}

void writeExtern(KeyWriter& w, const ExternCall& e) {
  w.tag("extern");
  w.str(e.name());
  w.str(e.target());
  w.i64((int64_t)e.inputs().size());
  for (const auto& in : e.inputs()) {
    w.str(in);
  }
  w.i64((int64_t)e.results().size());
  for (const auto& b : e.results()) {
    writeBuffer(w, b);
  }
  // Legacy scalar args: bit-exact doubles (the pretty printer truncates
  // these to f32 text, which is exactly the collision review F2 forbids
  // in cache keys).
  w.i64((int64_t)e.scalarArgs().size());
  for (double s : e.scalarArgs()) {
    w.bitExactDouble(s);
  }
  w.str(e.effects());
  w.boolean(e.mayAliasInputs());
  w.i64((int64_t)e.descriptors().size());
  for (const auto& d : e.descriptors()) {
    writeDescriptor(w, d);
  }
  w.i64((int64_t)e.arguments().size());
  for (const auto& a : e.arguments()) {
    writeCallArg(w, a);
  }
  w.i64((int64_t)e.keywordArgs().size());
  for (const auto& k : e.keywordArgs()) {
    w.str(k.name());
    writeCallArg(w, k.value());
  }
}

} // namespace

std::string canonicalKeyOf(const Program& program) {
  program.verify();
  KeyWriter w;
  w.tag("program");
  w.str(program.name());
  w.str(program.numerics());
  w.i64((int64_t)program.inputs().size());
  for (const auto& b : program.inputs()) {
    writeBuffer(w, b);
  }
  w.i64((int64_t)program.inputDescriptors().size());
  for (const auto& d : program.inputDescriptors()) {
    writeDescriptor(w, d);
  }
  w.i64((int64_t)program.nodes().size());
  for (const auto& n : program.nodes()) {
    if (const auto* v = std::get_if<View>(&n)) {
      writeView(w, *v);
    } else if (const auto* c = std::get_if<Compute>(&n)) {
      writeCompute(w, *c);
    } else {
      writeExtern(w, *std::get_if<ExternCall>(&n));
    }
  }
  w.i64((int64_t)program.outputs().size());
  for (const auto& d : program.outputs()) {
    writeDescriptor(w, d);
  }
  return w.str();
}

std::string canonicalKeyOf(const KernelProgram& program) {
  program.verify();
  KeyWriter w;
  w.tag("kernel_program");
  w.str(program.name());
  w.str(program.numerics());
  w.i64((int64_t)program.inputs().size());
  for (const auto& b : program.inputs()) {
    writeBuffer(w, b);
  }
  w.i64((int64_t)program.inputDescriptors().size());
  for (const auto& d : program.inputDescriptors()) {
    writeDescriptor(w, d);
  }
  // Views are declarative metadata with full typed fields (review F1:
  // the pretty dump omits them, which caused the cache hole).
  w.i64((int64_t)program.views().size());
  for (const auto& v : program.views()) {
    writeView(w, v);
  }
  w.i64((int64_t)program.nodes().size());
  for (const auto& n : program.nodes()) {
    if (const auto* f = std::get_if<KernelFunction>(&n)) {
      w.tag("kernel_function");
      w.str(f->name());
      w.str(f->numerics());
      w.i64((int64_t)f->regions().size());
      for (const auto& r : f->regions()) {
        writeCompute(w, r);
      }
    } else {
      writeExtern(w, *std::get_if<ExternCall>(&n));
    }
  }
  w.i64((int64_t)program.outputs().size());
  for (const auto& d : program.outputs()) {
    writeDescriptor(w, d);
  }
  // Dependency edges are part of the semantic state (review F1).
  w.i64((int64_t)program.dependencies().size());
  for (const auto& deps : program.dependencies()) {
    w.i64((int64_t)deps.size());
    for (size_t d : deps) {
      w.i64((int64_t)d);
    }
  }
  return w.str();
}

std::string canonicalKeyOf(const TiledProgram& program) {
  program.verify();
  KeyWriter w;
  w.tag("tiled_program");
  w.str(program.numerics());
  w.i64((int64_t)program.functions().size());
  for (const auto& f : program.functions()) {
    w.tag("tiled_function");
    w.str(f.name());
    w.str(f.numerics());
    w.i64((int64_t)f.parameters().size());
    for (const auto& p : f.parameters()) {
      writeBuffer(w, p);
    }
    w.i64(f.grid_size());
    w.i64(f.block_size());
    w.i64(f.num_warps());
    w.boolean(f.enable_fp_fusion());
    w.boolean(f.disable_ftz());
    w.i64((int64_t)f.blocks().size());
    for (const auto& b : f.blocks()) {
      w.tag("tiled_block");
      w.str(b.name());
      w.str(b.terminator());
      w.i64((int64_t)b.instructions().size());
      for (const auto& ins : b.instructions()) {
        w.tag("tiled_insn");
        w.str(ins.op());
        w.i64(ins.id());
        w.str(ins.dtype());
        w.vecI64(ins.shape());
        w.str(ins.buffer());
        w.i64((int64_t)ins.operands().size());
        for (int o : ins.operands()) {
          w.i64(o);
        }
        w.i64(ins.int_value());
        w.bitExactDouble(ins.float_value());
      }
    }
  }
  return w.str();
}

std::string canonicalKeyOf(const Compute& compute) {
  KeyWriter w;
  writeCompute(w, compute);
  return w.str();
}

std::string fingerprintOf(const TiledProgram& program) {
  return "tiled:" + program.numerics() + ":" +
      toHex16(fnv1a64(canonicalKeyOf(program)));
}

} // namespace analyses

} // namespace torch::inductor::loops_ir
