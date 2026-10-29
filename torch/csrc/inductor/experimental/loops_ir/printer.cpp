// Human-readable printer. dump() verifies first; output is for reading only
// and is never parsed.

#include "ir.h"

#include <cstdio>
#include <sstream>

namespace torch::inductor::loops_ir {
namespace {

std::string joinNames(const std::vector<std::string>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += v[i];
  }
  return s;
}

std::string vecStr(const std::vector<int64_t>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += std::to_string(v[i]);
  }
  return s + "]";
}

std::string bufferStr(const Buffer& b) {
  return "(" + b.name + " " + b.dtype + " " + std::to_string(b.elements) + ")";
}

std::string buffersStr(const std::vector<Buffer>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += bufferStr(v[i]);
  }
  return s + "]";
}

std::string rangeStr(const Range& r) {
  return r.name + ":" + std::to_string(r.start) + ":" + std::to_string(r.stop);
}

std::string rangesStr(const std::vector<Range>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += rangeStr(v[i]);
  }
  return s + "]";
}

std::string accStr(const Accumulator& a) {
  return "(" + a.name + " " + a.dtype + " " + formatFloat(a.identity) + " " +
      a.combine + ")";
}

std::string accsStr(const std::vector<Accumulator>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += accStr(v[i]);
  }
  return s + "]";
}

std::string scalarsStr(const std::vector<double>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += formatFloat(v[i]);
  }
  return s + "]";
}

// Structured extern arguments preserve exact values: ints, lists, bools,
// None, dtype/layout/memory_format names, and exact-double floats
// (not f32 dumps). Infinities print as inf/-inf (exact round-trip); NaN is
// rejected at construction.
std::string printCallArg(const CallArg& a) {
  switch (a.kind()) {
    case CallArg::Kind::None:
      return "None";
    case CallArg::Kind::Tensor:
      return "tensor(" + a.tensorName() + ")";
    case CallArg::Kind::Int:
      return std::to_string(a.intValue());
    case CallArg::Kind::Float:
      return formatDoubleExact(a.floatValue());
    case CallArg::Kind::Bool:
      return a.boolValue() ? "True" : "False";
    case CallArg::Kind::String:
      return "'" + a.text() + "'";
    case CallArg::Kind::Dtype:
      return "dtype(" + a.text() + ")";
    case CallArg::Kind::Layout:
      return "layout(" + a.text() + ")";
    case CallArg::Kind::MemoryFormat:
      return "memory_format(" + a.text() + ")";
    case CallArg::Kind::List: {
      std::string s = "[";
      bool first = true;
      for (const auto& it : a.items()) {
        if (!first) {
          s += ", ";
        }
        first = false;
        s += printCallArg(it);
      }
      return s + "]";
    }
    case CallArg::Kind::Tuple: {
      std::string s = "(";
      bool first = true;
      for (const auto& it : a.items()) {
        if (!first) {
          s += ", ";
        }
        first = false;
        s += printCallArg(it);
      }
      if (a.items().size() == 1) {
        s += ",";
      }
      return s + ")";
    }
  }
  return "<unknown>";
}

std::string printKeyword(const CallKeyword& k) {
  return k.name() + "=" + printCallArg(k.value());
}

std::string printInstruction(const Instruction& ins) {
  switch (ins.kind()) {
    case Instruction::Kind::Constant:
      return "v" + std::to_string(ins.id()) + " = constant " +
          formatFloat(ins.value());
    case Instruction::Kind::Load:
      return "v" + std::to_string(ins.id()) + " = load " + ins.buffer() + "[" +
          ins.offset().toString() + "]";
    case Instruction::Kind::ViewRead: {
      std::string s =
          "v" + std::to_string(ins.id()) + " = view_read " + ins.view() + "[";
      for (size_t i = 0; i < ins.coordinates().size(); ++i) {
        if (i) {
          s += " ";
        }
        s += "c" + std::to_string(i) + "=" + ins.coordinates()[i].toString();
      }
      return s + "]";
    }
    case Instruction::Kind::Binary:
      return "v" + std::to_string(ins.id()) + " = " + ins.op() + " v" +
          std::to_string(ins.lhs()) + " v" + std::to_string(ins.rhs());
    case Instruction::Kind::Relu:
      return "v" + std::to_string(ins.id()) + " = relu v" +
          std::to_string(ins.arg());
    case Instruction::Kind::Accumulate: {
      std::string s = "accumulate";
      for (int a : ins.args()) {
        s += " v" + std::to_string(a);
      }
      return s;
    }
    case Instruction::Kind::ReadAccumulator:
      return "v" + std::to_string(ins.id()) + " = read_accumulator " +
          std::to_string(ins.accumulator());
    case Instruction::Kind::Store:
      return "store " + ins.buffer() + "[" + ins.offset().toString() + "] v" +
          std::to_string(ins.arg());
  }
  return "<unknown>";
}

} // namespace

std::string formatFloat(double v) {
  // f32 representation, deterministic round-trip precision.
  float f = static_cast<float>(v);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(f));
  return std::string(buf);
}

std::string formatDoubleExact(double v) {
  // Exact round-trip for structured extern float args (not f32-truncated).
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return std::string(buf);
}

std::string callArgToString(const CallArg& a) {
  return printCallArg(a);
}

std::string Program::dump() const {
  verify();
  std::ostringstream os;
  os << "program " << name_ << " numerics=" << numerics_ << "\n";
  os << "inputs\n";
  for (const Buffer& b : inputs_) {
    os << "  buffer " << b.name << " " << b.dtype << " " << b.elements << "\n";
  }
  os << "input_descriptors\n";
  for (const Descriptor& d : input_descriptors_) {
    os << "  descriptor " << d.name << " -> " << d.buffer
       << " sizes=" << vecStr(d.sizes) << " strides=" << vecStr(d.strides)
       << " offset=" << d.offset << "\n";
  }
  for (const Node& n : nodes_) {
    if (const auto* v = std::get_if<View>(&n)) {
      os << "view " << v->name << " source=" << v->source
         << " sizes=" << vecStr(v->sizes) << " strides=" << vecStr(v->strides)
         << " offset=" << v->offset << "\n";
    } else if (const auto* c = std::get_if<Compute>(&n)) {
      os << "compute " << c->name() << " inputs=[" << joinNames(c->inputs())
         << "] results=" << buffersStr(c->results())
         << " pointwise_ranges=" << rangesStr(c->pointwiseRanges())
         << " reduction_ranges=" << rangesStr(c->reductionRanges())
         << " accumulators=" << accsStr(c->accumulators()) << "\n";
      os << "  reduction_body\n";
      for (const Instruction& ins : c->reductionBody()) {
        os << "    " << printInstruction(ins) << "\n";
      }
      os << "  post_body\n";
      for (const Instruction& ins : c->postBody()) {
        os << "    " << printInstruction(ins) << "\n";
      }
    } else if (const auto* e = std::get_if<ExternCall>(&n)) {
      os << "extern " << e->name() << " target=" << e->target() << " inputs=["
         << joinNames(e->inputs()) << "] results=" << buffersStr(e->results())
         << " scalar_args=" << scalarsStr(e->scalarArgs())
         << " effects=" << e->effects()
         << " may_alias_inputs=" << (e->mayAliasInputs() ? "true" : "false")
         << "\n";
      // Structured typed args: exact ints/lists/bools/None/dtype, exact
      // doubles. Preserved verbatim, never opaque dumps.
      if (!e->arguments().empty() || !e->keywordArgs().empty()) {
        os << "  arguments [";
        bool first = true;
        for (const auto& a : e->arguments()) {
          if (!first) {
            os << ", ";
          }
          first = false;
          os << printCallArg(a);
        }
        os << "]";
        if (!e->keywordArgs().empty()) {
          os << " keyword_args {";
          bool kfirst = true;
          for (const auto& k : e->keywordArgs()) {
            if (!kfirst) {
              os << ", ";
            }
            kfirst = false;
            os << printKeyword(k);
          }
          os << "}";
        }
        os << "\n";
      }
      // Boundary metadata: one tensor_argument per descriptor.
      for (const Descriptor& d : e->descriptors()) {
        os << "  tensor_argument " << d.name << " -> " << d.buffer
           << " sizes=" << vecStr(d.sizes) << " strides=" << vecStr(d.strides)
           << " offset=" << d.offset << "\n";
      }
      // Opaque output acquisition: no execution claim.
      os << "  # outputs acquired opaquely by this call\n";
    }
  }
  os << "outputs\n";
  for (const Descriptor& d : outputs_) {
    os << "  descriptor " << d.name << " -> " << d.buffer
       << " sizes=" << vecStr(d.sizes) << " strides=" << vecStr(d.strides)
       << " offset=" << d.offset << "\n";
  }
  return os.str();
}

} // namespace torch::inductor::loops_ir
