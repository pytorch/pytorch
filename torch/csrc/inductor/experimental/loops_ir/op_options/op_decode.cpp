#include "op_decode.h"

#include <cmath>
#include <map>
#include <variant>

#include "program_facts.h"

namespace torch::inductor::loops_ir::op_options {

namespace {

enum class ParamType { Tensor, Scalar, DimList, Bool, OptionalDtype };

struct SchemaParam {
  const char* name;
  ParamType type;
  bool kw_only;
  bool required;
};

struct OpSchema {
  const char* target;
  OpKind kind;
  std::vector<SchemaParam> params;
};

const std::vector<OpSchema>& schemas() {
  static const std::vector<SchemaParam> reductionParams{
      {"self", ParamType::Tensor, false, true},
      {"dim", ParamType::DimList, false, true},
      {"keepdim", ParamType::Bool, false, false},
      {"dtype", ParamType::OptionalDtype, true, false},
  };
  static const std::vector<OpSchema> s{
      {"aten.mm.default", OpKind::Mm, {{"self", ParamType::Tensor, false, true}, {"mat2", ParamType::Tensor, false, true}}},
      {"aten.addmm.default",
       OpKind::Addmm,
       {{"self", ParamType::Tensor, false, true},
        {"mat1", ParamType::Tensor, false, true},
        {"mat2", ParamType::Tensor, false, true},
        {"beta", ParamType::Scalar, true, false},
        {"alpha", ParamType::Scalar, true, false}}},
      {"aten.bmm.default", OpKind::Bmm, {{"self", ParamType::Tensor, false, true}, {"mat2", ParamType::Tensor, false, true}}},
      {"aten.sum.dim_IntList", OpKind::SumDim, reductionParams},
      {"aten.mean.dim", OpKind::MeanDim, reductionParams},
  };
  return s;
}

struct DecodeFailure {
  DecodeStatus status;
  RejectReason reason;
  std::string detail;
};

[[noreturn]] void malformed(const std::string& detail) {
  throw DecodeFailure{DecodeStatus::MalformedAbi, RejectReason::Signature, detail};
}

[[noreturn]] void unsupported(RejectReason reason, const std::string& detail) {
  throw DecodeFailure{DecodeStatus::Unsupported, reason, detail};
}

using Bound = std::vector<const CallArg*>;

Bound bindArguments(const OpSchema& schema, const ExternCall& call) {
  if (call.arguments().empty() && call.keywordArgs().empty()) {
    malformed("recognized target without structured arguments");
  }
  Bound bound(schema.params.size(), nullptr);
  size_t positional = 0;
  for (const CallArg& a : call.arguments()) {
    if (positional >= schema.params.size() || schema.params[positional].kw_only) {
      malformed("too many positional arguments");
    }
    bound[positional++] = &a;
  }
  for (const CallKeyword& k : call.keywordArgs()) {
    size_t i = 0;
    while (i < schema.params.size() && k.name() != schema.params[i].name) {
      ++i;
    }
    if (i == schema.params.size()) {
      malformed("unknown keyword '" + k.name() + "'");
    }
    if (bound[i]) {
      malformed("argument '" + k.name() + "' bound twice");
    }
    bound[i] = &k.value();
  }
  for (size_t i = 0; i < schema.params.size(); ++i) {
    if (!bound[i] && schema.params[i].required) {
      malformed(std::string("missing argument '") + schema.params[i].name + "'");
    }
  }
  return bound;
}

double scalarValue(const CallArg* a, double def, const char* name) {
  if (!a) {
    return def;
  }
  if (a->kind() == CallArg::Kind::Float) {
    return a->floatValue();
  }
  if (a->kind() == CallArg::Kind::Int) {
    const int64_t v = a->intValue();
    if (v > (int64_t{1} << 53) || v < -(int64_t{1} << 53)) {
      unsupported(RejectReason::Signature, std::string(name) + ": integer Scalar not exactly representable as f64");
    }
    return static_cast<double>(v);
  }
  if (a->kind() == CallArg::Kind::Bool) {
    unsupported(RejectReason::Signature, std::string(name) + ": bool Scalar not supported");
  }
  malformed(std::string(name) + ": expected a Scalar");
}

std::vector<int64_t> dimList(const CallArg* a, size_t rank) {
  std::vector<int64_t> dims;
  if (a->kind() == CallArg::Kind::None) {
    for (size_t d = 0; d < rank; ++d) {
      dims.push_back(static_cast<int64_t>(d));
    }
    return dims;
  }
  if (a->kind() == CallArg::Kind::Int) {
    return {a->intValue()};
  }
  if (a->kind() != CallArg::Kind::List && a->kind() != CallArg::Kind::Tuple) {
    malformed("dim: expected int[] or None");
  }
  for (const CallArg& item : a->items()) {
    if (item.kind() != CallArg::Kind::Int) {
      malformed("dim: non-integer entry");
    }
    dims.push_back(item.intValue());
  }
  if (dims.empty()) {
    unsupported(RejectReason::Signature, "dim=[] (empty dim list) is not decoded");
  }
  return dims;
}

bool boolValue(const CallArg* a, bool def) {
  if (!a) {
    return def;
  }
  if (a->kind() != CallArg::Kind::Bool) {
    malformed("keepdim: expected bool");
  }
  return a->boolValue();
}

void requireNoDtype(const CallArg* a) {
  if (!a || a->kind() == CallArg::Kind::None) {
    return;
  }
  if (a->kind() != CallArg::Kind::Dtype) {
    malformed("dtype: expected ScalarType or None");
  }
  unsupported(RejectReason::Dtype, "explicit reduction dtype '" + a->text() + "' is not decoded");
}

StorageDtype storageDtype(const std::string& core) {
  static const std::map<std::string, StorageDtype> m{
      {"f16", StorageDtype::F16},
      {"bf16", StorageDtype::BF16},
      {"f32", StorageDtype::F32},
      {"f64", StorageDtype::F64},
      {"i32", StorageDtype::I32},
      {"i64", StorageDtype::I64},
      {"bool", StorageDtype::Bool},
  };
  auto it = m.find(core);
  if (it == m.end()) {
    unsupported(RejectReason::Dtype, "dtype '" + core + "' has no option storage dtype");
  }
  return it->second;
}

const Descriptor& descriptorFor(const ExternCall& call, const std::string& name) {
  const Descriptor* found = nullptr;
  for (const Descriptor& d : call.descriptors()) {
    if (d.buffer == name) {
      if (found) {
        malformed("multiple descriptors for '" + name + "'");
      }
      found = &d;
    }
  }
  if (!found) {
    malformed("no descriptor for '" + name + "'");
  }
  return *found;
}

std::pair<TensorOperand, TensorBinding> bindTensor(const ExternCall& call,
                                                   const ProgramFacts& facts,
                                                   const std::string& name) {
  const Descriptor& d = descriptorFor(call, name);
  const RootRef& r = facts.resolve(name);
  TensorOperand op{name, TensorLayout{storageDtype(facts.root_dtype.at(r.root)), d.sizes, d.strides, ShapeKind::Static}};
  TensorBinding b{name, r.root, r.offset + d.offset, facts.root_elements.at(r.root), facts.producer.count(r.root) > 0};
  for (int64_t st : d.strides) {
    if (st < 0) {
      unsupported(RejectReason::Layout, "'" + name + "' has a negative stride");
    }
  }
  if (b.root_offset < 0 || b.root_offset + op.layout.extent() > b.root_elements) {
    malformed("descriptor for '" + name + "' exceeds its storage root");
  }
  return {std::move(op), std::move(b)};
}

const std::string& tensorName(const CallArg* a, const char* param) {
  if (a->kind() != CallArg::Kind::Tensor) {
    malformed(std::string(param) + ": expected a tensor");
  }
  return a->tensorName();
}

OpSignature buildSignature(const OpSchema& schema,
                           const Bound& bound,
                           const std::vector<TensorOperand>& tensors,
                           const TensorOperand& out,
                           DeviceKind device) {
  try {
    switch (schema.kind) {
      case OpKind::Mm:
        return OpSignature::Mm(device, tensors[0], tensors[1], out);
      case OpKind::Bmm:
        return OpSignature::Bmm(device, tensors[0], tensors[1], out);
      case OpKind::Addmm:
        return OpSignature::Addmm(
            device, tensors[0], tensors[1], tensors[2], out, scalarValue(bound[4], 1.0, "alpha"),
            scalarValue(bound[3], 1.0, "beta"));
      default: {
        requireNoDtype(bound[3]);
        const auto dims = dimList(bound[1], tensors[0].layout.rank());
        return OpSignature::Reduction(
            schema.kind, device, tensors[0], dims, boolValue(bound[2], false), std::nullopt, out);
      }
    }
  } catch (const OptionRejected& e) {
    if (e.reason() == RejectReason::Signature || e.reason() == RejectReason::Rank) {
      malformed(std::string("descriptor metadata disagrees with the op: ") + e.what());
    }
    unsupported(e.reason(), e.what());
  }
}

void decodeOne(const OpSchema& schema, const ExternCall& call, const ProgramFacts& facts, DeviceKind device, DecodedOp& out) {
  const Bound bound = bindArguments(schema, call);
  std::vector<TensorOperand> tensors;
  for (size_t i = 0; i < schema.params.size(); ++i) {
    if (schema.params[i].type == ParamType::Tensor) {
      auto [op, b] = bindTensor(call, facts, tensorName(bound[i], schema.params[i].name));
      for (const TensorOperand& prev : tensors) {
        if (prev.name == op.name) {
          op.name += "#" + std::to_string(tensors.size()); // repeated argument (x @ x)
        }
      }
      tensors.push_back(std::move(op));
      out.input_bindings.push_back(std::move(b));
    }
  }
  if (call.results().size() != 1) {
    malformed("expected exactly one result");
  }
  auto [result, rb] = bindTensor(call, facts, call.results().front().name);
  out.output_binding = std::move(rb);
  out.signature = buildSignature(schema, bound, tensors, result, device);
  std::vector<const TensorOperand*> all;
  for (const auto& t : out.signature->inputs()) {
    all.push_back(&t);
  }
  all.push_back(&out.signature->output());
  for (const TensorOperand* t : all) {
    if (t->layout.dtype != StorageDtype::F32) {
      unsupported(RejectReason::Dtype, "operand '" + t->name + "' is " + storageDtypeName(t->layout.dtype) +
                      "; options are f32-only");
    }
  }
}

} // namespace

const char* decodeStatusName(DecodeStatus status) {
  switch (status) {
    case DecodeStatus::Decoded:
      return "decoded";
    case DecodeStatus::UnknownTarget:
      return "unknown_target";
    case DecodeStatus::Unsupported:
      return "unsupported";
    case DecodeStatus::MalformedAbi:
      return "malformed_abi";
  }
  return "?";
}

DecodedOps decodeOpSignatures(const KernelProgram& program, DeviceKind device) {
  const ProgramFacts facts = analyzeProgram(program);
  DecodedOps out;
  out.numerics = program.numerics();
  for (size_t i = 0; i < program.nodes().size(); ++i) {
    const auto* call = std::get_if<ExternCall>(&program.nodes()[i]);
    if (!call) {
      continue;
    }
    DecodedOp op(*call);
    op.node_index = i;
    const OpSchema* schema = nullptr;
    for (const auto& s : schemas()) {
      if (call->target() == s.target) {
        schema = &s;
      }
    }
    if (schema) {
      try {
        decodeOne(*schema, *call, facts, device, op);
        op.status = DecodeStatus::Decoded;
      } catch (const DecodeFailure& f) {
        op.status = f.status;
        op.reason = f.reason;
        op.detail = f.detail;
        if (f.status == DecodeStatus::MalformedAbi) {
          op.signature.reset();
        }
      }
    }
    out.ops.push_back(std::move(op));
  }
  return out;
}

} // namespace torch::inductor::loops_ir::op_options
