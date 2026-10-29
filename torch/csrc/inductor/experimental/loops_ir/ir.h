#pragma once
// Standalone loops IR: typed IR + index tree. C++17, no torch/c10 deps.
// One namespace: torch::inductor::loops_ir.
//
// Typed prototype: generated Compute data values are f32; integer Index nodes
// are separate. Extern/input boundaries may use common opaque dtypes
// (f16, bf16, f32, f64, i8, u8, i16, i32, i64, bool, c64, c128) for
// explicit eager boundaries. Compute loaded/input/result data must be f32.
// Strict numerics policy is fixed to 'strict_eager': preserve operand order,
// result dtype, and observable rounding boundaries. No reassociation,
// contraction into FMA, FP identity rewrites, approximate functions, or
// subnormal flushing. No FP constant folding in passes.
// Store coverage/injectivity is proven narrowly where required for fusion;
// the base program verifier still enforces the nonempty-domain store rule.

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace torch::inductor::loops_ir {

// ---------------------------------------------------------------------------
// Index tree (integer indices, separate from f32 data).
// Central traversal lives here so passes share one walker.
// ---------------------------------------------------------------------------

struct IndexExpr {
  enum class Kind { Constant, Variable, Binary };
  Kind kind = Kind::Constant;
  // Binary op among: "add", "mul", "floordiv", "mod".
  std::string op;
  int64_t value = 0; // Kind::Constant
  std::string name; // Kind::Variable
  std::shared_ptr<const IndexExpr> lhs; // Kind::Binary
  std::shared_ptr<const IndexExpr> rhs; // Kind::Binary
};

class Index {
 public:
  Index() = default;
  explicit Index(std::shared_ptr<const IndexExpr> e) : expr_(std::move(e)) {
    if (!expr_) {
      throw std::invalid_argument("Index: null expression");
    }
  }

  static Index Constant(int64_t v) {
    auto e = std::make_shared<IndexExpr>();
    e->kind = IndexExpr::Kind::Constant;
    e->value = v;
    return Index(e);
  }
  static Index Variable(const std::string& n) {
    if (n.empty()) {
      throw std::invalid_argument("Index.variable: empty name");
    }
    auto e = std::make_shared<IndexExpr>();
    e->kind = IndexExpr::Kind::Variable;
    e->name = n;
    return Index(e);
  }
  static Index Binary(const std::string& op, const Index& lhs, const Index& rhs) {
    if (op != "add" && op != "mul" && op != "floordiv" && op != "mod") {
      throw std::invalid_argument("Index.binary: unsupported op '" + op + "' (allowed: add, mul, floordiv, mod)");
    }
    if (lhs.empty() || rhs.empty()) {
      throw std::invalid_argument("Index.binary: empty operand");
    }
    if ((op == "floordiv" || op == "mod") && rhs.expr_ && rhs.expr_->kind == IndexExpr::Kind::Constant &&
        rhs.expr_->value <= 0) {
      throw std::invalid_argument("Index.binary: '" + op + "' divisor constant must be > 0");
    }
    auto e = std::make_shared<IndexExpr>();
    e->kind = IndexExpr::Kind::Binary;
    e->op = op;
    e->lhs = lhs.expr_;
    e->rhs = rhs.expr_;
    return Index(e);
  }

  bool empty() const {
    return expr_ == nullptr;
  }
  const IndexExpr& expr() const {
    if (!expr_) {
      throw std::invalid_argument("Index: empty expression");
    }
    return *expr_;
  }
  const std::shared_ptr<const IndexExpr>& ptr() const {
    return expr_;
  }

  // Central index-tree traversal: pre-order walk over every node (shared
  // subtrees may be visited more than once). Passes should build on this.
  void walk(const std::function<void(const IndexExpr&)>& fn) const {
    walkImpl(expr_, fn);
  }

  // Deduped variable names in first-visit order.
  std::vector<std::string> collectVariables() const {
    std::vector<std::string> out;
    walk([&](const IndexExpr& e) {
      if (e.kind == IndexExpr::Kind::Variable) {
        bool seen = false;
        for (const auto& s : out) {
          if (s == e.name) {
            seen = true;
            break;
          }
        }
        if (!seen) {
          out.push_back(e.name);
        }
      }
    });
    return out;
  }

  std::string toString() const;

 private:
  std::shared_ptr<const IndexExpr> expr_;
  static void walkImpl(const std::shared_ptr<const IndexExpr>& e, const std::function<void(const IndexExpr&)>& fn) {
    if (!e) {
      return;
    }
    fn(*e);
    if (e->kind == IndexExpr::Kind::Binary) {
      if (!e->lhs || !e->rhs) {
        throw std::invalid_argument("Index: binary with empty operand");
      }
      walkImpl(e->lhs, fn);
      walkImpl(e->rhs, fn);
    }
  }
};

// Structural (tree) equality for Index. Must be used for CSE and indexing
// matching; do NOT use raw pointer equality on shared IndexExpr nodes.
bool indexEquals(const Index& a, const Index& b);

// Closed integer interval used for bounds checking.
struct Interval {
  int64_t lo = 0;
  int64_t hi = -1; // lo > hi means empty
};

// Evaluate the min/max of an index given inclusive lower/upper bounds per
// variable. Throws std::invalid_argument on undeclared variables, bad ops,
// int64 overflow (checked, never saturating), non-positive floordiv
// divisors, or non-constant-positive mod divisors.
Interval evalIndexInterval(const Index& idx, const std::map<std::string, Interval>& varBounds);

// ---------------------------------------------------------------------------
// Dtype helpers: boundary vs compute data.
// ---------------------------------------------------------------------------

// Opaque boundary dtypes allowed at program inputs and extern inputs/results.
// Includes f32 plus common non-f32 boundary types for eager calls.
bool isBoundaryDtype(const std::string& dtype);
// Generated Compute data must be f32 (loads, inputs, results, accumulators).
bool isComputeDataDtype(const std::string& dtype);

// Strict numerics policy name. Fixed default across Program/KernelIR.
inline const std::string& strictNumericsName() {
  static const std::string kName = "strict_eager";
  return kName;
}

// Alternative Inductor-following policy name.
inline const std::string& inductorNumericsName() {
  static const std::string kName = "inductor";
  return kName;
}

// True for 'strict_eager' or 'inductor'.
inline bool isValidNumerics(const std::string& numerics) {
  return numerics == strictNumericsName() || numerics == inductorNumericsName();
}

// Validated layout names. Only 'strided' carries full metadata today; other
// layout metadata stays behind the whitelist so the emitter never renders an
// arbitrary layout expression.
inline bool isValidLayoutName(const std::string& name) {
  return name == "strided" || name == "sparse_coo" || name == "sparse_csr" || name == "sparse_csc" ||
      name == "sparse_bsr" || name == "sparse_bsc" || name == "_mkldnn" || name == "mkldnn" || name == "jagged";
}

// Validated memory-format names (common contiguous/preserve/channels_last).
inline bool isValidMemoryFormatName(const std::string& name) {
  return name == "contiguous_format" || name == "preserve_format" || name == "channels_last" ||
      name == "channels_last_3d";
}

// ---------------------------------------------------------------------------
// Metadata structs
// ---------------------------------------------------------------------------

struct Buffer {
  Buffer() = default;
  Buffer(std::string n, std::string d, int64_t el) : name(std::move(n)), dtype(std::move(d)), elements(el) {}
  std::string name;
  std::string dtype = "f32"; // boundary dtypes allowed at inputs/externs; compute data must be f32
  int64_t elements = 0; // flat element count
};

struct Descriptor {
  Descriptor() = default;
  Descriptor(std::string n, std::string b, std::vector<int64_t> sz, std::vector<int64_t> st, int64_t off = 0)
      : name(std::move(n)), buffer(std::move(b)), sizes(std::move(sz)), strides(std::move(st)), offset(off) {}
  std::string name;
  std::string buffer; // storage buffer name, or alias view name (outputs)
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset = 0; // flat offset relative to buffer (or view pointer)
};

struct View {
  View() = default;
  View(std::string n, std::string src, std::vector<int64_t> sz, std::vector<int64_t> st, int64_t off = 0)
      : name(std::move(n)), source(std::move(src)), sizes(std::move(sz)), strides(std::move(st)), offset(off) {}
  std::string name;
  std::string source; // storage buffer or earlier view (alias chain)
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset = 0; // relative to source pointer
};

struct Range {
  Range() = default;
  Range(std::string n, int64_t s, int64_t e) : name(std::move(n)), start(s), stop(e) {}
  std::string name;
  int64_t start = 0;
  int64_t stop = 0; // exclusive; stop <= start means empty domain
};

struct Accumulator {
  Accumulator() = default;
  Accumulator(std::string n, std::string d, double ident, std::string comb = "add")
      : name(std::move(n)), dtype(std::move(d)), identity(ident), combine(std::move(comb)) {}
  std::string name;
  std::string dtype = "f32"; // only "f32" supported
  double identity = 0.0; // represented as f32; must be finite as double and
                         // as float
  std::string combine = "add"; // only "add" supported. Multiple independent
  // add accumulators (e.g. two sums) are supported; general coupled
  // reducers are NOT implemented and not claimed.
};

// ---------------------------------------------------------------------------
// Structured extern arguments: typed ordered args + keyword args.
// Preserves integers/lists/bools/None/dtype exactly (not opaque dumps).
// Tensor refs are ordered and may repeat; verifier checks every Tensor ref.
// ---------------------------------------------------------------------------

class CallArg {
 public:
  enum class Kind { None, Tensor, Int, Float, Bool, String, Dtype, Layout, MemoryFormat, List, Tuple };

  // C++ factories (bound to Python snakecase none/tensor/integer/floating/
  // boolean/string/dtype/layout/memory_format/list/tuple).
  static CallArg MakeNone();
  static CallArg MakeTensor(std::string name);
  static CallArg MakeInteger(int64_t v);
  static CallArg MakeFloating(double v);
  static CallArg MakeBoolean(bool v);
  static CallArg MakeString(std::string s);
  static CallArg MakeDtype(std::string d);
  static CallArg MakeLayout(std::string name);
  static CallArg MakeMemoryFormat(std::string name);
  static CallArg MakeList(std::vector<CallArg> items);
  static CallArg MakeTuple(std::vector<CallArg> items);

  CallArg() = default; // Kind::None

  Kind kind() const {
    return kind_;
  }
  // 'none','tensor','int','float','bool','string','dtype','layout',
  // 'memory_format','list','tuple'.
  std::string kindName() const;
  const std::string& tensorName() const {
    return tensor_name_;
  }
  int64_t intValue() const {
    return int_value_;
  }
  double floatValue() const {
    return float_value_;
  }
  bool boolValue() const {
    return bool_value_;
  }
  // String/dtype payload.
  const std::string& text() const {
    return text_;
  }
  const std::vector<CallArg>& items() const {
    return items_;
  }

 private:
  Kind kind_ = Kind::None;
  std::string tensor_name_;
  int64_t int_value_ = 0;
  double float_value_ = 0.0;
  bool bool_value_ = false;
  std::string text_;
  std::vector<CallArg> items_;
};

class CallKeyword {
 public:
  CallKeyword() = default;
  CallKeyword(std::string n, CallArg v) : name_(std::move(n)), value_(std::move(v)) {}
  const std::string& name() const {
    return name_;
  }
  const CallArg& value() const {
    return value_;
  }

 private:
  std::string name_;
  CallArg value_;
};

// ---------------------------------------------------------------------------
// Instruction (single f32 SSA value, except accumulate/store)
// ---------------------------------------------------------------------------

class Instruction {
 public:
  enum class Kind { Constant, Load, ViewRead, Binary, Relu, Accumulate, ReadAccumulator, Store };

  static Instruction Constant(int id, double value);
  static Instruction Load(int id, std::string buffer, Index offset);
  static Instruction ViewRead(int id, std::string view, std::vector<Index> coordinates);
  static Instruction Binary(int id, std::string op, int lhs, int rhs) {
    if (op != "add" && op != "mul") {
      throw std::invalid_argument("Instruction.binary: unsupported op '" + op + "' (allowed: add, mul)");
    }
    Instruction i;
    i.kind_ = Kind::Binary;
    i.id_ = id;
    i.op_ = std::move(op);
    i.lhs_ = lhs;
    i.rhs_ = rhs;
    return i;
  }
  static Instruction Relu(int id, int arg) {
    Instruction i;
    i.kind_ = Kind::Relu;
    i.id_ = id;
    i.arg_ = arg;
    return i;
  }
  // Updates the whole accumulator tuple; args.size() must equal acc count.
  static Instruction Accumulate(std::vector<int> args) {
    Instruction i;
    i.kind_ = Kind::Accumulate;
    i.id_ = -1;
    i.args_ = std::move(args);
    return i;
  }
  static Instruction ReadAccumulator(int id, int accumulator) {
    Instruction i;
    i.kind_ = Kind::ReadAccumulator;
    i.id_ = id;
    i.accumulator_ = accumulator;
    return i;
  }
  static Instruction Store(std::string buffer, Index offset, int value);

  Kind kind() const {
    return kind_;
  }
  std::string kindName() const;
  int id() const {
    return id_;
  }
  double value() const {
    return value_;
  }
  const std::string& buffer() const {
    return buffer_;
  }
  const Index& offset() const {
    return offset_;
  }
  const std::string& view() const {
    return view_;
  }
  const std::vector<Index>& coordinates() const {
    return coords_;
  }
  const std::string& op() const {
    return op_;
  }
  int lhs() const {
    return lhs_;
  }
  int rhs() const {
    return rhs_;
  }
  int arg() const {
    return arg_;
  }
  const std::vector<int>& args() const {
    return args_;
  }
  int accumulator() const {
    return accumulator_;
  }
  // True for constant/load/view_read/binary/relu/read_accumulator.
  bool definesValue() const {
    return kind_ == Kind::Constant || kind_ == Kind::Load || kind_ == Kind::ViewRead || kind_ == Kind::Binary ||
        kind_ == Kind::Relu || kind_ == Kind::ReadAccumulator;
  }

 private:
  Kind kind_ = Kind::Constant;
  int id_ = -1;
  double value_ = 0.0;
  std::string buffer_;
  Index offset_;
  std::string view_;
  std::vector<Index> coords_;
  std::string op_ = "add";
  int lhs_ = -1;
  int rhs_ = -1;
  int arg_ = -1;
  std::vector<int> args_;
  int accumulator_ = -1;
};

// ---------------------------------------------------------------------------
// Compute node
// ---------------------------------------------------------------------------

class Compute {
 public:
  Compute() = default;
  Compute(std::string n,
          std::vector<std::string> in,
          std::vector<Buffer> res,
          std::vector<Range> pw,
          std::vector<Range> red,
          std::vector<Accumulator> acc)
      : name_(std::move(n)),
        inputs_(std::move(in)),
        results_(std::move(res)),
        pointwise_ranges_(std::move(pw)),
        reduction_ranges_(std::move(red)),
        accumulators_(std::move(acc)) {}

  const std::string& name() const {
    return name_;
  }
  const std::vector<std::string>& inputs() const {
    return inputs_;
  }
  const std::vector<Buffer>& results() const {
    return results_;
  }
  const std::vector<Range>& pointwiseRanges() const {
    return pointwise_ranges_;
  }
  const std::vector<Range>& reductionRanges() const {
    return reduction_ranges_;
  }
  const std::vector<Accumulator>& accumulators() const {
    return accumulators_;
  }
  const std::vector<Instruction>& reductionBody() const {
    return reduction_body_;
  }
  const std::vector<Instruction>& postBody() const {
    return post_body_;
  }

  void addReductionInstruction(const Instruction& ins) {
    reduction_body_.push_back(ins);
  }
  void addPostInstruction(const Instruction& ins) {
    post_body_.push_back(ins);
  }

 private:
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<Buffer> results_;
  std::vector<Range> pointwise_ranges_;
  std::vector<Range> reduction_ranges_;
  std::vector<Accumulator> accumulators_;
  std::vector<Instruction> reduction_body_;
  std::vector<Instruction> post_body_;
};

// ---------------------------------------------------------------------------
// ExternCall node. Opaque call: no execution backend here, effects always
// preserved conservatively, no optim passes run over it. Descriptors are
// boundary metadata only (physical interpretation), covering every distinct
// input and result. Structured typed arguments preserve order and values.
// ---------------------------------------------------------------------------

class ExternCall {
 public:
  ExternCall() = default;
  ExternCall(std::string n,
             std::string tgt,
             std::vector<std::string> in,
             std::vector<Buffer> res,
             std::vector<double> sc,
             std::string eff = "unknown",
             bool mayAlias = true,
             std::vector<Descriptor> desc = {});
  // Structured factory: collects ordered tensor inputs recursively from
  // typed args (repeats allowed) + keyword arg values.
  static ExternCall Structured(std::string n,
                               std::string tgt,
                               std::vector<CallArg> args,
                               std::vector<CallKeyword> kwargs,
                               std::vector<Buffer> res,
                               std::vector<Descriptor> desc,
                               std::string eff = "unknown",
                               bool mayAlias = true);
  const std::string& name() const {
    return name_;
  }
  const std::string& target() const {
    return target_;
  }
  const std::vector<std::string>& inputs() const {
    return inputs_;
  }
  const std::vector<Buffer>& results() const {
    return results_;
  }
  const std::vector<double>& scalarArgs() const {
    return scalar_args_;
  }
  const std::string& effects() const {
    return effects_;
  }
  bool mayAliasInputs() const {
    return may_alias_inputs_;
  }
  const std::vector<Descriptor>& descriptors() const {
    return descriptors_;
  }
  const std::vector<CallArg>& arguments() const {
    return arguments_;
  }
  const std::vector<CallKeyword>& keywordArgs() const {
    return keyword_args_;
  }

 private:
  std::string name_;
  std::string target_;
  std::vector<std::string> inputs_;
  std::vector<Buffer> results_;
  std::vector<double> scalar_args_;
  std::string effects_ = "unknown";
  bool may_alias_inputs_ = true;
  std::vector<Descriptor> descriptors_;
  std::vector<CallArg> arguments_;
  std::vector<CallKeyword> keyword_args_;
};

// ---------------------------------------------------------------------------
// Program: metadata (inputs/outputs) + ordered nodes.
// ---------------------------------------------------------------------------

class Program {
 public:
  using Node = std::variant<View, Compute, ExternCall>;

  explicit Program(std::string n, std::string numerics = "strict_eager");

  const std::string& name() const {
    return name_;
  }
  const std::vector<Buffer>& inputs() const {
    return inputs_;
  }
  const std::vector<Descriptor>& inputDescriptors() const {
    return input_descriptors_;
  }
  const std::vector<Node>& nodes() const {
    return nodes_;
  }
  const std::vector<Descriptor>& outputs() const {
    return outputs_;
  }
  // Strict numerics policy; fixed default 'strict_eager'.
  const std::string& numerics() const {
    return numerics_;
  }

  void addInput(const Buffer& b) {
    inputs_.push_back(b);
  }
  void addInputDescriptor(const Descriptor& d) {
    input_descriptors_.push_back(d);
  }
  void addView(const View& v) {
    nodes_.emplace_back(v);
  }
  void addCompute(const Compute& c) {
    nodes_.emplace_back(c);
  }
  void addExternCall(const ExternCall& e) {
    nodes_.emplace_back(e);
  }
  void addOutput(const Descriptor& d) {
    outputs_.push_back(d);
  }

  // Full verification; throws std::invalid_argument with node context.
  void verify() const;
  // Verifies first, then returns human-readable text (never parsed).
  std::string dump() const;

 private:
  std::string name_;
  std::vector<Buffer> inputs_;
  std::vector<Descriptor> input_descriptors_;
  std::vector<Node> nodes_;
  std::vector<Descriptor> outputs_;
  std::string numerics_ = "strict_eager";
};

// Deterministic float printing for f32-representable values.
std::string formatFloat(double v);
// Exact double printing for structured extern float args (round-trip).
std::string formatDoubleExact(double v);
// Human-readable typed extern argument (preserves ints/lists/bools/None/dtype).
std::string callArgToString(const CallArg& a);

} // namespace torch::inductor::loops_ir
