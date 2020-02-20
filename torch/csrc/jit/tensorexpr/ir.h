#pragma once

#include <string>
#include <vector>

#include "torch/csrc/jit/tensorexpr/expr.h"

namespace torch {
namespace jit {
namespace tensorexpr {

enum IRNodeType {
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMod,
  kMax,
  kMin,
  kCompareSelect,
};

enum CompareSelectOperation {
  kEQ,
  kGT,
  kGE,
  kLT,
  kLE,
  kNE,
};

class Buffer;

class Cast : public ExprNode<Cast> {
 public:
  const Expr& src_value() const {
    return src_value_;
  }
  static Expr make(Dtype dtype, const Expr& src_value) {
    return Expr(new Cast(dtype, src_value));
  }

 private:
  Cast(Dtype dtype, const Expr& src_value)
      : ExprNodeBase(dtype), src_value_(src_value) {}
  Expr src_value_;
};

template <typename T>
Expr cast(const Expr& src_value) {
  return Cast::make(Dtype(ToDtype<T>(), src_value.dtype().lanes()), src_value);
}

// Represent the expression node for binary operators.
// A CRTP pattern to share common code among the operators.
template <typename Op>
class BinaryOpNode : public ExprNode<Op> {
 public:
  const Expr& lhs() const {
    return this->lhs_;
  }
  const Expr& rhs() const {
    return this->rhs_;
  }
  IRNodeType expr_type() const {
    return expr_type_;
  }

  static Expr make(const Expr& lhs, const Expr& rhs) {
    return Expr(new Op(lhs, rhs));
  }

 protected:
  BinaryOpNode(
      const Expr& lhs_v,
      const Expr& rhs_v,
      IRNodeType expr_type,
      ReturnType ret_type = ReturnType::knone)
      : ExprNode<Op>(BinaryOpDtype(lhs_v.dtype(), rhs_v.dtype(), ret_type)),
        lhs_(CastIfNeeded(lhs_v, ExprNode<Op>::dtype())),
        rhs_(CastIfNeeded(rhs_v, ExprNode<Op>::dtype())),
        expr_type_(expr_type) {}

 private:
  static Expr CastIfNeeded(const Expr& expr, Dtype dst_dtype) {
    if (expr.dtype() == dst_dtype) {
      return expr;
    }
    return Cast::make(dst_dtype, expr);
  }

  Expr lhs_;
  Expr rhs_;
  IRNodeType expr_type_;
};

class Add : public BinaryOpNode<Add> {
 private:
  Add(const Expr& lhs, const Expr& rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kAdd) {}
  friend class BinaryOpNode<Add>;
};

class Sub : public BinaryOpNode<Sub> {
 private:
  Sub(const Expr& lhs, const Expr& rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kSub) {}
  friend class BinaryOpNode<Sub>;
};

class Mul : public BinaryOpNode<Mul> {
 private:
  Mul(const Expr& lhs, const Expr& rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMul) {}
  friend class BinaryOpNode<Mul>;
};

class Div : public BinaryOpNode<Div> {
 private:
  Div(const Expr& lhs, const Expr& rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kDiv) {}
  friend class BinaryOpNode<Div>;
};

class Mod : public BinaryOpNode<Mod> {
 private:
  Mod(const Expr& lhs, const Expr& rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMod) {}
  friend class BinaryOpNode<Mod>;
};

class Max : public BinaryOpNode<Max> {
 private:
  bool propagate_nans_;
  Max(const Expr& lhs, const Expr& rhs, bool propagate_nans)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMax),
        propagate_nans_(propagate_nans) {}
  friend class BinaryOpNode<Max>;

 public:
  bool propagate_nans() const {
    return propagate_nans_;
  }

  static Expr make(const Expr& lhs, const Expr& rhs) = delete;
  static Expr make(const Expr& lhs, const Expr& rhs, bool propagate_nans) {
    return Expr(new Max(lhs, rhs, propagate_nans));
  }
};

class Min : public BinaryOpNode<Min> {
 private:
  bool propagate_nans_;
  Min(const Expr& lhs, const Expr& rhs, bool propagate_nans)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMin),
        propagate_nans_(propagate_nans) {}
  friend class BinaryOpNode<Min>;

 public:
  bool propagate_nans() const {
    return propagate_nans_;
  }

  static Expr make(const Expr& lhs, const Expr& rhs) = delete;
  static Expr make(const Expr& lhs, const Expr& rhs, bool propagate_nans) {
    return Expr(new Min(lhs, rhs, propagate_nans));
  }
};

class CompareSelect : public ExprNode<CompareSelect> {
 public:
  CompareSelectOperation compare_select_op() const {
    return compare_op_;
  }
  const Expr& lhs() const {
    return this->lhs_;
  }
  const Expr& rhs() const {
    return this->rhs_;
  }

  static Expr make(const Expr& lhs, const Expr& rhs) = delete;

  static Expr make(
      const Expr& lhs,
      const Expr& rhs,
      CompareSelectOperation cmp_op) {
    return Expr(new CompareSelect(lhs, rhs, cmp_op));
  }

 private:
  Expr lhs_;
  Expr rhs_;
  CompareSelectOperation compare_op_;
  CompareSelect(const Expr& lhs, const Expr& rhs, CompareSelectOperation cmp_op)
      : ExprNodeBase(ToDtype<int>()),
        lhs_(lhs),
        rhs_(rhs),
        compare_op_(cmp_op) {}
};

// Encode an integer immediate value.
class IntImm : public ExprNode<IntImm> {
 public:
  int value() const {
    return value_;
  }
  static Expr make(int value) {
    return Expr(new IntImm(value));
  }

 private:
  IntImm(int value) : ExprNodeBase(kInt32), value_(value) {}
  int value_;
};

// Encode an fp32 immediate value.
class FloatImm : public ExprNode<FloatImm> {
 public:
  float value() const {
    return value_;
  }
  static Expr make(float value) {
    return Expr(new FloatImm(value));
  }

 private:
  FloatImm(float value) : ExprNodeBase(kFloat32), value_(value) {}
  float value_;
};

// The underlying representation node to a Variable.
// Currently, each Variable object represents a unique variable, even though the
// names might be the same. We should consider add a unique_name as well.
class Variable : public ExprNode<Variable> {
 public:
  static Expr make(const std::string& name_hint, Dtype dtype) {
    return Expr(new Variable(name_hint, dtype));
  }
  static Expr make(Dtype dtype) {
    return Expr(new Variable("", dtype));
  }

  // TODO: unique_name
  const std::string& name_hint() const {
    return name_hint_;
  }

 private:
  Variable(const std::string& name_hint, Dtype dtype)
      : ExprNodeBase(dtype), name_hint_(name_hint) {}
  std::string name_hint_;
};

// An expression to construct the underlying variable node.
// Note: do not store any info here, since it is often possible to slice this
// object. For example: Var x('x'); Expr x2 = x;
class Var : public Expr {
 public:
  Var() : Expr(nullptr) {}
  explicit Var(Dtype dtype) : Expr(Variable::make(dtype)) {}
  Var(const std::string& name_hint, Dtype dtype)
      : Expr(Variable::make(name_hint, dtype)) {}
  explicit Var(Variable* node) : Expr(node) {}
  const Variable* node() const {
    return static_cast<const Variable*>(Expr::node());
  }
  bool operator==(const Var& other) const {
    return this->node() == other.node();
  }
  bool operator!=(const Var& other) const {
    return !(*this == other);
  }

  const std::string& name_hint() const {
    return this->node()->name_hint();
  }
  bool empty() const {
    return (this->node() == nullptr);
  }
};

// Bind the value to the var and evaluate the body.
class Let : public ExprNode<Let> {
 public:
  const Expr& var() const {
    return var_;
  }
  const Expr& value() const {
    return value_;
  }
  const Expr& body() const {
    return body_;
  }

  static Expr make(const Expr& var, const Expr& value, const Expr& body) {
    return Expr(new Let(var, value, body));
  }

 private:
  Let(const Expr& var, const Expr& value, const Expr& body)
      : ExprNodeBase(body.dtype()), var_(var), value_(value), body_(body) {}

  Expr var_;
  Expr value_;
  Expr body_;
};

class LetStmt : public StmtNode<LetStmt> {
 public:
  const Var& var() const {
    return var_;
  }

  const Expr& value() const {
    return value_;
  }

  const Stmt& body() const {
    return body_;
  }

  static Stmt make(const Var& var, const Expr& value, const Stmt& body) {
    return Stmt(new LetStmt(var, value, body));
  }

 private:
  LetStmt(const Var& var, const Expr& value, const Stmt& body)
      : var_(var), value_(value), body_(body) {}

  Var var_;
  Expr value_;
  Stmt body_;
};

class Block : public StmtNode<Block> {
 public:
  static Stmt make(const std::vector<Stmt>& stmts) {
    std::vector<Stmt> valid_stmts;
    for (size_t i = 0; i < stmts.size(); i++) {
      if (stmts[i].empty()) {
        continue;
      }
      valid_stmts.push_back(stmts[i]);
    }
    if (valid_stmts.empty()) {
      return Stmt();
    }
    return Stmt(new Block(valid_stmts));
  }
  int nstmts() const {
    return stmts_.size();
  }
  const Stmt& stmt(int index) const {
    return stmts_[index];
  }

 private:
  explicit Block(const std::vector<Stmt>& stmts) : stmts_(stmts) {}
  std::vector<Stmt> stmts_;
};

class LoopOptions {
 public:
  // GPU Block Index
  bool is_gpu_block_index() const {
    return gpu_block_index_ != -1;
  }

  bool gpu_block_index() const {
    return gpu_block_index_;
  }

  std::string gpu_block_index_str() const {
    DCHECK(is_gpu_block_index());
    static const char* kBlockIndexNames[] = {
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "blockIdx.w",
    };
    DCHECK(gpu_block_index_ >= 0 && gpu_block_index_ < 4);
    return kBlockIndexNames[gpu_block_index_];
  }

  void set_gpu_block_index(int index) {
    if (is_gpu_thread_index()) {
      throw std::runtime_error("Cannot set both gpu block and thread index");
    }
    if (is_gpu_block_index() && gpu_block_index() != index) {
      throw std::runtime_error(
          "Cannot set a previously set block index: " +
          std::to_string(gpu_block_index()) + " vs " + std::to_string(index));
    }
    gpu_block_index_ = index;
  }

  // GPU Thread Index
  bool is_gpu_thread_index() const {
    return gpu_thread_index() != -1;
  }

  int gpu_thread_index() const {
    return gpu_thread_index_;
  }

  std::string gpu_thread_index_str() const {
    DCHECK(is_gpu_thread_index());
    static const char* kThreadIndexNames[] = {
        "threadIdx.x", "threadIdx.y", "threadIdx.z", "threadIdx.w"};
    DCHECK(gpu_thread_index_ >= 0 && gpu_thread_index_ < 4);
    return kThreadIndexNames[gpu_thread_index_];
  }

  void set_gpu_thread_index(int index) {
    if (is_gpu_block_index()) {
      throw std::runtime_error("Cannot set both gpu thread and block index");
    }
    if (is_gpu_thread_index() && gpu_thread_index() != index) {
      throw std::runtime_error(
          "Cannot set a previously set thread index: " +
          std::to_string(gpu_thread_index()) + " vs " + std::to_string(index));
    }
    gpu_thread_index_ = index;
  }

  std::string ToString() const {
    std::ostringstream oss;
    if (is_gpu_block_index()) {
      oss << gpu_block_index_str();
    } else if (is_gpu_thread_index()) {
      oss << gpu_thread_index_str();
    }
    return oss.str();
  }

 private:
  int gpu_block_index_ = -1;
  int gpu_thread_index_ = -1;
};

class For : public StmtNode<For> {
 public:
  const Var& var() const {
    return var_;
  }
  const Expr& start() const {
    return start_;
  }
  const Expr& stop() const {
    return stop_;
  }
  const Stmt& body() const {
    return body_;
  }
  static Stmt make(
      const Var& var,
      const Expr& start,
      const Expr& stop,
      const Stmt& body) {
    if (body.empty()) {
      return Stmt();
    }
    return Stmt(new For(var, start, stop, body));
  }
  static Stmt make(
      const Var& var,
      const Expr& start,
      const Expr& stop,
      const Stmt& body,
      const LoopOptions& loop_options) {
    if (body.empty()) {
      return Stmt();
    }
    return Stmt(new For(var, start, stop, body, loop_options));
  }
  const LoopOptions loop_options() const {
    return loop_options_;
  }

 private:
  For(const Var& var, const Expr& start, const Expr& stop, const Stmt& body)
      : var_(var), start_(start), stop_(stop), body_(body) {}

  For(const Var& var,
      const Expr& start,
      const Expr& stop,
      const Stmt& body,
      const LoopOptions& loop_options)
      : var_(var),
        start_(start),
        stop_(stop),
        body_(body),
        loop_options_(loop_options) {}

  Var var_;
  Expr start_;
  Expr stop_;
  Stmt body_;
  LoopOptions loop_options_;
};

// Represents a ramp vector node:
//     [base, base + 1 * stride, ... , base + (lanes - 1) * stride]
class Ramp : public ExprNode<Ramp> {
 public:
  const Expr& base() const {
    return base_;
  }
  const Expr& stride() const {
    return stride_;
  }
  static Expr make(const Expr& base, const Expr& stride, int lanes) {
    return Expr(new Ramp(base, stride, lanes));
  }
  int lanes() const {
    return lanes_;
  }

 private:
  Ramp(const Expr& base, const Expr& stride, int lanes)
      : ExprNodeBase(Dtype(base.dtype(), lanes)),
        base_(base),
        stride_(stride),
        lanes_(lanes) {
    CHECK_EQ(stride.dtype(), base.dtype());
  }

  Expr base_;
  Expr stride_;
  int lanes_;
};

class TORCH_API Load : public ExprNode<Load> {
 public:
  const Var& base_handle() const {
    return base_handle_;
  }
  const Expr& index() const {
    return index_;
  }
  const Expr& mask() const {
    return mask_;
  }
  static Expr make(const Buffer& buffer, const Expr& index, const Expr& mask) {
    return Expr(new Load(buffer, index, mask));
  }
  static Expr make(
      Dtype dtype,
      const Var& base_handle,
      const Expr& index,
      const Expr& mask) {
    return Expr(new Load(dtype, base_handle, index, mask));
  }

 private:
  Load(const Buffer& buffer, const Expr& index, const Expr& mask);
  Load(
      Dtype dtype,
      const Var& base_handle,
      const Expr& index,
      const Expr& mask);

  Var base_handle_;
  Expr index_;
  Expr mask_;
};

class TORCH_API Store : public StmtNode<Store> {
 public:
  const Var& base_handle() const {
    return base_handle_;
  }
  const Expr& index() const {
    return index_;
  }
  const Expr& value() const {
    return value_;
  }
  const Expr& mask() const {
    return mask_;
  }

  static Stmt make(
      const Buffer& buffer,
      const Expr& index,
      const Expr& value,
      const Expr& mask) {
    return Stmt(new Store(buffer, index, value, mask));
  }

  static Stmt make(
      const Var& base_handle,
      const Expr& index,
      const Expr& value,
      const Expr& mask) {
    return Stmt(new Store(base_handle, index, value, mask));
  }

  static Stmt make(
      const Var& base_handle,
      const Expr& index,
      const Expr& value) {
    return Stmt(new Store(base_handle, index, value, Expr(1)));
  }

 private:
  // TODO: merge this with Load.
  Store(
      const Buffer& buffer,
      const Expr& index,
      const Expr& value,
      const Expr& mask);

  Store(
      const Var& base_handle,
      const Expr& index,
      const Expr& value,
      const Expr& mask)
      : base_handle_(base_handle), index_(index), value_(value), mask_(mask) {
    CHECK_EQ(base_handle_.dtype(), kHandle);
    CHECK_EQ(index.dtype().lanes(), mask.dtype().lanes());
    CHECK_EQ(index.dtype().lanes(), value.dtype().lanes());
    CHECK_EQ(index.dtype().scalar_type(), kInt32);
  }

  Var base_handle_;
  Expr index_;
  Expr value_;
  Expr mask_;
};

class Broadcast : public ExprNode<Broadcast> {
 public:
  const Expr& value() const {
    return value_;
  }
  int lanes() const {
    return lanes_;
  }
  static Expr make(const Expr& value, int lanes) {
    return Expr(new Broadcast(value, lanes));
  }

 private:
  Broadcast(const Expr& value, int lanes)
      : ExprNodeBase(Dtype(value.dtype(), lanes)),
        value_(value),
        lanes_(lanes) {}
  Expr value_;
  int lanes_;
};
class IfThenElse : public ExprNode<IfThenElse> {
 public:
  const Expr& condition() const {
    return condition_;
  }

  // Lazily evaluated only if condition is true
  const Expr& true_value() const {
    return true_;
  }

  // Lazily evaluated only if condition is false
  const Expr& false_value() const {
    return false_;
  }

  static Expr make(const Expr& c, const Expr& t, const Expr& f) {
    return Expr(new IfThenElse(c, t, f));
  }

 private:
  IfThenElse(const Expr& c, const Expr& t, const Expr& f)
      : ExprNodeBase(t.dtype()), condition_(c), true_(t), false_(f) {
    CHECK_EQ(c.dtype().scalar_type(), kInt32);
    CHECK_EQ(c.dtype().lanes(), 1);
    CHECK_EQ(t.dtype(), f.dtype());
  }
  Expr condition_;
  Expr true_;
  Expr false_;
};

class BaseCallNode : public BaseExprNode {
 public:
  enum CallType {
    kIntrinsics,
    kFunctionCall,
  };

  int nparams() const {
    return params_.size();
  }

  Expr& param(int index) {
    return params_[index];
  }
  const Expr& param(int index) const {
    return params_[index];
  }
  const std::vector<Expr>& params() const {
    return params_;
  }

  virtual std::string func_name() const = 0;

  CallType call_type() const {
    return call_type_;
  }

 protected:
  BaseCallNode(Dtype dtype, CallType call_type, const std::vector<Expr>& params)
      : BaseExprNode(dtype), call_type_(call_type), params_(params) {}

 private:
  // The handler for the default ir_mutator to make a copy of this node with new
  // params.
  virtual Expr DefaultMutator(const std::vector<Expr>& new_params) const = 0;

  template <class U, class B>
  friend class ExprNode;
  friend class IRMutator;

  CallType call_type_;
  std::vector<Expr> params_;
};

template <typename Op>
class CallNode : public ExprNode<Op, BaseCallNode> {
 public:
  using BaseClass = ExprNode<Op, BaseCallNode>;
  using BaseClass::BaseClass;
};

enum IntrinsicsOp {
  kSin,
  kCos,
  kTan,
  kAsin,
  kAcos,
  kAtan,
  kAtan2,
  kSinh,
  kCosh,
  kTanh,
  kExp,
  kExpm1,
  kFabs,
  kLog,
  kLog2,
  kLog10,
  kLog1p,
  kErf,
  kErfc,
  kSqrt,
  kRsqrt,
  kPow,
  kCeil,
  kFloor,
  kRound,
  kTrunc,
  kFmod,
  kRemainder,
  kLgamma,
  kFrac,
  kRand, // We need more discussions on this. Should we consider stateful?
};

class Intrinsics : public CallNode<Intrinsics> {
 public:
  static Expr make(IntrinsicsOp op_type, const Expr& v1) {
    return Expr(new Intrinsics(op_type, v1));
  }

  static Expr make(IntrinsicsOp op_type, const Expr& v1, const Expr& v2) {
    return Expr(new Intrinsics(op_type, v1, v2));
  }

  static Expr make(IntrinsicsOp op_type, const std::vector<Expr>& params) {
    return Expr(new Intrinsics(op_type, params));
  }

  IntrinsicsOp op_type() const {
    return op_type_;
  }

  std::string func_name() const override {
    switch (op_type()) {
      case kSin:
        return "sin";
      case kCos:
        return "cos";
      case kTan:
        return "tan";
      case kAsin:
        return "asin";
      case kAcos:
        return "acos";
      case kAtan:
        return "atan";
      case kAtan2:
        return "atan2";
      case kSinh:
        return "sinh";
      case kCosh:
        return "cosh";
      case kTanh:
        return "tanh";
      case kExp:
        return "exp";
      case kFabs:
        return "fabs";
      case kLog:
        return "log";
      case kLog2:
        return "log2";
      case kLog10:
        return "log10";
      case kLog1p:
        return "log1p";
      case kErf:
        return "erf";
      case kSqrt:
        return "sqrt";
      case kRsqrt:
        return "rsqrt";
      case kPow:
        return "pow";
      case kCeil:
        return "ceil";
      case kFloor:
        return "floor";
      case kRound:
        return "round";
      case kTrunc:
        return "trunc";
      case kRand:
        return "rand";
      case kFmod:
        return "fmod";
      case kRemainder:
        return "remainder";
      case kLgamma:
        return "lgamma";
      case kExpm1:
        return "expm1";
      case kErfc:
        return "erfc";
      case kFrac:
        return "frac";
      default:
        throw std::runtime_error(
            "invalid op_type: " + std::to_string(op_type()));
    }
  }

 private:
  using BaseClass = CallNode<Intrinsics>;

  TORCH_API static int OpArgCount(IntrinsicsOp op_type);

  Intrinsics(IntrinsicsOp op_type, const Expr& v1)
      : BaseClass(IntrinsicsDtype(op_type, v1.dtype()), kIntrinsics, {v1}),
        op_type_(op_type) {
    CHECK_EQ(OpArgCount(op_type), 1);
  }

  Intrinsics(IntrinsicsOp op_type, const Expr& v1, const Expr& v2)
      : BaseClass(
            IntrinsicsDtype(op_type, v1.dtype(), v2.dtype()),
            kIntrinsics,
            {v1, v2}),
        op_type_(op_type) {
    CHECK_EQ(OpArgCount(op_type), 2);
  }

  Intrinsics(IntrinsicsOp op_type, const std::vector<Expr>& params)
      : BaseClass(IntrinsicsDtype(op_type, params), kIntrinsics, params),
        op_type_(op_type) {
    CHECK_EQ(OpArgCount(op_type), nparams());
  }

  Expr DefaultMutator(const std::vector<Expr>& new_params) const override {
    return Intrinsics::make(this->op_type(), new_params);
  }

  TORCH_API static Dtype IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1);
  TORCH_API static Dtype IntrinsicsDtype(
      IntrinsicsOp op_type,
      Dtype dt1,
      Dtype dt2);
  TORCH_API static Dtype IntrinsicsDtype(
      IntrinsicsOp op_type,
      const std::vector<Expr>& params);

  IntrinsicsOp op_type_;
};

class FunctionCall;

// Allocate a buffer of given shapes and dtypes and bind it with the given
// buffer var. The life span is at most through the current program, until it is
// explicitly freed. An unfreed memory is likely considered an error.
class Allocate : public StmtNode<Allocate> {
 public:
  static Stmt make(
      const Var& buffer_var,
      Dtype dtype,
      const std::vector<Expr>& dims) {
    return Stmt(new Allocate(buffer_var, dtype, dims));
  }

  const Var& buffer_var() const {
    return buffer_var_;
  }

  Dtype dtype() const {
    return dtype_;
  }

  const std::vector<Expr>& dims() const {
    return dims_;
  }

 private:
  Allocate(const Var& buffer_var, Dtype dtype, const std::vector<Expr>& dims)
      : buffer_var_(buffer_var), dtype_(dtype), dims_(dims) {}

  Var buffer_var_;
  Dtype dtype_;
  std::vector<Expr> dims_;
  // TODO: add memory types.
};

// Free the specific buffer. It is an error.
class Free : public StmtNode<Free> {
 public:
  static Stmt make(const Var& buffer_var) {
    return Stmt(new Free(buffer_var));
  }

  const Var& buffer_var() const {
    return buffer_var_;
  }

 private:
  Free(const Var& buffer_var) : buffer_var_(buffer_var) {}

  Var buffer_var_;
};

class Cond : public StmtNode<Cond> {
 public:
  static Stmt make(
      const Expr& condition,
      const Stmt& true_stmt,
      const Stmt& false_stmt) {
    return Stmt(new Cond(condition, true_stmt, false_stmt));
  }

  const Expr& condition() const {
    return condition_;
  }

  const Stmt& true_stmt() const {
    return true_stmt_;
  }

  const Stmt& false_stmt() const {
    return false_stmt_;
  }

 private:
  Cond(const Expr& condition, const Stmt& true_stmt, const Stmt& false_stmt)
      : condition_(condition), true_stmt_(true_stmt), false_stmt_(false_stmt) {}

  Expr condition_;
  Stmt true_stmt_;
  Stmt false_stmt_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
