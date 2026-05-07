#pragma once

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include <functional>
#include <utility>
#include <vector>

namespace torch::jit::tensorexpr {

using ParameterList = const std::vector<VarHandle>;
using ReduceInteraction = std::function<ExprHandle(ExprHandle, ExprHandle)>;

// A Reducer is a user interface describing a particular reduction
// operation. It has three components: An initialization value, a way of
// interacting each value with the accumulation, and a method for obtaining the
// current value to be reduced. It is materialized into a ReduceOp when loop
// variables are known.
class TORCH_API Reducer {
 public:
  Reducer(ExprHandle init, ReduceInteraction& interaction)
      : init_(init.node()), interaction_(interaction) {}

  template <typename RI>
  Reducer(ExprHandle init, RI interaction)
      : init_(init.node()), interaction_(std::move(interaction)) {}

  ExprPtr initializer() const {
    return init_;
  }

  ExprHandle operator()(
      const BufHandle& result_buf,
      ExprHandle body,
      const std::vector<ExprHandle>& output,
      const std::vector<VarHandle>& inner) const;

  ReduceOpPtr operator()(
      const BufPtr& result_buf,
      ExprPtr body,
      const std::vector<ExprPtr>& output,
      const std::vector<VarPtr>& inner) const;

  ExprHandle operator()(
      const BufHandle& result_buf,
      BufHandle acc_buf,
      const ExprHandle& body,
      const std::vector<ExprHandle>& output,
      const std::vector<VarHandle>& inner) const;

  // Polymorphic handling of Body functions with a variety of parameters.
  static ExprHandle getReduceBody(
      const std::function<ExprHandle(ParameterList&)>& func,
      const std::vector<VarHandle>& vars) {
    return func(vars);
  }

  static ExprHandle getReduceBody(
      const std::function<ExprHandle(const VarHandle&)>& func,
      const std::vector<VarHandle>& vars) {
    if (vars.size() != 1) {
      throw malformed_input("mismatch between reduce body and arg size (1)");
    }

    return func(vars[0]);
  }

  static ExprHandle getReduceBody(
      const std::function<ExprHandle(const VarHandle&, const VarHandle&)>& func,
      const std::vector<VarHandle>& vars) {
    if (vars.size() != 2) {
      throw malformed_input("mismatch between reduce body and arg size (2)");
    }
    return func(vars[0], vars[1]);
  }

  static ExprHandle getReduceBody(
      const std::function<
          ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
          func,
      const std::vector<VarHandle>& vars) {
    if (vars.size() != 3) {
      throw malformed_input("mismatch between reduce body and arg size (3)");
    }
    return func(vars[0], vars[1], vars[2]);
  }

  static ExprHandle getReduceBody(
      const std::function<ExprHandle(
          const VarHandle&,
          const VarHandle&,
          const VarHandle&,
          const VarHandle&)>& func,
      const std::vector<VarHandle>& vars) {
    if (vars.size() != 4) {
      throw malformed_input("mismatch between reduce body and arg size (4)");
    }
    return func(vars[0], vars[1], vars[2], vars[3]);
  }

  // Completes the reduction operator by applying the interaction function to
  // the accumulation and the body expression.
  static ExprPtr complete(
      const BufPtr& accumulator,
      const ReduceInteraction& interaction,
      ExprHandle body,
      const std::vector<ExprPtr>& output_args,
      const std::vector<VarPtr>& reduce_args) {
    ExprHandle accum =
        ExprHandle(alloc<Load>(body.dtype(), accumulator, output_args));
    auto e = interaction(std::move(accum), std::move(body));
    return e.node();
  }
  static ExprHandle complete(
      const BufHandle& accumulator,
      const ReduceInteraction& interaction,
      ExprHandle body,
      const std::vector<ExprHandle>& output_args,
      const std::vector<VarHandle>& reduce_args) {
    ExprHandle accum = Load::make(body.dtype(), accumulator, output_args);
    auto e = interaction(std::move(accum), std::move(body));
    return e;
  }

 private:
  ExprPtr init_;
  ReduceInteraction interaction_;
};

// An expression representing a Reduction operation (e.g. Sum, Max) broken into
// it's component parts: initialization, accumulation var, acquisition of value
// to be reduced and interaction.
//
// This is intended to be expanded in the loopnest and not make it to codegen.
class TORCH_API ReduceOp : public ExprNode<ReduceOp> {
 public:
  ReduceOp(
      const ExprPtr& body,
      std::vector<VarPtr> reduce_args,
      Reducer reducer)
      : ExprNodeBase(body->dtype()),
        body_(body),
        reduce_args_(std::move(reduce_args)),
        reducer_(std::move(reducer)) {
    result_buf_ = nullptr;
    acc_buf_ = nullptr;
    ri_operand_ = nullptr;
  }

  ReduceOp(
      const ExprPtr& body,
      std::vector<VarPtr> reduce_args,
      BufPtr result_buf,
      BufPtr acc_buf,
      ExprPtr ri_operand,
      Reducer reducer)
      : ExprNodeBase(body->dtype()),
        body_(body),
        reduce_args_(std::move(reduce_args)),
        result_buf_(std::move(result_buf)),
        acc_buf_(std::move(acc_buf)),
        ri_operand_(std::move(ri_operand)),
        reducer_(std::move(reducer)) {}

  static ExprHandle make(
      ExprHandle body,
      const std::vector<VarHandle>& reduce_args,
      const Reducer& reducer);

  static ExprHandle make(
      ExprHandle body,
      const std::vector<VarHandle>& reduce_args,
      BufHandle result_buf,
      BufHandle acc_buf,
      ExprHandle ri_operand,
      const Reducer& reducer);

  // return the body expression which obtains the value to be reduced.
  ExprPtr body() const {
    return body_;
  }

  // Returns the original Reducer factory that can create ReduceOps.
  const Reducer& reducer() const {
    return reducer_;
  }

  // returns variables associated with the axes of reduction.
  const std::vector<VarPtr>& reduce_args() const {
    return reduce_args_;
  }

  void setAccBuf(BufHandle acc_buf) {
    acc_buf_ = acc_buf.node();
  }
  BufPtr getAccBuf() {
    return acc_buf_;
  }

  void setResultBuf(BufHandle buf) {
    result_buf_ = buf.node();
  }
  BufPtr getResultBuf() {
    return result_buf_;
  }

  void setRiOperand(ExprHandle ri_operand) {
    ri_operand_ = ri_operand.node();
  }
  ExprPtr getRiOperand() {
    return ri_operand_;
  }

 private:
  // body_ = reducer_->interaction_(result_buf_, ri_operand_)
  ExprPtr body_;
  std::vector<VarPtr> reduce_args_;

  BufPtr result_buf_;
  BufPtr acc_buf_;
  ExprPtr ri_operand_;

  const Reducer reducer_;
};

class Sum : public Reducer {
 public:
  Sum()
      : Reducer(ExprHandle(0), [](const ExprHandle& a, const ExprHandle& b) {
          return a + b;
        }) {}
};

inline ExprHandle maximumVal(ScalarType type) {
  switch (type) {
#define MAX_BY_TYPE_CASE(Type, Name) \
  case ScalarType::Name:             \
    return ExprHandle(std::numeric_limits<Type>::max());
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, MAX_BY_TYPE_CASE)
#undef MAX_BY_TYPE_CASE
    default:
      throw unsupported_dtype();
  }
  return ExprHandle();
}

inline ExprHandle minimumVal(ScalarType type) {
  switch (type) {
#define MAX_BY_TYPE_CASE(Type, Name) \
  case ScalarType::Name:             \
    return ExprHandle(std::numeric_limits<Type>::min());
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, MAX_BY_TYPE_CASE)
#undef MAX_BY_TYPE_CASE
    default:
      throw unsupported_dtype();
  }
}

class Maximum : public Reducer {
 public:
  // TODO possible to remove this arg by deferring the init value until we
  // know the dtype of the body.
  Maximum(Dtype dtype)
      : Reducer(
            minimumVal(dtype.scalar_type()),
            [](const ExprHandle& a, const ExprHandle& b) {
              return Max::make(a, b, true);
            }) {}
  Maximum(ExprHandle initializer)
      : Reducer(
            std::move(initializer),
            [](const ExprHandle& a, const ExprHandle& b) {
              return Max::make(a, b, true);
            }) {}
};

class Minimum : public Reducer {
 public:
  Minimum(Dtype dtype)
      : Reducer(
            maximumVal(dtype.scalar_type()),
            [](const ExprHandle& a, const ExprHandle& b) {
              return Min::make(a, b, true);
            }) {}
  Minimum(const ExprHandle& initializer)
      : Reducer(initializer, [](const ExprHandle& a, const ExprHandle& b) {
          return Min::make(a, b, true);
        }) {}
};

class ReductionExpander : public IRMutator {
 public:
  StmtPtr expand(const StmtPtr& s) {
    return s->accept_mutator(this);
  }

  ExprPtr mutate(const ReduceOpPtr& v) override {
    return v->body();
  }
};

} // namespace torch::jit::tensorexpr
