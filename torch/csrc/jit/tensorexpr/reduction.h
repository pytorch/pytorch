#pragma once

#include <torch/csrc/jit/tensorexpr/buffer.h>
#include <torch/csrc/jit/tensorexpr/dim_arg.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include <functional>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {

using ParameterList = const std::vector<VarHandle>;
using ReduceInteraction = std::function<ExprHandle(ExprHandle, ExprHandle)>;

// An expression representing a Reduction operation (e.g. Sum, Max) broken into
// it's component parts: initialization, accumulation var, acquisition of value
// to be reduced and interaction.
//
// This is intended to be expanded in the loopnest and not make it to codegen.
class ReduceOp : public ExprNode<ReduceOp> {
 public:
  ReduceOp(
      const Buf* accum,
      const Expr* init,
      ExprHandle body,
      ReduceInteraction c,
      const std::vector<const Expr*>& output_args,
      const std::vector<const Var*>& reduce_args)
      : ExprNodeBase(body.dtype()),
        accumulator_(accum),
        initializer_(init),
        body_(body),
        interaction_(c),
        output_args_(output_args),
        reduce_args_(reduce_args) {}

  // return the accumulation load expression.
  const Buf* accumulator() const {
    return accumulator_;
  }

  // return a Statement which stores the initializer into the accumulation
  // buffer.
  const Expr* initializer() const {
    return initializer_;
  }

  // return the body expression which obtains the value to be reduced.
  ExprHandle body() const {
    return body_;
  }

  // returns a function encoding the interaction between accumulator and the
  // reduction value.
  ReduceInteraction interaction() const {
    return interaction_;
  }

  // returns variables associated with the output Tensor.
  const std::vector<const Expr*>& output_args() const {
    return output_args_;
  }

  // returns variables associated with the axes of reduction.
  const std::vector<const Var*>& reduce_args() const {
    return reduce_args_;
  }

  // Completes the reduction operator by applying the interaction function to
  // the accumulation and the body expression.
  ExprHandle complete() const {
    std::vector<const Expr*> indices(output_args_.begin(), output_args_.end());
    ExprHandle accum = ExprHandle(
        new Load(body_.dtype(), accumulator_, indices, new IntImm(1)));
    auto e = interaction_(accum, body_);
    return e;
  }

 private:
  const Buf* accumulator_;
  const Expr* initializer_;
  ExprHandle body_;
  ReduceInteraction interaction_;
  std::vector<const Expr*> output_args_;
  std::vector<const Var*> reduce_args_;
};

// A Reducer is a user interface describing a particular reduction operation. It
// has three components: An initializtion value, a way of interacting each value
// with the accumulation, and a method for obtaining the current value to be
// reduced. It is materialized into a ReduceOp when loop variables are known.
class Reducer {
 public:
  Reducer(ExprHandle init, ReduceInteraction& interaction)
      : init_(init.node()), interaction_(interaction) {}

  Reducer(ExprHandle init, ReduceInteraction& interaction, Buffer& buf)
      : init_(init.node()), interaction_(interaction) {}

  template <typename RI>
  Reducer(ExprHandle init, RI interaction) : init_(init.node()) {
    interaction_ = interaction;
  }

  ReduceOp* operator()(
      Buf* result_buf,
      ExprHandle body,
      std::vector<const Expr*> output,
      std::vector<const Var*> inner) const {
    return new ReduceOp(
        result_buf,
        new Cast(body.dtype(), init_),
        body,
        interaction_,
        output,
        inner);
  }

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

 private:
  const Expr* init_;
  ReduceInteraction interaction_;
};

class Sum : public Reducer {
 public:
  Sum()
      : Reducer(ExprHandle(0), [](ExprHandle a, ExprHandle b) {
          return a + b;
        }) {}
};

namespace {
ExprHandle maximumVal(ScalarType type) {
  switch (type) {
#define MAX_BY_TYPE_CASE(Type, Name) \
  case ScalarType::Name:             \
    return ExprHandle(std::numeric_limits<Type>::max());
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, MAX_BY_TYPE_CASE)
#undef MAX_BY_TYPE_CASE
    default:
      throw unsupported_dtype();
  }
  return ExprHandle();
}

static ExprHandle minimumVal(ScalarType type) {
  switch (type) {
#define MAX_BY_TYPE_CASE(Type, Name) \
  case ScalarType::Name:             \
    return ExprHandle(std::numeric_limits<Type>::min());
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, MAX_BY_TYPE_CASE)
#undef MAX_BY_TYPE_CASE
    default:
      throw unsupported_dtype();
  }
}
} // namespace

class Maximum : public Reducer {
 public:
  // TODO possible to remove this arg by deferring the init value until we know
  // the dtype of the body.
  Maximum(Dtype dtype)
      : Reducer(
            minimumVal(dtype.scalar_type()),
            [](ExprHandle a, ExprHandle b) { return Max::make(a, b, true); }) {}
  Maximum(ExprHandle initializer)
      : Reducer(initializer, [](ExprHandle a, ExprHandle b) {
          return Max::make(a, b, true);
        }) {}
};

class Minimum : public Reducer {
 public:
  Minimum(Dtype dtype)
      : Reducer(
            maximumVal(dtype.scalar_type()),
            [](ExprHandle a, ExprHandle b) { return Min::make(a, b, true); }) {}
  Minimum(ExprHandle initializer)
      : Reducer(initializer, [](ExprHandle a, ExprHandle b) {
          return Min::make(a, b, true);
        }) {}
};

class ReductionInitCleaner : public IRMutator {
 public:
  Stmt* clean(Stmt* s) {
    return s->accept_mutator(this);
  }

  const Expr* mutate(const ReduceOp* v) override {
    if (v->initializer()->dtype() == kVoid) {
      return v;
    }

    return new ReduceOp(
        v->accumulator(),
        new NoOp(),
        v->body(),
        v->interaction(),
        v->output_args(),
        v->reduce_args());
    return v->complete().node();
  }
};

class ReductionExpander : public IRMutator {
 public:
  Stmt* expand(Stmt* s) {
    Stmt* s_new = s->accept_mutator(this);
    if (!initializers_.empty()) {
      throw std::runtime_error("failed to initialize all reductions");
    }

    return s_new;
  }

  Stmt* mutate(const For* v) override {
    Stmt* body_new = v->body()->accept_mutator(this);
    if (body_new == v->body()) {
      body_new = Stmt::clone(v->body());
    }

    Stmt* ret = v->cloneWithNewBody(body_new);

    for (size_t i = 0; i < initializers_.size();) {
      InitializerInfo& info = initializers_[i];

      auto end = std::remove(info.vars.begin(), info.vars.end(), v->var());
      if (end == info.vars.end()) {
        info.skipped_loops.push_back(v);
        i++;
        continue;
      }

      info.vars.erase(end);
      if (info.vars.empty()) {
        const ReduceOp* op = info.op;
        std::vector<const Expr*> indices(
            op->output_args().begin(), op->output_args().end());

        Stmt* init = new Store(
            op->accumulator(), indices, op->initializer(), new IntImm(1));

        for (auto it = info.skipped_loops.rbegin();
             it != info.skipped_loops.rend();
             it++) {
          const For* old_for = *it;
          init = old_for->cloneWithNewBody(init);
        }
        info.skipped_loops.clear();

        if (Block* b = dynamic_cast<Block*>(ret)) {
          b->prepend_stmt(init);
        } else {
          ret = new Block({init, ret});
        }
        initializers_.erase(initializers_.begin() + i);
        continue;
      }

      i++;
    }
    return ret;
  }

  const Expr* mutate(const ReduceOp* v) override {
    if (v->initializer()->dtype() != kVoid) {
      const std::vector<const Var*>& reduce_vars(v->reduce_args());
      initializers_.emplace_back(InitializerInfo(v, reduce_vars));
    }

    return v->complete().node();
  }

 private:
  struct InitializerInfo {
    InitializerInfo(const ReduceOp* o, std::vector<const Var*> v)
        : op(o), vars(std::move(v)) {}
    const ReduceOp* op;
    std::vector<const Var*> vars;
    std::vector<const For*> skipped_loops;
  };

  std::vector<InitializerInfo> initializers_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
