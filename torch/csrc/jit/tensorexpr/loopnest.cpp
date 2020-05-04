#include <torch/csrc/jit/tensorexpr/loopnest.h>

#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <c10/util/Logging.h>
#include <c10/util/string_utils.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {

// Evaluates a constant expression and returns its value.
template <typename T>
static T EvalConstExpr(const ExprHandle& expr) {
  ExprEval<SimpleIREvaluator> eval(expr);
  return eval.value<T>();
}

} // namespace

class IndexFlattener : public IRMutator {
 public:
  Stmt* flatten(Stmt* s) {
    return s->accept_mutator(this);
  }
  const Expr* mutate(const Load* v) override {
    if (v->indices().size() == 1) {
      return v;
    }
    return new Load(
        v->dtype(),
        v->buf(),
        {flatten_index(v->buf()->dims(), v->indices())},
        v->mask());
  }
  Stmt* mutate(const Store* v) override {
    const Expr* value = v->value();
    const Expr* new_value = value->accept_mutator(this);
    if (v->indices().size() == 1 && value == new_value) {
      return (Stmt*)v;
    }
    return new Store(
        v->buf(),
        {flatten_index(v->buf()->dims(), v->indices())},
        new_value,
        v->mask());
  }
};

class ReductionExpander : public IRMutator {
 public:
  Stmt* expand(Stmt* s) {
    return s->accept_mutator(this);
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
    const std::vector<const Var*>& reduce_vars(v->reduce_args());
    initializers_.emplace_back(InitializerInfo(v, reduce_vars));
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

class Vectorizer : public IRMutator {
 public:
  Stmt* vectorize(const For* v) {
    Stmt* body = v->body();
    const Var* var = v->var();
    const Expr* start = v->start();
    const Expr* stop = v->stop();

    const IntImm* start_imm = dynamic_cast<const IntImm*>(start);
    const IntImm* stop_imm = dynamic_cast<const IntImm*>(stop);
    if (!start_imm) {
      throw std::runtime_error(
          "Can't vectorize due to non-constant loop start!");
    }

    if (!stop_imm) {
      throw std::runtime_error(
          "Can't vectorize due to non-constant loop stop!");
    }

    var_ = var;
    start_ = start_imm;
    lanes_ = stop_imm->value();

    Stmt* new_body = body->accept_mutator(this);
    if (new_body == body) {
      throw std::runtime_error("Vectorization failed!");
    }

    return new_body;
  }

  const Expr* mutate(const Add* v) override {
    std::vector<const Expr*> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) + ExprHandle(inputs[1]);
    });
  }

  const Expr* mutate(const Sub* v) override {
    std::vector<const Expr*> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) - ExprHandle(inputs[1]);
    });
  }

  const Expr* mutate(const Mul* v) override {
    std::vector<const Expr*> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) * ExprHandle(inputs[1]);
    });
  }

  const Expr* mutate(const Div* v) override {
    std::vector<const Expr*> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) / ExprHandle(inputs[1]);
    });
  }

  const Expr* mutate(const Max* v) override {
    std::vector<const Expr*> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return Max::make(
          ExprHandle(inputs[0]), ExprHandle(inputs[1]), v->propagate_nans());
    });
  }

  const Expr* mutate(const Min* v) override {
    std::vector<const Expr*> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return Min::make(
          ExprHandle(inputs[0]), ExprHandle(inputs[1]), v->propagate_nans());
    });
  }

  const Expr* mutate(const CompareSelect* v) override {
    std::vector<const Expr*> inputs = {
        v->lhs(), v->rhs(), v->ret_val1(), v->ret_val2()};
    return try_vectorize(v, inputs, [&]() {
      return CompareSelect::make(
          ExprHandle(inputs[0]),
          ExprHandle(inputs[1]),
          ExprHandle(inputs[2]),
          ExprHandle(inputs[3]),
          v->compare_select_op());
    });
  }

  const Expr* mutate(const Cast* v) override {
    std::vector<const Expr*> inputs = {v->src_value()};
    return try_vectorize(v, inputs, [&]() {
      return Cast::make(
          Dtype(v->dtype().scalar_type(), lanes_), ExprHandle(inputs[0]));
    });
  }

  const Expr* mutate(const Var* v) override {
    if (v == var_) {
      return Ramp::make(ExprHandle(start_), 1, lanes_).node();
    }

    return v;
  }

  const Expr* mutate(const Let* v) override {
    const Expr* var = v->var();
    const Expr* value = v->value();
    const Expr* body = v->body();

    std::vector<const Expr*> inputs = {body};
    return try_vectorize(v, inputs, [&]() {
      return Let::make(
          ExprHandle(var), ExprHandle(value), ExprHandle(inputs[0]));
    });
  }

  const Expr* mutate(const Ramp* v) override {
    const Expr* base = v->base();
    const Expr* stride = v->stride();

    const Expr* base_new = base->accept_mutator(this);
    const Expr* stride_new = stride->accept_mutator(this);

    if (base_new == base && stride_new == stride) {
      return v;
    }

    throw std::runtime_error("Can't vectorize a Ramp!");
  }

  const Expr* mutate(const Load* v) override {
    Dtype dtype(v->dtype().scalar_type(), lanes_);
    const Buf* buf = v->buf();
    std::vector<const Expr*> inputs = {v->flat_index(), v->mask()};
    return try_vectorize(v, inputs, [&]() {
      return Load::make(
          dtype,
          BufHandle(buf),
          {ExprHandle(inputs[0])},
          ExprHandle(inputs[1]));
    });
  }

  const Expr* mutate(const Broadcast* v) override {
    const Expr* val = v->value();
    const Expr* new_val = val->accept_mutator(this);
    if (new_val == val) {
      return v;
    }

    throw std::runtime_error("Can't vectorize a Broadcast!");
  }

  const Expr* mutate(const IfThenElse* v) override {
    const Expr* condition = v->condition();
    const Expr* new_condition = condition->accept_mutator(this);
    if (new_condition != condition) {
      throw std::runtime_error("Can't vectorize an IfThenElse condition!");
    }

    std::vector<const Expr*> inputs = {v->true_value(), v->false_value()};
    return try_vectorize(v, inputs, [&]() {
      return IfThenElse::make(
          ExprHandle(condition), ExprHandle(inputs[0]), ExprHandle(inputs[1]));
    });
  }

  const Expr* mutate(const BaseCallNode* v) override {
    std::vector<const Expr*> inputs = v->params();
    return try_vectorize(
        v, inputs, [&]() { return ExprHandle(DefaultMutator(v, inputs)); });
  }

  Stmt* mutate(const Store* v) override {
    const Buf* buf = v->buf();
    std::vector<const Expr*> inputs = {v->flat_index(), v->value(), v->mask()};
    return try_vectorize(v, inputs, [&]() {
      return Store::make(
          BufHandle(buf),
          {ExprHandle(inputs[0])},
          ExprHandle(inputs[1]),
          ExprHandle(inputs[2]));
    });
  }

  Stmt* mutate(const For* v) override {
    const Var* var = v->var();
    const Expr* start = v->start();
    const Expr* stop = v->stop();
    LoopOptions loop_options = v->loop_options();

    const Expr* new_start = start->accept_mutator(this);
    const Expr* new_stop = stop->accept_mutator(this);

    if (new_start != start || new_stop != stop) {
      throw std::runtime_error(
          "Can't vectorize nested For with dependent loop bounds!");
    }

    Stmt* body = v->body();
    Stmt* new_body = body->accept_mutator(this);

    if (new_body == body) {
      return (For*)v;
    }

    return new For(var, new_start, new_stop, new_body, loop_options);
  }

  template <typename T>
  const Expr* try_vectorize(
      const Expr* e,
      std::vector<const Expr*>& inputs,
      T&& vec_ctor) {
    bool vectorize = vectorize_inputs(inputs);
    if (vectorize) {
      return vec_ctor().node();
    }

    return e;
  }

  template <typename T>
  Stmt* try_vectorize(
      const Stmt* s,
      std::vector<const Expr*>& inputs,
      T&& vec_ctor) {
    bool vectorize = vectorize_inputs(inputs);
    if (vectorize) {
      return vec_ctor();
    }

    return (Stmt*)s;
  }

  bool vectorize_inputs(std::vector<const Expr*>& inputs) {
    bool any_vectorized = false;
    bool all_vectorized = true;
    std::vector<const Expr*> new_inputs;

    // Attempt to vectorize each input.
    for (const Expr*& in : inputs) {
      const Expr* new_in = in->accept_mutator(this);
      new_inputs.push_back(new_in);
      if (new_in != in) {
        any_vectorized = true;
      } else {
        all_vectorized = false;
      }
    }

    // If none of them vectorized, then don't vectorize this.
    if (!any_vectorized) {
      return false;
    }

    // Insert broadcasts for any inputs that weren't vectorized.
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] == new_inputs[i]) {
        inputs[i] = Broadcast::make(ExprHandle(inputs[i]), lanes_).node();
      } else {
        inputs[i] = new_inputs[i];
      }
    }

    // And then vectorize this node.
    return true;
  }

  const Var* var_ = nullptr;
  int lanes_ = 0;
  const Expr* start_ = nullptr;
};

void LoopNest::vectorize(Stmt* stmt) {
  For* f = dynamic_cast<For*>(stmt);
  if (!f) {
    return;
  }

  Block* b = dynamic_cast<Block*>(f->get_parent());
  if (!b) {
    return;
  }

  Vectorizer v;
  Stmt* old_f = Stmt::clone(f);
  Stmt* new_f = nullptr;
  try {
    new_f = FlattenIndexes(f);
    new_f = v.vectorize(dynamic_cast<For*>(new_f));
  } catch (std::runtime_error& e) {
    // Partial vectorization may have corrupted f
    new_f = old_f;
  }

  b->replace_stmt(f, new_f);
}

class Flattener : public IRMutator {
 private:
  Expr* mutate(const FunctionCall* v) override {
    const Tensor* t = v->tensor();
    const Buf* b = t->buf();
    Buffer buffer(BufHandle(b), t->body()->dtype());
    const std::vector<const Expr*>& params = v->params();
    std::vector<ExprHandle> params_expr(params.size());
    for (size_t i = 0; i < params.size(); i++) {
      params_expr[i] = ExprHandle(params[i]);
    }
    return buffer(params_expr).node();
  }
};

class FunctionInliner : public IRMutator {
 public:
  FunctionInliner(const std::vector<Function*>& funcs) : funcs_(funcs) {
    for (Function* func : funcs) {
      // TODO: Support multiple-output functions
      if (func->func_vars().size() != 1) {
        throw unimplemented_lowering();
      }
      func_var_set_.insert(func->func_var(0)->base_handle());
    }
  }

 protected:
  bool should_inline(Function* func) const {
    return func_var_set_.count(func->func_var(0)->base_handle()) > 0;
  }

  // For the target function, insert the caller/callee pair into the replacement
  // mapping.
  const Expr* mutate(const FunctionCall* v) override {
    Function* func = v->tensor()->function();
    const Buf* buf = v->tensor()->buf();
    // TODO: Support multiple-output functions
    if (func->func_vars().size() != 1) {
      throw unimplemented_lowering();
    }

    if (should_inline(func)) {
      // Insert the caller/callee pair into the mapping.
      for (size_t i = 0; i < buf->ndim(); i++) {
        const Var* func_callee_arg = dynamic_cast<const Var*>(func->arg(i));
        const Expr* func_caller_param = v->param(i);
        auto iter = inline_mapping_.find(func_callee_arg);
        if (iter != inline_mapping_.end()) {
          throw std::runtime_error(
              "Duplicated variables: " + func_callee_arg->name_hint());
        }
        inline_mapping_[func_callee_arg] = func_caller_param;
      }

      // Call the actual replacement.
      const Expr* body = func->body(v->tensor()->output_index());
      const Expr* result = body->accept_mutator(this);

      // Remove the caller/callee relationship.
      for (size_t i = 0; i < buf->ndim(); i++) {
        const Var* func_callee_arg = dynamic_cast<const Var*>(func->arg(i));
        auto iter = inline_mapping_.find(func_callee_arg);
        if (iter == inline_mapping_.end()) {
          throw std::runtime_error(
              "Var already removed: " + func_callee_arg->name_hint());
        }
        inline_mapping_.erase(iter);
      }
      return result;
    } else {
      return IRMutator::mutate(v);
    }
  }

  // Replace the target variable with the caller expressions.
  const Expr* mutate(const Var* v) override {
    auto iter = inline_mapping_.find(v);
    if (iter == inline_mapping_.end()) {
      return IRMutator::mutate(v);
    } else {
      const Expr* expr = iter->second;
      // Continue to transform the value from the lookup table.
      return expr->accept_mutator(this);
    }
  }

  // Remove the buffer write the inlined function.
  Stmt* mutate(const Store* v) override {
    if (func_var_set_.count(v->base_handle()) > 0) {
      return nullptr;
    } else {
      return IRMutator::mutate(v);
    }
  }

 private:
  std::unordered_map<const Var*, const Expr*> inline_mapping_;
  std::vector<Function*> funcs_;
  std::unordered_set<const Var*> func_var_set_;
};

// Inlining for functions containing rand().  Since rand() is stateful we can't
// simply inline it everywhere, or else we may generate new randoms where we
// should us a previously generated one.  As a contrived example:
//   %1 = rand()
//   %2 = %1 + 1
//   %3 = %1 - 1
//   %4 = %2 - %3
// Fully inlining this expr would, incorrectly, yield:
//   %4 = (rand() + 1) - (rand() - 1)
// when in fact the two uses of %1 should cancel.  To avoid this issue, we
// instead generate:
//   %4 = (let x = rand(); (x + 1) - (x - 1))
//
// The overall approach is to replace every rand() intrinsic with a newly
// generated variable, and then bind those variables to rand() calls in the
// body of the innermost control structure.
class RandomInliner : public FunctionInliner {
 public:
  explicit RandomInliner(const std::vector<Function*>& funcs)
      : FunctionInliner(funcs) {}

  using FunctionInliner::mutate;

  // Bind random vars in the true and false branches of a conditional.
  Stmt* mutate(const Cond* v) override {
    const Expr* cond = v->condition();
    Stmt* true_stmt = v->true_stmt();
    Stmt* false_stmt = v->false_stmt();

    const Expr* cond_new = cond->accept_mutator(this);
    Stmt* true_new = true_stmt ? true_stmt->accept_mutator(this) : true_stmt;
    true_new = bind_random_vars(true_new);
    Stmt* false_new =
        false_stmt ? false_stmt->accept_mutator(this) : false_stmt;
    false_new = bind_random_vars(false_new);

    if (cond_new == cond && true_new == true_stmt && false_new == false_stmt) {
      return const_cast<Cond*>(v); // NOLINT
    }
    return new Cond(cond_new, true_new, false_new);
  }

  // Bind random vars in the innermost loop where they are used.
  Stmt* mutate(const For* v) override {
    const Var* var = v->var();
    const Expr* start = v->start();
    const Expr* stop = v->stop();
    Stmt* body = v->body();
    LoopOptions loop_options = v->loop_options();

    Stmt* orig_body = Stmt::clone(body);
    Stmt* new_body = orig_body->accept_mutator(this);
    new_body = bind_random_vars(new_body);
    if (new_body == orig_body) {
      return const_cast<For*>(v); // NOLINT
    }
    if (new_body == nullptr) {
      return nullptr;
    }
    return new For(var, start, stop, new_body, loop_options);
  }

  // Inline calls containing rand().  Create a new random variable for each
  // call being inlined, and remember which function is currently being inlined
  // so we can look up the right variable to replace it with.
  const Expr* mutate(const FunctionCall* v) override {
    if (!should_inline(v->tensor()->function())) {
      return v;
    }
    Function* prev_func = current_func_;
    current_func_ = v->tensor()->function();

    // Remember the calling args; if we find another call with different args,
    // bail out because this case is too complicated.
    auto it = call_args_.find(current_func_);
    if (it == call_args_.end()) {
      call_args_.emplace(current_func_, std::cref(v->params()));
    } else {
      if (v->params() != it->second.get()) {
        throw std::runtime_error("Complex indexing pattern in rand() tensor");
      }
    }

    // Assign a new random variable for this function, if needed.
    if (!random_vars_.count(current_func_)) {
      const std::string& name = current_func_->func_var(0)->name_hint();
      random_vars_.emplace(current_func_, new Var(name, v->dtype()));
    }
    const Expr* result = FunctionInliner::mutate(v);
    current_func_ = prev_func;
    return result;
  }

  // Replace rand() intrinsics.
  const Expr* mutate(const Intrinsics* v) override {
    if (v->op_type() != kRand) {
      return v;
    }
    if (!current_func_) {
      return v;
    }
    auto it = random_vars_.find(current_func_);
    if (it == random_vars_.end()) {
      return v;
    }
    return it->second;
  }

 private:
  // Emit let statements for all encountered random vars, thenclear them.
  Stmt* bind_random_vars(Stmt* s) {
    for (auto const& p : random_vars_) {
      Var* v = p.second;
      s = new LetStmt(v, new Intrinsics(kRand, v->dtype()), s);
    }
    random_vars_.clear();
    return s;
  }

  // Track the function currently being inlined.
  Function* current_func_ = nullptr;

  // Map functions being inlined to the generated random variable.
  std::unordered_map<Function*, Var*> random_vars_;

  // Remember arguments of calls containing rand, and force all calls to have
  // the same argument list.  We use pointer equality of Exprs, which is
  // extremely strict but works for simple cases.
  using ArgVec = std::reference_wrapper<const std::vector<const Expr*>>;
  std::unordered_map<Function*, ArgVec> call_args_;
};

static Stmt* InjectInlines(
    Stmt* stmt,
    const std::vector<Function*>& inlined_funcs) {
  FunctionInliner inliner(inlined_funcs);
  Stmt* stmt_old = stmt;
  Stmt* stmt_new = stmt_old->accept_mutator(&inliner);
  return stmt_new;
}

static Stmt* InlineRandom(Stmt* stmt, const std::vector<Function*>& funcs) {
  RandomInliner inliner(funcs);
  return stmt->accept_mutator(&inliner);
}

class DepTracker : public IRVisitor {
 public:
  std::vector<Tensor*> findUsedTensors(Tensor* tensor) {
    used_tensors.clear();
    tensor->body()->accept(this);
    return used_tensors;
  }

 private:
  void visit(const FunctionCall* v) override {
    used_tensors.push_back(const_cast<Tensor*>(v->tensor())); // NOLINT
  }

  std::vector<Tensor*> used_tensors;
};

std::vector<Tensor*> LoopNest::findAllNeededTensors(
    const std::vector<Tensor*>& tensors) {
  DepTracker d;
  std::queue<Tensor*> q;
  std::unordered_set<Tensor*> queued;
  std::vector<Tensor*> result;
  std::unordered_set<Tensor*> processed;
  for (Tensor* t : tensors) {
    if (queued.insert(t).second) {
      q.push(t);
    }
  }
  while (!q.empty()) {
    Tensor* t = q.front();
    q.pop();
    queued.erase(t);
    std::vector<Tensor*> deps = d.findUsedTensors(t);
    bool all_processed = true;
    for (Tensor* dep : deps) {
      if (!processed.count(dep)) {
        if (queued.insert(dep).second) {
          q.push(dep);
        }
        all_processed = false;
      }
    }
    if (all_processed) {
      result.push_back(t);
      if (processed.count(t)) {
        throw malformed_input("failure to find all processed Tensors");
      }

      processed.insert(t);
    } else {
      if (queued.count(t)) {
        throw malformed_input("failure to find all queued Tensors");
      }

      q.push(t);
      queued.insert(t);
    }
  }

  return result;
}

LoopNest::LoopNest(const std::vector<Tensor*>& output_tensors)
    : output_tensors_(output_tensors.begin(), output_tensors.end()) {
  // Find all tensors we need to compute (including dependencies) and put them
  // in a topological order
  std::vector<Tensor*> tensors_to_compute =
      findAllNeededTensors(output_tensors);

  // Find all intermediate tensors, we'll need that for inserting alloc/free
  // statements
  std::unordered_set<Tensor*> tensors_to_compute_set(
      tensors_to_compute.begin(), tensors_to_compute.end());
  for (Tensor* t : tensors_to_compute) {
    if (!output_tensors_.count(t)) {
      intermediate_tensors_.insert(t);
    }
  }

  std::vector<Stmt*> loops;
  for (Tensor* t : tensors_to_compute) {
    Stmt* loop = lowerToStmt(t);
    loops.push_back(loop);
  }
  root_stmt_ = new Block(loops);
}

Stmt* LoopNest::lowerToStmt(Tensor* t) {
  Function* f = t->function();
  // TODO: Support multiple-output functions
  Stmt* body = f->ElementStmt(0);

  stmt_to_tensor_[body] = t;
  tensor_to_stmt_[t] = body;

  if (f->ndim() == 0) {
    return body;
  }

  if (f->ndim() == 0) {
    throw malformed_input("Tensor lowered to zero dimensions");
  }

  for (size_t i = 0; i < f->ndim(); i++) {
    // Going in reverse order: from innermost loop to the outermost
    size_t dim_index = f->ndim() - i - 1;
    body = new For(f->arg(dim_index), new IntImm(0), f->dim(dim_index), body);
  }
  return body;
}

void LoopNest::computeInline(Stmt* s) {
  // TODO: check if `s` is a body of a loop
  inlined_functions_.insert(stmt_to_tensor_.at(s)->function());
}

void LoopNest::computeInlineWithRandom(Stmt* s) {
  inlined_random_functions_.insert(stmt_to_tensor_.at(s)->function());
}

// TODO: Unify with DepTracker
class UseFinder : public IRVisitor {
 public:
  std::unordered_map<const Buf*, std::vector<BufUse>> findUses(Stmt* s) {
    uses_.clear();
    s->accept(this);
    return uses_;
  }

 private:
  void visit(const Store* v) override {
    if (stores_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({(Stmt*)v, true});
    }
    last_stmt_ = (Stmt*)v;
    IRVisitor::visit(v);
  }
  void visit(const Load* v) override {
    if (loads_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({last_stmt_, false});
    }
    IRVisitor::visit(v);
  }

  Stmt* last_stmt_ = nullptr;
  std::unordered_map<const Buf*, std::vector<BufUse>> uses_;

  // Sets of loads and stores in order to keep the results unique
  std::unordered_map<const Buf*, std::unordered_set<Stmt*>> loads_;
  std::unordered_map<const Buf*, std::unordered_set<Stmt*>> stores_;
};

std::unordered_map<const Buf*, std::vector<BufUse>> findUses(Stmt* s) {
  UseFinder uf;
  return uf.findUses(s);
}

class ContainedStmtsFinder : public IRVisitor {
 public:
  // Simply list all Stores and LetStmts that are children of the given stmt
  const std::unordered_set<Stmt*>& findContainedStmts(Stmt* s) {
    contained_.clear();
    s->accept(this);
    return contained_;
  }

 private:
  void visit(const Store* v) override {
    contained_.insert((Stmt*)v);
    IRVisitor::visit(v);
  }
  void visit(const LetStmt* v) override {
    contained_.insert((Stmt*)v);
    IRVisitor::visit(v);
  }

  std::unordered_set<Stmt*> contained_;
};

bool containsAll(const std::vector<BufUse>& uses, Block* b) {
  std::unordered_set<Stmt*> not_found;
  for (auto use : uses) {
    not_found.insert(use.s);
  }

  ContainedStmtsFinder csf;
  const std::unordered_set<Stmt*>& contained = csf.findContainedStmts(b);
  for (auto s : contained) {
    not_found.erase(s);
  }
  return not_found.empty();
}

Block* findParentBlock(Stmt* s) {
  while (s) {
    if (auto b = dynamic_cast<Block*>(s)) {
      return b;
    }
    s = s->get_parent();
  }
  return nullptr;
}

Block* findLowestContainingBlock(const std::vector<BufUse>& uses) {
  // TODO: we're not using the most efficient algorithm here for simplicity.
  // Replace with something more performant in case it becomes a bottleneck.
  Block* b = findParentBlock(uses[0].s);
  while (b && !containsAll(uses, b)) {
    b = findParentBlock(b->get_parent());
  }
  return b;
}

Stmt* LoopNest::insertAllocFree(Stmt* stmt) {
  // Add allocs and frees for intermediate buffers at the global level.
  // TODO: move allocs and frees to the imemediate areas to reuse buffers.
  if (intermediate_tensors_.size() == 0ULL && temp_bufs_.size() == 0ULL) {
    return stmt;
  }

  Block* b = dynamic_cast<Block*>(stmt);
  if (!b) {
    b = new Block({stmt});
  }

  // TODO: Fix the traversal, currently the order is non-deterministic
  for (Tensor* tensor : intermediate_tensors_) {
    if (inlined_functions_.count(tensor->function()) ||
        inlined_random_functions_.count(tensor->function())) {
      // No need to allocate memory for intermediate tensors.
      continue;
    }
    if (output_tensors_.count(tensor) > 0) {
      // No need to allocate memory if the tensors are given as input/output.
      continue;
    }
    Stmt* alloc = new Allocate(
        tensor->buf()->base_handle(), tensor->body()->dtype(), tensor->dims());
    Stmt* free = new Free(tensor->buf()->base_handle());
    b->prepend_stmt(alloc);
    b->append_stmt(free);
  }

  // Now insert allocations and frees for temporary buffers. Do that in the
  // innermost possible scope.
  std::unordered_map<const Buf*, std::vector<BufUse>> uses = findUses(stmt);

  for (const auto& temp_buf : temp_bufs_) {
    const Buf* buf = temp_buf.first;
    Stmt* alloc =
        new Allocate(buf->base_handle(), temp_buf.second, buf->dims());
    Stmt* free = new Free(buf->base_handle());

    Block* alloc_block = findLowestContainingBlock(uses.at(buf));
    alloc_block->prepend_stmt(alloc);
    alloc_block->append_stmt(free);
  }
  return b;
}

void LoopNest::prepareForCodegen() {
  std::vector<Function*> inlined_functions_vec(
      inlined_functions_.begin(), inlined_functions_.end());
  std::vector<Function*> inlined_randoms_vec(
      inlined_random_functions_.begin(), inlined_random_functions_.end());
  root_stmt_ = InjectInlines(root_stmt_, inlined_functions_vec);
  root_stmt_ = InlineRandom(root_stmt_, inlined_randoms_vec);

  // Expand reduction ops.
  ReductionExpander reduceExpander;
  root_stmt_ = reduceExpander.expand(root_stmt_);

  // Flatten function calls.
  Flattener flattener;
  root_stmt_ = root_stmt_->accept_mutator(&flattener);

  root_stmt_ = FlattenIndexes(root_stmt_);

  // Add allocs and frees for intermediate buffers at the global level.
  root_stmt_ = insertAllocFree(root_stmt_);
}

void LoopNest::splitWithTail(
    For* f,
    int factor,
    For** outer,
    For** inner,
    For** tail) {
  Block* p = dynamic_cast<Block*>(f->get_parent());
  if (!f) {
    throw malformed_input("splitWithTail attempted on null loop", f);
  } else if (!p) {
    throw malformed_input("splitWithTail attempted on loop with no parent", p);
  }

  bool tail_is_needed = true;
  if (dynamic_cast<const IntImm*>(f->start()) &&
      dynamic_cast<const IntImm*>(f->stop())) {
    int start_val = dynamic_cast<const IntImm*>(f->start())->value();
    int stop_val = dynamic_cast<const IntImm*>(f->stop())->value();
    int size_val = stop_val - start_val;
    int tail_size = size_val % factor;
    if (tail_size == 0) {
      tail_is_needed = false;
    }
  }

  const IntImm* factor_expr = new IntImm(factor);
  const Expr* size = new Sub(f->stop(), f->start());
  const Expr* split_count = new Div(size, factor_expr);
  const Expr* tail_size = new Mod(size, factor_expr);

  const std::string& loop_var_name = f->var()->name_hint();
  Dtype loop_var_dtype = f->var()->dtype();

  const Var* i_inner = new Var(loop_var_name + "_inner", loop_var_dtype);
  const Var* i_outer = new Var(loop_var_name + "_outer", loop_var_dtype);

  // x -> x.outer * inner.size + x.inner
  const Expr* combined_index1 = new Add(new Mul(i_outer, factor_expr), i_inner);

  Stmt* body_inner =
      Substitute(Stmt::clone(f->body()), {{f->var(), combined_index1}});

  *inner = new For(i_inner, new IntImm(0), factor_expr, body_inner);
  *outer = new For(i_outer, new IntImm(0), split_count, *inner);

  // TODO: cleanup API for adding/removing statements
  p->replace_stmt(f, *outer);

  if (tail_is_needed) {
    const Var* i_tail = new Var(loop_var_name + "_tail", loop_var_dtype);
    // x -> x.tail + outer.size * inner.size
    const Expr* combined_index2 =
        new Add(i_tail, new Mul(split_count, factor_expr));

    Stmt* body_tail =
        Substitute(Stmt::clone(f->body()), {{f->var(), combined_index2}});
    *tail = new For(i_tail, new IntImm(0), tail_size, body_tail);

    p->append_stmt(*tail);
  } else {
    *tail = nullptr;
  }

  // TODO: record history of transformations
}

void LoopNest::splitWithMask(For* f, int factor, For** outer, For** inner) {
  Block* p = dynamic_cast<Block*>(f->get_parent());
  if (!p) {
    std::cerr << "Parent is not a Block!\n";
    return;
  }

  bool tail_is_needed = true;
  if (dynamic_cast<const IntImm*>(f->start()) &&
      dynamic_cast<const IntImm*>(f->stop())) {
    int start_val = dynamic_cast<const IntImm*>(f->start())->value();
    int stop_val = dynamic_cast<const IntImm*>(f->stop())->value();
    int size_val = stop_val - start_val;
    int tail_size = size_val % factor;
    if (tail_size == 0) {
      tail_is_needed = false;
    }
  }

  const IntImm* factor_expr = new IntImm(factor);
  const Expr* size = new Sub(f->stop(), f->start());
  // split_count = (size + factor - 1) / factor
  const Expr* split_count =
      new Div(new Sub(new Add(size, factor_expr), new IntImm(1)), factor_expr);

  const std::string& loop_var_name = f->var()->name_hint();
  Dtype loop_var_dtype = f->var()->dtype();

  const Var* i_inner = new Var(loop_var_name + "_inner", loop_var_dtype);
  const Var* i_outer = new Var(loop_var_name + "_outer", loop_var_dtype);

  // x -> x.outer * inner.size + x.inner
  const Expr* combined_index = new Add(new Mul(i_outer, factor_expr), i_inner);

  Stmt* body_inner = Stmt::clone(f->body());
  // TODO: is it ok that we're doing it eagerly? In the other implementation we
  // are only materializing predicates at the last, lowering, step.
  if (tail_is_needed) {
    const IntImm* start = dynamic_cast<const IntImm*>(f->start());
    if (!start || start->value() != 0) {
      throw unimplemented_lowering();
    }

    const Expr* predicate =
        CompareSelect::make(ExprHandle(f->var()), ExprHandle(f->stop()), kLT)
            .node();
    body_inner = Cond::make(ExprHandle(predicate), body_inner, nullptr);
  }
  body_inner = Substitute(body_inner, {{f->var(), combined_index}});

  *inner = new For(i_inner, new IntImm(0), factor_expr, body_inner);
  *outer = new For(i_outer, new IntImm(0), split_count, *inner);

  // TODO: cleanup API for adding/removing statements
  p->replace_stmt(f, *outer);

  // TODO: record history of transformations
}

void LoopNest::reorderAxis(Tensor* t, For* a, For* b) {
  if (a == b) {
    // nothing to do.
    return;
  }
  // find inner and outer.
  For* outer{nullptr};
  For* inner{nullptr};
  std::deque<For*> internal_axes;

  // Find relevant axes, store reversed.
  for (For* loop : getLoopStmtsFor(t)) {
    if (loop == a || loop == b) {
      if (outer == nullptr) {
        outer = loop;
        internal_axes.push_front(loop);
      } else {
        inner = loop;
        internal_axes.push_front(loop);
      }
    } else if (outer && !inner) {
      internal_axes.push_front(loop);
    }
  }

  if (!inner || !outer) {
    throw std::runtime_error("Reordered a loop not in LoopNest");
  }

  Block* root = dynamic_cast<Block*>(outer->get_parent());
  CHECK(root);

  // Do a shallow copy of the inner blocks.
  Block* body = new Block({});
  body->splice(body->end(), inner->body());

  For* before{outer};
  For* after{nullptr};
  For* last = internal_axes.front();
  Stmt* newInner = body;

  // This is the major complexity in loop reordering: handling statements not in
  // the straight line of the reorder. To handle this we partition the tree into
  // the section before the critical path and after the critical path.
  //
  // An example of this pattern is:
  // for i in ..
  //   Statement A
  //   for j in ..
  //     Statement B
  //   Statement C
  //
  // When reordering loop i and j we need to ensure that Statement A and C are
  // still both executed with the loop extents of i, and that the three
  // statements are not reordered (as much as possible).
  for (auto* loop : internal_axes) {
    // If the inner loop had a component after the loop we must wrap it in a For
    // loop matching this level of the tree.
    if (after != nullptr) {
      after = loop->cloneWithNewBody(after);
    }

    bool pastMidpoint = false;
    bool hadBeforeStmts = false;
    for (auto I = loop->body()->begin(), E = loop->body()->end(); I != E;) {
      // Be careful not to invalidate the iterator.
      Stmt* s = *(I++);
      if (s == last) {
        // This is the midpoint.
        loop->body()->remove_stmt(s);
        if (!hadBeforeStmts) {
          // If there were no existing statements this loop does not need  to be
          // preserved and we can roll it into the above loop.
          last = loop;
        }
        pastMidpoint = true;
      } else if (pastMidpoint) {
        // Statements after the reordered path must be moved to a new tree after
        // the reordered statement has occurred to preserve ordering.
        loop->body()->remove_stmt(s);
        if (after == nullptr) {
          after = loop->cloneWithNewBody(s);
        } else {
          after->body()->append_stmt(s);
        }
      } else {
        // We can leave any statements before the reordered loop alone, so long
        // as we preserve the loop structure.
        hadBeforeStmts = true;
      }
    }
  }

  // If the top level is now empty, eliminate it.
  if (before->body()->nstmts() == 0) {
    root->remove_stmt(before);
    before = nullptr;
  }

  // now we can actually reorder the chosen axes.
  std::swap(internal_axes.front(), internal_axes.back());

  // Create the reordered internals:
  for (auto* loop : internal_axes) {
    newInner = loop->cloneWithNewBody(newInner);
  }

  // Append the new statements to the root of the tree.
  root->append_stmt(newInner);
  if (after) {
    root->append_stmt(after);
  }
} // namespace tensorexpr

std::vector<For*> LoopNest::getLoopStmtsFor(Tensor* t) const {
  std::vector<For*> result;
  Stmt* cur_stmt = tensor_to_stmt_.at(t);
  while (cur_stmt) {
    if (auto* loop = dynamic_cast<For*>(cur_stmt)) {
      result.push_back(loop);
    }
    cur_stmt = cur_stmt->get_parent();
  }
  return std::vector<For*>(result.rbegin(), result.rend());
}

void LoopNest::setGPUBlockIndex(For* f, int block_index) {
  f->set_gpu_block_index(block_index);
}

void LoopNest::setGPUThreadIndex(For* f, int thread_index) {
  f->set_gpu_thread_index(thread_index);
}

Stmt* LoopNest::getLoopBodyFor(Tensor* t) const {
  return tensor_to_stmt_.at(t);
}

bool LoopNest::hasLoopBodyFor(Tensor* t) const {
  return tensor_to_stmt_.count(t) > 0;
}

Stmt* FlattenIndexes(Stmt* s) {
  IndexFlattener idx_flattener;
  return idx_flattener.flatten(s);
}

// Auxiliary class for rewriting we're doing in `compute_at`. See
// LoopNest::computeAt for more details.
class LoopComputeAtRewriter : public IRMutator {
 public:
  LoopComputeAtRewriter(
      const Buf* buf,
      const Buf* new_buf,
      std::vector<const Expr*> offsets)
      : buf_(buf), new_buf_(new_buf), offsets_(std::move(offsets)) {}

 private:
  const Buf* buf_;
  const Buf* new_buf_;
  std::vector<const Expr*> offsets_;

  const Expr* mutate(const Load* v) override {
    if (v->buf() != buf_) {
      return v;
    }
    std::vector<const Expr*> new_indices(v->indices().size());
    for (size_t i = 0; i < v->indices().size(); i++) {
      new_indices[i] =
          IRSimplifier::simplify(new Sub(v->indices()[i], offsets_[i]));
    }
    return new Load(v->dtype(), new_buf_, new_indices, v->mask());
  }
  const Expr* mutate(const FunctionCall* v) override {
    if (v->tensor()->func_var() != buf_) {
      return v;
    }
    std::vector<const Expr*> new_indices;
    for (size_t i = 0; i < v->nparams(); i++) {
      new_indices.push_back(
          IRSimplifier::simplify(new Sub(v->param(i), offsets_[i])));
    }
    return new Load(v->dtype(), new_buf_, new_indices, new IntImm(1));
  }
};

static Store* getStoreStmtOfProducer(Stmt* s) {
  if (Store* st = dynamic_cast<Store*>(s)) {
    return st;
  }
  if (Block* b = dynamic_cast<Block*>(s)) {
    for (Stmt* ss : *b) {
      if (Store* st = dynamic_cast<Store*>(ss)) {
        return st;
      }
    }
  }
  return nullptr;
}

static std::vector<const Var*> getOuterLoopIndexes(Stmt* s) {
  std::vector<const Var*> res;
  Stmt* cur = s;
  while (cur) {
    if (auto l = dynamic_cast<For*>(cur)) {
      res.push_back(l->var());
    }
    cur = cur->get_parent();
  }
  return res;
}

/*
 * WHAT COMPUTE_AT DOES
 * ====================
 *
 * Suppose we have two loops:
 *
 * for i in 0..100:
 *   for j in 0..200:
 *     A[i,j] = sin(i*j)
 * for i in 0..100:
 *   for j in 0..199:
 *     B[i,j] = A[i,j] + A[i, j+1]
 *
 * If we compute these loops as is, we would have to allocate two buffers:
 * 100x200 for A and 100x199 for B. To decrease the memory usage one can use
 * compute_inline primitive, which would result in the following:
 *
 * for i in 0..100:
 *   for j in 0..199:
 *     B[i,j] = sin(i*j) + sin(i*(j+1))
 *
 * We now need only one buffer - 100x199 for B. However, we're now doing some
 * redundant computations: we're calling `sin` twice as much as in the first
 * version.
 *
 * Ultimately, we nede to choose at what point we prefer to compute values of
 * A[i,j] - we can do it in the very beginning for the entire buffer A (the
 * first option) or compute it on the fly when we compute B (the second option).
 * There are also options in between those two: we can compute a part of B which
 * is required for a computation of part of B, e.g. for a single row of B. The
 * code would then look like:
 *
 * for i in 0..100:
 *   for j in 0..200:
 *     A[j] = sin(i*j)
 *   for j in 0..199:
 *     B[i,j] = A[j] + A[j+1]
 *
 * In this case we're only using 1x200 for A, and we're avoiding redundant
 * computations.
 *
 * The purpose of `compute_at` is to achieve exactly this transformation.
 *
 * compute_at requires to specify What to compute and Where to compute: in our
 * example we would call compute_at(What=`A[i,j] = sin(i*j)`, Where=`for i in
 * 0..100`).
 *
 * More info about compute_at could be found in Halide's tutorials:
 * https://halide-lang.org/tutorials/tutorial_lesson_08_scheduling_2.html
 *
 * HOW COMPUTE_AT WORKS
 * ====================
 *
 * The most important part of compute_at is bounds inference: we need to figure
 * out what part of the used tensors we need to compute when we move the
 * computation to a new scope. In the example above, we need bounds inference to
 * tell us that in order to compute A at each iteration of the outer loop, we
 * need to compute A within indices [i:i+1,0:200].
 *
 * This info allows us to conclude that we need a temp buffer of size 1x200.
 *
 * Once this is known we need to insert statements for allocation and freeing
 * the temporary buffer and copy the original computation to fill the temp
 * buffer with proper values. When we copy the computation we also must rewrite
 * indices used in it: old indices are referring to the old loop and are not
 * valid in the new loop.
 *
 * To easier follow the logic, let's examine an example. Suppose we start from
 * the following loop nest:
 *   for py in 0..100:
 *     for px in 0..100:
 *       producer[py,px] = py*px
 *   for cy in 0..100:
 *     for cx in 0..100:
 *       consumer[cy,cx] = producer[cy,cx]
 *
 * And then we're running `compute_at(producer, cy)`.
 *
 * What we would like to get is the following loop nest:
 *   for py in 0..100:
 *     for px in 0..100:
 *       producer[py,px] = py*px
 *   for cy in 0..100:
 *     Allocate(temp, {1, 100})
 *     for ty in 0..1:
 *       for tx in 0..100:
 *         temp[ty,tx] = (ty+cy)*(tx+0)
 *     for cx in 0..100:
 *       consumer[cy,cx] = temp[0,cx]
 *     Free(temp)
 *
 * NB: this loop nest can and should be simplified (e.g. the producer loop can
 * be removed since its result is no longer used), but this clean-up
 * optimization is performed separately (currently, not performed at all).
 *
 * If we examine the final loop nest, we can identify that the following steps
 * needs to be performed:
 *   - Bounds inference needs to tell us that we need a 1x100 buffer for temp.
 *   - Allocate and Free statements for this buffer need to be inserted to the
 *   loop.
 *   - A new loop-nest should be inserted to the loop CY for computing `temp`
 *   and it should replicate the loopnest of producer (PY,PX loops). The indices
 *   in the loop body need to be offset by (cy, 0) - the offsets come from
 *   bounds inference too.
 *   - The computation of `consumer` needs to be rewritten so that it uses
 *   `temp` instead of `producer`. The indices in the corresponding accesses
 *   also need to be offset.
 */
void LoopNest::computeAt(Stmt* s, For* f) {
  Store* st = getStoreStmtOfProducer(s);
  if (!st) {
    return;
  }

  // Infer bounds info for all accesses that we make in the loop
  auto loop_bounds_info = inferBounds(f->body());

  // store_bounds_info holds bounds info for the store we're trying to move to
  // the loop. If its result isn't accessed in the loop at all - do nothing and
  // exit early.
  TensorAccessBoundsInfo store_bounds_info;
  bool found = false;
  for (const TensorAccessBoundsInfo& p : loop_bounds_info) {
    if (p.buf == st->buf()) {
      store_bounds_info = p;
      found = true;
    }
  }
  if (!found) {
    return;
  }

  // Compute dimensions of the temp buffer we would need to allocate
  std::vector<const Expr*> dims;
  for (size_t i = 0; i < store_bounds_info.start.size(); i++) {
    const Expr* dim = IRSimplifier::simplify(new Add(
        new Sub(store_bounds_info.stop[i], store_bounds_info.start[i]),
        new IntImm(1)));
    dims.push_back(dim);
  }

  // TODO: Use name-hint of the producer instead of "temp"
  const Buf* temp_buf =
      new Buf(new Var("temp", store_bounds_info.buf->dtype()), dims);

  // Generate index variables for 'temp'
  std::vector<const Expr*> temp_indices(dims.size());
  for (size_t i = 0; i < dims.size(); i++) {
    // TODO: Use name-hint of the producer indices instead of 'idx'
    temp_indices[i] = new Var(std::string("idx") + c10::to_string(i), kInt);
  }

  // Prepare substitute rules for constructing the temp statement from the prod
  // statement
  // TODO: Instead of going up the loop nest we should go through the indices in
  // the original tensor expression. The loops in the nest might've been
  // modified (e.g. split or merged) so that the loop indices no longer
  // correspond to the indices of the original expression and even their number
  // might be different. In that case, the loop below would crash.
  std::vector<const Var*> prod_indices = getOuterLoopIndexes(s);
  std::vector<std::pair<const Var*, const Expr*>> rewrite_indices_map;
  for (size_t i = 0; i < prod_indices.size(); i++) {
    const Expr* offset = store_bounds_info.start[i];
    rewrite_indices_map.push_back(
        {prod_indices[i], new Add(temp_indices[i], offset)});
  }
  // Construct the temp statement
  Stmt* bd = new Store(
      temp_buf,
      temp_indices,
      Substitute(st->value(), rewrite_indices_map),
      st->mask());

  // Construct the loop nest for the temp computation
  for (size_t i = 0; i < dims.size(); i++) {
    // We're creating loops from innermost to outermost, so we need to access
    // dimensions in reversed order.
    size_t dim_idx = dims.size() - 1 - i;
    bd = new For(
        dynamic_cast<const Var*>(temp_indices[dim_idx]),
        new IntImm(0),
        dims[dim_idx],
        bd);
  }

  // Add constructed stmts to the consumer loop
  f->body()->prepend_stmt(bd);

  // Rewrite accesses to producer in consumer with accesses to temp
  LoopComputeAtRewriter lr(
      store_bounds_info.buf, temp_buf, store_bounds_info.start);
  Stmt* new_f = f->accept_mutator(&lr);
  if (f != new_f) {
    Block* bb = dynamic_cast<Block*>(f->get_parent());
    bb->replace_stmt(f, new_f);
  }

  // Mark the new temp buffer as requiring an alloc (it will be inserted as a
  // part of prepareForCodegen).
  temp_bufs_.emplace_back(std::make_pair(temp_buf, st->value()->dtype()));
}

class SwapReduce : public IRMutator {
 public:
  SwapReduce(ReduceOp* new_reduce) : new_reduce_(new_reduce) {}

  Stmt* mutate(const Store* v) override {
    if (dynamic_cast<const ReduceOp*>(v->value())) {
      auto buf = new_reduce_->accumulator();
      return new Store(
          buf, new_reduce_->output_args(), new_reduce_, new IntImm(1));
    }
    return IRMutator::mutate(v);
  }

 private:
  ReduceOp* new_reduce_;
};

class StoreFinder : public IRVisitor {
 public:
  StoreFinder(Expr* t) : target_(t), store_(nullptr) {}
  Store* store() {
    return const_cast<Store*>(store_); // NOLINT: TODO fix up const correctness
  }
  void visit(const Store* s) override {
    if (s->value() == target_) {
      store_ = s;
    }
    IRVisitor::visit(s);
  }

 private:
  Expr* target_;
  const Store* store_;
};

void LoopNest::rfactor(
    const Expr* r,
    const Var* reduction_var,
    Block* insertion_point) {
  ReduceOp* reduce_op = dynamic_cast<ReduceOp*>(
      const_cast<Expr*>(r)); // NOLINT: TODO add update()
  if (!reduce_op) {
    std::cerr << "Must pass in reduce op\n";
    return;
  }
  StoreFinder sf(reduce_op);
  root_stmt()->accept(&sf);
  Stmt* st = sf.store();
  if (!st) {
    std::cerr << "Can't find reduction to rfactor " << *reduce_op << "\n";
    return;
  }

  For* root_for = nullptr;
  For* target_for = nullptr;
  std::set<const Var*> reduce_args = {reduce_op->reduce_args().begin(),
                                      reduce_op->reduce_args().end()};
  while (st) {
    auto f = dynamic_cast<For*>(st);
    if (f) {
      if (f->var() == reduction_var) {
        target_for = f;
      }
      if (reduce_args.count(f->var())) {
        reduce_args.erase(f->var());
        root_for = f;
      }
    }
    st = st->get_parent();
  };
  if (!target_for) {
    std::cerr << "Couldn't find loop over variable: " << *reduction_var << "\n";
    return;
  }

  if (reduce_args.size()) {
    std::cerr << "Couldn't find all variables associated with the reduction.\n";
    return;
  }

  if (!root_for) {
    std::cerr << "Couldn't deduce the root For loop for this rfactor\n";
    return;
  }

  auto& dims = reduce_op->reduce_args();
  if (dims.size() < 2) {
    std::cerr
        << "Cannot rfactor reduction with a single reduce variable.  Use split first.\n";
    return;
  }

  std::vector<const Expr*> new_dims = {};
  Buf* tmp_buf = new Buf(new Var("tmp_buf", kHandle), new_dims);

  auto old_acc = reduce_op->accumulator();
  auto old_init_expr = reduce_op->initializer();
  auto new_inner = reduce_op->reduce_args();
  auto new_outer = reduce_op->output_args();
  bool found = false;
  for (size_t i = 0; i < new_inner.size(); ++i) {
    if (new_inner[i] == reduction_var) {
      new_inner.erase(new_inner.begin() + i);
      found = true;
      break;
    }
  }
  if (!found) {
    std::stringstream ss;
    for (auto& v : new_inner) {
      ss << *v;
      if (&v != &new_inner.back()) {
        ss << ", ";
      }
    }
    std::cerr << "Couldn't find target reduction var " << *reduction_var
              << " in the reduce operation, which reduces over " << ss.str()
              << "\n";
    return;
  }
  new_outer.emplace_back(reduction_var);

  auto first_reduce = new ReduceOp(
      tmp_buf,
      old_init_expr,
      reduce_op->body(),
      reduce_op->interaction(),
      new_outer,
      new_inner);

  auto second_reduce_load_indices = reduce_op->output_args();
  second_reduce_load_indices.emplace_back(reduction_var);
  auto second_reduce_load = ExprHandle(new Load(
      reduce_op->body().dtype(),
      tmp_buf,
      second_reduce_load_indices,
      new IntImm(1)));
  auto second_reduce = new ReduceOp(
      old_acc,
      reduce_op->initializer(),
      second_reduce_load,
      reduce_op->interaction(),
      reduce_op->output_args(),
      {reduction_var});

  // 1) replace target for loop (which is a reduction loop)
  // with an iterative for loop by removing the reduction var from the
  // innermost op and creating a new temporary output buffer.
  //
  // 2) append a clone of the target for loop (which reduces over multiple
  // variables) with a reduce over only its var by replacing the reduction op
  // buffer input with the temporary output buffer and removing other reductions
  // variables.
  SwapReduce sr(first_reduce);
  auto root_block = dynamic_cast<Block*>(root_stmt());
  auto parent_block = dynamic_cast<Block*>(root_for->get_parent());
  if (!parent_block) {
    std::cerr << "Cannot rfactor a loop whose parent is not a block.\n";
    return;
  }
  auto new_root_for = root_for->accept_mutator(&sr);
  auto res = parent_block->replace_stmt(root_for, new_root_for);
  if (!res) {
    std::cerr << "Couldn't find target loop within parent block of loop nest\n";
    return;
  };

  if (insertion_point && insertion_point == root_for->body()) {
    insertion_point = dynamic_cast<For*>(new_root_for)->body();
  } else if (insertion_point) {
    throw std::runtime_error("TODO: enable non-root insertion points");
  }

  // From this point forward any errors cannot be handled silently.
  auto second_buf = dynamic_cast<const Buf*>(second_reduce->accumulator());
  std::vector<const Expr*> second_indices = {second_reduce->output_args()};
  if (insertion_point &&
      dynamic_cast<For*>(insertion_point->get_parent())->var() ==
          target_for->var()) {
    insertion_point->append_stmt(
        new Store(second_buf, second_indices, second_reduce, new IntImm(1)));
  } else {
    For* new_for = new For(
        target_for->var(),
        target_for->start(),
        target_for->stop(),
        new Store(second_buf, second_indices, second_reduce, new IntImm(1)),
        target_for->loop_options());
    if (insertion_point) {
      insertion_point->append_stmt(new_for);
    } else {
      parent_block->append_stmt(new_for);
    }
  }

  auto loop_bounds_info = inferBounds(root_stmt_);
  found = false;
  for (const TensorAccessBoundsInfo& p : loop_bounds_info) {
    if (p.buf == tmp_buf) {
      found = true;
      std::vector<const Expr*> dims;
      for (size_t i = 0; i < p.start.size(); i++) {
        const Expr* dim = IRSimplifier::simplify(
            new Add(new Sub(p.stop[i], p.start[i]), new IntImm(1)));
        dims.push_back(dim);
      }
      tmp_buf->set_dims(dims);
    }
  }
  if (!found) {
    throw std::runtime_error(
        "Hit undefined behavior in rfactor -- couldn't infer bounds.");
  }

  temp_bufs_.emplace_back(std::make_pair(tmp_buf, reduce_op->body().dtype()));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
