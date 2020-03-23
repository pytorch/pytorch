#include <torch/csrc/jit/tensorexpr/loopnest.h>

#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
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
    const Var* base_handle = v->base_handle();
    std::vector<const Expr*> inputs = {v->index(), v->mask()};
    return try_vectorize(v, inputs, [&]() {
      return Load::make(
          dtype,
          VarHandle(base_handle),
          ExprHandle(inputs[0]),
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
    const Var* base_handle = v->base_handle();
    std::vector<const Expr*> inputs = {v->index(), v->value(), v->mask()};
    return try_vectorize(v, inputs, [&]() {
      return Store::make(
          VarHandle(base_handle),
          ExprHandle(inputs[0]),
          ExprHandle(inputs[1]),
          ExprHandle(inputs[2]));
    });
  }

  Stmt* mutate(const For* v) override {
    throw std::runtime_error("Can't vectorize nested For!");
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

void LoopNest::Vectorize(Stmt* stmt) {
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
    new_f = v.vectorize(f);
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
    Buffer buffer(
        VarHandle(t->func_var()),
        t->body()->dtype(),
        ExprVectorToExprHandleVector(t->dims()));
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
      func_var_set_.insert(func->func_var(0));
    }
  }

 protected:
  bool should_inline(Function* func) const {
    return func_var_set_.count(func->func_var(0)) > 0;
  }

  // For the target function, insert the caller/callee pair into the replacement
  // mapping.
  const Expr* mutate(const FunctionCall* v) override {
    Function* func = v->tensor()->function();
    // TODO: Support multiple-output functions
    if (func->func_vars().size() != 1) {
      throw unimplemented_lowering();
    }

    if (should_inline(func)) {
      // Insert the caller/callee pair into the mapping.
      for (int i = 0; i < func->ndim(); i++) {
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
      for (int i = 0; i < func->ndim(); i++) {
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

std::vector<Tensor*> LoopNest::FindAllNeededTensors(
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
        throw malformed_input();
      }

      processed.insert(t);
    } else {
      if (queued.count(t)) {
        throw malformed_input();
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
      FindAllNeededTensors(output_tensors);

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
    Stmt* loop = LowerToStmt(t);
    loops.push_back(loop);
  }
  root_stmt_ = new Block(loops);
}

Stmt* LoopNest::LowerToStmt(Tensor* t) {
  Function* f = t->function();
  // TODO: Support multiple-output functions
  Stmt* body = f->ElementStmt(0);

  stmt_to_tensor_[body] = t;
  tensor_to_stmt_[t] = body;

  if (f->ndim() == 0) {
    return body;
  }

  if (f->ndim() == 0) {
    throw malformed_input();
  }

  for (size_t i = 0; i < f->ndim(); i++) {
    // Going in reverse order: from innermost loop to the outermost
    size_t dim_index = f->ndim() - i - 1;
    Range r(new IntImm(0), f->dim(dim_index));
    body = new For(f->arg(dim_index), r.start(), r.stop(), body);
  }
  return body;
}

void LoopNest::ComputeInline(Stmt* s) {
  // TODO: check if `s` is a body of a loop
  inlined_functions_.insert(stmt_to_tensor_.at(s)->function());
}

void LoopNest::ComputeInlineWithRandom(Stmt* s) {
  inlined_random_functions_.insert(stmt_to_tensor_.at(s)->function());
}

void LoopNest::ApplyInlines() {
  // TODO: check if `s` is a body of a loop
  std::vector<Function*> inlined_functions_vec(
      inlined_functions_.begin(), inlined_functions_.end());
  std::vector<Function*> inlined_randoms_vec(
      inlined_random_functions_.begin(), inlined_random_functions_.end());
  root_stmt_ = InjectInlines(root_stmt_, inlined_functions_vec);
  root_stmt_ = InlineRandom(root_stmt_, inlined_randoms_vec);

  // Flatten function calls.
  Flattener flattener;
  Stmt* core_stmt = root_stmt_->accept_mutator(&flattener);

  // Add allocs and frees for intermediate buffers at the global level.
  // TODO: move allocs and frees to the imemediate areas to reuse buffers.
  if (intermediate_tensors_.size() == 0ULL) {
    root_stmt_ = core_stmt;
    return;
  }
  std::vector<Stmt*> allocs;
  std::vector<Stmt*> frees;

  // TODO: Fix the traversal, currently the order is non-deterministic
  for (Tensor* tensor : intermediate_tensors_) {
    if (inlined_functions_.count(tensor->function()) ||
        inlined_random_functions_.count(tensor->function())) {
      // No need to allocation memory for intermediate tensors.
      continue;
    }
    if (output_tensors_.count(tensor) > 0) {
      // No need to allocate memory if the tensors are given as input/output.
      continue;
    }
    Stmt* alloc = new Allocate(
        tensor->func_var(), tensor->body()->dtype(), tensor->dims());
    allocs.push_back(alloc);
    Stmt* free = new Free(tensor->func_var());
    frees.push_back(free);
  }
  std::reverse(frees.begin(), frees.end());
  Stmt* alloc_block = Block::make(allocs);
  Stmt* free_block = Block::make(frees);
  Stmt* combined_stmt = Block::make({alloc_block, core_stmt, free_block});
  root_stmt_ = combined_stmt;
}

void LoopNest::SplitWithTail(
    For* f,
    int factor,
    For** outer,
    For** inner,
    For** tail) {
  Block* p = dynamic_cast<Block*>(f->get_parent());
  if (!f) {
    throw malformed_input(f);
  } else if (!p) {
    throw malformed_input(p);
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

  auto const& size = ExprHandle(f->stop()) - ExprHandle(f->start());
  auto const& split_count = size / factor;
  auto const& tail_size = size % factor;

  const std::string& loop_var_name = f->var()->name_hint();
  Dtype loop_var_dtype = f->var()->dtype();

  VarHandle i_inner(loop_var_name + "_inner", loop_var_dtype);
  VarHandle i_outer(loop_var_name + "_outer", loop_var_dtype);

  // x -> x.outer * inner.size + x.inner
  auto combined_index1 = i_outer * factor + i_inner;

  Stmt* body_inner =
      Substitute(Stmt::clone(f->body()), {{f->var(), combined_index1}});

  *inner = For::make(i_inner, 0, factor, body_inner);
  *outer = For::make(i_outer, 0, split_count, *inner);

  // TODO: cleanup API for adding/removing statements
  p->replace_stmt(f, *outer);

  if (tail_is_needed) {
    VarHandle i_tail(loop_var_name + "_tail", loop_var_dtype);
    // x -> x.tail + outer.size * inner.size
    auto combined_index2 = i_tail + split_count * factor;

    Stmt* body_tail =
        Substitute(Stmt::clone(f->body()), {{f->var(), combined_index2}});
    *tail = For::make(i_tail, 0, tail_size, body_tail);

    p->append_stmt(*tail);
  } else {
    *tail = nullptr;
  }

  // TODO: record history of transformations
}

void LoopNest::SplitWithMask(For* f, int factor, For** outer, For** inner) {
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

  auto const& size = ExprHandle(f->stop()) - ExprHandle(f->start());
  auto const& split_count = (size + factor - 1) / factor;

  const std::string& loop_var_name = f->var()->name_hint();
  Dtype loop_var_dtype = f->var()->dtype();

  VarHandle i_inner(loop_var_name + "_inner", loop_var_dtype);
  VarHandle i_outer(loop_var_name + "_outer", loop_var_dtype);

  // x -> x.outer * inner.size + x.inner
  auto combined_index = i_outer * factor + i_inner;

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

  *inner = For::make(i_inner, 0, factor, body_inner);
  *outer = For::make(i_outer, 0, split_count, *inner);

  // TODO: cleanup API for adding/removing statements
  p->replace_stmt(f, *outer);

  // TODO: record history of transformations
}

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

void LoopNest::SetGPUBlockIndex(For* f, int block_index) {
  f->set_gpu_block_index(block_index);
}

void LoopNest::SetGPUThreadIndex(For* f, int thread_index) {
  f->set_gpu_thread_index(thread_index);
}

Stmt* LoopNest::getLoopBodyFor(Tensor* t) const {
  return tensor_to_stmt_.at(t);
}

bool LoopNest::hasLoopBodyFor(Tensor* t) const {
  return tensor_to_stmt_.count(t) > 0;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
