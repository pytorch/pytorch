#include "torch/csrc/jit/tensorexpr/schedule.h"

#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {
namespace schedule {

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

Stmt* Vectorize(const Stmt* stmt) {
  const For* f = dynamic_cast<const For*>(stmt);
  if (!f) {
    throw std::runtime_error("Statement is not a For loop!");
  }

  Vectorizer v;
  return v.vectorize(f);
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
      CHECK(func->func_vars().size() == 1);
      func_var_set_.insert(func->func_var(0));
    }
  }

 private:
  // For the target function, insert the caller/callee pair into the replacement
  // mapping.
  const Expr* mutate(const FunctionCall* v) override {
    Function* func = v->tensor()->function();
    // TODO: Support multiple-output functions
    CHECK(func->func_vars().size() == 1);
    if (func_var_set_.count(func->func_var(0)) > 0) {
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

  std::unordered_map<const Var*, const Expr*> inline_mapping_;
  std::vector<Function*> funcs_;
  std::unordered_set<const Var*> func_var_set_;
};

static Stmt* InjectInlines(
    Stmt* stmt,
    const std::vector<Function*>& inlined_funcs) {
  FunctionInliner inliner(inlined_funcs);
  Stmt* stmt_old = stmt;
  Stmt* stmt_new = stmt_old->accept_mutator(&inliner);
  return stmt_new;
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
      CHECK(!processed.count(t));
      processed.insert(t);
    } else {
      CHECK(!queued.count(t));
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
  CHECK(f->ndim() >= 1);
  for (size_t i = 0; i < f->ndim(); i++) {
    // Going in reverse order: from innermost loop to the outermost
    size_t dim_index = f->ndim() - i - 1;
    Range r(0, ExprHandle(f->dim(dim_index)));
    body = For::make(VarHandle(f->arg(dim_index)), r.start(), r.stop(), body);
  }
  return body;
}

void LoopNest::ComputeInline(Stmt* s) {
  // TODO: check if `s` is a body of a loop
  inlined_functions_.insert(stmt_to_tensor_.at(s)->function());
}

void LoopNest::ApplyInlines() {
  // TODO: check if `s` is a body of a loop
  std::vector<Function*> inlined_functions_vec(
      inlined_functions_.begin(), inlined_functions_.end());
  root_stmt_ = InjectInlines(root_stmt_, inlined_functions_vec);

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
    if (inlined_functions_.count(tensor->function()) > 0) {
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
    Stmt* s,
    int factor,
    Stmt** outer,
    Stmt** inner,
    Stmt** tail) {
  Block* p = dynamic_cast<Block*>(s->get_parent());
  For* f = dynamic_cast<For*>(s);
  CHECK(f && p);

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

  Stmt* body_inner = Substitute(Stmt::clone(f->body()), {{f->var(), combined_index1}});

  *inner = For::make(i_inner, 0, factor, body_inner);
  *outer = For::make(i_outer, 0, split_count, *inner);

  // TODO: cleanup API for adding/removing statements
  p->replace_stmt(s, *outer);

  if (tail_is_needed) {
    VarHandle i_tail(loop_var_name + "_tail", loop_var_dtype);
    // x -> x.tail + outer.size * inner.size
    auto combined_index2 = i_tail + split_count * factor;

    Stmt* body_tail = Substitute(Stmt::clone(f->body()), {{f->var(), combined_index2}});
    *tail = For::make(i_tail, 0, tail_size, body_tail);

    p->append_stmt(*tail);
  }

  // TODO: record history of transformations
}

void LoopNest::SplitWithMask(Stmt* s, int factor, Stmt** outer, Stmt** inner) {
  Block* p = dynamic_cast<Block*>(s->get_parent());
  For* f = dynamic_cast<For*>(s);
  if (!f) {
    std::cerr << "Stmt is not a For loop!\n";
    return;
  }
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
    CHECK(start && start->value() == 0)
        << "Non-zero start is not implemented yet";
    const Expr* predicate =
        CompareSelect::make(ExprHandle(f->var()), ExprHandle(f->stop()), kLT)
            .node();
    body_inner = Cond::make(ExprHandle(predicate), body_inner, nullptr);
  }
  body_inner = Substitute(body_inner, {{f->var(), combined_index}});

  *inner = For::make(i_inner, 0, factor, body_inner);
  *outer = For::make(i_outer, 0, split_count, *inner);

  // TODO: cleanup API for adding/removing statements
  p->replace_stmt(s, *outer);

  // TODO: record history of transformations
}

std::vector<Stmt*> LoopNest::getLoopStmtsFor(Tensor* t) const {
  std::vector<Stmt*> result;
  Stmt* cur_stmt = tensor_to_stmt_.at(t);
  while (cur_stmt) {
    if (auto* loop = dynamic_cast<For*>(cur_stmt)) {
      result.push_back(cur_stmt);
    }
    cur_stmt = cur_stmt->get_parent();
  }
  return std::vector<Stmt*>(result.rbegin(), result.rend());
}

void LoopNest::SetGPUBlockIndex(Stmt* s, int block_index) {
  For* f = dynamic_cast<For*>(s);
  if (!f) {
    std::cerr << "Stmt is not a For loop!\n";
    return;
  }
  f->set_gpu_block_index(block_index);
}

void LoopNest::SetGPUThreadIndex(Stmt* s, int thread_index) {
  For* f = dynamic_cast<For*>(s);
  if (!f) {
    std::cerr << "Stmt is not a For loop!\n";
    return;
  }
  f->set_gpu_thread_index(thread_index);
}

Stmt* LoopNest::getLoopBodyFor(Tensor* t) const {
  return tensor_to_stmt_.at(t);
}

} // namespace schedule
} // namespace tensorexpr
} // namespace jit
} // namespace torch
