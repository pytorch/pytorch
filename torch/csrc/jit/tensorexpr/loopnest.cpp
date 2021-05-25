#include <torch/csrc/jit/tensorexpr/loopnest.h>

#include <algorithm>
#include <stdexcept>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <c10/util/Logging.h>
#include <c10/util/string_utils.h>

#include <ATen/core/functional.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_verifier.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

LoopNest::LoopNest(const LoopNest& other)
    : root_stmt_(Stmt::clone(other.root_stmt_)),
      output_bufs_(other.output_bufs_) {
  verify(root_stmt_);
}

LoopNest::LoopNest(Stmt* stmt, std::unordered_set<const Buf*> output_bufs)
    : root_stmt_(stmt), output_bufs_(std::move(output_bufs)) {
  verify(root_stmt_);
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
LoopNest::LoopNest(
    const std::vector<Tensor*>& output_tensors,
    const std::vector<Tensor*>& tensors_to_compute) {
  initialize(output_tensors, tensors_to_compute);
  verify(root_stmt_);
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
LoopNest::LoopNest(const std::vector<Tensor*>& output_tensors) {
  initialize(output_tensors, output_tensors);
  verify(root_stmt_);
}

const std::unordered_set<const Buf*> LoopNest::getIntermediateBufs() const {
  std::unordered_set<const Buf*> result;
  auto input_bufs = getInputBufs();
  auto bufs = NodeFinder<Buf>::find(root_stmt_);
  for (auto* buf : bufs) {
    if (!output_bufs_.count(buf) && !input_bufs.count(buf)) {
      result.insert(buf);
    }
  }
  return result;
}

const std::unordered_set<const Buf*> LoopNest::getInputBufs() const {
  std::unordered_set<const Buf*> result;
  auto buf_load_store_uses = findLoadOrStoreUses(root_stmt_);
  for (const auto& kv : buf_load_store_uses) {
    bool has_store = false;
    for (const auto& use : kv.second) {
      if (use.isStore) {
        has_store = true;
        break;
      }
    }
    if (!has_store) {
      result.insert(kv.first);
    }
  }
  return result;
}

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
        v->dtype(), v->buf(), {flatten_index(v->buf()->dims(), v->indices())});
  }

  Stmt* mutate(const Store* v) override {
    const Expr* value = v->value();
    const Expr* new_value = value->accept_mutator(this);
    if (v->indices().size() == 1 && value == new_value) {
      return (Stmt*)v;
    }
    return new Store(
        v->buf(), {flatten_index(v->buf()->dims(), v->indices())}, new_value);
  }
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

  const Expr* mutate(const And* v) override {
    std::vector<const Expr*> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) & ExprHandle(inputs[1]);
    });
  }

  const Expr* mutate(const Or* v) override {
    std::vector<const Expr*> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) | ExprHandle(inputs[1]);
    });
  }

  const Expr* mutate(const Xor* v) override {
    std::vector<const Expr*> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) ^ ExprHandle(inputs[1]);
    });
  }

  const Expr* mutate(const Lshift* v) override {
    std::vector<const Expr*> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) << ExprHandle(inputs[1]);
    });
  }

  const Expr* mutate(const Rshift* v) override {
    std::vector<const Expr*> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) >> ExprHandle(inputs[1]);
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
          v->compare_select_op(),
          v->bias());
    });
  }

  const Expr* mutate(const BitCast* v) override {
    std::vector<const Expr*> inputs = {v->src_value()};
    return try_vectorize(v, inputs, [&]() {
      return BitCast::make(
          Dtype(v->dtype().scalar_type(), lanes_), ExprHandle(inputs[0]));
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
    std::vector<const Expr*> inputs = {v->flat_index()};
    return try_vectorize(v, inputs, [&]() {
      return Load::make(dtype, BufHandle(buf), {ExprHandle(inputs[0])});
    });
  }

  const Expr* mutate(const ReduceOp* v) override {
    Dtype dtype(v->dtype().scalar_type(), lanes_);

    std::vector<const Expr*> inputs = {v->body()};

    auto* out = try_vectorize(v, inputs, [&]() {
      return ExprHandle(
          new ReduceOp(inputs[0], v->reduce_args(), v->reducer()));
    });
    return out;
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

  const Expr* mutate(const Intrinsics* v) override {
    std::vector<const Expr*> inputs = v->params();
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(new Intrinsics(v->op_type(), inputs));
    });
  }

  Stmt* mutate(const Store* v) override {
    const Buf* buf = v->buf();
    std::vector<const Expr*> inputs = {v->flat_index(), v->value()};
    return try_vectorize(v, inputs, [&]() {
      return Store::make(
          BufHandle(buf), {ExprHandle(inputs[0])}, ExprHandle(inputs[1]));
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
    std::vector<const Expr*> new_inputs;

    // Attempt to vectorize each input.
    for (const Expr*& in : inputs) {
      const Expr* new_in = in->accept_mutator(this);
      new_inputs.push_back(new_in);
      if (new_in != in) {
        any_vectorized = true;
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

void LoopNest::vectorize(For* f) {
  Block* b = dynamic_cast<Block*>(f->get_parent());
  if (!b) {
    return;
  }

  // Can't vectorize reduction axes.
  auto reductions = NodeFinder<ReduceOp>::find(f);
  for (auto* r : reductions) {
    if (std::find(r->reduce_args().begin(), r->reduce_args().end(), f->var()) !=
        r->reduce_args().end()) {
      throw std::logic_error("Cannot vectorize reduction axis - rfactor first");
    }
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

  b->replace_stmt(f, IRSimplifier::simplify(new_f));
}

void LoopNest::initialize(
    const std::vector<Tensor*>& output_tensors,
    const std::vector<Tensor*>& tensors_to_compute) {
  for (auto t : output_tensors) {
    output_bufs_.insert(t->buf());
  }

  std::vector<Stmt*> loops;
  for (Tensor* t : tensors_to_compute) {
    Stmt* loop = t->stmt();
    if (loop->get_parent()) {
      std::cerr << "Error: creating a loopnest from already used Tensors\n";
      loops = {};
      break;
    }
    // Flatten initializers.
    if (Block* block = dynamic_cast<Block*>(loop)) {
      for (auto* s : block->stmts()) {
        block->remove_stmt(s);
        loops.push_back(s);
      }
    } else {
      loops.push_back(loop);
    }
  }

  root_stmt_ = new Block(loops);
}

class FunctionInliner : public IRMutator {
 public:
  FunctionInliner(Store* producer, std::unordered_set<const Buf*> outputs)
      : buf_(producer->buf()),
        producer_(producer),
        outputs_(std::move(outputs)) {
    for (auto* i : producer->indices()) {
      if (auto index_var = dynamic_cast<const Var*>(i)) {
        index_vars_.insert(index_var);
        producer_index_vars_.push_back(index_var);
      } else if (dynamic_cast<const IntImm*>(i) != nullptr) {
        // If the index can be a constant, then that dimension must have size 1
        // (since we don't support in-place writes). Resolves issue 52581.
        TORCH_INTERNAL_ASSERT(
            dynamic_cast<const IntImm*>(i)->value() == 0,
            "Constant index impression should always be zero");
        producer_index_vars_.push_back(nullptr);
      } else {
        throw std::logic_error("cannot inline Buf with compound indices");
      }
    }
  }

 private:
  const Expr* mutate_loads(const Buf* buf, std::vector<const Expr*> dims) {
    std::vector<const Var*> index_vars;
    TORCH_INTERNAL_ASSERT(buf->ndim() == producer_index_vars_.size());
    for (size_t i = 0; i < buf->ndim(); i++) {
      const Var* func_callee_arg = producer_index_vars_.at(i);
      const Expr* func_caller_param = dims.at(i);
      if (func_callee_arg == nullptr) {
        TORCH_INTERNAL_ASSERT(
            dynamic_cast<const IntImm*>(func_caller_param) != nullptr &&
                dynamic_cast<const IntImm*>(func_caller_param)->value() == 0,
            "We are implicitly assuming that if you have an index of 0, that must also be inlined into an index of 0");
        continue;
      }
      if (func_callee_arg == nullptr)
        continue;
      auto iter = inline_mapping_.find(func_callee_arg);
      if (iter != inline_mapping_.end()) {
        throw std::runtime_error(
            "Duplicated variables: " + func_callee_arg->name_hint());
      }
      // Add a mapping for each function parameter to it's source name.
      inline_mapping_[func_callee_arg] = func_caller_param;
      index_vars.push_back(func_callee_arg);
    }

    // Call the actual replacement.
    const Expr* body = producer_->value();
    const Expr* result = body->accept_mutator(this);

    // Remove the mappings we created for this function parameters.
    for (auto* v : index_vars) {
      for (auto& pair : random_bindings_) {
        if (pair.second.erase(v)) {
          const Expr* inlined = inline_mapping_[v];
          for (auto* nv : VarFinder::find(inlined)) {
            pair.second.insert(nv);
          }
        }
      }
      inline_mapping_.erase(v);
    }
    return result;
  }

  const Expr* mutate(const Load* v) override {
    const Buf* buf = v->buf();
    if (buf != buf_) {
      return IRMutator::mutate(v);
    }

    if (v->indices().size() != buf->ndim()) {
      throw malformed_input(
          "Placeholder indexed access is inconsistent with its rank", v);
    }
    return mutate_loads(buf, v->indices());
  }

  // Replace the target variable with the caller expressions.
  const Expr* mutate(const Var* v) override {
    auto iter = inline_mapping_.find(v);
    if (iter == inline_mapping_.end()) {
      return v;
    } else {
      const Expr* expr = iter->second;
      // Continue to transform the value from the lookup table.
      return expr->accept_mutator(this);
    }
  }

  // Handle random intrinsics which should be cached.
  const Expr* mutate(const Intrinsics* v) override {
    if (!in_producer_ || v->op_type() != kRand) {
      return IRMutator::mutate(v);
    }

    // Create a new Let Statment for the random variable, which we can refer to
    // multiple times and resolve the same value (ie. store it in a scalar
    // rather than the Tensor).
    const std::string& name = buf_->name_hint();
    Var* new_var = new Var(name, v->dtype());
    random_bindings_[new Let(new_var, v)] = index_vars_;
    return new_var;
  }

  // Remove the buffer write from the inlined function.
  Stmt* mutate(const Store* v) override {
    // If the buf_ is in the outputs set, keep its statement intact. Otherwise,
    // remove it.
    if (v == producer_ && !outputs_.count(buf_)) {
      in_producer_ = true;
      producer_ = dynamic_cast<const Store*>(IRMutator::mutate(v));
      TORCH_INTERNAL_ASSERT(producer_ != nullptr);
      in_producer_ = false;
      return nullptr;
    } else {
      return IRMutator::mutate(v);
    }
  }

  // Any Random Instrinsics that were turned into vars must be inserted here.
  Stmt* mutate(const Block* v) override {
    std::vector<Stmt*> stmts;
    for (Stmt* stmt : *v) {
      Stmt* stmt_new = stmt->accept_mutator(this);
      if (!stmt_new) {
        continue;
      }

      if (stmt == stmt_new) {
        stmt_new = Stmt::clone(stmt);
      }

      stmts.push_back(stmt_new);
    }

    return Block::make(stmts);
  }

  Stmt* mutate(const For* v) override {
    For* res = dynamic_cast<For*>(IRMutator::mutate(v));
    if (!res) {
      return nullptr;
    }

    // Find any random bindings that should be defined in this loops body.
    std::vector<Let*> bindings_this_loop;
    const Var* fv = v->var();
    for (auto& pair : random_bindings_) {
      auto& index_var = pair.second;
      if (index_var.erase(fv)) {
        bindings_this_loop.push_back(pair.first);
      }
    }

    for (auto* l : bindings_this_loop) {
      res->body()->prepend_stmt(l);
      random_bindings_.erase(l);
    }
    return res;
  }

 private:
  const Buf* buf_;
  const Store* producer_;

  // Index Vars present in the producer.
  std::unordered_set<const Var*> index_vars_;
  std::vector<const Var*> producer_index_vars_;

  std::unordered_map<const Var*, const Expr*> inline_mapping_;

  // In the producer's scope - we need to bind any calls to rand().
  bool in_producer_ = false;
  std::unordered_map<Let*, std::unordered_set<const Var*>> random_bindings_;
  std::unordered_set<const Buf*> outputs_;
};

bool LoopNest::computeInline(Stmt* s) {
  auto* s_store = dynamic_cast<Store*>(s);
  if (s_store == nullptr) {
    throw std::logic_error("Could not find buffer producer to inline");
  }
  return computeInline(s_store->buf());
}

bool LoopNest::computeInline(const Buf* b) {
  // If buf is used or defined in an ExternalCall, we cannot inline it
  auto buf_load_store_uses = findLoadOrStoreUses(root_stmt_);
  for (const auto& use : buf_load_store_uses.at(b)) {
    Stmt* s = use.s;
    if (dynamic_cast<ExternalCall*>(s)) {
      return false;
    }
  }

  // Find producers.
  Store* relevant_store{nullptr};
  auto stores = NodeFinder<Store>::find(root_stmt_);
  for (auto* s : stores) {
    if (s->buf() == b) {
      auto reductions = NodeFinder<ReduceOp>::find(s);
      if (!reductions.empty()) {
        // Cannot inline a reduction computation
        return false;
      }
      if (relevant_store != nullptr) {
        // Cannot inline Buf with multiple Tensors
        return false;
      }
      relevant_store = s;
    }
  }

  TORCH_INTERNAL_ASSERT(relevant_store);

  FunctionInliner inliner(relevant_store, output_bufs_);
  root_stmt_ = root_stmt_->accept_mutator(&inliner);

  return true;
}

// inlining buffers with multiple uses can create duplicated work, which can
// slow down cpu code generation but is enabled on gpu because it avoids
// difficult synchronization logic across blocks. Inlining trivial reads does
// not duplicate work
void LoopNest::inlineIntermediateBufs(bool allow_duplicated_work) {
  std::unordered_set<const Buf*> bufs_to_inline;

  auto intermediate_bufs = getIntermediateBufs();
  if (allow_duplicated_work) {
    bufs_to_inline.insert(intermediate_bufs.begin(), intermediate_bufs.end());
  } else {
    auto buf_load_store_uses = findLoadOrStoreUses(root_stmt_);
    auto input_bufs = getInputBufs();

    for (auto buf : intermediate_bufs) {
      TORCH_INTERNAL_ASSERT(buf_load_store_uses.count(buf));
      std::vector<BufLoadOrStoreUse>& uses = buf_load_store_uses[buf];
      auto stores = c10::filter(
          uses, [](const BufLoadOrStoreUse& use) { return use.isStore; });

      // if the intermediate is the buffer formed from reading in the input
      // tensors, always inline, bc we are not duplicating any work
      // and avoiding an intermediary buffer
      if (stores.size() == 1) {
        if (auto store = dynamic_cast<Store*>(stores[0].s)) {
          auto input_as_load = dynamic_cast<const Load*>(store->value());
          if (input_as_load && input_bufs.count(input_as_load->buf())) {
            bufs_to_inline.insert(buf);
            continue;
          }
        } else {
          // If S is not a store, it must be an ExternalCall.
          TORCH_INTERNAL_ASSERT(dynamic_cast<ExternalCall*>(stores[0].s));
        }
      }

      // all bufs will have at least one store (if they have > 1 they cant be
      // inlined anyway)
      size_t reads = uses.size() - 1;
      // if only one read, we can inline it without duplicating work
      if (reads <= 1) {
        bufs_to_inline.insert(buf);
      }
    }
  }

  if (allow_duplicated_work) {
    bufs_to_inline.insert(output_bufs_.begin(), output_bufs_.end());
  }

  for (auto b : bufs_to_inline) {
    computeInline(b);
  }
}

// TODO: Unify with DepTracker
class LoadOrStoreUseFinder : public IRVisitor {
 public:
  std::unordered_map<const Buf*, std::vector<BufLoadOrStoreUse>> findUses(
      Stmt* s) {
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

  void visit(const ExternalCall* v) override {
    if (stores_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({(Stmt*)v, true});
    }
    last_stmt_ = (Stmt*)v;

    for (const Buf* input_buf : v->buf_args()) {
      if (loads_[input_buf].insert(last_stmt_).second) {
        uses_[input_buf].push_back({last_stmt_, false});
      }
    }

    IRVisitor::visit(v);
  }

  void visit(const Load* v) override {
    if (loads_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({last_stmt_, false});
    }
    IRVisitor::visit(v);
  }

  Stmt* last_stmt_ = nullptr;
  std::unordered_map<const Buf*, std::vector<BufLoadOrStoreUse>> uses_;

  // Sets of loads and stores in order to keep the results unique
  std::unordered_map<const Buf*, std::unordered_set<Stmt*>> loads_;
  std::unordered_map<const Buf*, std::unordered_set<Stmt*>> stores_;
};

std::unordered_map<const Buf*, std::vector<BufLoadOrStoreUse>>
findLoadOrStoreUses(Stmt* s) {
  LoadOrStoreUseFinder uf;
  return uf.findUses(s);
}

class ContainedStmtsFinder : public IRVisitor {
 public:
  // Simply list all Stores and Block that are children of the given stmt
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
  void visit(const ExternalCall* v) override {
    contained_.insert((Stmt*)v);
    IRVisitor::visit(v);
  }
  void visit(const Block* v) override {
    contained_.insert((Stmt*)v);
    IRVisitor::visit(v);
  }

  std::unordered_set<Stmt*> contained_;
};

bool containsAll(const std::vector<BufLoadOrStoreUse>& uses, Block* b) {
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

Block* findLowestContainingBlock(const std::vector<BufLoadOrStoreUse>& uses) {
  // TODO: we're not using the most efficient algorithm here for simplicity.
  // Replace with something more performant in case it becomes a bottleneck.
  Block* b = findParentBlock(uses[0].s);
  while (b && !containsAll(uses, b)) {
    b = findParentBlock(b->get_parent());
  }
  return b;
}

Stmt* LoopNest::insertAllocFree(Stmt* stmt) {
  auto intermediate_bufs = getIntermediateBufs();
  if (intermediate_bufs.size() == 0ULL) {
    return stmt;
  }

  Block* b = dynamic_cast<Block*>(stmt);
  if (!b) {
    b = new Block({stmt});
  }

  std::unordered_map<const Buf*, std::vector<BufLoadOrStoreUse>> uses =
      findLoadOrStoreUses(stmt);
  // Insert allocations and frees for temporary buffers in the innermost
  // possible scope.
  for (const Buf* buf : intermediate_bufs) {
    Stmt* alloc = new Allocate(buf);
    Stmt* free = new Free(buf);
    Block* alloc_block = findLowestContainingBlock(uses.at(buf));
    alloc_block->prepend_stmt(alloc);
    alloc_block->append_stmt(free);
  }

  return b;
}

class StmtDeleter : public IRMutator {
 public:
  StmtDeleter(const std::unordered_set<const Stmt*>& targets)
      : targets_(targets) {}

 private:
  Stmt* mutate(const Block* v) override {
    std::vector<Stmt*> stmts;

    for (auto* s : v->stmts()) {
      if (targets_.count(s) == 0) {
        Stmt* ns = s->accept_mutator(this);
        if (ns) {
          stmts.push_back(Stmt::clone(ns));
        }
      }
    }

    return Block::make(stmts);
  }

  const std::unordered_set<const Stmt*>& targets_;
};

void LoopNest::eliminateDeadStores() {
  using namespace analysis;
  MemDependencyChecker checker(getInputBufs(), getOutputBufs());
  root_stmt_->accept(&checker);

  std::unordered_set<const Stmt*> deadStores;
  std::vector<std::shared_ptr<AccessInfo>> outputAccesses;
  for (auto* o : getOutputBufs()) {
    outputAccesses.push_back(checker.output(o));
  }

  for (auto& info : checker.getHistory()) {
    if (!info->isWrite()) {
      continue;
    }
    bool found = false;

    for (auto& output : outputAccesses) {
      if (checker.dependsIndirectly(output, info)) {
        found = true;
        break;
      }
    }

    if (!found) {
      deadStores.insert(info->stmt());
    }
  }

  StmtDeleter deleter(deadStores);
  root_stmt_ = root_stmt_->accept_mutator(&deleter);
}

void LoopNest::prepareForCodegen() {
  // Expand reduction ops.
  ReductionExpander reduceExpander;
  root_stmt_ = reduceExpander.expand(root_stmt_);

  root_stmt_ = FlattenIndexes(root_stmt_);

  // Add allocs and frees for intermediate buffers at the global level.
  root_stmt_ = insertAllocFree(root_stmt_);
}

namespace {

class IfThenElseReplacer : public IRMutator {
 public:
  IfThenElseReplacer(const IfThenElse* to_replace, const Expr* new_expr)
      : to_replace_(to_replace), new_expr_(new_expr) {}

  const Expr* mutate(const IfThenElse* i) override {
    if (i == to_replace_) {
      return new_expr_;
    }
    return i;
  }

 private:
  const IfThenElse* to_replace_;
  const Expr* new_expr_;
};

// Check if the given condition is optimizable.
// Specifically, this function looks for the following pattern:
//    "var < expr"
//
// If this pattern is found, then this function:
//   * sets `cond_var` to `var`,
//   * sets `compared_value` to `expr`, and
//   * returns true.
bool isConditionOptimizable(
    const Expr* condition,
    const Var** cond_var,
    const Expr** compared_value) {
  auto cs = dynamic_cast<const CompareSelect*>(condition);
  if (cs && cs->compare_select_op() == kLT) {
    auto var = dynamic_cast<const Var*>(cs->lhs());
    if (var) {
      *cond_var = var;
      *compared_value = cs->rhs();
      return true;
    }
  }
  return false;
}

// Checks if the given if-then-else expression is a conditional that is
// generated from `aten::cat`.
//
// The expected format of conditionals is:
//     IfThenElse(var < val1? 1 : 0,
//       IfThenElse (var < val2? 1 : 0,
//         IfThenElse (var < val3? 1 : 0,
//           sub-expr1,
//           sub-expr2),
//         sub-expr3),
//       sub-expr4)
//
// If such a conditional is found, this function also sets:
//   * cond_var to the condition variable found in this expression.
//   * comp_values to the list of compared values in the condition expressions.
//   * sub_exprs to the list of sub-expressions that are the result of this
//     if-then-else expression.
bool isConditionalFromCat(
    const IfThenElse* ite,
    const Var** cond_var,
    std::vector<const Expr*>* comp_values,
    std::vector<const Expr*>* sub_exprs) {
  const Var* var = nullptr;
  const Expr* comp_value;
  if (isConditionOptimizable(ite->condition(), &var, &comp_value)) {
    if (*cond_var == nullptr) {
      *cond_var = var;
    } else if (*cond_var != var) {
      // Different condition variables found in nested if-then-else
      // expressions. Can not optimize such cases.
      return false;
    }
    auto true_ite = dynamic_cast<const IfThenElse*>(ite->true_value());
    if (true_ite) {
      if (!isConditionalFromCat(true_ite, cond_var, comp_values, sub_exprs)) {
        return false;
      }
    } else {
      sub_exprs->push_back(ite->true_value());
    }
    auto false_ite = dynamic_cast<const IfThenElse*>(ite->false_value());
    if (false_ite) {
      return false;
    }
    comp_values->push_back(comp_value);
    sub_exprs->push_back(ite->false_value());
    return true;
  }
  return false;
}

bool areConstantsAndSorted(const std::vector<const Expr*>& comp_values) {
  std::vector<int> comp_consts;
  comp_consts.reserve(comp_values.size());
  for (auto c : comp_values) {
    if (!c->isConstant()) {
      return false;
    }
    comp_consts.push_back(immediateAs<int>(c));
  }
  return std::is_sorted(comp_consts.begin(), comp_consts.end());
}

} // namespace

bool LoopNest::optimizeConditionals() {
  // Consider every store in the root_stmt_ and try to optimize the
  // conditionals in that store.
  auto stores = NodeFinder<Store>::find(root_stmt_);
  std::unordered_set<For*> split_fors;
  for (auto store : stores) {
    const Var* cond_var = nullptr;
    // `comp_values` represent the list of compared values that will be
    // collected as we check for the expected pattern. Since that will
    // only include the RHS of the conditions in the if-then-else expressions
    // we need to start with `0` which is the initial bound, given that we
    // only handle normalized loops (check for this is done below).
    std::vector<const Expr*> comp_values = {new IntImm(0)};
    std::vector<const Expr*> sub_exprs;
    auto ifthenelse_exprs = NodeFinder<IfThenElse>::find(store);
    if (ifthenelse_exprs.empty()) {
      continue;
    }
    // We only check if the first if-then-else expression in this store
    // corresponds to a conditional of the required format. If there are more
    // than one such conditional, optimizing them requires checking if the
    // conditions are exactly the same across them and handling all of them
    // together. Currently, this is not handled.
    if (!isConditionalFromCat(
            ifthenelse_exprs.front(), &cond_var, &comp_values, &sub_exprs)) {
      continue;
    }

    auto fors = getLoopStmtsFor(store);
    if (cond_var != fors.back()->var()) {
      // Currently, we only handle the case where the condition variable
      // is the same as the inner-most loop variable.
      // TODO: Handle all other cases here.
      //
      // In order to handle all other cases, the method `clone_and_replace`
      // called below to clone the body of the loop with a new store needs
      // to recursively handle cloning of the loops and other blocks it
      // contains.
      continue;
    }

    auto for_to_split = fors.back();
    if (!LoopNest::isNormalized(for_to_split)) {
      // Do not optimize this conditional since the condition variable
      // refers to a loop that is not normalized.
      continue;
    }
    if (split_fors.count(for_to_split)) {
      // This loop has already been split while optimizing conditionals
      // earlier.
      //
      // Optimizing multiple conditionals that require splitting the same loop
      // is tricky. It requires checking if the conditions are exactly the same
      // across them and handling all of them together by splitting the loop
      // exactly once.
      //
      // Currently, this case is not supported.
      continue;
    }
    split_fors.insert(for_to_split);

    // `comp_values` needs to include the end bound, which is `for_to_split`
    // stop value.
    comp_values.push_back(for_to_split->stop());

    // Check if all `comp_values` are constants and they are sorted.
    if (!areConstantsAndSorted(comp_values)) {
      continue;
    }

    // Remove all the if-then-else expressions from this store and create
    // one loop per sub-expression.
    std::vector<Stmt*> split_loops;
    auto cond_to_replace = ifthenelse_exprs.front();
    for (size_t i = 0; i < sub_exprs.size(); ++i) {
      IfThenElseReplacer ifthenelseReplacer(cond_to_replace, sub_exprs[i]);
      auto new_store = store->accept_mutator(&ifthenelseReplacer);
      auto new_for_body =
          for_to_split->body()->clone_and_replace(store, new_store);
      auto new_for = new For(
          for_to_split->var(),
          comp_values[i],
          comp_values[i + 1],
          new_for_body);
      LoopNest::normalize(new_for);
      split_loops.push_back(new_for);
    }
    auto par = dynamic_cast<Block*>(for_to_split->get_parent());
    par->replace_stmt(for_to_split, new Block(split_loops));
  }
  root_stmt_ = IRSimplifier::simplify(root_stmt_);
  return true;
}

void LoopNest::vectorizeInnerLoops() {
  std::vector<For*> innerLoops;
  std::vector<For*> worklist;

  // Find outer-most For loops
  if (For* rootF = dynamic_cast<For*>(root_stmt_)) {
    worklist.push_back(rootF);
  } else if (Block* body = dynamic_cast<Block*>(root_stmt_)) {
    std::vector<Block*> blocks = {body};
    while (blocks.size()) {
      Block* b = blocks.back();
      blocks.pop_back();

      for (Stmt* s : *b) {
        if (For* f = dynamic_cast<For*>(s)) {
          worklist.push_back(f);
        } else if (Block* b2 = dynamic_cast<Block*>(s)) {
          blocks.push_back(b2);
        }
      }
    }
  }

  // Traverse the For loop nest find inner-most loops, which are
  // vectorization candidates.
  while (worklist.size()) {
    For* f = worklist.back();
    worklist.pop_back();

    bool containsSubLoops = false;
    if (Block* body = dynamic_cast<Block*>(f->body())) {
      for (Stmt* s2 : *body) {
        if (For* f2 = dynamic_cast<For*>(s2)) {
          containsSubLoops = true;
          worklist.push_back(f2);
        }
      }
    }

    if (!containsSubLoops) {
      innerLoops.push_back(f);
    }
  }

  // vectorize inner loops.
  for (For* loop : innerLoops) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    For* split1;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    For* tail1;

    static const int kBodyVectorWidth = 8;
    splitWithTail(loop, kBodyVectorWidth, &split1, &tail1);
    vectorize(split1);

    if (tail1) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      For* split2;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      For* tail2;
      static const int kTailVectorWidth = 4;
      splitWithTail(tail1, kTailVectorWidth, &split2, &tail2);
      vectorize(split2);
    }
  }
}

void LoopNest::sliceHead(For* f, int factor, For** head, For** tail) {
  if (dynamic_cast<const IntImm*>(f->start()) &&
      dynamic_cast<const IntImm*>(f->stop())) {
    int start_val = dynamic_cast<const IntImm*>(f->start())->value();
    int stop_val = dynamic_cast<const IntImm*>(f->stop())->value();
    int size_val = stop_val - start_val;
    if (factor >= size_val) {
      *head = f;
      *tail = nullptr;
      return;
    }
  }

  if (!f) {
    throw malformed_input("sliceHead attempted on null loop", f);
  }

  Block* p = dynamic_cast<Block*>(f->get_parent());
  if (!p) {
    throw malformed_input("sliceHead attempted on loop with no parent", p);
  }

  const Expr* head_end =
      new Min(new Add(f->start(), new IntImm(factor)), f->stop(), true);
  *head = new For(f->var(), f->start(), head_end, Stmt::clone(f->body()));
  *tail = new For(
      f->var(), head_end, f->stop(), Stmt::clone(f->body()), f->loop_options());

  p->replace_stmt(f, *head);
  p->insert_stmt_after(*tail, *head);

  if (f->loop_options().is_gpu_block_index() ||
      f->loop_options().is_gpu_thread_index()) {
    LoopNest::normalize(*tail);
  }
}
void LoopNest::sliceHead(For* f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For *head, *tail;
  sliceHead(f, factor, &head, &tail);
}

void LoopNest::sliceTail(For* f, int factor, For** head, For** tail) {
  if (dynamic_cast<const IntImm*>(f->start()) &&
      dynamic_cast<const IntImm*>(f->stop())) {
    int start_val = dynamic_cast<const IntImm*>(f->start())->value();
    int stop_val = dynamic_cast<const IntImm*>(f->stop())->value();
    int size_val = stop_val - start_val;
    if (factor >= size_val) {
      *head = nullptr;
      *tail = f;
      return;
    }
  }

  if (!f) {
    throw malformed_input("sliceTail attempted on null loop", f);
  }

  Block* p = dynamic_cast<Block*>(f->get_parent());
  if (!p) {
    throw malformed_input("sliceTail attempted on loop with no parent", p);
  }

  const Expr* tail_start =
      new Max(f->start(), new Sub(f->stop(), new IntImm(factor)), true);
  *head = new For(
      f->var(),
      f->start(),
      tail_start,
      Stmt::clone(f->body()),
      f->loop_options());
  *tail = new For(f->var(), tail_start, f->stop(), Stmt::clone(f->body()));

  p->replace_stmt(f, *head);
  p->insert_stmt_after(*tail, *head);

  if (f->loop_options().is_gpu_block_index() ||
      f->loop_options().is_gpu_thread_index()) {
    LoopNest::normalize(*head);
  }
}
void LoopNest::sliceTail(For* f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For *head, *tail;
  sliceTail(f, factor, &head, &tail);
}

void LoopNest::splitWithTail(For* f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For *inner, *tail;
  splitWithTail(f, factor, &inner, &tail);
}

void LoopNest::splitWithTail(
    For* f,
    int factor,
    For** inner,
    For** tail) {
  if (!f) {
    throw malformed_input("splitWithTail attempted on null loop", f);
  }

  Block* p = dynamic_cast<Block*>(f->get_parent());
  if (!p) {
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

  if (tail_is_needed) {
    const Var* i_tail = new Var(loop_var_name + "_tail", loop_var_dtype);
    // x -> x.tail + outer.size * inner.size
    const Expr* combined_index2 =
        new Add(i_tail, new Mul(split_count, factor_expr));

    Stmt* body_tail =
        Substitute(Stmt::clone(f->body()), {{f->var(), combined_index2}});
    *tail = new For(i_tail, new IntImm(0), tail_size, body_tail);

    p->insert_stmt_after(*tail, f);
  } else {
    *tail = nullptr;
  }

  Stmt* body_inner = Substitute(f->removeBody(), {{f->var(), combined_index1}});

  *inner = new For(i_inner, new IntImm(0), factor_expr, body_inner);
  // The input loop `f` will be the outer loop after split.
  f->setVar(i_outer);
  f->setStart(new IntImm(0));
  f->setStop(split_count);
  f->setBody(*inner);
}

void LoopNest::splitWithMask(For* f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* inner;
  splitWithMask(f, factor, &inner);
}

void LoopNest::splitWithMask(For* f, int factor, For** inner) {
  Block* p = dynamic_cast<Block*>(f->get_parent());
  if (!p) {
    std::cerr << "Parent is not a Block!\n";
    return;
  }

  bool tail_is_needed = true;
  const Expr* start = IRSimplifier::simplify(f->start());
  const Expr* stop = IRSimplifier::simplify(f->stop());
  if (start->isConstant() && stop->isConstant()) {
    int start_val = immediateAs<int>(start);
    int stop_val = immediateAs<int>(stop);
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

  Stmt* body_inner = f->removeBody();
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
  // The input loop `f` will be the outer loop after split.
  f->setVar(i_outer);
  f->setStart(new IntImm(0));
  f->setStop(split_count);
  f->setBody(*inner);
}

std::vector<For*> LoopNest::distributeLoop(
    For* loop,
    const std::unordered_set<Stmt*>& pivots) {
  TORCH_INTERNAL_ASSERT(loop);
  auto root = loop->get_parent();
  if (root == nullptr) {
    throw malformed_input("Loop without parent: ", loop);
  }
  auto root_block = dynamic_cast<Block*>(root);
  if (root_block == nullptr) {
    throw malformed_input(
        "Loop's parent must be a Block, instead found ", root);
  }

  // Extract bodies for all the loops after distribution.
  std::vector<Block*> new_loop_bodies;
  auto new_loop_body = new Block({});
  while (!loop->body()->empty()) {
    auto s = loop->body()->front();
    loop->body()->remove_stmt(s);
    new_loop_body->append_stmt(s);
    if (pivots.count(s)) {
      new_loop_bodies.push_back(new_loop_body);
      new_loop_body = new Block({});
    }
  }
  if (!new_loop_body->empty()) {
    new_loop_bodies.push_back(new_loop_body);
  }

  // The first loop body has to be in the original loop.
  loop->body()->splice(loop->body()->begin(), new_loop_bodies.front());
  std::vector<For*> new_loops = {loop};

  // Create loops for all the remaining blocks.
  // Add all the new loops to the parent block.
  for (size_t i = 1; i < new_loop_bodies.size(); ++i) {
    auto new_loop = loop->cloneWithNewBody(new_loop_bodies[i]);
    root_block->insert_stmt_after(new_loop, new_loops.back());
    new_loops.push_back(new_loop);
  }

  return new_loops;
}

std::vector<For*> LoopNest::distributeLoop(For* loop) {
  std::unordered_set<Stmt*> stmtsInBlock(
      loop->body()->begin(), loop->body()->end());
  return distributeLoop(loop, stmtsInBlock);
}

std::vector<For*> LoopNest::distributeLoopOverInnerLoops(For* loop) {
  auto loops = NodeFinder<For>::find(loop);
  std::unordered_set<Stmt*> loopsSet(loops.begin(), loops.end());
  return distributeLoop(loop, loopsSet);
}

bool areEqual(const Expr* expr1, const Expr* expr2) {
  auto diff = IRSimplifier::simplify(new Sub(expr1, expr2));
  return diff->isConstant() && (immediateAs<int>(diff) == 0);
};

bool areEqual(
    const std::vector<const Expr*>& expr_list1,
    const std::vector<const Expr*>& expr_list2) {
  if (expr_list1.size() != expr_list2.size()) {
    return false;
  }
  for (size_t i = 0; i < expr_list1.size(); ++i) {
    if (!areEqual(expr_list1[i], expr_list2[i])) {
      return false;
    }
  }
  return true;
}

bool LoopNest::hasLoopCarriedDependence(For* loop) {
  analysis::MemDependencyChecker analyzer;
  loop->accept(&analyzer);
  // High-level algorithm to check if two accesses to a buffer, A and B, one of
  // which is a Store, result in a loop-carried dependence:
  //   1. If the index expressions are equal in A and B, then that is a
  //      loop-independent dependence.
  //   2. If the index expressions are not equal in A and B:
  //       a) if the bounds on the accesses overlap, then this is a
  //          loop-carried dependence.
  //       b) if the bounds on the accesses do not overlap, then there is no
  //          dependence.
  //
  // Implementation:
  // For every pair of statements, S1 and S2, in the loop:
  //  * Get the loads and stores in S1 and S2.
  //  * For every store in S1 and load in S2 to the same buffer, if the index
  //    expressions are not equal and there is an overlap in accesses, return
  //    true to indicate a loop-carried dependence.
  //  * For every load in S1 and store in S2 to the same buffer, if the index
  //    expressions are not equal and there is an overlap in accesses, return
  //    true to indicate a loop-carried dependence.
  //  * For every store in S1 and store in S2 to the same buffer, if the index
  //    expressions are not equal and there is an overlap in accesses, return
  //    true to indicate a loop-carried dependence.
  for (auto it1 = loop->body()->begin(); it1 != loop->body()->end(); ++it1) {
    for (auto it2 = std::next(it1); it2 != loop->body()->end(); ++it2) {
      auto aStores = NodeFinder<Store>::find(*it1);
      auto aLoads = NodeFinder<Load>::find(*it1);
      auto bStores = NodeFinder<Store>::find(*it2);
      auto bLoads = NodeFinder<Load>::find(*it2);
      // ReadAfterWrite
      for (auto& aStore : aStores) {
        for (auto& bLoad : bLoads) {
          if (aStore->buf() == bLoad->buf()) {
            if (!areEqual(aStore->indices(), bLoad->indices())) {
              if (isOverlapping(analyzer, aStore, bLoad)) {
                return true;
              }
            }
          }
        }
      }
      // WriteAfterRead
      for (auto& bStore : bStores) {
        for (auto& aLoad : aLoads) {
          if (bStore->buf() == aLoad->buf()) {
            if (!areEqual(bStore->indices(), aLoad->indices())) {
              if (isOverlapping(analyzer, bStore, aLoad)) {
                return true;
              }
            }
          }
        }
      }
      // WriteAfterWrite
      for (auto& aStore : aStores) {
        for (auto& bStore : bStores) {
          if (aStore->buf() == bStore->buf()) {
            if (!areEqual(aStore->indices(), bStore->indices())) {
              if (isOverlapping(analyzer, aStore, bStore)) {
                return true;
              }
            }
          }
        }
      }
    }
  }
  return false;
}

bool LoopNest::fuseLoops(const std::vector<For*>& loops, For** fused) {
  if (loops.empty()) {
    return false;
  }
  if (loops.size() == 1) {
    *fused = loops.front();
    return true;
  }

  // Check if all the loops have the same parent.
  auto root = loops.front()->get_parent();
  for (auto l : loops) {
    auto par = l->get_parent();
    if (par == nullptr) {
      return false;
    }
    if (par != root) {
      return false;
    }
  }
  auto root_block = dynamic_cast<Block*>(root);
  if (root_block == nullptr) {
    return false;
  }

  // Currently, we only handle cases where there are no statements between
  // the given loops in their parents body. We can possibly relax this
  // constraint by allowing statements that do not affect the loops being
  // fused by performing some dependency analysis. TODO.
  auto it = root_block->begin();
  for (; it != root_block->end(); ++it) {
    if (*it == loops.front()) {
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(it != root_block->end());
  for (auto l : loops) {
    if (*it != l) {
      return false;
    }
    ++it;
  }

  // Check if bounds are the same for all the loops.
  auto first_loop = loops.front();
  auto first_loop_start = IRSimplifier::simplify(first_loop->start());
  auto first_loop_stop = IRSimplifier::simplify(first_loop->stop());
  for (size_t i = 1; i < loops.size(); ++i) {
    auto curr_loop = loops[i];
    auto curr_loop_start = IRSimplifier::simplify(curr_loop->start());
    auto curr_loop_stop = IRSimplifier::simplify(curr_loop->stop());
    if (!areEqual(curr_loop_start, first_loop_start)) {
      return false;
    }
    if (!areEqual(curr_loop_stop, first_loop_stop)) {
      return false;
    }
  }

  // A lambda to fuse all the given loops.
  auto fuse_all_loops = [](const std::vector<For*>& loops) {
    auto first_loop = loops.front();
    // Fuse the loops by taking all the statements from the second loops
    // onwards and moving them into the first loop's body.
    // This way the final fused loop will be the same as the first loop.
    for (size_t i = 1; i < loops.size(); ++i) {
      auto body = dynamic_cast<Block*>(Substitute(
          Stmt::clone(loops[i]->body()),
          {{loops[i]->var(), first_loop->var()}}));
      first_loop->body()->splice(first_loop->body()->end(), body);
    }
  };

  // We need to check if fusing the loops results in a loop-carried dependence.
  // This check can be done only after the loops are fused into one. But if the
  // check is violated, we need to return the given loops in the original form.
  // So, we create a clone of all the loops, fuse them and check for this.
  std::vector<For*> loops_copy;
  loops_copy.reserve(loops.size());
  for (const auto& l : loops) {
    loops_copy.push_back(dynamic_cast<For*>(Stmt::clone(l)));
  }
  fuse_all_loops(loops_copy);
  if (hasLoopCarriedDependence(loops_copy.front())) {
    return false;
  }

  // Now that all conditions are satisfied, we fuse the given loops.
  fuse_all_loops(loops);
  *fused = loops.front();
  for (size_t i = 1; i < loops.size(); ++i) {
    root_block->remove_stmt(loops[i]);
  }
  return true;
}

For* findOuterFor(For* a, For* b) {
  Stmt* s = b; // guess b is the latter.
  while (s != nullptr) {
    if (s == a) {
      // yes, b is after a.
      return a;
    }
    s = s->get_parent();
  }

  // check that the two are in the same loop nest.
  s = a;
  while (s != nullptr) {
    if (s == b) {
      // a is after b.
      return b;
    }
    s = s->get_parent();
  }

  // a and b have no relationship.
  return nullptr;
}

void LoopNest::reorderAxis(For* a, For* b) {
  if (a == b) {
    // nothing to do.
    return;
  }
  // find inner and outer.
  For* outer = findOuterFor(a, b);
  if (outer == nullptr) {
    throw std::runtime_error("Reordered a loop not in LoopNest");
  }

  For* inner = a == outer ? b : a;
  std::deque<For*> internal_axes;

  // Find relevant axes, store reversed.
  Stmt* s = inner;
  while (s != outer) {
    if (For* f = dynamic_cast<For*>(s)) {
      internal_axes.push_back(f);
    }

    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    s = s->get_parent();
  }

  internal_axes.push_back(outer);

  Block* root = dynamic_cast<Block*>(outer->get_parent());
  CHECK(root);

  // Do a shallow copy of the inner blocks.
  Block* body = new Block({});
  body->splice(body->end(), inner->body());

  For* before{outer};
  For* after{nullptr};
  For* last = internal_axes.front();
  Stmt* newInner = body;

  s = inner;
  while (s != outer) {
    if (auto cond = dynamic_cast<Cond*>(s->get_parent())) {
      if (s == cond->true_stmt()) {
        newInner = cond->cloneWithNewBody(newInner);
      } else {
        // s is the false branch of Cond
        newInner = cond->cloneWithNewBodies(new Block({}), newInner);
      }
    }
    s = s->get_parent();
  }

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

  // now we can actually reorder the chosen axes.
  std::swap(internal_axes.front(), internal_axes.back());

  // Create the reordered internals:
  for (auto* loop : internal_axes) {
    newInner = loop->cloneWithNewBody(newInner);
  }

  // Append the new statements to the root of the tree.
  if (before->body()->nstmts() == 0) {
    // If the top level is now empty, eliminate it.
    root->replace_stmt(before, newInner);
  } else {
    root->insert_stmt_after(newInner, before);
  }

  if (after) {
    root->insert_stmt_after(after, newInner);
  }
}

bool isTrivialPermutation(const std::vector<size_t>& permutation) {
  for (size_t i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != i) {
      return false;
    }
  }
  return true;
}

bool isValidPermutation(std::vector<size_t> permutation) {
  std::sort(permutation.begin(), permutation.end());
  return isTrivialPermutation(permutation);
}

std::vector<For*> LoopNest::reorder(
    const std::vector<For*>& loops,
    const std::vector<size_t>& permutation) {
  if (loops.size() != permutation.size()) {
    throw malformed_input("invalid permutation size");
  }
  if (isTrivialPermutation(permutation)) {
    return loops;
  }
  if (!isValidPermutation(permutation)) {
    throw malformed_input("invalid permutation for reorder");
  }
  if (loops.size() < 2) {
    return loops;
  }
  if (!areLoopsPerfectlyNested(loops)) {
    throw malformed_input("reorder is only allowed on perfectly nested loops");
  }

  auto parent = dynamic_cast<Block*>(loops.front()->get_parent());
  if (parent == nullptr) {
    throw malformed_input("parent of the loops must be a Block");
  }

  // Reorder the loops according to the permutation.
  std::vector<For*> result(loops.size());
  for (size_t i = 0; i < loops.size(); ++i) {
    result[permutation[i]] = loops[i];
  }

  // Remove the bodies from all the loops.
  auto innermost_body = loops.back()->removeBody();
  // We use an empty block statement to replace the outermost loop
  // so that we know the position where the outermost reordered loop
  // is to be inserted.
  auto empty_block = new Block({});
  parent->replace_stmt(loops.front(), empty_block);
  for (size_t i = 1; i < loops.size(); ++i) {
    auto block = dynamic_cast<Block*>(loops[i]->get_parent());
    TORCH_INTERNAL_ASSERT(block);
    block->remove_stmt(loops[i]);
  }

  // Set the new bodies after reorder for all the loops.
  for (size_t i = 0; i < result.size() - 1; ++i) {
    result[i]->setBody(result[i + 1]);
  }
  result.back()->setBody(innermost_body);
  parent->replace_stmt(empty_block, result.front());
  return result;
}

bool LoopNest::areLoopsPerfectlyNested(const std::vector<For*>& loops) {
  if (loops.size() < 2) {
    return true;
  }
  for (size_t i = 0; i < loops.size() - 1; ++i) {
    auto loop_body = loops[i]->body();
    if (loop_body->nstmts() != 1 || loop_body->front() != loops[i + 1]) {
      return false;
    }
  }
  return true;
}

void LoopNest::unroll(For* f, Stmt** unrolled) {
  Block* p = dynamic_cast<Block*>(f->get_parent());
  if (!f) {
    throw malformed_input("unroll attempted on null loop");
  } else if (!p) {
    throw malformed_input("unroll attempted on loop with no parent");
  }

  auto start_expr = IRSimplifier::simplify(f->start());
  auto stop_expr = IRSimplifier::simplify(f->stop());
  if (!start_expr->isConstant()) {
    throw std::runtime_error("Can't unroll due to non-constant loop start!");
  }
  if (!stop_expr->isConstant()) {
    throw std::runtime_error("Can't unroll due to non-constant loop stop!");
  }

  std::vector<Stmt*> unrolled_stmts;
  int start_val = immediateAs<int>(start_expr);
  int stop_val = immediateAs<int>(stop_expr);
  for (int current = start_val; current < stop_val; ++current) {
    for (const auto stmt : f->body()->stmts()) {
      auto stmt_copy = Stmt::clone(stmt);
      unrolled_stmts.push_back(Substitute(
          stmt_copy,
          {{f->var(), getImmediateByType(f->var()->dtype(), current)}}));
    }
  }
  *unrolled = new Block(unrolled_stmts);
  *unrolled = IRSimplifier::simplify(*unrolled);

  p->replace_stmt(f, *unrolled);
}

void LoopNest::unroll(For* f) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Stmt* unrolled;
  unroll(f, &unrolled);
}

bool LoopNest::isNormalized(For* f) {
  if (f->start()->isConstant()) {
    return immediateAs<int>(f->start()) == 0;
  }
  return false;
}

bool LoopNest::normalize(For* f) {
  if (!f) {
    throw malformed_input("normalize attempted on null loop");
  }

  if (isNormalized(f)) {
    // No need to normalize anymore here.
    return false;
  }

  auto for_body_normalized = Substitute(
      f->body(),
      {{f->var(), (VarHandle(f->var()) + ExprHandle(f->start())).node()}});
  f->setBody(for_body_normalized);
  f->setStop(new Sub(f->stop(), f->start()));
  f->setStart(new IntImm(0));
  return true;
}

// This function expects that there are 'num' loops perfectly nested within
// and including 'f'.
std::vector<For*> LoopNest::getLoopStmtsInLoopNest(For* f, size_t num) {
  std::vector<For*> loops(num);
  For* curr_for = f;
  loops[0] = curr_for;
  for (size_t i = 1; i < num; ++i) {
    TORCH_INTERNAL_ASSERT(curr_for->body()->nstmts() == 1);
    curr_for = dynamic_cast<For*>(curr_for->body()->front());
    TORCH_INTERNAL_ASSERT(curr_for);
    loops[i] = curr_for;
  }
  return loops;
}

bool LoopNest::flatten(const std::vector<For*>& loops, For** flattened) {
  if (loops.empty()) {
    throw malformed_input("flatten attempted on empty set of loops");
  }
  Block* p = dynamic_cast<Block*>(loops[0]->get_parent());
  if (!p) {
    throw malformed_input("flatten attempted on loops with no parent");
  }

  if (loops.size() == 1) {
    // This loop nest is already flattened.
    *flattened = loops[0];
    return false;
  }

  // Check if all the loops correspond to a perfect loopnest:
  //  * every loop except the inner-most should have only one stmt, the For.
  // Do not flatten, otherwise.
  // This check also ensures we do not flatten reduction loops.
  for (size_t i = 0; i < loops.size() - 1; ++i) {
    if ((loops[i]->body()->nstmts() != 1) ||
        (loops[i]->body()->front() != loops[i + 1])) {
      return false;
    }
  }

  // Normalize the loops before flattening.
  // We need to normalize them from inner-most to outer because once the outer
  // loop is normalized, the given pointers to inner loops point to old code.
  // For the same reason, we can't store the normalized inner loops until after
  // the outer-most loop is normalized.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  for (size_t i = 0; i < loops.size(); ++i) {
    size_t idx = loops.size() - i - 1;
    LoopNest::normalize(loops[idx]);
  }

  // 'normalized' points to the outer-most loop in the normalized loopnest.
  // Collect all the normalized loops.
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  auto normalized_loops = getLoopStmtsInLoopNest(loops.front(), loops.size());

  auto flat_var = new Var(
      normalized_loops[0]->var()->name_hint() + "_flat",
      normalized_loops[0]->var()->dtype());
  VarMapping var_mapping;
  Expr* stop = new IntImm(1);
  for (size_t i = 0; i < normalized_loops.size(); ++i) {
    size_t idx = normalized_loops.size() - i - 1;
    auto curr_loop = normalized_loops[idx];
    Expr* div = new Div(flat_var, stop);
    Expr* sub_expr = idx == 0 ? div : new Mod(div, curr_loop->stop());
    var_mapping.push_back(std::make_pair(curr_loop->var(), sub_expr));
    stop = new Mul(curr_loop->stop(), stop);
  }
  auto flattened_body =
      Substitute(normalized_loops.back()->removeBody(), var_mapping);

  normalized_loops.front()->setVar(flat_var);
  normalized_loops.front()->setStart(new IntImm(0));
  normalized_loops.front()->setStop(stop);
  normalized_loops.front()->setBody(flattened_body);
  *flattened = normalized_loops.front();
  return true;
}

bool LoopNest::flatten(const std::vector<For*>& loops) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  For* flattened;
  return flatten(loops, &flattened);
}

void LoopNest::compressBuffer(Buf* buf, Stmt* stmt) {
  if (buf->initializer()) {
    throw malformed_input("Can't compress buffer whose initializer is set");
  }

  // Loop iterations in NNC IR do not follow sequential semantics by default.
  // In other words, the iterations of the loops could be executed in any
  // random order without affecting correctness. This constraint in turn
  // implies that there can’t be any *inter-iteration* dependences
  // (or *loop-carried* dependences) in NNC loops. So, any NNC IR with such
  // dependences is considered invalid.
  //
  // Given the constraint above, for any pair of accesses to a buffer (where
  // at least one of the access is a write), the accesses must be
  // loop-independent on the innermost loop containing the accesses as well as
  // all the loops above it. So, any dimension that uses only those loop
  // variables to access the given buffer could be optimized away.
  //
  // Algorithm:
  //   * Find all the accesses to the given buf. (A)
  //   * Find the parent common to all accesses in A. (P)
  //   * Collect all the loops above P. (L)
  //   * Collect all the loop variables corresponding to L. (LV)
  //   * For every access a in A:
  //      * For the index I in every dimension of a:
  //          * If the variables in I are all in LV, mark this dimension
  //            for deletion.
  //   * For every dimension that is marked for deletion in ALL accesses in A:
  //      * Update the buffer to set the size of that dimension to 1.
  //      * Update all accesses in A to set the index in that dimension to 0.

  auto writes = WritesToBuf::find(stmt, buf);
  auto reads = StmtsReadingBuf::find(stmt, buf);

  // All buffers must be read and written at least once.
  // Is this a valid assumption? TODO
  TORCH_INTERNAL_ASSERT(!writes.empty());
  TORCH_INTERNAL_ASSERT(!reads.empty());

  // Find the parent common to all the buffer accesses.
  const Block* parent = dynamic_cast<Block*>(writes.front()->get_parent());
  TORCH_INTERNAL_ASSERT(parent);
  for (auto w : writes) {
    parent = Block::getSharedParent(parent, w);
  }
  for (auto r : reads) {
    parent = Block::getSharedParent(parent, r);
  }

  // Collect all the loops that are above the common parent.
  auto loops = LoopNest::getEnclosingLoopNest(parent);
  std::unordered_set<const Var*> loop_vars;
  for (auto l : loops) {
    loop_vars.insert(l->var());
  }

  // TODO: Need to handle other Stmts / Exprs that read / write buffers.
  auto stores = NodeFinder<Store>::find(stmt);
  auto loads = NodeFinder<Load>::find(stmt);

  // Vector to indicate which dimensions could be compressed away.
  std::vector<bool> dims(buf->dims().size(), true);
  auto check_indices = [&](const std::vector<const Expr*>& indices) {
    TORCH_INTERNAL_ASSERT(indices.size() == dims.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      auto index_vars = NodeFinder<Var>::find(indices[i]);
      for (auto iv : index_vars) {
        if (loop_vars.count(iv) == 0) {
          // A variable in this index is not in loop_vars.
          // This implies that this dimension cannot be optimized away.
          dims[i] = false;
          break;
        }
      }
    }
  };
  for (auto s : stores) {
    if (s->buf() == buf) {
      check_indices(s->indices());
    }
  }
  for (auto l : loads) {
    if (l->buf() == buf) {
      check_indices(l->indices());
    }
  }
  bool any_dim_to_compress = false;
  for (auto d : dims) {
    any_dim_to_compress |= d;
  }
  if (!any_dim_to_compress) {
    return;
  }

  // Compress buffer by removing the marked dims.
  std::vector<const Expr*> new_dims(buf->dims());
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i]) {
      new_dims[i] = new IntImm(1);
    }
  }
  buf->set_dims(new_dims);

  // Modify all access to reflect the removed dims.
  auto get_new_indices = [&](const std::vector<const Expr*>& indices) {
    TORCH_INTERNAL_ASSERT(indices.size() == dims.size());
    std::vector<const Expr*> new_indices(indices);
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i]) {
        new_indices[i] = new IntImm(0);
      }
    }
    return new_indices;
  };
  for (auto s : stores) {
    if (s->buf() == buf) {
      s->set_indices(get_new_indices(s->indices()));
    }
  }
  for (auto l : loads) {
    if (l->buf() == buf) {
      l->set_indices(get_new_indices(l->indices()));
    }
  }
}

std::vector<For*> LoopNest::getLoopStmtsFor(Tensor* t) const {
  Stmt* cur_stmt = getLoopBodyFor(t);
  return getLoopStmtsFor(cur_stmt);
}

std::vector<For*> LoopNest::getLoopStmtsFor(const Buf* buf) const {
  Stmt* cur_stmt = getLoopBodyFor(buf);
  return getLoopStmtsFor(cur_stmt);
}

std::vector<For*> LoopNest::getLoopStmtsFor(Stmt* s) const {
  std::vector<For*> result;

  while (s) {
    if (auto* loop = dynamic_cast<For*>(s)) {
      result.push_back(loop);
    }
    s = s->get_parent();
  }
  std::reverse(result.begin(), result.end());
  return result;
}

void LoopNest::setGPUBlockIndex(For* f, int block_index) {
  f->set_gpu_block_index(block_index);
}

void LoopNest::setGPUThreadIndex(For* f, int thread_index) {
  f->set_gpu_thread_index(thread_index);
}

void LoopNest::setBufferMap(
    For* f,
    const std::unordered_map<std::string, const Buf*>& map) {
  f->set_buffer_map(map);
}

Stmt* LoopNest::getLoopBodyFor(Tensor* t) const {
  return getLoopBodyFor(t->buf());
}

Stmt* LoopNest::getLoopBodyFor(const Buf* buf) const {
  auto writes = WritesToBuf::find(root_stmt_, buf);

  // special case for reduction Tensors, ignore the initializer if it's the only
  // op:
  if (writes.size() == 2) {
    if (const Store* s = dynamic_cast<const Store*>(writes.back())) {
      if (const ReduceOp* r = dynamic_cast<const ReduceOp*>(s->value())) {
        return (Stmt*)s; // NOLINT
      }
    }
  }

  const Stmt* res = nullptr;
  for (const auto* s : writes) {
    if (!res) {
      res = s;
      continue;
    }

    res = Block::getSharedParent(res, s);
  }

  return (Stmt*)res; // NOLINT
}

For* LoopNest::getParentLoop(const Stmt* st) {
  if (st == nullptr) {
    return nullptr;
  }
  auto par = st->get_parent();
  if (auto f = dynamic_cast<For*>(par)) {
    return f;
  }
  return getParentLoop(par);
}

std::vector<For*> LoopNest::getEnclosingLoopNest(const Stmt* st) {
  std::vector<For*> loops;
  auto f = getParentLoop(st);
  while (f) {
    loops.push_back(f);
    f = getParentLoop(f);
  }
  std::reverse(loops.begin(), loops.end());
  return loops;
}

std::vector<const Stmt*> LoopNest::getAllWritesToBuf(const Buf* buf) const {
  return WritesToBuf::find(root_stmt_, buf);
}

std::vector<For*> LoopNest::getAllInnermostLoopsWritingToBuf(
    const Buf* buf) const {
  auto writes = getAllWritesToBuf(buf);
  std::vector<For*> innermost_loops;
  innermost_loops.reserve(writes.size());
  for (auto w : writes) {
    innermost_loops.push_back(LoopNest::getParentLoop(w));
  }
  return innermost_loops;
}

std::vector<std::vector<For*>> LoopNest::getAllLoopNestsWritingToBuf(
    const Buf* buf) const {
  auto writes = getAllWritesToBuf(buf);
  std::vector<std::vector<For*>> loopnests;
  loopnests.reserve(writes.size());
  for (auto w : writes) {
    loopnests.emplace_back(LoopNest::getEnclosingLoopNest(w));
  }
  return loopnests;
}

Stmt* LoopNest::simplify() {
  root_stmt_ = IRSimplifier::simplify(root_stmt_);
  return root_stmt_;
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
    return new Load(v->dtype(), new_buf_, new_indices);
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

class CacheReplacer : public IRMutator {
 public:
  CacheReplacer(
      const Buf* buffer,
      const Buf* cache,
      std::vector<const Expr*>& offsets)
      : buf_(buffer), cache_(cache), offsets_(offsets) {}

 private:
  const Expr* mutate(const Load* v) override {
    const Buf* buf = v->buf();
    if (buf != buf_) {
      return IRMutator::mutate(v);
    }

    // Map indices to call-parameters.
    std::vector<const Expr*> newIndices;
    TORCH_INTERNAL_ASSERT(offsets_.size() == v->indices().size());
    for (size_t i = 0; i < v->indices().size(); ++i) {
      const Expr* index = v->indices()[i]->accept_mutator(this);
      const Expr* offset = offsets_[i];
      const Expr* sub = IRSimplifier::simplify(new Sub(index, offset));
      newIndices.push_back(sub);
    }

    return new Load(cache_, newIndices);
  }

  Stmt* mutate(const Store* v) override {
    const Buf* buf = v->buf();
    if (buf != buf_) {
      return IRMutator::mutate(v);
    }

    const Expr* newValue = v->value()->accept_mutator(this);

    // Map indices to call-parameters.
    std::vector<const Expr*> newIndices;
    TORCH_INTERNAL_ASSERT(offsets_.size() == v->indices().size());
    for (size_t i = 0; i < v->indices().size(); ++i) {
      const Expr* index = v->indices()[i]->accept_mutator(this);
      const Expr* offset = offsets_[i];
      const Expr* sub = IRSimplifier::simplify(new Sub(index, offset));
      newIndices.push_back(sub);
    }

    return new Store(cache_, newIndices, newValue);
  }

  const Buf* buf_;
  const Buf* cache_;
  std::vector<const Expr*>& offsets_;
};

LoopNest::AccessResult LoopNest::cacheAccesses(
    const Buf* producer,
    const std::string& name,
    Stmt* consumer) {
  const ReduceOp* reduceOp{nullptr};
  auto stores = NodeFinder<Store>::find(consumer);
  for (auto* store : stores) {
    if (auto ro = dynamic_cast<const ReduceOp*>(store->value())) {
      if (store->buf() != producer) {
        continue;
      }

      if (reduceOp) {
        throw std::runtime_error(
            "can only cache accesses used by at most a single reduceOp");
        return {nullptr, nullptr};
      }

      reduceOp = ro;
    }
  }

  // Check bounds but don't care about AccessKind.
  auto consumer_bounds_info = inferBounds(consumer, false);
  auto bounds_it = consumer_bounds_info.find(producer);
  if (bounds_it == consumer_bounds_info.end()) {
    throw std::runtime_error("consumer does not use the Tensor produced");
    return {nullptr, nullptr};
  }

  TORCH_INTERNAL_ASSERT(bounds_it->second.size() == 1);
  TensorAccessBoundsInfo& info = bounds_it->second[0];
  bool hasReads = info.kind == kLoad || info.kind == kMutate;
  bool hasWrites = info.kind == kStore || info.kind == kMutate;

  std::vector<std::string> var_names = {"i", "j", "k", "l", "m", "n", "o", "p"};
  std::vector<const Expr*> tmp_dims;
  std::vector<Var*> new_loop_vars;
  std::vector<const Expr*> new_loop_vars_expr;

  // Determine the size of the cache, and create a loop var for each dimension.
  for (size_t i = 0; i < info.start.size(); ++i) {
    const Expr* dim = IRSimplifier::simplify(
        new Add(new Sub(info.stop[i], info.start[i]), new IntImm(1)));

    tmp_dims.push_back(dim);

    new_loop_vars.push_back(new Var(var_names[i % var_names.size()], kInt));
    new_loop_vars_expr.push_back(new_loop_vars[i]);
  }

  // Create the var.
  Buf* tmp_buf = new Buf(new Var(name, kHandle), tmp_dims, producer->dtype());

  // determine the offsets for calls into the cache based off the loop start of
  // each axis.
  std::vector<const Expr*> tmp_params;
  for (size_t i = 0; i < new_loop_vars.size(); ++i) {
    tmp_params.push_back(new Add(new_loop_vars[i], info.start[i]));
  }

  // Replace acceses to the producer in the consumer with the cache.
  CacheReplacer replacer(producer, tmp_buf, info.start);
  Stmt* new_consumer =
      IRSimplifier::simplify(consumer->accept_mutator(&replacer));

  // replace the old consumer with the replaced consumer.
  Block* consumer_block = nullptr;
  // if the consumer is a block, we should mutate it in place.
  if ((consumer_block = dynamic_cast<Block*>(consumer))) {
    consumer_block->clear();
    consumer_block->append_stmt(new_consumer);
  } else {
    consumer_block = dynamic_cast<Block*>(consumer->get_parent());
    assert(consumer_block);
    consumer_block->replace_stmt(consumer, new_consumer);
  }

  // If there's a reduction we can't just write the result straight back to the
  // original buffer, since after parallelism the writes will race. Instead we
  // need to create a new ReduceOp.
  if (reduceOp) {
    // reduceOp means we had both loads and stores.

    // Init cache to 0.
    Stmt* tmp_init = new Store(
        tmp_buf, new_loop_vars_expr, getImmediateByType(tmp_buf->dtype(), 0));

    for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
      tmp_init =
          new For(new_loop_vars[i], new IntImm(0), tmp_dims[i], tmp_init);
    }

    consumer_block->insert_stmt_before(tmp_init, new_consumer);

    // Reduce back to the original buffer:
    Stmt* tmp_store = new Store(
        producer,
        tmp_params,
        reduceOp->reducer()(
            producer,
            ExprHandle(new Load(tmp_buf, new_loop_vars_expr)),
            tmp_params,
            {}));

    for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
      tmp_store =
          new For(new_loop_vars[i], new IntImm(0), tmp_dims[i], tmp_store);
    }

    consumer_block->insert_stmt_after(tmp_store, new_consumer);

    return std::make_pair(tmp_buf, new_consumer);
  }

  if (hasReads) {
    // Fill the cache with values from the consumer.
    Stmt* tmp_store =
        new Store(tmp_buf, new_loop_vars_expr, new Load(producer, tmp_params));

    for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
      tmp_store =
          new For(new_loop_vars[i], new IntImm(0), tmp_dims[i], tmp_store);
    }

    consumer_block->insert_stmt_before(tmp_store, new_consumer);
  }

  if (hasWrites) {
    // sync the cache back to the producer buf.
    Stmt* tmp_store =
        new Store(producer, tmp_params, new Load(tmp_buf, new_loop_vars_expr));

    for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
      tmp_store =
          new For(new_loop_vars[i], new IntImm(0), tmp_dims[i], tmp_store);
    }

    consumer_block->insert_stmt_after(tmp_store, new_consumer);
  }

  return std::make_pair(tmp_buf, new_consumer);
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

  // bounds_it holds bounds info for the store we're trying to move to
  // the loop. If its result isn't accessed in the loop at all - do nothing and
  // exit early.
  auto bounds_it = loop_bounds_info.find(st->buf());
  if (bounds_it == loop_bounds_info.end()) {
    return;
  }

  // Compute dimensions of the temp buffer we would need to allocate
  std::vector<const Expr*> dims = getBoundExtents(bounds_it->second);

  // TODO: Use name-hint of the producer instead of "temp"
  const Buf* temp_buf = new Buf("temp", dims, st->value()->dtype());

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
  std::vector<const Expr*> offsets;
  for (const TensorAccessBoundsInfo& p : bounds_it->second) {
    for (size_t i = 0; i < p.start.size(); i++) {
      if (offsets.size() <= i) {
        offsets.push_back(p.start[i]);
      } else {
        offsets[i] =
            IRSimplifier::simplify(new Min(offsets[i], p.start[i], true));
      }
    }
  }

  for (size_t i = 0; i < prod_indices.size(); i++) {
    rewrite_indices_map.push_back(
        {prod_indices[i], new Add(temp_indices[i], offsets[i])});
  }

  // Construct the temp statement
  Stmt* bd = new Store(
      temp_buf, temp_indices, Substitute(st->value(), rewrite_indices_map));

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
  LoopComputeAtRewriter lr(st->buf(), temp_buf, offsets);
  Stmt* new_f = f->accept_mutator(&lr);
  if (f != new_f) {
    Block* bb = dynamic_cast<Block*>(f->get_parent());
    bb->replace_stmt(f, new_f);
  }
}

class RfactorStoreRewriter : public IRMutator {
 public:
  RfactorStoreRewriter(
      const Buf* old_buf,
      const std::vector<const Expr*>& old_indices,
      const Buf* new_buf,
      const Var* reduction_var)
      : old_buf_(old_buf),
        old_indices_(old_indices),
        new_buf_(new_buf),
        reduction_var_(reduction_var),
        new_indices_(old_indices) {
    new_indices_.push_back(reduction_var_);
  }

  const Expr* mutate(const Load* v) override {
    if (v->buf() != old_buf_) {
      return IRMutator::mutate(v);
    }

    TORCH_INTERNAL_ASSERT(old_indices_.size() == v->indices().size());

    bool equal_indices = true;
    for (size_t i = 0; i < v->indices().size(); ++i) {
      if (!exprEquals(v->indices()[i], old_indices_[i])) {
        equal_indices = false;
        break;
      }
    }
    if (!equal_indices) {
      return IRMutator::mutate(v);
    }

    return new Load(new_buf_, new_indices_);
  }

  const Expr* mutate(const ReduceOp* v) override {
    const Expr* body_new = v->body()->accept_mutator(this);

    std::vector<const Var*> new_reduce_args;
    for (auto* r : v->reduce_args()) {
      if (r != reduction_var_) {
        new_reduce_args.push_back(r);
      }
    }

    return new ReduceOp(body_new, new_reduce_args, v->reducer());
  }

  Stmt* mutate(const Store* v) override {
    if (v->buf() != old_buf_) {
      return IRMutator::mutate(v);
    }

    TORCH_INTERNAL_ASSERT(old_indices_.size() == v->indices().size());

    bool equal_indices = true;
    for (size_t i = 0; i < v->indices().size(); ++i) {
      if (!exprEquals(v->indices()[i], old_indices_[i])) {
        equal_indices = false;
        break;
      }
    }
    if (!equal_indices) {
      return IRMutator::mutate(v);
    }

    const Expr* new_value = v->value()->accept_mutator(this);
    return new Store(new_buf_, new_indices_, new_value);
  }

 private:
  const Buf* old_buf_;
  const std::vector<const Expr*>& old_indices_;
  const Buf* new_buf_;
  const Var* reduction_var_;
  std::vector<const Expr*> new_indices_;
};

bool LoopNest::rfactor(Stmt* st, For* target_for) {
  Buf* tmp_buf = nullptr;
  return rfactor(st, target_for, &tmp_buf);
}

bool LoopNest::rfactor(Stmt* st, For* outer_reduction_for, Buf** rfac_buf_ptr) {
  Store* reduction_store = dynamic_cast<Store*>(st);
  const ReduceOp* reduce_op =
      dynamic_cast<const ReduceOp*>(reduction_store->value());
  if (!reduce_op) {
    // Not a reduction store
    return false;
  }

  auto orig_buf = reduction_store->buf();
  auto orig_buf_indices = reduction_store->indices();
  const Var* reduction_var = outer_reduction_for->var();

  std::set<const Var*> reduce_args = {
      reduce_op->reduce_args().begin(), reduce_op->reduce_args().end()};

  if (reduce_args.size() < 2) {
    // Not enough reduction axis to do rfactor
    return false;
  }

  // Verify that outer_reduction_for is a perfect loop nest with all loops being
  // reductions
  Stmt* cur = outer_reduction_for;
  while (For* cur_for = dynamic_cast<For*>(cur)) {
    if (!reduce_args.count(cur_for->var())) {
      // output axis inside outer_reduction_for are not allowed
      return false;
    }
    reduce_args.erase(cur_for->var());

    Block* b = cur_for->body();
    if (b->nstmts() != 1) {
      return false;
    }
    cur = b->stmts().front();
  }
  if (cur != st) {
    // The reduction store is not a single stmt in the innermost loop - bail in
    // that case
    return false;
  }
  if (!reduce_args.empty()) {
    // This is not the outermost reduction axis
    return false;
  }

  // assert: reduce_axis match loop vars from outer_reduction_for and inside
  // assert: no other stmts in outer_reduction_for or its child loops

  std::vector<const Expr*> rfac_dims = orig_buf->dims();
  const Expr* extra_dim = IRSimplifier::simplify(
      new Sub(outer_reduction_for->stop(), outer_reduction_for->start()));
  rfac_dims.push_back(extra_dim);
  const Expr* rfac_init =
      new Cast(reduce_op->dtype(), reduce_op->reducer().initializer());

  *rfac_buf_ptr = new Buf(
      orig_buf->name_hint() + "_rfac",
      rfac_dims,
      reduce_op->dtype(),
      rfac_init);
  Buf* rfac_buf = *rfac_buf_ptr;

  // Rewrite the original reduction store to use the temporary rfac buffer:
  //   1) X[*indexes] --> T[*indexes + {reduction_var}]
  //   2) reduce_axis -= {reduction_var}
  RfactorStoreRewriter rfac_rewriter(
      orig_buf, orig_buf_indices, rfac_buf, reduction_var);
  dynamic_cast<Block*>(st->get_parent())
      ->replace_stmt(st, st->accept_mutator(&rfac_rewriter));

  // Insert a store for the final reduction over the temp buffer into the
  // original buffer:
  //   X[*indexes] = ReduceOp(X[*indexes] + T[*indexes + {reduction_var}],
  //                          reduce_axis={reduction_var})
  Block* b = outer_reduction_for->body();
  TORCH_INTERNAL_ASSERT(b->nstmts() == 1);
  Stmt* first_reduction_loop = b->stmts().front();
  auto rfac_buf_indices = orig_buf_indices;
  rfac_buf_indices.emplace_back(reduction_var);

  const Expr* final_reduce_load = new Load(rfac_buf, rfac_buf_indices);
  outer_reduction_for->body()->insert_stmt_after(
      new Store(
          orig_buf,
          orig_buf_indices,
          reduce_op->reducer()(
              orig_buf, final_reduce_load, orig_buf_indices, {reduction_var})),
      first_reduction_loop);

  // Insert an initialization store for the temp buffer:
  //   T[a,b,c] = init
  outer_reduction_for->body()->insert_stmt_before(
      new Store(rfac_buf, rfac_buf_indices, rfac_init), first_reduction_loop);
  return true;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
