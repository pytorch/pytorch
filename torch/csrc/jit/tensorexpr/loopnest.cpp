#include <torch/csrc/jit/tensorexpr/loopnest.h>

#include <algorithm>
#include <stdexcept>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <c10/util/string_utils.h>

#include <ATen/core/functional.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_cloner.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_verifier.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {

LoopNest::LoopNest(const LoopNest& other)
    : root_stmt_(Stmt::clone(other.root_stmt_)),
      output_bufs_(other.output_bufs_) {
  verify(root_stmt_);
}

LoopNest::LoopNest(StmtPtr stmt, std::unordered_set<BufPtr> output_bufs)
    : root_stmt_(stmt), output_bufs_(std::move(output_bufs)) {
  verify(root_stmt_);
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
LoopNest::LoopNest(
    const std::vector<Tensor>& output_tensors,
    const std::vector<Tensor>& tensors_to_compute) {
  initialize(output_tensors, tensors_to_compute);
  verify(root_stmt_);
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
LoopNest::LoopNest(const std::vector<Tensor>& output_tensors) {
  initialize(output_tensors, output_tensors);
  verify(root_stmt_);
}

std::vector<BufPtr> LoopNest::getIntermediateBufs() const {
  std::vector<BufPtr> result;
  std::unordered_set<BufPtr> result_set;
  auto input_bufs = getInputBufs();
  auto bufs = NodeFinder<Buf>::find(root_stmt_);
  for (auto buf : bufs) {
    if (!output_bufs_.count(buf) && !input_bufs.count(buf) &&
        !result_set.count(buf)) {
      result.push_back(buf);
      result_set.insert(buf);
    }
  }
  return result;
}

const std::unordered_set<BufPtr> LoopNest::getInputBufs() const {
  std::unordered_set<BufPtr> result;
  auto buf_load_store_uses = findLoadOrStoreUses(root_stmt_);
  for (auto& kv : buf_load_store_uses) {
    bool has_store = false;
    for (auto& use : kv.second) {
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
  StmtPtr flatten(StmtPtr s) {
    return s->accept_mutator(this);
  }

  ExprPtr mutate(LoadPtr v) override {
    if (v->indices().size() == 1) {
      return v;
    }
    return alloc<Load>(
        v->dtype(),
        v->buf(),
        std::vector<ExprPtr>({flatten_index(
            v->buf()->dims(), v->indices(), v->buf()->strides())}));
  }

  StmtPtr mutate(StorePtr v) override {
    ExprPtr value = v->value();
    ExprPtr new_value = value->accept_mutator(this);
    if (v->indices().size() == 1 && value == new_value) {
      return v;
    }
    std::vector<ExprPtr> indices = {
        flatten_index(v->buf()->dims(), v->indices(), v->buf()->strides())};
    v->set_indices(indices);
    v->set_value(new_value);
    return v;
  }
};

static bool isValidIdentifierChar(char c, size_t pos) {
  return islower(c) || isupper(c) || c == '_' || (pos > 0 && isdigit(c));
}

// replaces all invalid characters with underscore
std::string sanitizeName(const std::string& input_name) {
  std::stringstream sanitized_name;
  for (size_t i = 0; i < input_name.size(); ++i) {
    if (isValidIdentifierChar(input_name[i], i)) {
      sanitized_name << input_name[i];
    } else {
      if (i == 0) {
        // Don't start names with underscore
        sanitized_name << "v";
      }
      sanitized_name << "_";
    }
  }
  return sanitized_name.str();
}

class VarNameSanitizer : public IRMutator {
 public:
  ExprPtr mutate(BufPtr v) override {
    if (seen_bufs_.count(v)) {
      return v;
    }
    const std::string& name = v->name_hint();
    auto new_name = sanitizeName(name);
    if (taken_names_.count(new_name)) {
      new_name = getNextAvailableName(new_name);
    }
    v->set_name_hint(new_name);
    taken_names_.insert(new_name);
    seen_bufs_.insert(v);
    return v;
  }

  ExprPtr mutate(VarPtr v) override {
    if (seen_vars_.count(v)) {
      return v;
    }
    const std::string& name = v->name_hint();
    auto new_name = sanitizeName(name);
    if (taken_names_.count(new_name)) {
      new_name = getNextAvailableName(new_name);
    }
    v->set_name_hint(new_name);
    taken_names_.insert(new_name);
    seen_vars_.insert(v);
    return v;
  }

  StmtPtr mutate(ForPtr v) override {
    auto new_name = getNextAvailableName(getIndexVarNameAtLevel(level_));
    if (seen_index_vars_.count(v->var())) {
      auto new_var = alloc<Var>("", v->var()->dtype());
      Substitute(v, {{v->var(), new_var}});
    }
    v->var()->set_name_hint(new_name);
    seen_index_vars_.insert(v->var());
    seen_vars_.insert(v->var());
    taken_names_.insert(new_name);
    level_++;
    v->body()->accept_mutator(this);
    level_--;
    v->start()->accept_mutator(this);
    v->stop()->accept_mutator(this);
    return v;
  }

  std::string getIndexVarNameAtLevel(int level_) {
    int names_num = index_var_names_.size();
    int counter = level_ / names_num;
    if (counter == 0) {
      return index_var_names_[level_ % names_num];
    } else {
      return index_var_names_[level_ % names_num] + std::to_string(counter);
    }
  }
  std::string getNextAvailableName(const std::string& base_name) {
    std::string name = base_name;
    int counter = 0;
    while (taken_names_.count(name)) {
      counter++;
      name = base_name + "_" + std::to_string(counter);
    }
    return name;
  }

 private:
  std::vector<std::string> index_var_names_ =
      {"i", "j", "k", "l", "m", "n", "o", "p"};
  std::unordered_set<std::string> taken_names_;
  std::unordered_set<VarPtr> seen_index_vars_;
  std::unordered_set<VarPtr> seen_vars_;
  std::unordered_set<BufPtr> seen_bufs_;
  int level_ = 0;
};

StmtPtr LoopNest::sanitizeNames(StmtPtr s) {
  VarNameSanitizer r;
  s->accept_mutator(&r);
  return s;
}

class Vectorizer : public IRMutator {
 public:
  StmtPtr vectorize(ForPtr v) {
    StmtPtr body = v->body();
    VarPtr var = v->var();
    ExprPtr start = v->start();
    ExprPtr stop = v->stop();

    auto start_imm = intValue(start);
    auto stop_imm = intValue(stop);
    if (!start_imm) {
      // Can't vectorize due to non-constant loop start!
      success_ = false;
      return v;
    }

    if (!stop_imm) {
      // Can't vectorize due to non-constant loop stop!
      success_ = false;
      return v;
    }

    var_ = var;
    start_ = immLike(start, *start_imm);
    lanes_ = *stop_imm;

    StmtPtr new_body = body->accept_mutator(this);
    if (new_body == body) {
      // Vectorization failed!
      success_ = false;
      return v;
    }

    return new_body;
  }

  bool success() const {
    return success_;
  }

  ExprPtr mutate(AddPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) + ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(SubPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) - ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(MulPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) * ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(DivPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) / ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(ModPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) % ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(AndPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) & ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(OrPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) | ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(XorPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) ^ ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(LshiftPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) << ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(RshiftPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) >> ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(MaxPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return Max::make(
          ExprHandle(inputs[0]), ExprHandle(inputs[1]), v->propagate_nans());
    });
  }

  ExprPtr mutate(MinPtr v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return Min::make(
          ExprHandle(inputs[0]), ExprHandle(inputs[1]), v->propagate_nans());
    });
  }

  ExprPtr mutate(CompareSelectPtr v) override {
    std::vector<ExprPtr> inputs = {
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

  ExprPtr mutate(BitCastPtr v) override {
    std::vector<ExprPtr> inputs = {v->src_value()};
    return try_vectorize(v, inputs, [&]() {
      return BitCast::make(
          Dtype(v->dtype().scalar_type(), lanes_), ExprHandle(inputs[0]));
    });
  }

  ExprPtr mutate(CastPtr v) override {
    std::vector<ExprPtr> inputs = {v->src_value()};
    return try_vectorize(v, inputs, [&]() {
      return Cast::make(
          Dtype(v->dtype().scalar_type(), lanes_), ExprHandle(inputs[0]));
    });
  }

  ExprPtr mutate(VarPtr v) override {
    if (v == var_) {
      return Ramp::make(
                 ExprHandle(start_), ExprHandle(immLike(start_, 1)), lanes_)
          .node();
    }

    return v;
  }

  ExprPtr mutate(RampPtr v) override {
    ExprPtr base = v->base();
    ExprPtr stride = v->stride();

    ExprPtr base_new = base->accept_mutator(this);
    ExprPtr stride_new = stride->accept_mutator(this);

    if (base_new == base && stride_new == stride) {
      return v;
    }

    // Can't vectorize a Ramp!
    success_ = false;
    return v;
  }

  ExprPtr mutate(LoadPtr v) override {
    Dtype dtype(v->dtype().scalar_type(), lanes_);
    BufPtr buf = v->buf();
    std::vector<ExprPtr> inputs = {v->flat_index()};
    return try_vectorize(v, inputs, [&]() {
      return Load::make(dtype, BufHandle(buf), {ExprHandle(inputs[0])});
    });
  }

  ExprPtr mutate(ReduceOpPtr v) override {
    Dtype dtype(v->dtype().scalar_type(), lanes_);

    std::vector<ExprPtr> inputs = {v->body()};

    auto out = try_vectorize(v, inputs, [&]() {
      return ExprHandle(
          alloc<ReduceOp>(inputs[0], v->reduce_args(), v->reducer()));
    });
    return out;
  }

  ExprPtr mutate(BroadcastPtr v) override {
    ExprPtr val = v->value();
    ExprPtr new_val = val->accept_mutator(this);
    if (new_val == val) {
      return v;
    }

    // Can't vectorize a Broadcast!
    success_ = false;
    return v;
  }

  ExprPtr mutate(IfThenElsePtr v) override {
    ExprPtr condition = v->condition();
    ExprPtr new_condition = condition->accept_mutator(this);
    if (new_condition != condition) {
      // Can't vectorize an IfThenElse condition!
      success_ = false;
      return v;
    }

    std::vector<ExprPtr> inputs = {v->true_value(), v->false_value()};
    return try_vectorize(v, inputs, [&]() {
      return IfThenElse::make(
          ExprHandle(condition), ExprHandle(inputs[0]), ExprHandle(inputs[1]));
    });
  }

  ExprPtr mutate(IntrinsicsPtr v) override {
    std::vector<ExprPtr> inputs = v->params();
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(alloc<Intrinsics>(v->op_type(), inputs));
    });
  }

  StmtPtr mutate(StorePtr v) override {
    BufPtr buf = v->buf();
    std::vector<ExprPtr> inputs = {v->flat_index(), v->value()};
    return try_vectorize(v, inputs, [&]() {
      return Store::make(
          BufHandle(buf), {ExprHandle(inputs[0])}, ExprHandle(inputs[1]));
    });
  }

  StmtPtr mutate(ForPtr v) override {
    VarPtr var = v->var();
    ExprPtr start = v->start();
    ExprPtr stop = v->stop();
    LoopOptions loop_options = v->loop_options();

    ExprPtr new_start = start->accept_mutator(this);
    ExprPtr new_stop = stop->accept_mutator(this);

    if (new_start != start || new_stop != stop) {
      // Can't vectorize nested For with dependent loop bounds!
      success_ = false;
      return v;
    }

    StmtPtr body = v->body();
    StmtPtr new_body = body->accept_mutator(this);

    if (new_body == body) {
      return (ForPtr)v;
    }

    return alloc<For>(var, new_start, new_stop, new_body, loop_options);
  }

  StmtPtr mutate(BlockPtr v) override {
    // IRMutator does in-place mutations. But the logic in vectorization checks
    // for success by looking for a new stmt. So, we override the in-place
    // mutations and create a clone here if any of its statements change.
    // TODO: Can we change the logic of vectorizer so that we don't need this?
    bool any_change = false;
    std::vector<StmtPtr> stmts;
    for (StmtPtr stmt : *v) {
      StmtPtr stmt_new = stmt->accept_mutator(this);
      if (stmt != stmt_new) {
        any_change = true;
      } else {
        stmt_new = Stmt::clone(stmt);
      }
      if (stmt_new) {
        stmts.push_back(stmt_new);
      }
    }
    if (any_change) {
      return alloc<Block>(stmts);
    }
    return v;
  }

  template <typename T>
  ExprPtr try_vectorize(ExprPtr e, std::vector<ExprPtr>& inputs, T&& vec_ctor) {
    bool vectorize = vectorize_inputs(inputs);
    if (vectorize) {
      return vec_ctor().node();
    }

    return e;
  }

  template <typename T>
  StmtPtr try_vectorize(StmtPtr s, std::vector<ExprPtr>& inputs, T&& vec_ctor) {
    bool vectorize = vectorize_inputs(inputs);
    if (vectorize) {
      return vec_ctor();
    }

    return (StmtPtr)s;
  }

  bool vectorize_inputs(std::vector<ExprPtr>& inputs) {
    bool any_vectorized = false;
    std::vector<ExprPtr> new_inputs;

    // Attempt to vectorize each input.
    for (ExprPtr& in : inputs) {
      ExprPtr new_in = in->accept_mutator(this);
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

  VarPtr var_ = nullptr;
  int lanes_ = 0;
  ExprPtr start_ = nullptr;
  bool success_ = true;
};

bool LoopNest::vectorize(ForPtr f) {
  BlockPtr b = to<Block>(f->get_parent());
  if (!b) {
    return false;
  }

  // Can't vectorize reduction axes.
  auto reductions = NodeFinder<ReduceOp>::find(f);
  for (auto r : reductions) {
    if (std::find(r->reduce_args().begin(), r->reduce_args().end(), f->var()) !=
        r->reduce_args().end()) {
      return false;
    }
  }

  Vectorizer v;
  StmtPtr new_f = nullptr;
  new_f = Stmt::clone(f);
  normalize(to<For>(new_f));
  new_f = FlattenIndexes(new_f);
  new_f = v.vectorize(to<For>(new_f));
  if (!v.success()) {
    // We clone f before vectorizing. So, any partial vectorization will
    // have modified the clone. In case of an exception, we can continue
    // using f.
    new_f = f;
  }

  if (new_f != f) {
    b->replace_stmt(f, IRSimplifier::simplify(new_f));
    return true;
  }

  // Vectorization was not successful.
  return false;
}

void LoopNest::initialize(
    const std::vector<Tensor>& output_tensors,
    const std::vector<Tensor>& tensors_to_compute) {
  for (auto t : output_tensors) {
    output_bufs_.insert(t.buf());
  }

  std::vector<StmtPtr> loops;
  for (Tensor t : tensors_to_compute) {
    StmtPtr loop = t.stmt();
    if (loop->get_parent()) {
      std::cerr << "Error: creating a loopnest from already used Tensors\n";
      loops = {};
      break;
    }
    // Flatten initializers.
    if (BlockPtr block = to<Block>(loop)) {
      for (auto s : block->stmts()) {
        block->remove_stmt(s);
        loops.push_back(s);
      }
    } else {
      loops.push_back(loop);
    }
  }

  root_stmt_ = alloc<Block>(loops);
}

class FunctionInliner : public IRMutator {
 public:
  FunctionInliner(StorePtr producer, std::unordered_set<BufPtr> outputs)
      : buf_(producer->buf()),
        producer_(producer),
        outputs_(std::move(outputs)) {
    success_ = true;
    for (auto i : producer->indices()) {
      if (auto index_var = to<Var>(i)) {
        index_vars_.insert(index_var);
        producer_index_vars_.push_back(index_var);
      } else {
        // If the index can be a constant, then that dimension must have size 1
        // (since we don't support in-place writes). Resolves issue 52581.
        auto index_val = evalInt(i);
        if (!index_val || *index_val != 0) {
          success_ = false;
          break;
        }
        producer_index_vars_.push_back(nullptr);
      }
    }
  }

  bool success() const {
    return success_;
  }

 private:
  ExprPtr mutate_loads(BufPtr buf, std::vector<ExprPtr> dims) {
    std::vector<VarPtr> index_vars;
    if (buf->ndim() != producer_index_vars_.size()) {
      // Dimensions of producer and consumer expressions do not match in inliner
      // in the fuser
      success_ = false;
      return nullptr;
    }
    for (const auto i : c10::irange(buf->ndim())) {
      VarPtr func_callee_arg = producer_index_vars_.at(i);
      ExprPtr func_caller_param = dims.at(i);
      if (func_callee_arg == nullptr) {
        continue;
      }
      auto iter = inline_mapping_.find(func_callee_arg);
      if (iter != inline_mapping_.end()) {
        // Duplicated variables
        success_ = false;
        return nullptr;
      }
      // Add a mapping for each function parameter to it's source name.
      inline_mapping_[func_callee_arg] = func_caller_param;
      GRAPH_DEBUG(
          "ComputeInline: Inline mapping: ",
          std::to_string(func_callee_arg),
          " -> ",
          std::to_string(func_caller_param));
      index_vars.push_back(func_callee_arg);
    }

    // Call the actual replacement.
    ExprPtr body = producer_->value();
    GRAPH_DEBUG("ComputeInline: Before rewriting body: ", std::to_string(body));
    ExprPtr result = Expr::clone(body)->accept_mutator(this);
    GRAPH_DEBUG(
        "ComputeInline: After rewriting body: ", std::to_string(result));

    // Remove the mappings we created for this function parameters.
    for (auto v : index_vars) {
      for (auto& pair : random_bindings_) {
        if (pair.second.erase(v)) {
          ExprPtr inlined = inline_mapping_[v];
          for (auto nv : VarFinder::find(inlined)) {
            pair.second.insert(nv);
          }
        }
      }
      GRAPH_DEBUG("ComputeInline: Inline mapping: erasing", std::to_string(v));
      inline_mapping_.erase(v);
    }
    return result;
  }

  ExprPtr mutate(LoadPtr v) override {
    if (!success()) {
      return v;
    }
    BufPtr buf = v->buf();
    if (buf != buf_) {
      return IRMutator::mutate(v);
    }

    if (v->indices().size() != buf->ndim()) {
      // Number of indices doesn't match buf rank in the fuser
      success_ = false;
      return v;
    }
    auto result = mutate_loads(buf, v->indices());
    if (!result) {
      // If we don't inline successfully return the given load.
      success_ = false;
      return v;
    }
    return result;
  }

  // Replace the target variable with the caller expressions.
  ExprPtr mutate(VarPtr v) override {
    if (!success()) {
      return v;
    }
    auto iter = inline_mapping_.find(v);
    if (iter == inline_mapping_.end()) {
      return v;
    } else {
      ExprPtr expr = iter->second;
      // Continue to transform the value from the lookup table.
      return expr->accept_mutator(this);
    }
  }

  // Handle random intrinsics which should be cached.
  ExprPtr mutate(IntrinsicsPtr v) override {
    if (!success()) {
      return v;
    }
    if (!in_producer_ || v->op_type() != kRand) {
      return IRMutator::mutate(v);
    }

    // Create a new Let Statement for the random variable, which we can refer
    // to multiple times and resolve the same value (ie. store it in a scalar
    // rather than the Tensor).
    const std::string& name = buf_->name_hint();
    VarPtr new_var = alloc<Var>(name, v->dtype());
    random_bindings_[alloc<Let>(new_var, v)] = index_vars_;
    GRAPH_DEBUG(
        "ComputeInline: created random bindings for ", std::to_string(new_var));
    return new_var;
  }

  // Remove the buffer write from the inlined function.
  StmtPtr mutate(StorePtr v) override {
    if (!success()) {
      return v;
    }
    // If the buf_ is in the outputs set, keep its statement intact. Otherwise,
    // remove it.
    if (v == producer_ && !outputs_.count(buf_)) {
      in_producer_ = true;
      producer_ = to<Store>(IRMutator::mutate(v));
      if (!producer_) {
        // Producer statement for output buf should remain non-null in the fuser
        success_ = false;
        return v;
      }
      in_producer_ = false;
      return nullptr;
    } else {
      return IRMutator::mutate(v);
    }
  }

  // Any Random Instrinsics that were turned into vars must be inserted here.
  StmtPtr mutate(BlockPtr v) override {
    if (!success()) {
      return v;
    }
    std::vector<StmtPtr> stmts;
    for (StmtPtr stmt : *v) {
      StmtPtr stmt_new = stmt->accept_mutator(this);
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

  StmtPtr mutate(ForPtr v) override {
    if (!success()) {
      return v;
    }
    ForPtr res = to<For>(IRMutator::mutate(v));
    if (!res) {
      return nullptr;
    }

    // Find any random bindings that should be defined in this loops body.
    std::vector<LetPtr> bindings_this_loop;
    VarPtr fv = v->var();
    for (auto& pair : random_bindings_) {
      auto& index_var = pair.second;
      if (index_var.erase(fv)) {
        bindings_this_loop.push_back(pair.first);
      }
    }

    for (auto l : bindings_this_loop) {
      res->body()->prepend_stmt(l);
      random_bindings_.erase(l);
    }
    return res;
  }

 private:
  BufPtr buf_;
  StorePtr producer_;

  // Index Vars present in the producer.
  std::unordered_set<VarPtr> index_vars_;
  std::vector<VarPtr> producer_index_vars_;

  std::unordered_map<VarPtr, ExprPtr> inline_mapping_;

  // In the producer's scope - we need to bind any calls to rand().
  bool in_producer_ = false;
  std::unordered_map<LetPtr, std::unordered_set<VarPtr>> random_bindings_;
  std::unordered_set<BufPtr> outputs_;
  bool success_ = true;
};

StmtPtr computeInlineImpl(
    BufPtr b,
    StmtPtr stmt,
    const std::unordered_set<BufPtr>& output_bufs) {
  // If buf is used or defined in an ExternalCall, we cannot inline it
  auto buf_load_store_uses = findLoadOrStoreUses(stmt);
  if (!buf_load_store_uses.count(b)) {
    return nullptr;
  }
  for (auto& use : buf_load_store_uses.at(b)) {
    StmtPtr s = use.s;
    if (to<ExternalCall>(s) || to<ExternalCall2>(s)) {
      return nullptr;
    }
  }

  // Find producers.
  StorePtr relevant_store{nullptr};
  auto stores = NodeFinder<Store>::find(stmt);
  for (auto s : stores) {
    if (s->buf() == b) {
      auto reductions = NodeFinder<ReduceOp>::find(s);
      if (!reductions.empty()) {
        // Cannot inline a reduction computation
        return nullptr;
      }
      if (relevant_store != nullptr) {
        // Cannot inline Buf with multiple Tensors
        return nullptr;
      }
      relevant_store = s;
    }
  }

  if (!relevant_store) {
    // Cannot find a relevant store to inline a buf in the fuser
    return nullptr;
  }

  GRAPH_DEBUG("ComputeInline: Def: ", std::to_string(relevant_store));
  FunctionInliner inliner(relevant_store, output_bufs);
  auto result = stmt->accept_mutator(&inliner);
  if (inliner.success()) {
    return result;
  }
  return nullptr;
}

bool LoopNest::computeInline(BufPtr b) {
  // Inlining may not always be successful. Since all mutations now happen
  // in-place, an unsuccessful inlining transformation might leave the IR
  // in an invalid state. To get around this problem, we clone the root stmt,
  // try inlining on the clone, and if it succeeds, we proceed to perform
  // inlining on the actual root stmt. This way the root stmt will always be
  // in a valid state.
  auto stmt_copy = Stmt::clone(root_stmt_);
  auto try_inline = computeInlineImpl(b, stmt_copy, output_bufs_);
  if (!try_inline) {
    return false;
  }
  root_stmt_ = computeInlineImpl(b, root_stmt_, output_bufs_);
  return true;
}

bool LoopNest::computeInline(StmtPtr s) {
  auto s_store = to<Store>(s);
  if (s_store == nullptr) {
    // Could not find buffer producer to inline
    return false;
  }
  return computeInline(s_store->buf());
}

// inlining buffers with multiple uses can create duplicated work, which can
// slow down cpu code generation but is enabled on gpu because it avoids
// difficult synchronization logic across blocks. Inlining trivial reads does
// not duplicate work
void LoopNest::inlineIntermediateBufs(bool allow_duplicated_work) {
  std::unordered_set<BufPtr> bufs_to_inline;

  auto intermediate_bufs = getIntermediateBufs();
  if (allow_duplicated_work) {
    bufs_to_inline.insert(intermediate_bufs.begin(), intermediate_bufs.end());
  } else {
    auto buf_load_store_uses = findLoadOrStoreUses(root_stmt_);
    auto input_bufs = getInputBufs();

    for (auto buf : intermediate_bufs) {
      TORCH_INTERNAL_ASSERT(
          buf_load_store_uses.count(buf),
          buildErrorMessage(
              "Could not find uses of buf '" + buf->name_hint() +
              "' in the fuser."));
      std::vector<BufLoadOrStoreUse>& uses = buf_load_store_uses[buf];
      auto stores = c10::filter(
          uses, [](const BufLoadOrStoreUse& use) { return use.isStore; });

      // if the intermediate is the buffer formed from reading in the input
      // tensors, always inline, bc we are not duplicating any work
      // and avoiding an intermediary buffer
      if (stores.size() == 1) {
        if (auto store = to<Store>(stores[0].s)) {
          auto input_as_load = to<Load>(store->value());
          if (input_as_load && input_bufs.count(input_as_load->buf())) {
            bufs_to_inline.insert(buf);
            continue;
          }
        } else {
          // If S is not a store, it must be an ExternalCall.
          TORCH_INTERNAL_ASSERT(
              to<ExternalCall>(stores[0].s) || to<ExternalCall2>(stores[0].s),
              buildErrorMessage(
                  "Expected stmt: " + std::to_string(stores[0].s) +
                  "\nto be either a Store or an ExternalCall in the fuser."));
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
  std::unordered_map<BufPtr, std::vector<BufLoadOrStoreUse>> findUses(
      StmtPtr s) {
    uses_.clear();
    s->accept(this);
    return uses_;
  }

 private:
  void visit(StorePtr v) override {
    if (stores_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({(StmtPtr)v, true});
    }
    last_stmt_ = (StmtPtr)v;
    IRVisitor::visit(v);
  }

  void visit(ExternalCallPtr v) override {
    if (stores_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({(StmtPtr)v, true});
    }
    last_stmt_ = (StmtPtr)v;

    for (BufPtr input_buf : v->buf_args()) {
      if (loads_[input_buf].insert(last_stmt_).second) {
        uses_[input_buf].push_back({last_stmt_, false});
      }
    }

    IRVisitor::visit(v);
  }

  void visit(ExternalCall2Ptr v) override {
    for (BufPtr out_buf : v->buf_out_args()) {
      if (stores_[out_buf].insert(last_stmt_).second) {
        uses_[out_buf].push_back({(StmtPtr)v, true});
      }
    }
    last_stmt_ = (StmtPtr)v;

    for (BufPtr input_buf : v->buf_args()) {
      if (loads_[input_buf].insert(last_stmt_).second) {
        uses_[input_buf].push_back({last_stmt_, false});
      }
    }

    IRVisitor::visit(v);
  }

  void visit(LoadPtr v) override {
    if (loads_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({last_stmt_, false});
    }
    IRVisitor::visit(v);
  }

  StmtPtr last_stmt_ = nullptr;
  std::unordered_map<BufPtr, std::vector<BufLoadOrStoreUse>> uses_;

  // Sets of loads and stores in order to keep the results unique
  std::unordered_map<BufPtr, std::unordered_set<StmtPtr>> loads_;
  std::unordered_map<BufPtr, std::unordered_set<StmtPtr>> stores_;
};

std::unordered_map<BufPtr, std::vector<BufLoadOrStoreUse>> findLoadOrStoreUses(
    StmtPtr s) {
  LoadOrStoreUseFinder uf;
  return uf.findUses(s);
}

class ContainedStmtsFinder : public IRVisitor {
 public:
  // Simply list all Stores and Block that are children of the given stmt
  const std::unordered_set<StmtPtr>& findContainedStmts(StmtPtr s) {
    contained_.clear();
    s->accept(this);
    return contained_;
  }

 private:
  void visit(StorePtr v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }
  void visit(ExternalCallPtr v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }
  void visit(ExternalCall2Ptr v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }
  void visit(BlockPtr v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }

  std::unordered_set<StmtPtr> contained_;
};

bool containsAll(const std::vector<BufLoadOrStoreUse>& uses, BlockPtr b) {
  std::unordered_set<StmtPtr> not_found;
  for (auto use : uses) {
    not_found.insert(use.s);
  }

  ContainedStmtsFinder csf;
  const std::unordered_set<StmtPtr>& contained = csf.findContainedStmts(b);
  for (auto s : contained) {
    not_found.erase(s);
  }
  return not_found.empty();
}

BlockPtr findParentBlock(StmtPtr s) {
  while (s) {
    if (auto b = to<Block>(s)) {
      return b;
    }
    s = s->get_parent();
  }
  return nullptr;
}

BlockPtr findLowestContainingBlock(const std::vector<BufLoadOrStoreUse>& uses) {
  // TODO: we're not using the most efficient algorithm here for simplicity.
  // Replace with something more performant in case it becomes a bottleneck.
  BlockPtr b = findParentBlock(uses[0].s);
  while (b && !containsAll(uses, b)) {
    b = findParentBlock(b->get_parent());
  }
  return b;
}

class StmtDeleter : public IRMutator {
 public:
  StmtDeleter(const std::unordered_set<StmtPtr>& targets) : targets_(targets) {}

 private:
  StmtPtr mutate(BlockPtr v) override {
    std::vector<StmtPtr> stmts;

    for (auto s : v->stmts()) {
      if (targets_.count(s) == 0) {
        StmtPtr ns = s->accept_mutator(this);
        if (ns) {
          stmts.push_back(Stmt::clone(ns));
        }
      }
    }

    return Block::make(stmts);
  }

  const std::unordered_set<StmtPtr>& targets_;
};

void LoopNest::eliminateDeadStores() {
  using namespace analysis;
  MemDependencyChecker checker(getInputBufs(), getOutputBufs());
  root_stmt_->accept(&checker);

  std::unordered_set<StmtPtr> deadStores;
  std::vector<std::shared_ptr<AccessInfo>> outputAccesses;
  for (auto o : getOutputBufs()) {
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
}

namespace {

// This is extended from IRCloner instead of IRMutator because we want all
// the rest of the IR nodes (the ones not touched directly) to be cloned.
class IfThenElseReplacer : public IRCloner {
 public:
  IfThenElseReplacer(IfThenElsePtr to_replace, ExprPtr new_expr)
      : to_replace_(to_replace), new_expr_(new_expr) {}

  ExprPtr mutate(IfThenElsePtr i) override {
    if (i == to_replace_) {
      return new_expr_;
    }
    return IRCloner::mutate(i);
  }

 private:
  IfThenElsePtr to_replace_;
  ExprPtr new_expr_;
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
    ExprPtr condition,
    VarPtr* cond_var,
    ExprPtr* compared_value) {
  auto cs = to<CompareSelect>(condition);
  if (cs && cs->compare_select_op() == kLT) {
    auto var = to<Var>(cs->lhs());
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
    IfThenElsePtr ite,
    VarPtr* cond_var,
    std::vector<ExprPtr>* comp_values,
    std::vector<ExprPtr>* sub_exprs) {
  VarPtr var = nullptr;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ExprPtr comp_value;
  if (isConditionOptimizable(ite->condition(), &var, &comp_value)) {
    if (*cond_var == nullptr) {
      *cond_var = var;
    } else if (*cond_var != var) {
      // Different condition variables found in nested if-then-else
      // expressions. Can not optimize such cases.
      return false;
    }
    auto true_ite = to<IfThenElse>(ite->true_value());
    if (true_ite) {
      if (!isConditionalFromCat(true_ite, cond_var, comp_values, sub_exprs)) {
        return false;
      }
    } else {
      sub_exprs->push_back(ite->true_value());
    }
    auto false_ite = to<IfThenElse>(ite->false_value());
    if (false_ite) {
      return false;
    }
    comp_values->push_back(comp_value);
    sub_exprs->push_back(ite->false_value());
    return true;
  }
  return false;
}

bool areConstantsAndSorted(const std::vector<ExprPtr>& comp_values) {
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
  std::unordered_set<ForPtr> split_fors;
  for (auto store : stores) {
    VarPtr cond_var = nullptr;
    // `comp_values` represent the list of compared values that will be
    // collected as we check for the expected pattern. Since that will
    // only include the RHS of the conditions in the if-then-else expressions
    // we need to start with `0` which is the initial bound, given that we
    // only handle normalized loops (check for this is done below).
    std::vector<ExprPtr> comp_values;
    std::vector<ExprPtr> sub_exprs;
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
    TORCH_INTERNAL_ASSERT(
        comp_values.size() >= 1,
        buildErrorMessage(
            "Expected at least one expression in optimizeConditional in the fuser."));
    comp_values.insert(comp_values.begin(), immLike(comp_values[0], 0));

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
    std::vector<StmtPtr> split_loops;
    auto cond_to_replace = ifthenelse_exprs.front();
    for (size_t i = 0; i < sub_exprs.size(); ++i) {
      IfThenElseReplacer ifthenelseReplacer(cond_to_replace, sub_exprs[i]);
      auto new_store = store->accept_mutator(&ifthenelseReplacer);
      auto new_for_body =
          for_to_split->body()->clone_and_replace(store, new_store);
      auto new_for = alloc<For>(
          for_to_split->var(),
          comp_values[i],
          comp_values[i + 1],
          new_for_body);
      LoopNest::normalize(new_for);
      split_loops.push_back(new_for);
    }
    auto par = to<Block>(for_to_split->get_parent());
    par->replace_stmt(for_to_split, alloc<Block>(split_loops));
  }
  root_stmt_ = IRSimplifier::simplify(root_stmt_);
  return true;
}

void LoopNest::vectorizeInnerLoops() {
  std::vector<ForPtr> innerLoops;
  std::vector<ForPtr> worklist;

  // Find outer-most For loops
  if (ForPtr rootF = to<For>(root_stmt_)) {
    worklist.push_back(rootF);
  } else if (BlockPtr body = to<Block>(root_stmt_)) {
    std::vector<BlockPtr> blocks = {body};
    while (blocks.size()) {
      BlockPtr b = blocks.back();
      blocks.pop_back();

      for (StmtPtr s : *b) {
        if (ForPtr f = to<For>(s)) {
          worklist.push_back(f);
        } else if (BlockPtr b2 = to<Block>(s)) {
          blocks.push_back(b2);
        }
      }
    }
  }

  // Traverse the For loop nest find inner-most loops, which are
  // vectorization candidates.
  while (worklist.size()) {
    ForPtr f = worklist.back();
    worklist.pop_back();

    bool containsSubLoops = false;
    if (BlockPtr body = to<Block>(f->body())) {
      for (StmtPtr s2 : *body) {
        if (ForPtr f2 = to<For>(s2)) {
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
  for (ForPtr loop : innerLoops) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ForPtr split1;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ForPtr tail1;

    static const int kBodyVectorWidth = 8;
    splitWithTail(loop, kBodyVectorWidth, &split1, &tail1);
    vectorize(split1);

    if (tail1) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      ForPtr split2;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      ForPtr tail2;
      static const int kTailVectorWidth = 4;
      splitWithTail(tail1, kTailVectorWidth, &split2, &tail2);
      vectorize(split2);
    }
  }
}

void LoopNest::sliceHead(ForPtr f, int factor, ForPtr* head, ForPtr* tail) {
  if (intValue(f->start()) && intValue(f->stop())) {
    auto start_val = *intValue(f->start());
    auto stop_val = *intValue(f->stop());
    auto size_val = stop_val - start_val;
    if (factor >= size_val) {
      *head = f;
      *tail = nullptr;
      return;
    }
  }

  if (!f) {
    throw malformed_input("sliceHead attempted on null loop", f);
  }

  BlockPtr p = to<Block>(f->get_parent());
  if (!p) {
    throw malformed_input("sliceHead attempted on loop with no parent", p);
  }

  ExprPtr head_end = alloc<Min>(
      alloc<Add>(f->start(), immLike(f->stop(), factor)), f->stop(), true);
  *head = alloc<For>(f->var(), f->start(), head_end, Stmt::clone(f->body()));
  p->insert_stmt_before(*head, f);

  f->set_start(head_end);
  *tail = f;

  if (f->loop_options().is_gpu_block_index() ||
      f->loop_options().is_gpu_thread_index()) {
    LoopNest::normalize(*tail);
  }
}
void LoopNest::sliceHead(ForPtr f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head, tail;
  sliceHead(f, factor, &head, &tail);
}

void LoopNest::sliceTail(ForPtr f, int factor, ForPtr* head, ForPtr* tail) {
  if (intValue(f->start()) && intValue(f->stop())) {
    auto start_val = *intValue(f->start());
    auto stop_val = *intValue(f->stop());
    auto size_val = stop_val - start_val;
    if (factor >= size_val) {
      *head = nullptr;
      *tail = f;
      return;
    }
  }

  if (!f) {
    throw malformed_input("sliceTail attempted on null loop", f);
  }

  BlockPtr p = to<Block>(f->get_parent());
  if (!p) {
    throw malformed_input("sliceTail attempted on loop with no parent", p);
  }

  ExprPtr tail_start = alloc<Max>(
      f->start(), alloc<Sub>(f->stop(), immLike(f->stop(), factor)), true);
  *tail = alloc<For>(f->var(), tail_start, f->stop(), Stmt::clone(f->body()));
  p->insert_stmt_after(*tail, f);

  f->set_stop(tail_start);
  *head = f;

  if (f->loop_options().is_gpu_block_index() ||
      f->loop_options().is_gpu_thread_index()) {
    LoopNest::normalize(*head);
  }
}
void LoopNest::sliceTail(ForPtr f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr head, tail;
  sliceTail(f, factor, &head, &tail);
}

void LoopNest::splitWithTail(ForPtr f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner, tail;
  splitWithTail(f, factor, &inner, &tail);
}

void LoopNest::splitWithTail(
    ForPtr f,
    int factor,
    ForPtr* inner,
    ForPtr* tail) {
  if (!f) {
    throw malformed_input("splitWithTail attempted on null loop", f);
  }

  BlockPtr p = to<Block>(f->get_parent());
  if (!p) {
    throw malformed_input("splitWithTail attempted on loop with no parent", p);
  }

  // Normalize the loop to simplify start and stop bound computation
  normalize(f);

  bool tail_is_needed = true;
  if (intValue(f->start()) && intValue(f->stop())) {
    auto const start_val = *intValue(f->start());
    auto const stop_val = *intValue(f->stop());
    auto const size_val = stop_val - start_val;
    auto const tail_size = size_val % factor;
    if (tail_size == 0) {
      tail_is_needed = false;
    }
  }

  ExprPtr factor_expr = immLike(f->stop(), factor);
  ExprPtr size = alloc<Sub>(f->stop(), f->start());
  ExprPtr split_count = alloc<Div>(size, factor_expr);
  ExprPtr tail_size = alloc<Mod>(size, factor_expr);

  const std::string& loop_var_name = f->var()->name_hint();
  Dtype loop_var_dtype = f->var()->dtype();

  VarPtr i_inner = alloc<Var>(loop_var_name + "_inner", loop_var_dtype);
  VarPtr i_outer = alloc<Var>(loop_var_name + "_outer", loop_var_dtype);

  // x -> x.outer * inner.size + x.inner
  ExprPtr combined_index1 =
      alloc<Add>(alloc<Mul>(i_outer, factor_expr), i_inner);

  if (tail_is_needed) {
    VarPtr i_tail = alloc<Var>(loop_var_name + "_tail", loop_var_dtype);
    // x -> x.tail + outer.size * inner.size
    ExprPtr combined_index2 =
        alloc<Add>(i_tail, alloc<Mul>(split_count, factor_expr));

    StmtPtr body_tail =
        SubstituteInClone(f->body(), {{f->var(), combined_index2}});
    *tail = alloc<For>(i_tail, immLike(tail_size, 0), tail_size, body_tail);

    p->insert_stmt_after(*tail, f);
  } else {
    *tail = nullptr;
  }

  StmtPtr body_inner =
      Substitute(f->removeBody(), {{f->var(), combined_index1}});

  *inner =
      alloc<For>(i_inner, immLike(factor_expr, 0), factor_expr, body_inner);
  // The input loop `f` will be the outer loop after split.
  f->set_var(i_outer);
  f->set_start(immLike(split_count, 0));
  f->set_stop(split_count);
  f->set_body(*inner);
}

void LoopNest::splitWithMask(ForPtr f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner;
  splitWithMask(f, factor, &inner);
}

void LoopNest::splitWithMask(ForPtr f, int factor, ForPtr* inner) {
  BlockPtr p = to<Block>(f->get_parent());
  if (!p) {
    std::cerr << "Parent is not a Block!\n";
    return;
  }

  bool tail_is_needed = true;
  ExprPtr start = IRSimplifier::simplify(f->start());
  ExprPtr stop = IRSimplifier::simplify(f->stop());
  if (start->isConstant() && stop->isConstant()) {
    auto start_val = *intValue(start);
    auto stop_val = *intValue(stop);
    auto size_val = stop_val - start_val;
    auto tail_size = size_val % factor;
    if (tail_size == 0) {
      tail_is_needed = false;
    }
  }

  auto factor_expr = immLike(f->stop(), factor);
  ExprPtr size = alloc<Sub>(f->stop(), f->start());
  // split_count = (size + factor - 1) / factor
  ExprPtr split_count = alloc<Div>(
      alloc<Sub>(alloc<Add>(size, factor_expr), immLike(size, 1)), factor_expr);

  const std::string& loop_var_name = f->var()->name_hint();
  Dtype loop_var_dtype = f->var()->dtype();

  VarPtr i_inner = alloc<Var>(loop_var_name + "_inner", loop_var_dtype);
  VarPtr i_outer = alloc<Var>(loop_var_name + "_outer", loop_var_dtype);

  // x -> x.outer * inner.size + x.inner
  ExprPtr combined_index =
      alloc<Add>(alloc<Mul>(i_outer, factor_expr), i_inner);

  StmtPtr body_inner = f->removeBody();
  // TODO: is it ok that we're doing it eagerly? In the other implementation we
  // are only materializing predicates at the last, lowering, step.
  if (tail_is_needed) {
    auto start = intValue(f->start());
    if (!start || *start != 0) {
      throw unimplemented_lowering();
    }

    ExprPtr predicate =
        CompareSelect::make(ExprHandle(f->var()), ExprHandle(f->stop()), kLT)
            .node();
    body_inner = Cond::make(ExprHandle(predicate), body_inner, nullptr);
  }
  body_inner = Substitute(body_inner, {{f->var(), combined_index}});

  *inner =
      alloc<For>(i_inner, immLike(factor_expr, 0), factor_expr, body_inner);
  // The input loop `f` will be the outer loop after split.
  f->set_var(i_outer);
  f->set_start(immLike(split_count, 0));
  f->set_stop(split_count);
  f->set_body(*inner);
}

std::vector<ForPtr> LoopNest::distributeLoop(
    ForPtr loop,
    const std::unordered_set<StmtPtr>& pivots) {
  TORCH_INTERNAL_ASSERT(
      loop,
      buildErrorMessage(
          "Expected non-null loop in distributeLoop in the fuser."));
  auto root = loop->get_parent();
  if (root == nullptr) {
    throw malformed_input("Loop without parent: ", loop);
  }
  auto root_block = to<Block>(root);
  if (root_block == nullptr) {
    throw malformed_input(
        "Loop's parent must be a Block, instead found ", root);
  }

  // Extract bodies for all the loops after distribution.
  std::vector<BlockPtr> new_loop_bodies;
  auto new_loop_body = alloc<Block>(std::vector<StmtPtr>({}));
  while (!loop->body()->empty()) {
    auto s = loop->body()->front();
    loop->body()->remove_stmt(s);
    new_loop_body->append_stmt(s);
    if (pivots.count(s)) {
      new_loop_bodies.push_back(new_loop_body);
      new_loop_body = alloc<Block>(std::vector<StmtPtr>({}));
    }
  }
  if (!new_loop_body->empty()) {
    new_loop_bodies.push_back(new_loop_body);
  }

  // The first loop body has to be in the original loop.
  loop->body()->splice(loop->body()->begin(), new_loop_bodies.front());
  std::vector<ForPtr> new_loops = {loop};

  // Create loops for all the remaining blocks.
  // Add all the new loops to the parent block.
  for (size_t i = 1; i < new_loop_bodies.size(); ++i) {
    auto new_loop = loop->cloneWithNewBody(new_loop_bodies[i]);
    root_block->insert_stmt_after(new_loop, new_loops.back());
    new_loops.push_back(new_loop);
  }

  return new_loops;
}

std::vector<ForPtr> LoopNest::distributeLoop(ForPtr loop) {
  std::unordered_set<StmtPtr> stmtsInBlock(
      loop->body()->begin(), loop->body()->end());
  return distributeLoop(loop, stmtsInBlock);
}

std::vector<ForPtr> LoopNest::distributeLoopAndParents(ForPtr loop) {
  auto parentLoop = getParentLoop(loop);
  auto result = distributeLoop(loop);
  if (parentLoop) {
    return distributeLoopAndParents(parentLoop);
  }
  return result;
}

std::vector<ForPtr> LoopNest::distributeLoopOverInnerLoops(ForPtr loop) {
  auto loops = NodeFinder<For>::find(loop);
  std::unordered_set<StmtPtr> loopsSet(loops.begin(), loops.end());
  return distributeLoop(loop, loopsSet);
}

std::vector<ForPtr> LoopNest::distributeLoopAndParentsOverInnerLoops(
    ForPtr loop) {
  auto parentLoop = getParentLoop(loop);
  auto result = distributeLoopOverInnerLoops(loop);
  if (parentLoop) {
    return distributeLoopAndParentsOverInnerLoops(parentLoop);
  }
  return result;
}

bool areEqual(ExprPtr expr1, ExprPtr expr2) {
  auto diff = IRSimplifier::simplify(alloc<Sub>(expr1, expr2));
  return diff->isConstant() && (immediateAs<int>(diff) == 0);
};

bool doesExprContainAnyVar(
    ExprPtr expr,
    const std::unordered_set<VarPtr>& vars) {
  for (auto v : VarFinder::find(expr)) {
    if (vars.count(v)) {
      return true;
    }
  }
  return false;
}

// Returns true if the given list of indices refer to two accesses
// that are loop-independent w.r.t. the given list of outer loop
// variables.
bool areIndicesLoopIndependent(
    const std::vector<ExprPtr>& expr_list1,
    const std::vector<ExprPtr>& expr_list2,
    const std::unordered_set<VarPtr>& outer_loop_vars) {
  if (expr_list1.size() != expr_list2.size()) {
    return false;
  }
  for (size_t i = 0; i < expr_list1.size(); ++i) {
    auto expr1 = expr_list1[i];
    auto expr2 = expr_list2[i];
    if (doesExprContainAnyVar(expr1, outer_loop_vars) ||
        doesExprContainAnyVar(expr2, outer_loop_vars)) {
      if (!areEqual(expr1, expr2)) {
        return false;
      }
    }
  }
  return true;
}

bool LoopNest::hasLoopCarriedDependence(ForPtr loop) {
  analysis::MemDependencyChecker analyzer;
  loop->accept(&analyzer);

  std::unordered_set<VarPtr> outer_loop_vars = {loop->var()};
  auto outer_loops = LoopNest::getEnclosingLoopNest(loop);
  for (auto l : outer_loops) {
    outer_loop_vars.insert(l->var());
  }

  // High-level algorithm to check if two accesses to a buffer, A and B, one of
  // which is a Store, result in a loop-carried dependence:
  //   1. For every pair of index expressions, Ai and Bi, that refer to a dim
  //      of A and B, if one of the following conditions are satisfied:
  //       a) Ai and Bi are equal (OR)
  //       b) Both Ai and Bi do not contain any outer-loop variables
  //      then, the dependence between A and B is a loop-independent
  //      dependence. This is because, in the case of b), those index
  //      expressions do not affect the ordering of accesses A and B.
  //   2. If condition 1) is not satisfied:
  //       a) if the bounds on the accesses overlap, then this is a
  //          loop-carried dependence.
  //       b) if the bounds on the accesses do not overlap, then there is no
  //          dependence.
  //
  // NOTE: Since we check for equality of index expressions whenever outer
  //     loop variables are involved, this may incorrectly report some cases as
  //     having a loop-carried dependence. It is impractical to handle all
  //     possible cases here, so, we are being conservative and allow for
  //     some false positives. While this will prevent some loop fusion
  //     opportunities, that should be a small fraction of the cases that are
  //     allowed.
  //
  // Implementation:
  //
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
            if (!areIndicesLoopIndependent(
                    aStore->indices(), bLoad->indices(), outer_loop_vars)) {
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
            if (!areIndicesLoopIndependent(
                    bStore->indices(), aLoad->indices(), outer_loop_vars)) {
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
            if (!areIndicesLoopIndependent(
                    aStore->indices(), bStore->indices(), outer_loop_vars)) {
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

bool LoopNest::unsafeFuseLoops(
    const std::vector<ForPtr>& loops,
    ForPtr* fused) {
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
  auto root_block = to<Block>(root);
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
  TORCH_INTERNAL_ASSERT(
      it != root_block->end(),
      buildErrorMessage(
          "Could not find the given loop in the root stmt in unsafeFuseLoop the fuser."));
  for (auto l : loops) {
    if (*it != l) {
      return false;
    }
    ++it;
  }

  auto first_loop = loops.front();
  // Fuse the loops by taking all the statements from the second loops
  // onwards and moving them into the first loop's body.
  // This way the final fused loop will be the same as the first loop.
  for (size_t i = 1; i < loops.size(); ++i) {
    auto body = to<Block>(SubstituteInClone(
        loops[i]->body(), {{loops[i]->var(), first_loop->var()}}));
    first_loop->body()->splice(first_loop->body()->end(), body);
    root_block->remove_stmt(loops[i]);
  }

  *fused = loops.front();
  return true;
}

bool LoopNest::fuseLoops(const std::vector<ForPtr>& loops, ForPtr* fused) {
  if (loops.empty()) {
    return false;
  }
  if (loops.size() == 1) {
    *fused = loops.front();
    return true;
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

  // We need to check if fusing the loops results in a loop-carried dependence.
  // This check can be done only after the loops are fused into one. But if the
  // check is violated, we need to return the given loops in the original form.
  // So, we create a clone of all the loops, fuse them and check for this.
  std::vector<ForPtr> loops_copy;
  loops_copy.reserve(loops.size());
  BlockPtr parent = alloc<Block>(std::vector<StmtPtr>({}));
  for (auto& l : loops) {
    auto l_copy = Stmt::clone(l);
    loops_copy.push_back(to<For>(l_copy));
    parent->append_stmt(l_copy);
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr fused_copy;
  bool ret = unsafeFuseLoops(loops_copy, &fused_copy);
  if (!ret || hasLoopCarriedDependence(fused_copy)) {
    return false;
  }

  // Now that all conditions are satisfied, we fuse the given loops.
  return unsafeFuseLoops(loops, fused);
}

ForPtr LoopNest::findOuterFor(ForPtr a, ForPtr b) {
  StmtPtr s = b; // guess b is the latter.
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

void LoopNest::reorderAxis(ForPtr a, ForPtr b) {
  if (a == b) {
    // nothing to do.
    return;
  }
  // find inner and outer.
  ForPtr outer = findOuterFor(a, b);
  if (outer == nullptr) {
    throw std::runtime_error("Reordered a loop not in LoopNest");
  }

  ForPtr inner = a == outer ? b : a;
  std::deque<ForPtr> internal_axes;

  // Find relevant axes, store reversed.
  StmtPtr s = inner;
  while (s != outer) {
    if (ForPtr f = to<For>(s)) {
      internal_axes.push_back(f);
    }

    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    s = s->get_parent();
  }

  internal_axes.push_back(outer);

  BlockPtr root = to<Block>(outer->get_parent());
  CHECK(root);

  // Do a shallow copy of the inner blocks.
  BlockPtr body = alloc<Block>(std::vector<StmtPtr>({}));
  body->splice(body->end(), inner->body());

  ForPtr before{outer};
  ForPtr after{nullptr};
  ForPtr last = internal_axes.front();
  StmtPtr newInner = body;

  s = inner;
  while (s != outer) {
    if (auto cond = to<Cond>(s->get_parent())) {
      if (s == cond->true_stmt()) {
        newInner = cond->cloneWithNewBody(newInner);
      } else {
        // s is the false branch of Cond
        newInner = cond->cloneWithNewBodies(
            alloc<Block>(std::vector<StmtPtr>({})), newInner);
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
  for (auto loop : internal_axes) {
    // If the inner loop had a component after the loop we must wrap it in a For
    // loop matching this level of the tree.
    if (after != nullptr) {
      after = loop->cloneWithNewBody(after);
    }

    bool pastMidpoint = false;
    bool hadBeforeStmts = false;
    for (auto I = loop->body()->begin(), E = loop->body()->end(); I != E;) {
      // Be careful not to invalidate the iterator.
      StmtPtr s = *(I++);
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
  for (auto loop : internal_axes) {
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

std::vector<ForPtr> LoopNest::reorder(
    const std::vector<ForPtr>& loops,
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

  auto parent = to<Block>(loops.front()->get_parent());
  if (parent == nullptr) {
    throw malformed_input("parent of the loops must be a Block");
  }

  // Reorder the loops according to the permutation.
  std::vector<ForPtr> result(loops.size());
  for (size_t i = 0; i < loops.size(); ++i) {
    result[i] = loops[permutation[i]];
  }

  // Remove the bodies from all the loops.
  auto innermost_body = loops.back()->removeBody();
  // We use an empty block statement to replace the outermost loop
  // so that we know the position where the outermost reordered loop
  // is to be inserted.
  auto empty_block = alloc<Block>(std::vector<StmtPtr>({}));
  parent->replace_stmt(loops.front(), empty_block);
  for (size_t i = 1; i < loops.size(); ++i) {
    auto block = to<Block>(loops[i]->get_parent());
    TORCH_INTERNAL_ASSERT(
        block,
        buildErrorMessage(
            "Expected parent stmt to be a non-null Block in reorder transformation the fuser."));
    block->remove_stmt(loops[i]);
  }

  // Set the new bodies after reorder for all the loops.
  for (size_t i = 0; i < result.size() - 1; ++i) {
    result[i]->set_body(result[i + 1]);
  }
  result.back()->set_body(innermost_body);
  parent->replace_stmt(empty_block, result.front());
  return result;
}

ForPtr LoopNest::getLoopAt(ForPtr root, const std::vector<int>& indices) const {
  if (indices.empty()) {
    return root;
  }
  if (root == nullptr) {
    throw malformed_input("root loop is null");
  }

  ForPtr curr = root;
  for (auto i : indices) {
    if (i < 0 || curr->body()->nstmts() <= i) {
      return nullptr;
    }
    std::list<StmtPtr>::iterator stmtp = curr->body()->begin();
    std::advance(stmtp, i);
    curr = to<For>(*stmtp);
    if (curr == nullptr) {
      return nullptr;
    }
  }

  return curr;
}

ForPtr LoopNest::tile(ForPtr x, ForPtr y, int x_factor, int y_factor) {
  auto parent = to<Block>(x->get_parent());
  if (parent == nullptr) {
    throw malformed_input("parent of the loops must be a Block");
  }
  if (!areLoopsPerfectlyNested({x, y})) {
    throw malformed_input("two loops must be perfectly nested");
  }

  // Split x, y axes by x_factor and y_factor
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr yi, ytail;
  splitWithTail(y, y_factor, &yi, &ytail);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr xi, xtail;
  splitWithTail(x, x_factor, &xi, &xtail);

  // Distribute xi over yo and ytail so we can manipulate the loop order of {xo,
  // xi, yo, yi}
  auto loops = distributeLoop(xi);

  // For {xi, yo, yi}, reorder the axes to be yo, xi, yi
  xi = loops.front();
  ForPtr yo = to<For>(xi->body()->stmts().front());
  CHECK(yo);
  reorder({xi, yo}, {1, 0});

  // For {xi, ytail}, reorder the axes to be ytail, xi
  if (loops.size() == 2) {
    xi = loops.back();
    ytail = to<For>(xi->body()->stmts().front());
    CHECK(ytail);
    reorder({xi, ytail}, {1, 0});
  }

  return xtail;
}

bool LoopNest::areLoopsPerfectlyNested(const std::vector<ForPtr>& loops) {
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

void LoopNest::fullUnroll(ForPtr f, StmtPtr* unrolled) {
  BlockPtr p = to<Block>(f->get_parent());
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

  std::vector<StmtPtr> unrolled_stmts;
  int start_val = immediateAs<int>(start_expr);
  int stop_val = immediateAs<int>(stop_expr);
  for (int current = start_val; current < stop_val; ++current) {
    for (auto stmt : f->body()->stmts()) {
      unrolled_stmts.push_back(SubstituteInClone(
          stmt, {{f->var(), getImmediateByType(f->var()->dtype(), current)}}));
    }
  }
  *unrolled = alloc<Block>(unrolled_stmts);
  *unrolled = IRSimplifier::simplify(*unrolled);

  p->replace_stmt(f, *unrolled);
}

void LoopNest::fullUnroll(ForPtr f) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  StmtPtr unrolled;
  fullUnroll(f, &unrolled);
}

void LoopNest::unroll(ForPtr f, int factor, ForPtr* tail) {
  if (factor < 2) {
    return;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner;
  splitWithTail(f, factor, &inner, tail);
  fullUnroll(inner);
}

void LoopNest::unroll(ForPtr f, int factor) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  unroll(f, factor, &tail);
}

bool LoopNest::isNormalized(ForPtr f) {
  if (f->start()->isConstant()) {
    return immediateAs<int>(f->start()) == 0;
  }
  return false;
}

bool LoopNest::normalize(ForPtr f) {
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
  f->set_body(IRSimplifier::simplify(for_body_normalized));
  f->set_stop(IRSimplifier::simplify(alloc<Sub>(f->stop(), f->start())));
  f->set_start(immLike(f->stop(), 0));
  return true;
}

// This function expects that there are 'num' loops perfectly nested within
// and including 'f'.
std::vector<ForPtr> LoopNest::getLoopStmtsInLoopNest(ForPtr f, size_t num) {
  std::vector<ForPtr> loops(num);
  ForPtr curr_for = f;
  loops[0] = curr_for;
  for (size_t i = 1; i < num; ++i) {
    TORCH_INTERNAL_ASSERT(
        curr_for->body()->nstmts() == 1,
        buildErrorMessage("Expected a single stmt in the loop body."));
    curr_for = to<For>(curr_for->body()->front());
    TORCH_INTERNAL_ASSERT(
        curr_for,
        buildErrorMessage("Expected the only child stmt to be a For loop."));
    loops[i] = curr_for;
  }
  return loops;
}

bool LoopNest::flatten(const std::vector<ForPtr>& loops, ForPtr* flattened) {
  if (loops.empty()) {
    throw malformed_input("flatten attempted on empty set of loops");
  }
  BlockPtr p = to<Block>(loops[0]->get_parent());
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

  auto flat_var = alloc<Var>(
      normalized_loops[0]->var()->name_hint() + "_flat",
      normalized_loops[0]->var()->dtype());
  VarMapping var_mapping;
  ExprPtr stop = immLike(flat_var, 1);
  for (size_t i = 0; i < normalized_loops.size(); ++i) {
    size_t idx = normalized_loops.size() - i - 1;
    auto curr_loop = normalized_loops[idx];
    ExprPtr div = alloc<Div>(flat_var, stop);
    ExprPtr sub_expr = idx == 0 ? div : alloc<Mod>(div, curr_loop->stop());
    var_mapping.push_back(std::make_pair(curr_loop->var(), sub_expr));
    stop = alloc<Mul>(curr_loop->stop(), stop);
  }
  auto flattened_body =
      Substitute(normalized_loops.back()->removeBody(), var_mapping);

  normalized_loops.front()->set_var(flat_var);
  normalized_loops.front()->set_start(immLike(stop, 0));
  normalized_loops.front()->set_stop(stop);
  normalized_loops.front()->set_body(flattened_body);
  *flattened = normalized_loops.front();
  return true;
}

bool LoopNest::flatten(const std::vector<ForPtr>& loops) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr flattened;
  return flatten(loops, &flattened);
}

void LoopNest::compressBuffer(BufPtr buf, StmtPtr stmt) {
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

  // Find the parent common to all the buffer accesses.
  BlockPtr parent = to<Block>(writes.front()->get_parent());
  TORCH_INTERNAL_ASSERT(
      parent,
      buildErrorMessage(
          "Expected parent stmt to be a non-null block in compressBuffer in the fuser."));
  for (auto w : writes) {
    parent = Block::getSharedParent(parent, w);
  }
  for (auto r : reads) {
    parent = Block::getSharedParent(parent, r);
  }

  // Collect all the loops that are above the common parent.
  auto loops = LoopNest::getEnclosingLoopNest(parent);
  std::unordered_set<VarPtr> loop_vars;
  for (auto l : loops) {
    loop_vars.insert(l->var());
  }

  // TODO: Need to handle other Stmts / Exprs that read / write buffers.
  auto stores = NodeFinder<Store>::find(stmt);
  auto loads = NodeFinder<Load>::find(stmt);

  // Vector to indicate which dimensions could be compressed away.
  std::vector<bool> dims(buf->dims().size(), true);
  auto check_indices = [&](const std::vector<ExprPtr>& indices) {
    TORCH_INTERNAL_ASSERT(
        indices.size() == dims.size(),
        buildErrorMessage(
            "Expected ranks to match in compressBuffer in the fuser."));
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
  std::vector<ExprPtr> new_dims(buf->dims());
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i]) {
      new_dims[i] = immLike(buf->dims()[i], 1);
    }
  }
  buf->set_dims(new_dims);

  // Modify all access to reflect the removed dims.
  auto get_new_indices = [&](const std::vector<ExprPtr>& indices) {
    TORCH_INTERNAL_ASSERT(
        indices.size() == dims.size(),
        buildErrorMessage(
            "Expected ranks to match in compressBuffer in the fuser."));
    std::vector<ExprPtr> new_indices(indices);
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i]) {
        new_indices[i] = immLike(indices[i], 0);
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

void LoopNest::compressAllBuffers(StmtPtr stmt) {
  for (auto buf : BufFinder::find(stmt)) {
    compressBuffer(buf, stmt);
  }
}

std::vector<ForPtr> LoopNest::getLoopStmtsFor(Tensor t) const {
  StmtPtr cur_stmt = getLoopBodyFor(t);
  return getLoopStmtsFor(cur_stmt);
}

std::vector<ForPtr> LoopNest::getLoopStmtsFor(BufPtr buf) const {
  StmtPtr cur_stmt = getLoopBodyFor(buf);
  return getLoopStmtsFor(cur_stmt);
}

std::vector<ForPtr> LoopNest::getLoopStmtsFor(StmtPtr s) const {
  std::vector<ForPtr> result;

  while (s) {
    if (auto loop = to<For>(s)) {
      result.push_back(loop);
    }
    s = s->get_parent();
  }
  std::reverse(result.begin(), result.end());
  return result;
}

StmtPtr LoopNest::getLoopBodyFor(Tensor t) const {
  return getLoopBodyFor(t.buf());
}

StmtPtr LoopNest::getLoopBodyFor(BufPtr buf) const {
  auto writes = WritesToBuf::find(root_stmt_, buf);

  // special case for reduction Tensors, ignore the initializer if it's the only
  // op:
  if (writes.size() == 2) {
    if (StorePtr s = to<Store>(writes.back())) {
      if (ReduceOpPtr r = to<ReduceOp>(s->value())) {
        return (StmtPtr)s; // NOLINT
      }
    }
  }

  StmtPtr res = nullptr;
  for (auto s : writes) {
    if (!res) {
      res = s;
      continue;
    }

    res = Block::getSharedParent(res, s);
  }

  return (StmtPtr)res; // NOLINT
}

ForPtr LoopNest::getParentLoop(StmtPtr st) {
  if (st == nullptr) {
    return nullptr;
  }
  auto par = st->get_parent();
  if (auto f = to<For>(par)) {
    return f;
  }
  return getParentLoop(par);
}

std::vector<ForPtr> LoopNest::getEnclosingLoopNest(StmtPtr st) {
  std::vector<ForPtr> loops;
  auto f = getParentLoop(st);
  while (f) {
    loops.push_back(f);
    f = getParentLoop(f);
  }
  std::reverse(loops.begin(), loops.end());
  return loops;
}

std::vector<StmtPtr> LoopNest::getAllWritesToBuf(BufPtr buf) const {
  return WritesToBuf::find(root_stmt_, buf);
}

std::vector<ForPtr> LoopNest::getAllInnermostLoopsWritingToBuf(
    BufPtr buf) const {
  auto writes = getAllWritesToBuf(buf);
  std::vector<ForPtr> innermost_loops;
  innermost_loops.reserve(writes.size());
  for (auto w : writes) {
    innermost_loops.push_back(LoopNest::getParentLoop(w));
  }
  return innermost_loops;
}

std::vector<std::vector<ForPtr>> LoopNest::getAllLoopNestsWritingToBuf(
    BufPtr buf) const {
  auto writes = getAllWritesToBuf(buf);
  std::vector<std::vector<ForPtr>> loopnests;
  loopnests.reserve(writes.size());
  for (auto w : writes) {
    loopnests.emplace_back(LoopNest::getEnclosingLoopNest(w));
  }
  return loopnests;
}

StmtPtr LoopNest::simplify() {
  root_stmt_ = IRSimplifier::simplify(root_stmt_);
  return root_stmt_;
}

StmtPtr FlattenIndexes(StmtPtr s) {
  IndexFlattener idx_flattener;
  return idx_flattener.flatten(s);
}

// Auxiliary class for rewriting we're doing in `compute_at`. See
// LoopNest::computeAt for more details.
class LoopComputeAtRewriter : public IRMutator {
 public:
  LoopComputeAtRewriter(
      BufPtr buf,
      BufPtr new_buf,
      std::vector<ExprPtr> offsets)
      : buf_(buf), new_buf_(new_buf), offsets_(std::move(offsets)) {}

 private:
  BufPtr buf_;
  BufPtr new_buf_;
  std::vector<ExprPtr> offsets_;

  ExprPtr mutate(LoadPtr v) override {
    if (v->buf() != buf_) {
      return v;
    }
    std::vector<ExprPtr> new_indices(v->indices().size());
    for (const auto i : c10::irange(v->indices().size())) {
      new_indices[i] =
          IRSimplifier::simplify(alloc<Sub>(v->indices()[i], offsets_[i]));
    }
    return alloc<Load>(v->dtype(), new_buf_, new_indices);
  }
};

static StorePtr getStoreStmtOfProducer(StmtPtr s) {
  if (StorePtr st = to<Store>(s)) {
    return st;
  }
  if (BlockPtr b = to<Block>(s)) {
    for (StmtPtr ss : *b) {
      if (StorePtr st = to<Store>(ss)) {
        return st;
      }
    }
  }
  return nullptr;
}

static std::vector<VarPtr> getOuterLoopIndexes(StmtPtr s) {
  std::vector<VarPtr> res;
  StmtPtr cur = s;
  while (cur) {
    if (auto l = to<For>(cur)) {
      res.push_back(l->var());
    }
    cur = cur->get_parent();
  }
  return res;
}

class CacheReplacer : public IRMutator {
 public:
  CacheReplacer(BufPtr buffer, BufPtr cache, std::vector<ExprPtr>& offsets)
      : buf_(buffer), cache_(cache), offsets_(offsets) {}

 private:
  ExprPtr mutate(LoadPtr v) override {
    BufPtr buf = v->buf();
    if (buf != buf_) {
      return IRMutator::mutate(v);
    }

    // Map indices to call-parameters.
    std::vector<ExprPtr> newIndices;
    TORCH_INTERNAL_ASSERT(
        offsets_.size() == v->indices().size(),
        buildErrorMessage(
            "Expected ranks to match in CacheReplacer in the fuser."));
    for (size_t i = 0; i < v->indices().size(); ++i) {
      ExprPtr index = v->indices()[i]->accept_mutator(this);
      ExprPtr offset = offsets_[i];
      ExprPtr sub = IRSimplifier::simplify(alloc<Sub>(index, offset));
      newIndices.push_back(sub);
    }
    v->set_buf(cache_);
    v->set_indices(newIndices);
    return v;
  }

  StmtPtr mutate(StorePtr v) override {
    BufPtr buf = v->buf();
    if (buf != buf_) {
      return IRMutator::mutate(v);
    }

    ExprPtr newValue = v->value()->accept_mutator(this);

    // Map indices to call-parameters.
    std::vector<ExprPtr> newIndices;
    TORCH_INTERNAL_ASSERT(
        offsets_.size() == v->indices().size(),
        buildErrorMessage(
            "Expected ranks to match in CacheReplacer in the fuser."));
    for (size_t i = 0; i < v->indices().size(); ++i) {
      ExprPtr index = v->indices()[i]->accept_mutator(this);
      ExprPtr offset = offsets_[i];
      ExprPtr sub = IRSimplifier::simplify(alloc<Sub>(index, offset));
      newIndices.push_back(sub);
    }
    v->set_buf(cache_);
    v->set_indices(newIndices);
    v->set_value(newValue);
    return v;
  }

  BufPtr buf_;
  BufPtr cache_;
  std::vector<ExprPtr>& offsets_;
};

LoopNest::AccessResult LoopNest::cacheAccesses(
    BufPtr producer,
    const std::string& name,
    StmtPtr consumer) {
  ReduceOpPtr reduceOp{nullptr};
  auto stores = NodeFinder<Store>::find(consumer);
  for (auto store : stores) {
    if (auto ro = to<ReduceOp>(store->value())) {
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

  TORCH_INTERNAL_ASSERT(
      bounds_it->second.size() == 1,
      buildErrorMessage(
          "Unexpected number of bound info entries in cacheAccesses in the fuser."));
  TensorAccessBoundsInfo& info = bounds_it->second[0];
  bool hasReads = info.kind == kLoad || info.kind == kMutate;
  bool hasWrites = info.kind == kStore || info.kind == kMutate;

  std::vector<std::string> var_names = {"i", "j", "k", "l", "m", "n", "o", "p"};
  std::vector<ExprPtr> tmp_dims;
  std::vector<VarPtr> new_loop_vars;
  std::vector<ExprPtr> new_loop_vars_expr;

  // Determine the size of the cache, and create a loop var for each dimension.
  for (size_t i = 0; i < info.start.size(); ++i) {
    ExprPtr dim = IRSimplifier::simplify(alloc<Add>(
        alloc<Sub>(info.stop[i], info.start[i]), immLike(info.stop[i], 1)));

    tmp_dims.push_back(dim);

    new_loop_vars.push_back(
        alloc<Var>(var_names[i % var_names.size()], info.stop[i]->dtype()));
    new_loop_vars_expr.push_back(new_loop_vars[i]);
  }

  // Create the var.
  BufPtr tmp_buf =
      alloc<Buf>(alloc<Var>(name, kHandle), tmp_dims, producer->dtype());

  // determine the offsets for calls into the cache based off the loop start of
  // each axis.
  std::vector<ExprPtr> tmp_params;
  for (size_t i = 0; i < new_loop_vars.size(); ++i) {
    tmp_params.push_back(alloc<Add>(new_loop_vars[i], info.start[i]));
  }

  // Replace acceses to the producer in the consumer with the cache.
  CacheReplacer replacer(producer, tmp_buf, info.start);
  consumer->accept_mutator(&replacer);

  // replace the old consumer with the replaced consumer.
  BlockPtr consumer_block = to<Block>(consumer);
  BlockPtr parent_block = to<Block>(consumer->get_parent());
  // if the consumer is a block, we should mutate it in place.
  bool is_block = consumer_block != nullptr;

  // If there's a reduction and we are operating on the reduce axis, we need to
  // initialize the cache with 0s. Also, we can't just write the result straight
  // back to the original buffer, since after parallelism the writes will race.
  // Instead we need to create a new ReduceOp.
  bool on_reduce_axis = false;
  if (reduceOp) {
    std::set<VarPtr> reduce_args(
        reduceOp->reduce_args().begin(), reduceOp->reduce_args().end());
    std::set<VarPtr> enclosing_vars;
    for (auto enclosing_for_stmt : NodeFinder<For>::find(consumer)) {
      enclosing_vars.insert(enclosing_for_stmt->var());
    }
    for (auto reduce_arg : reduce_args) {
      if (enclosing_vars.find(reduce_arg) == enclosing_vars.end()) {
        on_reduce_axis = true;
      }
    }
  }
  if (reduceOp && on_reduce_axis) {
    // reduceOp means we had both loads and stores.

    // Init cache to 0.
    StmtPtr tmp_init = alloc<Store>(
        tmp_buf, new_loop_vars_expr, getImmediateByType(tmp_buf->dtype(), 0));

    for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
      tmp_init = alloc<For>(
          new_loop_vars[i], immLike(tmp_dims[i], 0), tmp_dims[i], tmp_init);
    }

    if (is_block) {
      consumer_block->prepend_stmt(tmp_init);
    } else {
      parent_block->insert_stmt_before(tmp_init, consumer);
    }

    // Reduce back to the original buffer:
    StmtPtr tmp_store = alloc<Store>(
        producer,
        tmp_params,
        reduceOp->reducer()(
            producer,
            ExprHandle(alloc<Load>(tmp_buf, new_loop_vars_expr)),
            tmp_params,
            {}));

    for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
      tmp_store = alloc<For>(
          new_loop_vars[i], immLike(tmp_dims[i], 0), tmp_dims[i], tmp_store);
    }

    if (is_block) {
      consumer_block->append_stmt(tmp_store);
    } else {
      parent_block->insert_stmt_after(tmp_store, consumer);
    }

    return std::make_pair(tmp_buf, consumer);
  }

  if (hasReads) {
    // Fill the cache with values from the consumer.
    StmtPtr tmp_store = alloc<Store>(
        tmp_buf, new_loop_vars_expr, alloc<Load>(producer, tmp_params));

    for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
      tmp_store = alloc<For>(
          new_loop_vars[i], immLike(tmp_dims[i], 0), tmp_dims[i], tmp_store);
    }

    if (is_block) {
      consumer_block->prepend_stmt(tmp_store);
    } else {
      parent_block->insert_stmt_before(tmp_store, consumer);
    }
  }

  if (hasWrites) {
    // sync the cache back to the producer buf.
    StmtPtr tmp_store = alloc<Store>(
        producer, tmp_params, alloc<Load>(tmp_buf, new_loop_vars_expr));

    for (int64_t i = new_loop_vars.size() - 1; i >= 0; --i) {
      tmp_store = alloc<For>(
          new_loop_vars[i], immLike(tmp_dims[i], 0), tmp_dims[i], tmp_store);
    }

    if (is_block) {
      consumer_block->append_stmt(tmp_store);
    } else {
      parent_block->insert_stmt_after(tmp_store, consumer);
    }
  }

  return std::make_pair(tmp_buf, consumer);
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
void LoopNest::computeAt(StmtPtr s, ForPtr f) {
  StorePtr st = getStoreStmtOfProducer(s);
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
  std::vector<ExprPtr> dims = getBoundExtents(bounds_it->second);

  // TODO: Use name-hint of the producer instead of "temp"
  BufPtr temp_buf = alloc<Buf>("temp", dims, st->value()->dtype());

  // Generate index variables for 'temp'
  std::vector<ExprPtr> temp_indices(dims.size());
  for (const auto i : c10::irange(dims.size())) {
    // TODO: Use name-hint of the producer indices instead of 'idx'
    temp_indices[i] =
        alloc<Var>(std::string("idx") + c10::to_string(i), dims[i]->dtype());
  }

  // Prepare substitute rules for constructing the temp statement from the prod
  // statement
  // TODO: Instead of going up the loop nest we should go through the indices in
  // the original tensor expression. The loops in the nest might've been
  // modified (e.g. split or merged) so that the loop indices no longer
  // correspond to the indices of the original expression and even their number
  // might be different. In that case, the loop below would crash.
  std::vector<VarPtr> prod_indices = getOuterLoopIndexes(s);
  std::vector<std::pair<VarPtr, ExprPtr>> rewrite_indices_map;
  std::vector<ExprPtr> offsets;
  for (const TensorAccessBoundsInfo& p : bounds_it->second) {
    for (const auto i : c10::irange(p.start.size())) {
      if (offsets.size() <= i) {
        offsets.push_back(p.start[i]);
      } else {
        offsets[i] =
            IRSimplifier::simplify(alloc<Min>(offsets[i], p.start[i], true));
      }
    }
  }

  for (const auto i : c10::irange(prod_indices.size())) {
    rewrite_indices_map.push_back(
        {prod_indices[i], alloc<Add>(temp_indices[i], offsets[i])});
  }

  // Construct the temp statement
  StmtPtr bd = alloc<Store>(
      temp_buf,
      temp_indices,
      SubstituteInClone(st->value(), rewrite_indices_map));

  // Construct the loop nest for the temp computation
  for (const auto i : c10::irange(dims.size())) {
    // We're creating loops from innermost to outermost, so we need to access
    // dimensions in reversed order.
    size_t dim_idx = dims.size() - 1 - i;
    bd = alloc<For>(
        to<Var>(temp_indices[dim_idx]),
        immLike(dims[dim_idx], 0),
        dims[dim_idx],
        bd);
  }

  // Add constructed stmts to the consumer loop
  f->body()->prepend_stmt(bd);

  // Rewrite accesses to producer in consumer with accesses to temp
  LoopComputeAtRewriter lr(st->buf(), temp_buf, offsets);
  StmtPtr new_f = f->accept_mutator(&lr);
  if (f != new_f) {
    BlockPtr bb = to<Block>(f->get_parent());
    bb->replace_stmt(f, new_f);
  }
}

class RfactorStoreRewriter : public IRMutator {
 public:
  RfactorStoreRewriter(
      BufPtr old_buf,
      const std::vector<ExprPtr>& old_indices,
      BufPtr new_buf,
      VarPtr reduction_var)
      : old_buf_(old_buf),
        old_indices_(old_indices),
        new_buf_(new_buf),
        reduction_var_(reduction_var),
        new_indices_(old_indices) {
    new_indices_.push_back(reduction_var_);
  }

  ExprPtr mutate(LoadPtr v) override {
    if (v->buf() != old_buf_) {
      return IRMutator::mutate(v);
    }

    TORCH_INTERNAL_ASSERT(
        old_indices_.size() == v->indices().size(),
        buildErrorMessage(
            "Expected ranks to match in RfactorStoreRewriter in the fuser."));

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

    return alloc<Load>(new_buf_, new_indices_);
  }

  ExprPtr mutate(ReduceOpPtr v) override {
    ExprPtr body_new = v->body()->accept_mutator(this);

    std::vector<VarPtr> new_reduce_args;
    for (auto r : v->reduce_args()) {
      if (r != reduction_var_) {
        new_reduce_args.push_back(r);
      }
    }

    return alloc<ReduceOp>(body_new, new_reduce_args, v->reducer());
  }

  StmtPtr mutate(StorePtr v) override {
    if (v->buf() != old_buf_) {
      return IRMutator::mutate(v);
    }

    TORCH_INTERNAL_ASSERT(
        old_indices_.size() == v->indices().size(),
        buildErrorMessage(
            "Expected ranks to match in RfactorStoreRewriter in the fuser."));

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

    ExprPtr new_value = v->value()->accept_mutator(this);
    return alloc<Store>(new_buf_, new_indices_, new_value);
  }

 private:
  BufPtr old_buf_;
  const std::vector<ExprPtr>& old_indices_;
  BufPtr new_buf_;
  VarPtr reduction_var_;
  std::vector<ExprPtr> new_indices_;
};

bool LoopNest::rfactor(StmtPtr st, ForPtr target_for) {
  BufPtr tmp_buf = nullptr;
  return rfactor(st, target_for, &tmp_buf);
}

bool LoopNest::rfactor(
    StmtPtr st,
    ForPtr outer_reduction_for,
    BufPtr* rfac_buf_ptr) {
  StorePtr reduction_store = to<Store>(st);
  ReduceOpPtr reduce_op = to<ReduceOp>(reduction_store->value());
  if (!reduce_op) {
    // Not a reduction store
    return false;
  }

  auto orig_buf = reduction_store->buf();
  auto orig_buf_indices = reduction_store->indices();
  VarPtr reduction_var = outer_reduction_for->var();

  std::set<VarPtr> reduce_args = {
      reduce_op->reduce_args().begin(), reduce_op->reduce_args().end()};

  if (reduce_args.size() < 2) {
    // Not enough reduction axis to do rfactor
    return false;
  }

  // Verify that outer_reduction_for is a perfect loop nest with all loops being
  // reductions
  StmtPtr cur = outer_reduction_for;
  while (ForPtr cur_for = to<For>(cur)) {
    if (!reduce_args.count(cur_for->var())) {
      // output axis inside outer_reduction_for are not allowed
      return false;
    }
    reduce_args.erase(cur_for->var());

    BlockPtr b = cur_for->body();
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

  std::vector<ExprPtr> rfac_dims = orig_buf->dims();
  ExprPtr extra_dim = IRSimplifier::simplify(
      alloc<Sub>(outer_reduction_for->stop(), outer_reduction_for->start()));
  rfac_dims.push_back(extra_dim);
  ExprPtr rfac_init =
      alloc<Cast>(reduce_op->dtype(), reduce_op->reducer().initializer());

  *rfac_buf_ptr = alloc<Buf>(
      orig_buf->name_hint() + "_rfac",
      rfac_dims,
      reduce_op->dtype(),
      rfac_init);
  BufPtr rfac_buf = *rfac_buf_ptr;

  // Rewrite the original reduction store to use the temporary rfac buffer:
  //   1) X[*indexes] --> T[*indexes + {reduction_var}]
  //   2) reduce_axis -= {reduction_var}
  RfactorStoreRewriter rfac_rewriter(
      orig_buf, orig_buf_indices, rfac_buf, reduction_var);
  to<Block>(st->get_parent())
      ->replace_stmt(st, st->accept_mutator(&rfac_rewriter));

  // Insert a store for the final reduction over the temp buffer into the
  // original buffer:
  //   X[*indexes] = ReduceOp(X[*indexes] + T[*indexes + {reduction_var}],
  //                          reduce_axis={reduction_var})
  BlockPtr b = outer_reduction_for->body();
  TORCH_INTERNAL_ASSERT(
      b->nstmts() == 1,
      buildErrorMessage(
          "Expected to have a single stmt in the block in rfactor transformation in the fuser."));
  StmtPtr first_reduction_loop = b->stmts().front();
  auto rfac_buf_indices = orig_buf_indices;
  rfac_buf_indices.emplace_back(reduction_var);

  ExprPtr final_reduce_load = alloc<Load>(rfac_buf, rfac_buf_indices);
  outer_reduction_for->body()->insert_stmt_after(
      alloc<Store>(
          orig_buf,
          orig_buf_indices,
          reduce_op->reducer()(
              orig_buf, final_reduce_load, orig_buf_indices, {reduction_var})),
      first_reduction_loop);

  // Insert an initialization store for the temp buffer:
  //   T[a,b,c] = init
  outer_reduction_for->body()->insert_stmt_before(
      alloc<Store>(rfac_buf, rfac_buf_indices, rfac_init),
      first_reduction_loop);
  return true;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
