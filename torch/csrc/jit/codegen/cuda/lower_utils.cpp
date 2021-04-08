#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

#include <algorithm>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace scope_utils {

// START SCOPE HELPER SYSTEMS
namespace {

class Loops : private OptInDispatch {
 private:
  std::deque<kir::ForLoop*> loops;
  void handle(kir::ForLoop* fl) final {
    loops.insert(loops.begin(), fl);
  }

  void handle(kir::IfThenElse* ite) final {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static std::vector<kir::ForLoop*> getLoops(Expr* scope) {
    Loops loops;
    Expr* it = scope;
    while (it != nullptr) {
      loops.handle(it);
      it = scope_utils::getParent(it);
    }
    return std::vector<kir::ForLoop*>(loops.loops.begin(), loops.loops.end());
  }
};

class scopePushBack : private OptInDispatch {
 private:
  Expr* expr_;
  void handle(kir::ForLoop* fl) final {
    fl->body().push_back(expr_);
  }

  void handle(kir::IfThenElse* ite) final {
    ite->thenBody().push_back(expr_);
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  scopePushBack(Expr* expr) : expr_(expr) {}

 public:
  static void push(Expr* scope, Expr* expr) {
    scopePushBack pb(expr);
    TORCH_INTERNAL_ASSERT(
        expr != nullptr && scope != nullptr,
        "Cannot push back, scope or expr is a nullptr.");
    pb.handle(scope);
  }
};

class scopeInsertBefore : private OptInDispatch {
 private:
  Expr* ref_;
  Expr* expr_;
  void handle(kir::ForLoop* fl) final {
    fl->body().insert_before(ref_, expr_);
  }

  void handle(kir::IfThenElse* ite) final {
    ite->thenBody().insert_before(ref_, expr_);
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  scopeInsertBefore(Expr* ref, Expr* expr) : ref_(ref), expr_(expr) {}

 public:
  static void insert(Expr* scope, Expr* ref, Expr* expr) {
    scopeInsertBefore scb(ref, expr);
    TORCH_INTERNAL_ASSERT(
        expr != nullptr && scope != nullptr,
        "Cannot push back, scope or expr is a nullptr.");
    scb.handle(scope);
  }
};

class ExprInScope : private OptInDispatch {
 private:
  Expr* expr_;
  bool contains_ = false;

  void handle(kir::ForLoop* fl) final {
    if (fl->body().contains(expr_)) {
      contains_ = true;
    }
  }

  void handle(kir::IfThenElse* ite) final {
    if (ite->thenBody().contains(expr_)) {
      contains_ = true;
    }
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  ExprInScope(Expr* expr) : expr_(expr) {}

 public:
  static bool find(Expr* scope, Expr* expr) {
    ExprInScope eis(expr);
    TORCH_INTERNAL_ASSERT(
        expr != nullptr && scope != nullptr,
        "Cannot push back, scope or expr is a nullptr.");
    eis.handle(scope);
    return eis.contains_;
  }
};

class parentScope : private OptInDispatch {
 private:
  Expr* parent_ = nullptr;

  void handle(kir::ForLoop* fl) final {
    parent_ = fl->parentScope();
  }

  void handle(kir::IfThenElse* ite) final {
    parent_ = ite->parentScope();
  }

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

 public:
  static Expr* get(Expr* scope) {
    parentScope sp;
    sp.handle(scope);
    return sp.parent_;
  }
};

void assertScope(Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      expr->getExprType() == ExprType::ForLoop ||
          expr->getExprType() == ExprType::IfThenElse,
      "Assert Scope failed when calling a scope_util function.");
}

class CloneLoopNest : public OptOutMutator {
 private:
  Expr* parent_scope_ = nullptr;
  Expr* to_clone_ = nullptr;

  Statement* mutate(kir::ForLoop* fl) final {
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());
    const auto parent_scope =
        fl == to_clone_ ? parent_scope_ : fl->parentScope();
    auto new_loop = ir_builder.create<kir::ForLoop>(
        fl->index(), fl->iter_domain(), parent_scope);
    for (Expr* expr : fl->body().exprs()) {
      new_loop->body().push_back(ir_utils::asExpr(OptOutMutator::mutate(expr)));
    }
    return new_loop;
  }

  CloneLoopNest(Expr* _to_clone, Expr* _parent_scope)
      : parent_scope_(_parent_scope), to_clone_(_to_clone) {}

 public:
  static kir::ForLoop* getClone(kir::ForLoop* _to_clone, Expr* _parent_scope) {
    TORCH_INTERNAL_ASSERT(
        _to_clone != nullptr,
        "Tried to clone a scope, but received a nullptr.");
    CloneLoopNest cln(_to_clone, _parent_scope);
    return ir_utils::asForLoop(ir_utils::asExpr(cln.mutate(_to_clone)));
  }
};

class ReplaceExprsInScope : public OptOutDispatch {
 public:
  static void replace(
      Expr* scope,
      std::unordered_map<Expr*, Expr*> replacement_map) {
    ReplaceExprsInScope reis(std::move(replacement_map));
    reis.handle(scope);
  }

 private:
  explicit ReplaceExprsInScope(std::unordered_map<Expr*, Expr*> replacement_map)
      : replacement_map_(std::move(replacement_map)) {}

  void handleScope(kir::Scope& scope) {
    for (size_t i = 0; i < scope.size(); ++i) {
      const auto it = replacement_map_.find(scope[i]);
      if (it == replacement_map_.end()) {
        handle(scope[i]);
        continue;
      }
      scope[i] = it->second;
    }
  }

  void handle(Expr* expr) final {
    OptOutDispatch::handle(expr);
  }

  void handle(kir::ForLoop* fl) final {
    handleScope(fl->body());
  }

  void handle(kir::IfThenElse* ite) final {
    handleScope(ite->thenBody());
    handleScope(ite->elseBody());
  }

 private:
  std::unordered_map<Expr*, Expr*> replacement_map_;
};

class FirstInnerMostScope : private OptInDispatch {
 private:
  Expr* active_scope = nullptr;

  void handle(kir::ForLoop* fl) final {
    for (auto expr : fl->body().exprs()) {
      if (ir_utils::isScope(expr)) {
        active_scope = expr;
        return;
      }
    }
    active_scope = nullptr;
  }

  void handle(kir::IfThenElse* ite) final {
    for (auto expr : ite->thenBody().exprs()) {
      if (ir_utils::isScope(expr)) {
        active_scope = expr;
        return;
      }
    }
    for (auto expr : ite->elseBody().exprs()) {
      if (ir_utils::isScope(expr)) {
        active_scope = expr;
        return;
      }
    }
    active_scope = nullptr;
  }

  Expr* getInner(Expr* expr) {
    OptInDispatch::handle(expr);
    return active_scope;
  }

 public:
  static Expr* get(Expr* scope) {
    TORCH_INTERNAL_ASSERT(
        scope != nullptr,
        "Tried to get inner most scope, but was provided nullptr.");

    FirstInnerMostScope fims;
    Expr* inner = fims.getInner(scope);

    if (inner == nullptr)
      return scope;

    while (fims.getInner(inner) != nullptr)
      inner = fims.getInner(inner);
    return inner;
  }
};

// END SCOPE HELPER SYSTEMS
} // namespace

// Grab the ForLoop starting from scope working out
std::vector<kir::ForLoop*> getLoops(Expr* scope) {
  if (scope == nullptr)
    return std::vector<kir::ForLoop*>();
  assertScope(scope);
  return Loops::getLoops(scope);
}

// Push back an expr to scope
void pushBack(Expr* scope, Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      scope != nullptr, "Scope is a nullptr, cannot push an expr to it.");
  assertScope(scope);
  scopePushBack::push(scope, expr);
}

// Insert expr in scope before ref
void insertBefore(Expr* scope, Expr* ref, Expr* expr) {
  scopeInsertBefore::insert(scope, ref, expr);
}

bool exprInScope(Expr* scope, Expr* expr) {
  return ExprInScope::find(scope, expr);
}

// Return the parent of the active scope
Expr* getParent(Expr* scope) {
  TORCH_INTERNAL_ASSERT(
      scope != nullptr,
      "Tried to close the active scope, but there isn't one set.");
  assertScope(scope);
  return parentScope::get(scope);
}

// Open a new inner most for loop
kir::ForLoop* openFor(Expr* scope, IterDomain* id) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  const auto kir_id = GpuLower::lowerValue(id)->as<kir::IterDomain>();
  kir::ForLoop* new_scope = nullptr;
  if (id->isThread()) {
    std::stringstream ss;
    ss << id->getParallelType();
    new_scope = ir_builder.create<kir::ForLoop>(
        ir_builder.create<kir::NamedScalar>(ss.str(), DataType::Int),
        kir_id,
        scope);
  } else {
    new_scope = ir_builder.create<kir::ForLoop>(
        ir_builder.create<kir::Int>(c10::nullopt), kir_id, scope);
  }
  if (scope != nullptr)
    pushBack(scope, new_scope);
  return new_scope;
}

kir::ForLoop* cloneLoopNest(kir::ForLoop* to_clone, Expr* parent_scope) {
  return CloneLoopNest::getClone(to_clone, parent_scope);
}

void replaceExprsInScope(
    Expr* scope,
    std::unordered_map<Expr*, Expr*> replacement_map) {
  TORCH_INTERNAL_ASSERT(
      replacement_map.find(scope) == replacement_map.end(),
      "Error trying to replace expressions in a scope, scope wants to be replaced entirely.");
  ReplaceExprsInScope::replace(scope, std::move(replacement_map));
}

Expr* firstInnerMostScope(Expr* scope) {
  return FirstInnerMostScope::get(scope);
}

} // namespace scope_utils

namespace ir_utils {

TVDomainGuard::TVDomainGuard(TensorView* _tv, TensorDomain* td)
    : tv_(_tv), prev_domain(tv_->domain()) {
  tv_->setDomain(td);
}

TVDomainGuard::~TVDomainGuard() {
  tv_->setDomain(prev_domain);
}

std::vector<IterDomain*> iterDomainInputsOf(
    const std::vector<IterDomain*>& input_ids) {
  auto inputs = IterVisitor::getInputsTo({input_ids.begin(), input_ids.end()});
  std::vector<IterDomain*> id_inputs(
      ir_utils::filterByType<IterDomain>(inputs).begin(),
      ir_utils::filterByType<IterDomain>(inputs).end());
  return id_inputs;
}

std::vector<IterDomain*> iterDomainInputsOfOrderedAs(
    const std::vector<IterDomain*>& of,
    const std::vector<IterDomain*>& order) {
  auto inputs_vec = iterDomainInputsOf(of);

  std::unordered_set<IterDomain*> inputs_set(
      inputs_vec.begin(), inputs_vec.end());

  std::vector<IterDomain*> ordered_inputs;
  std::copy_if(
      order.begin(),
      order.end(),
      std::back_inserter(ordered_inputs),
      [&inputs_set](const auto& id) {
        return inputs_set.find(id) != inputs_set.end();
      });

  return ordered_inputs;
}

std::vector<Val*> indices(std::vector<kir::ForLoop*> loops) {
  std::vector<Val*> inds(loops.size());
  std::transform(
      loops.begin(), loops.end(), inds.begin(), [](kir::ForLoop* fl) {
        return fl->index();
      });
  return inds;
}

bool isTV(const Val* val) {
  return val->getValType().value() == ValType::TensorView;
}

// Check if we're a TensorView op that we can generate code for.
bool isTVOp(const Expr* expr) {
  if (expr->outputs().size() == 1 && isTV(expr->output(0)) &&
      (expr->getExprType().value() == ExprType::BinaryOp ||
       expr->getExprType().value() == ExprType::UnaryOp ||
       expr->getExprType().value() == ExprType::TernaryOp ||
       expr->getExprType().value() == ExprType::ReductionOp ||
       expr->getExprType().value() == ExprType::BroadcastOp))
    return true;
  return false;
}

TensorView* getTVOutput(const Expr* expr) {
  for (auto out : expr->outputs()) {
    if (out->getValType().value() == ValType::TensorView) {
      return out->as<TensorView>();
    }
  }
  return nullptr;
}

bool isScalarOp(const Expr* expr) {
  for (auto out : expr->outputs())
    if (!out->isScalar())
      return false;
  return true;
}

void ASSERT_EXPR(Statement* stmt) {
  TORCH_INTERNAL_ASSERT(
      stmt->isExpr(),
      "Tried to generate a kernel but hit a non expression during lowering: ",
      stmt);
}

Expr* asExpr(Statement* stmt) {
  ASSERT_EXPR(stmt);
  return stmt->as<Expr>();
}

TensorView* asTV(Val* val) {
  TORCH_INTERNAL_ASSERT(isTV(val));
  return val->as<TensorView>();
}

bool isScope(const Expr* expr) {
  return expr->getExprType() == ExprType::ForLoop ||
      expr->getExprType() == ExprType::IfThenElse;
}

kir::ForLoop* asForLoop(Statement* stmt) {
  Expr* expr = asExpr(stmt);
  TORCH_INTERNAL_ASSERT(expr->getExprType() == ExprType::ForLoop);
  return expr->as<kir::ForLoop>();
}

const TensorView* asConstTV(const Val* val) {
  TORCH_INTERNAL_ASSERT(isTV(val));
  return val->as<TensorView>();
}

bool isUnrolledFor(const Expr* expr) {
  if (expr->getExprType() != ExprType::ForLoop) {
    return false;
  }
  return expr->as<kir::ForLoop>()->iter_domain()->getParallelType() ==
      ParallelType::Unroll;
}

const std::unordered_map<ParallelType, int, TypeHash>
    ParallelTypeBitmap::pt_to_offset_{
        {ParallelType::BIDx, 0},
        {ParallelType::BIDy, 1},
        {ParallelType::BIDz, 2},
        {ParallelType::TIDx, 3},
        {ParallelType::TIDy, 4},
        {ParallelType::TIDz, 5}};

const std::unordered_map<int, ParallelType> ParallelTypeBitmap::offset_to_pt_ =
    {{0, ParallelType::BIDx},
     {1, ParallelType::BIDy},
     {2, ParallelType::BIDz},
     {3, ParallelType::TIDx},
     {4, ParallelType::TIDy},
     {5, ParallelType::TIDz}};

bool ParallelTypeBitmap::get(ParallelType pt) const {
  if (pt_to_offset_.find(pt) == pt_to_offset_.end()) {
    TORCH_INTERNAL_ASSERT(false, "Could not recognize parallel type.");
  }
  return bitset_[pt_to_offset_.at(pt)];
}

bool ParallelTypeBitmap::set(ParallelType pt, bool new_val) {
  if (pt_to_offset_.find(pt) == pt_to_offset_.end()) {
    TORCH_INTERNAL_ASSERT(false, "Could not recognize parallel type.");
  }
  bool old_val = bitset_[pt_to_offset_.at(pt)];
  bitset_[pt_to_offset_.at(pt)] = new_val;
  return old_val;
}

ParallelTypeBitmap ParallelTypeBitmap::operator&=(
    const ParallelTypeBitmap& other) {
  bitset_ &= other.bitset_;
  return *this;
}

ParallelTypeBitmap ParallelTypeBitmap::operator|=(
    const ParallelTypeBitmap& other) {
  bitset_ |= other.bitset_;
  return *this;
}

ParallelTypeBitmap ParallelTypeBitmap::operator^=(
    const ParallelTypeBitmap& other) {
  bitset_ ^= other.bitset_;
  return *this;
}

ParallelTypeBitmap ParallelTypeBitmap::operator~() const {
  return ParallelTypeBitmap(~bitset_);
}

bool ParallelTypeBitmap::none() const {
  return bitset_.none();
}

bool ParallelTypeBitmap::any() const {
  return bitset_.any();
}

bool ParallelTypeBitmap::all() const {
  return bitset_.all();
}

bool ParallelTypeBitmap::operator[](size_t pos) const {
  TORCH_INTERNAL_ASSERT(
      pos < num_p_type, "Invalid index to ParallelTypeBitset: ", pos);
  return bitset_[pos];
}

std::map<ParallelType, bool> ParallelTypeBitmap::getMap() const {
  std::map<ParallelType, bool> map;
  for (const auto& pt_offset : pt_to_offset_) {
    map.emplace(pt_offset.first, bitset_[pt_offset.second]);
  }
  return map;
}

ParallelTypeBitmap operator&(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x &= rhs;
  return x;
}

ParallelTypeBitmap operator|(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x |= rhs;
  return x;
}

ParallelTypeBitmap operator^(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x ^= rhs;
  return x;
}

ParallelTypeBitmap getParallelBroadcastDomains(
    const Val* bop_out,
    const ThreadPredicateMap& preds) {
  if (bop_out->getValType().value() == ValType::TensorIndex) {
    bop_out = bop_out->as<kir::TensorIndex>()->view()->fuserTv();
  }
  TORCH_INTERNAL_ASSERT(
      bop_out->getValType().value() == ValType::TensorView,
      "Out is not tensor view");
  auto out_tv = bop_out->as<TensorView>();
  // If no pred is found for out_tv, no predicate is necessary
  if (preds.find(out_tv) == preds.end()) {
    return ParallelTypeBitmap();
  }
  const ParallelTypeBitmap& out_pred = preds.at(out_tv).first;

  ParallelTypeBitmap parallel_broadcast;
  const auto& iter_domains = out_tv->domain()->domain();
  // If the output is on shared memory, assume that all subsequent
  // reads from all threads in its CTA can be done with no parallel
  // broadcast. Only one thread will write to shared memory followed
  // by a proper _syncthreads.
  const bool output_smem = out_tv->getMemoryType() == MemoryType::Shared;
  for (auto id : iter_domains) {
    if (!id->isBroadcast()) {
      continue;
    }
    if (id->isBlockDim() || (!output_smem && id->isThreadDim())) {
      parallel_broadcast.set(id->getParallelType(), true);
    }
  }

  return parallel_broadcast & out_pred;
}

} // namespace ir_utils

namespace loop_utils {

std::pair<kir::ForLoop*, int64_t> getAllocPoint(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops) {
  // If in global memory, it can be all the way outside the loops.
  if (tv->getMemoryType() == MemoryType::Global) {
    return {nullptr, 0};
  }

  // Figure out where we want to place alloc/reduction initialization. We want
  // outside an unroll loop, or inside our computeAt point.
  kir::ForLoop* alloc_loop = nullptr;

  auto loops_it = loops.begin();

  // Look at each axis individually in out's domain
  for (const auto tv_i : c10::irange((int64_t)tv->getThisComputeAtAxis())) {
    // Grab the axis ID

    auto ca_id = tv->getComputeAtAxis(tv_i).first;
    auto kir_ca_id = GpuLower::lowerValue(ca_id)->as<kir::IterDomain>();

    loops_it =
        std::find_if(loops_it, loops.end(), [&kir_ca_id](const auto& loop) {
          return kir_ca_id == loop->iter_domain() ||
              loop->iter_domain()->getParallelType() == ParallelType::Unroll;
        });

    if (loops_it == loops.end()) {
      for (auto loop : loops) {
        std::cout << kir::toString(loop->iter_domain()) << "  ";
      }
      std::cout << std::endl;
    }
    TORCH_INTERNAL_ASSERT(
        loops_it != loops.end(),
        "Could not find all required axes for indexing when trying to index into ",
        tv);

    if (kir_ca_id->getParallelType() == ParallelType::Unroll) {
      return {alloc_loop, tv_i};
    }

    alloc_loop = *loops_it;
    ++loops_it;
  }

  return {alloc_loop, (int64_t)tv->getThisComputeAtAxis()};
}

std::unordered_map<IterDomain*, IterDomain*> p2cRootMap(
    const std::vector<Expr*>& exprs) {
  std::unordered_map<IterDomain*, IterDomain*> p2c_root_map;

  for (auto expr : exprs) {
    auto out_tv = ir_utils::getTVOutput(expr);
    for (auto inp : expr->inputs()) {
      if (inp->getValType().value() != ValType::TensorView) {
        continue;
      }

      auto root_p2c = TensorDomain::mapRootPtoC(
          inp->as<TensorView>()->domain(), out_tv->domain());
      for (auto entry : root_p2c) {
        auto p_id = entry.first;
        auto c_id = entry.second;
        // Careful we don't allow circular references
        if (p_id != c_id) {
          p2c_root_map[p_id] = c_id;
        }
      }
    }
  }

  return p2c_root_map;
}

IterDomain* getTermIDInMap(
    IterDomain* root_id,
    std::unordered_map<IterDomain*, IterDomain*> p2c_root_map) {
  auto entry = root_id;
  while (p2c_root_map.find(entry) != p2c_root_map.end()) {
    entry = p2c_root_map.at(entry);
  }
  return entry;
}

} // namespace loop_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
