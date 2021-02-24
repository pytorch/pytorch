#include <torch/csrc/jit/tensorexpr/mem_dependency_checker.h>
#include <fstream>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace analysis {

const char* AccessToString(AccessType a) {
  switch (a) {
    case AccessType::Input:
      return "Input";
    case AccessType::Output:
      return "Output";
    case AccessType::Load:
      return "Load";
    case AccessType::Store:
      return "Store";
    case AccessType::Call:
      return "Call";
    case AccessType::AtomicAdd:
      return "AtomicAdd";
    case AccessType::Alloc:
      return "Alloc";
    case AccessType::Free:
      return "Free";
    default:
      break;
  }
  return "Unknown";
}

void getDependencyChain(
    const std::shared_ptr<AccessInfo>& info,
    DependencySet& dependencies) {
  if (!dependencies.insert(info).second) {
    return;
  }

  for (auto& dep : info->dependencies()) {
    getDependencyChain(dep.second, dependencies);
  }
}

void getDependentsChain(
    const std::shared_ptr<AccessInfo>& info,
    DependencySet& dependents) {
  if (!dependents.insert(info).second) {
    return;
  }

  for (auto& dep : info->dependents()) {
    getDependencyChain(dep.second, dependents);
  }
}

// AccessInfo

std::vector<const Expr*> AccessInfo::getIndices() const {
  std::vector<const Expr*> indices;

  if (expr_) {
    if (auto* load = dynamic_cast<const Load*>(expr_)) {
      indices = load->indices();
    } else if (auto* call = dynamic_cast<const FunctionCall*>(expr_)) {
      indices = call->params();
    }
  } else {
    if (auto* store = dynamic_cast<const Store*>(stmt_)) {
      indices = store->indices();
    }
  }
  return indices;
}

void AccessInfo::addDependency(const std::shared_ptr<AccessInfo>& write) {
  auto res = dependencies_.emplace(write->id(), write);
  TORCH_INTERNAL_ASSERT(res.second);
}

void AccessInfo::addDependent(const std::shared_ptr<AccessInfo>& read) {
  auto res = dependents_.emplace(read->id(), read);
  TORCH_INTERNAL_ASSERT(res.second);
}

bool AccessInfo::hasDependency(const std::shared_ptr<AccessInfo>& info) const {
  return dependencies_.count(info->id()) != 0;
}

DependencySet AccessInfo::getDirectDependencies() {
  DependencySet res;
  for (auto& depPair : dependencies_) {
    res.insert(depPair.second);
  }
  return res;
}

DependencySet AccessInfo::getIndirectDependencies() {
  DependencySet res;
  for (auto& depPair : dependencies_) {
    getDependencyChain(depPair.second, res);
  }
  return res;
}

DependencySet AccessInfo::getDirectDependents() {
  DependencySet res;
  for (auto& depPair : dependents_) {
    res.insert(depPair.second);
  }
  return res;
}

DependencySet AccessInfo::getIndirectDependents() {
  DependencySet res;
  for (auto& depPair : dependencies_) {
    getDependentsChain(depPair.second, res);
  }
  return res;
}

bool AccessInfo::isRead() const {
  switch (type_) {
    case AccessType::Output:
    case AccessType::Load:
    case AccessType::Call:
    case AccessType::AtomicAdd:
      return true;
    default:
      break;
  }
  return false;
}

bool AccessInfo::isWrite() const {
  switch (type_) {
    case AccessType::Input:
    case AccessType::Store:
    case AccessType::AtomicAdd:
    case AccessType::Alloc:
    case AccessType::Free:
      return true;
    default:
      break;
  }
  return false;
}

void AccessInfo::print() const {
  std::cout << id_ << ". " << AccessToString(type_) << ": " << *var_ << "[";
  if (bounds_.size() > 0) {
    for (size_t i = 0; i < bounds_.size() - 1; ++i) {
      bounds_[i].print();
      std::cout << ", ";
    }

    size_t i = bounds_.size() - 1;
    bounds_[i].print();
  }
  std::cout << "]";

  if (!dependencies_.empty()) {
    std::cout << " - depends on: ";
    for (auto& pair : dependencies_) {
      std::cout << pair.second->id() << " ";
    }
  }

  if (!dependents_.empty()) {
    std::cout << " - dependents: ";
    for (auto& pair : dependents_) {
      std::cout << pair.second->id() << " ";
    }
  }

  std::cout << "\n";
}

void AccessInfo::dumpDOT(std::ostream& os) const {
  if (type_ == AccessType::Input || type_ == AccessType::Output ||
      type_ == AccessType::Alloc) {
    os << "n" << id_ << " [\n";
    os << "label = \"" << AccessToString(type_) << "\\n " << *var_ << "[";
    if (bounds_.size() > 0) {
      for (size_t i = 0; i < bounds_.size() - 1; ++i) {
        os << *IRSimplifier::simplify(new Add(bounds_[i].end, new IntImm(1)))
           << ", ";
      }

      size_t i = bounds_.size() - 1;
      os << *IRSimplifier::simplify(new Add(bounds_[i].end, new IntImm(1)));
      os << "]\"\n ";
    }
    if (isWrite()) {
      os << "\tshape = \"invhouse\"\n";
    } else {
      os << "\tshape = \"house\"\n";
    }
  } else {
    os << "n" << id_ << " [\n";
    os << "label = \"" << AccessToString(type_) << " (#" << id_ << ")\\n";
    os << "buf : " << *var_ << "\\n";
    os << "bounds : \[";
    if (bounds_.size() > 0) {
      for (size_t i = 0; i < bounds_.size() - 1; ++i) {
        os << "(" << *bounds_[i].start << ", " << *bounds_[i].end << "), ";
      }

      size_t i = bounds_.size() - 1;
      os << "(" << *bounds_[i].start << ", " << *bounds_[i].end << ")]";
    }
    os << "\"\n";
    os << "\tshape = \"box\"\n";
  }
  os << "\tstyle=\"filled\"\n";
  os << "\tcolor=\"" << AccessTypeColour() << "\"\n";
  std::string edgeColour;
  if (isWrite()) {
    edgeColour = "cornflowerblue";
  } else {
    edgeColour = "goldenrod";
  }
  os << "]\n";
  for (auto& pair : dependencies_) {
    os << "n" << pair.second->id() << " -> "
       << "n" << id_ << " [color=\"" << edgeColour << "\"]\n";
  }
}

const char* AccessInfo::AccessTypeColour() const {
  switch (type_) {
    case AccessType::Input:
    case AccessType::Output:
      return "palegreen";
    case AccessType::Load:
      return "peachpuff";
    case AccessType::Store:
      return "dodgerblue";
    case AccessType::Call:
      return "violet";
    case AccessType::Alloc:
    case AccessType::Free:
      return "sandybrown";
    default:
      break;
  }
  return "white";
}

// MemDependencyChecker
//
MemDependencyChecker::MemDependencyChecker() {
  currentScope_ = std::make_shared<Scope>(nullptr, nullptr);
}

MemDependencyChecker::MemDependencyChecker(
    const std::unordered_set<const Buf*>& inputs,
    const std::unordered_set<const Buf*>& outputs) {
  for (auto* s : inputs) {
    inputs_[s] = nullptr;
  }
  for (auto* s : outputs) {
    outputs_[s] = nullptr;
  }

  currentScope_ = std::make_shared<Scope>(nullptr, nullptr);
}

MemDependencyChecker::MemDependencyChecker(
    const std::vector<BufHandle>& inputs,
    const std::vector<BufHandle>& outputs) {
  for (auto& s : inputs) {
    inputs_[s.node()] = nullptr;
  }
  for (auto& s : outputs) {
    outputs_[s.node()] = nullptr;
  }

  currentScope_ = std::make_shared<Scope>(nullptr, nullptr);
}

bool MemDependencyChecker::allowLoopExecutionOrderAnalysis(bool allow) {
  std::swap(allowExecutionOrderAnalysis_, allow);
  return allow;
}

const std::vector<std::shared_ptr<AccessInfo>>& MemDependencyChecker::
    getHistory() const {
  return currentScope_->accesses_;
}

void MemDependencyChecker::dumpDAG(const std::string& filename) const {
  std::ofstream dotfile(filename);

  dotfile << "digraph {\n";
  for (auto& wi : getHistory()) {
    wi->dumpDOT(dotfile);
  }
  dotfile << "}\n";
  dotfile.close();
}

// dependsDirectly, dependsIndirectly and friends:

DependencySet MemDependencyChecker::getAllWriteDependencies(
    const DependencySet& products) {
  DependencySet writes;

  for (auto& info : products) {
    DependencySet dependencies;
    getDependencyChain(info, dependencies);
    for (auto& other : dependencies) {
      if (other->isWrite()) {
        writes.insert(other);
      }
    }
  }

  return writes;
}

bool MemDependencyChecker::dependsDirectly(const Expr* A, const Stmt* B) {
  return dependsDirectlyHelper(A, B);
}

bool MemDependencyChecker::dependsDirectly(const Stmt* A, const Stmt* B) {
  return dependsDirectlyHelper(A, B);
}

bool MemDependencyChecker::dependsDirectly(const Buf* O, const Stmt* B) {
  auto outputAccess = output(O);
  auto bWrites = getAllWritesWithin(B);

  for (auto& depPair : outputAccess->dependencies()) {
    if (bWrites.count(depPair.second) != 0) {
      return true;
    }
  }

  return false;
}

bool MemDependencyChecker::dependsDirectly(const Stmt* A, const Buf* I) {
  auto aReads = getAllReadsWithin(A);
  auto inputAccess = input(I);

  for (auto& depPair : inputAccess->dependents()) {
    if (aReads.count(depPair.second) != 0) {
      return true;
    }
  }

  return false;
}

bool MemDependencyChecker::dependsDirectly(const Expr* A, const Buf* I) {
  auto aReads = getAllReadsWithin(A);
  auto inputAccess = input(I);

  for (auto& depPair : inputAccess->dependents()) {
    if (aReads.count(depPair.second) != 0) {
      return true;
    }
  }

  return false;
}

bool MemDependencyChecker::dependsDirectly(
    const std::shared_ptr<AccessInfo>& A,
    const std::shared_ptr<AccessInfo>& B) {
  return A->hasDependency(B) && B->isWrite();
}

bool MemDependencyChecker::dependsIndirectly(const Expr* A, const Stmt* B) {
  return dependsIndirectlyHelper(A, B);
}

bool MemDependencyChecker::dependsIndirectly(const Stmt* A, const Stmt* B) {
  return dependsIndirectlyHelper(A, B);
}

bool MemDependencyChecker::dependsIndirectly(const Buf* O, const Stmt* B) {
  auto outputAccess = output(O);

  DependencySet dependencies;
  getDependencyChain(outputAccess, dependencies);

  auto bWrites = getAllWritesWithin(B);
  for (auto& dep : dependencies) {
    if (bWrites.count(dep) != 0) {
      return true;
    }
  }

  return false;
}

bool MemDependencyChecker::dependsIndirectly(const Stmt* A, const Buf* I) {
  auto aReads = getAllReadsWithin(A);
  auto inputAccess = input(I);

  auto aDeps = getAllWriteDependencies(aReads);

  return aDeps.count(inputAccess) != 0;
}

bool MemDependencyChecker::dependsIndirectly(const Expr* A, const Buf* I) {
  auto aReads = getAllReadsWithin(A);
  auto inputAccess = input(I);

  auto aDeps = getAllWriteDependencies(aReads);

  return aDeps.count(inputAccess) != 0;
}

bool MemDependencyChecker::dependsIndirectly(const Buf* O, const Buf* I) {
  auto outputAccess = output(O);
  auto inputAccess = input(I);

  return dependsIndirectly(outputAccess, inputAccess);
}

bool MemDependencyChecker::dependsIndirectly(
    const std::shared_ptr<AccessInfo>& A,
    const std::shared_ptr<AccessInfo>& B) {
  if (!B->isWrite()) {
    return false;
  }

  DependencySet dependencies;
  getDependencyChain(A, dependencies);
  if (dependencies.count(B) == 0) {
    return false;
  }

  return true;
}

std::shared_ptr<AccessInfo> MemDependencyChecker::accessFor(
    const Stmt* A) const {
  auto bound = stmtToAccess_.equal_range(A);
  for (auto it = bound.first; it != bound.second; ++it) {
    if (it->second->expr() == nullptr) {
      return it->second;
    }
  }
  return nullptr;
}

std::shared_ptr<AccessInfo> MemDependencyChecker::accessFor(
    const Expr* A) const {
  // TODO exprs can have multiple accesses... we're returning the first but that
  // isn't great. Can't do much here.
  auto bound = exprToAccess_.equal_range(A);
  if (bound.first != exprToAccess_.end()) {
    return bound.first->second;
  }

  return nullptr;
}

std::unordered_set<std::shared_ptr<AccessInfo>> MemDependencyChecker::
    accessesWithin(const Stmt* A) const {
  auto it = scopeToAccesses_.find(A);
  if (it != scopeToAccesses_.end()) {
    return std::unordered_set<std::shared_ptr<AccessInfo>>(
        it->second.begin(), it->second.end());
  }

  std::unordered_set<std::shared_ptr<AccessInfo>> ret;
  auto bound = stmtToAccess_.equal_range(A);
  for (auto it = bound.first; it != bound.second; ++it) {
    ret.insert(it->second);
  }
  return ret;
}

std::unordered_set<std::shared_ptr<AccessInfo>> MemDependencyChecker::
    accessesWithin(const Expr* A) const {
  return {accessFor(A)};
}

std::shared_ptr<AccessInfo> MemDependencyChecker::input(const Buf* b) const {
  auto it = inputs_.find(b);
  if (it == inputs_.end()) {
    return nullptr;
  }
  return it->second;
}

std::shared_ptr<AccessInfo> MemDependencyChecker::output(const Buf* b) const {
  auto it = outputs_.find(b);
  if (it == outputs_.end()) {
    return nullptr;
  }
  return it->second;
}

// Node visitors:

void MemDependencyChecker::visit(const Store* v) {
  const Stmt* last = lastStmt_;
  lastStmt_ = v;
  v->value()->accept(this);

  for (const Expr* ind : v->indices()) {
    ind->accept(this);
  }
  lastStmt_ = last;

  // Create a new AccessInfo for the store.
  const Var* var = v->buf()->base_handle();
  auto info = std::make_shared<AccessInfo>(
      nextAccess_++, AccessType::Store, v, var, getIndicesBounds(v->indices()));

  // Add a dependency to any accesses that are within the scope of this store
  // (ie. the RHS).
  auto bound = stmtToAccess_.equal_range(v);
  for (auto it = bound.first; it != bound.second; ++it) {
    info->addDependency(it->second);
    it->second->addDependent(info);
  }

  stmtToAccess_.emplace(v, info);

  // This write is open, and will close any open writes that it totally
  // overlaps.
  auto& history = currentScope_->openWrites_[var];
  updateWriteHistory(history, info, info->id());
  currentScope_->accesses_.push_back(info);
}

void MemDependencyChecker::visit(const Load* v) {
  // Create a temporary scope to hold any loads that occur within the indices of
  // this load.
  auto indicesScope =
      std::make_shared<Scope>(currentScope_->block, currentScope_);
  currentScope_ = indicesScope;

  for (const Expr* ind : v->indices()) {
    ind->accept(this);
  }

  // Create a new AccessInfo for the load.
  const Var* var = v->buf()->base_handle();
  auto load = std::make_shared<AccessInfo>(
      nextAccess_++,
      AccessType::Load,
      v,
      lastStmt_,
      var,
      getIndicesBounds(v->indices()));

  // If there were loads in the indices, this load depends on them, and merge
  // them in.
  if (!indicesScope->accesses_.empty()) {
    for (auto& access : indicesScope->accesses_) {
      load->addDependency(access);
      access->addDependent(load);
    }
    mergeScope(indicesScope, indicesScope->parent, false);
  }

  currentScope_ = indicesScope->parent;

  stmtToAccess_.emplace(lastStmt_, load);
  exprToAccess_.emplace(v, load);

  // This is a read, and does not close any accesses - but we need to establish
  // dependencies on accesses in the same scope.
  // Intentionally using operator[], we want it to be created if it does not
  // exist.
  auto& writeHistory = currentScope_->openWrites_[var];
  updateWriteHistory(writeHistory, load, load->id());
  currentScope_->accesses_.push_back(load);
}

void MemDependencyChecker::visit(const FunctionCall* v) {
  // This is essentially the same as Load.
  auto paramScope =
      std::make_shared<Scope>(currentScope_->block, currentScope_);
  currentScope_ = paramScope;

  for (const Expr* param : v->params()) {
    param->accept(this);
  }

  const Var* var = v->tensor()->buf()->base_handle();
  auto call = std::make_shared<AccessInfo>(
      nextAccess_++,
      AccessType::Call,
      v,
      lastStmt_,
      var,
      getIndicesBounds(v->params()));

  // If there were loads in the parameters, this call depends on them, also
  // merge.
  if (!paramScope->accesses_.empty()) {
    for (auto& access : paramScope->accesses_) {
      call->addDependency(access);
      access->addDependent(call);
    }
    mergeScope(paramScope, paramScope->parent, false);
  }

  currentScope_ = paramScope->parent;

  stmtToAccess_.emplace(lastStmt_, call);
  exprToAccess_.emplace(v, call);

  // Intentionally using operator[], we want it to be created if it does not
  // exist.
  auto& writeHistory = currentScope_->openWrites_[var];
  updateWriteHistory(writeHistory, call, call->id());
  currentScope_->accesses_.push_back(call);
}

// This check determines if two accesses within a loop are "safe" from loop-self
// dependence. This function does not consider overlap in bound range, but
// rather the stride of the bound relative to the loop variable. This is the
// section of the code which considers iteration order, if allowed.
bool executionSafetyCheck(
    const std::shared_ptr<AccessInfo>& info,
    const std::shared_ptr<AccessInfo>& other,
    const std::vector<const Expr*>& aStrides,
    const std::vector<const Expr*>& oStrides,
    bool parallelized) {
  if (aStrides.empty() || oStrides.empty()) {
    return false;
  }
  TORCH_INTERNAL_ASSERT(info->bounds().size() == other->bounds().size());
  for (size_t b = 0; b < info->bounds().size(); ++b) {
    const Expr* aIndexStride = aStrides[b];
    const Expr* oIndexStride = oStrides[b];
    // can't be safe on this index if we can't determine stride.
    if (!aIndexStride->isConstant() || !oIndexStride->isConstant()) {
      continue;
    }

    const Expr* minStride =
        IRSimplifier::simplify(new Min(aIndexStride, oIndexStride, true));
    const Expr* maxStride =
        IRSimplifier::simplify(new Max(aIndexStride, oIndexStride, true));

    // If the first access has no stride don't apply safety).
    if (immediateEquals(minStride, 0)) {
      continue;
    }

    const Expr* modCheck =
        IRSimplifier::simplify(new Mod(maxStride, minStride));

    // if the strides can't have easily inferable distinct offsets, they're not
    // safe.
    if (!immediateEquals(modCheck, 0)) {
      continue;
    }

    // If the loop has a defined execution order (ie. sequential for) then
    // the order of execution can provide safety from overlaps.
    // Specifically if the difference in first access position for any
    // axis is the same sign as the common stride, then they will not
    // overlap.

    const Expr* startDiff = IRSimplifier::simplify(
        new Sub(info->bounds()[b].start, other->bounds()[b].start));

    bool diffNegative = immediateIsNegative(startDiff);
    bool strideNegative = immediateIsNegative(minStride);

    // Invert the startDiff so mod works.
    if (diffNegative != strideNegative) {
      startDiff = IRSimplifier::simplify(new Sub(new IntImm(0), startDiff));
    }

    // If both accesses have the same stride, and the difference in start
    // element is smaller than this stride then the entire range is distinct.
    if (exprEquals(minStride, maxStride)) {
      const Expr* check1 =
          IRSimplifier::simplify(new CompareSelect(startDiff, minStride, kLT));
      if (check1->isConstant() && immediateEquals(check1, 1)) {
        return true;
      }
    }

    startDiff = IRSimplifier::simplify(new Mod(startDiff, minStride));

    CompareSelectOperation op = strideNegative ? kLT : kGT;

    const Expr* check =
        IRSimplifier::simplify(new CompareSelect(startDiff, new IntImm(0), op));

    // If the start difference modulo the minimum stride is offset from that
    // stride, then the ranges have distinct strides.
    if (check->isConstant() && immediateEquals<int>(check, 1)) {
      return true;
    }

    // If we can consider execution order and the difference in offset is
    // opposite signed to the stride then the read occurs in the past and we can
    // infer safety.
    if (!parallelized && diffNegative == strideNegative &&
        immediateEquals(startDiff, 0)) {
      return true;
    }
  }

  return false;
}

void MemDependencyChecker::visit(const For* v) {
  const Var* var = v->var();

  const Stmt* last = lastStmt_;
  lastStmt_ = v;

  v->var()->accept(this);

  // Loads inside the For's start and stop expression are special.
  // They exist in the enclosing scope, but accesses within the loop body may
  // depend on them via usage of the loop variable.
  // The way we handle this is to create a new scope so we have an easily
  // accessible list of the acceses within the extents.
  auto extentsScope =
      std::make_shared<Scope>(currentScope_->block, currentScope_);
  currentScope_ = extentsScope;

  v->start()->accept(this);
  v->stop()->accept(this);

  currentScope_ = currentScope_->parent;

  auto newScope = std::make_shared<Scope>(v->body(), currentScope_);
  currentScope_ = newScope;

  v->body()->accept(this);

  lastStmt_ = last;

  // Ok now we need to determine whether accesses in the loop depend on
  // other loop iterations.
  //
  // This is the real challenge here, it depends on both the fully expanded
  // bounds and the symbolic bounds.

  // The indices must change monotonically to avoid intersection. This is
  // hard to determine, so here's our heuristic I hope it's conservative
  // enough.

  // the size of at least one dependent index must be >= the size of the
  // loop.

  // First step is to infer the stride relative to each dimension of each
  // access, which we do via substituting the loop var with (var+1) into the
  // indices expr.

  std::vector<std::vector<const Expr*>> loopStrides;
  loopStrides.resize(currentScope_->accesses_.size());

  for (size_t a = 0; a < currentScope_->accesses_.size(); ++a) {
    auto& info = currentScope_->accesses_[a];

    std::vector<const Expr*> indices = info->getIndices();

    std::vector<const Expr*>& loopIndicesStride = loopStrides[a];
    loopIndicesStride.resize(indices.size());

    // index expr must depend on the loop var in some way to have a stride.
    for (size_t i = 0; i < indices.size(); i++) {
      VarFinder vf;
      if (vf.find(indices[i]).count(var) == 0) {
        loopIndicesStride[i] = new IntImm(0);
      } else {
        // If we've previously swapped the start and end of this bound, we
        // should apply the substitution to the reverse of the bounds.
        if (info->bounds()[i].swapped) {
          info->bounds()[i].end = IRSimplifier::simplify(
              Substitute(info->bounds()[i].end, {{var, v->start()}}));
          info->bounds()[i].start = IRSimplifier::simplify(Substitute(
              info->bounds()[i].start,
              {{var, new Sub(v->stop(), new IntImm(1))}}));

        } else {
          info->bounds()[i].start = IRSimplifier::simplify(
              Substitute(info->bounds()[i].start, {{var, v->start()}}));
          info->bounds()[i].end = IRSimplifier::simplify(Substitute(
              info->bounds()[i].end,
              {{var, new Sub(v->stop(), new IntImm(1))}}));
        }

        const Expr* zeroStep = indices[i];
        const Expr* oneStep =
            Substitute(indices[i], {{var, new Add(var, new IntImm(1))}});
        loopIndicesStride[i] =
            IRSimplifier::simplify(new Sub(oneStep, zeroStep));

        // If the start < end then swap the order of the bound.
        const Expr* diff = IRSimplifier::simplify(
            new Sub(info->bounds()[i].end, info->bounds()[i].start));
        if (diff->isConstant() && immediateIsNegative(diff)) {
          info->bounds()[i].swap();
        }

        // If this access uses the loop var, it depends on loads used to compute
        // the loop var.
        for (auto& extentLoad : extentsScope->accesses_) {
          info->addDependency(extentLoad);
          extentLoad->addDependent(info);
        }
      }
    }
  }

  // Now we need to update the bounds in openWrites since that is what we use to
  // merge.
  for (auto& openWritePair : currentScope_->openWrites_) {
    for (auto& pair : openWritePair.second) {
      IndexBounds& bounds = pair.first;

      // The bounds may not contain the loop var, but in that case Substitute
      // does nothing.
      for (auto& bound : bounds) {
        bound.start = IRSimplifier::simplify(
            Substitute(bound.start, {{var, v->start()}}));
        bound.end = IRSimplifier::simplify(
            Substitute(bound.end, {{var, new Sub(v->stop(), new IntImm(1))}}));

        // If the start < end then swap the order of the bound.
        const Expr* diff =
            IRSimplifier::simplify(new Sub(bound.end, bound.start));
        if (diff->isConstant() && immediateIsNegative(diff)) {
          bound.swap();
        }
      }
    }
  }

  // TODO this isn't a scalable way to determine parallelism.
  bool parallelized = v->loop_options().is_gpu_block_index() ||
      v->loop_options().is_gpu_thread_index();

  // Store buffers allocated at this scope.
  std::unordered_set<const Var*> local_intermediates;

  // Scanning from the top of the loop, we look for accesses which may depend
  // on a previous or parallel loop iteration.
  for (size_t a = 0; a < currentScope_->accesses_.size(); ++a) {
    auto& info = currentScope_->accesses_[a];
    if (info->type() == AccessType::Alloc) {
      local_intermediates.insert(info->var());
      continue;
    }

    if (!info->isRead()) {
      continue;
    }

    // Vars that don't carry outside this scope can't have loop self dependence.
    if (local_intermediates.count(info->var())) {
      continue;
    }

    // Copy the bounds so we can keep track of open bounds internally without
    // affecting the merge into the enclosing scope. The open portion of the
    // bounds may be cut into multiple independent slices.
    std::vector<IndexBounds> openBounds({info->bounds()});

    // Scan from the bottom of the loop.
    for (size_t j = currentScope_->accesses_.size() - 1; j > a; --j) {
      std::shared_ptr<AccessInfo> other = currentScope_->accesses_[j];
      if (!other->isWrite()) {
        continue;
      }

      if (info->var() != other->var()) {
        continue;
      }

      if (info->hasDependency(other)) {
        continue;
      }

      // Whether or not the accesses within the loop are dependent on other
      // iterations depends whether the loop could be parallelized, the
      // difference in their strides and their start offset.
      bool iterationsDistinct = executionSafetyCheck(
          info,
          other,
          loopStrides[a],
          loopStrides[j],
          !allowExecutionOrderAnalysis_ || parallelized);

      if (iterationsDistinct) {
        continue;
      }

      std::vector<IndexBounds> newBoundSlices;
      for (auto& b : openBounds) {
        OverlapKind overlap = overlaps(b, other->bounds());
        if (overlap == NoOverlap) {
          newBoundSlices.push_back(b);
          continue;
        }

        // It's dependent, link it to other.
        info->addDependency(other);
        other->addDependent(info);

        if (overlap == Contains) {
          continue;
        }

        // Otherwise update openBounds.
        auto slices = subtractIndicesBounds(b, other->bounds(), overlap);
        std::move(
            slices.begin(), slices.end(), std::back_inserter(newBoundSlices));
      }

      if (newBoundSlices.empty()) {
        break;
      }
      openBounds.swap(newBoundSlices);
    }
  }

  std::vector<std::shared_ptr<AccessInfo>> mergedAccesses;
  mergedAccesses.reserve(
      extentsScope->accesses_.size() + currentScope_->accesses_.size());
  std::copy(
      extentsScope->accesses_.begin(),
      extentsScope->accesses_.end(),
      std::back_inserter(mergedAccesses));
  std::copy(
      currentScope_->accesses_.begin(),
      currentScope_->accesses_.end(),
      std::back_inserter(mergedAccesses));
  scopeToAccesses_.emplace(v, mergedAccesses);

  // it's a little faster to merge without closing, and since no writes can
  // occur within the start and stop exprs we'll do that.
  mergeScope(extentsScope, extentsScope->parent, false);
  mergeScope(currentScope_, currentScope_->parent, true);
  currentScope_ = currentScope_->parent;
}

void MemDependencyChecker::visit(const Cond* v) {
  const Stmt* last = lastStmt_;
  lastStmt_ = v;

  auto enclosingScope =
      std::make_shared<Scope>(currentScope_->block, currentScope_);

  // condition is in enclosing scope.
  v->condition()->accept(this);

  Block* true_stmt = v->true_stmt();
  Block* false_stmt = v->false_stmt();

  // Create scopes so the Block visitor doesn't create and merge a new scope.
  auto trueScope = std::make_shared<Scope>(true_stmt, enclosingScope);
  auto falseScope = std::make_shared<Scope>(false_stmt, enclosingScope);

  if (true_stmt) {
    currentScope_ = trueScope;
    true_stmt->accept(this);
  }

  if (false_stmt) {
    currentScope_ = falseScope;
    false_stmt->accept(this);
  }

  // TODO(nickg): this logic isn't quite correct, if a write's Bound range is
  // present in both the true and false branches then we can close overlapping
  // accesses in the enclosing scope. Without that analysis future accesses
  // may be dependent on a write of a common range in all three of the
  // enclosing, true and false scope. This is a false positve so not too bad
  // in the short term, I think.

  // Merge both true and false branches into the parent, but don't close any
  // accesses.
  mergeScope(trueScope, enclosingScope, false);
  mergeScope(falseScope, enclosingScope, false);

  // Merge the enclosing scope into it's parent.
  mergeScope(enclosingScope, enclosingScope->parent, false);

  currentScope_ = enclosingScope;
  scopeToAccesses_.emplace(v, enclosingScope->accesses_);

  currentScope_ = enclosingScope->parent;
  lastStmt_ = last;
}

void MemDependencyChecker::visit(const IfThenElse* v) {
  // condition is in enclosing scope.
  v->condition()->accept(this);

  const Expr* true_value = v->true_value();
  const Expr* false_value = v->false_value();

  auto enclosingScope = currentScope_;

  // Create scopes to hold downstream Loads. It's safe to put nullptr for the
  // Scope's Block as it is only used by Stmts, not Exprs.
  auto trueScope = std::make_shared<Scope>(nullptr, enclosingScope);
  auto falseScope = std::make_shared<Scope>(nullptr, enclosingScope);

  if (true_value) {
    currentScope_ = trueScope;
    true_value->accept(this);
  }

  if (false_value) {
    currentScope_ = falseScope;
    false_value->accept(this);
  }

  // This doesn't have the same issue as Cond where there could be false
  // positives from the enclosing scope since there are no Exprs which are
  // writes.

  // Merge both true and false branches into the parent, but don't close any
  // accesses.
  mergeScope(trueScope, enclosingScope, false);
  mergeScope(falseScope, enclosingScope, false);

  currentScope_ = enclosingScope;
}

void MemDependencyChecker::visit(const CompareSelect* v) {
  // condition is in enclosing scope.
  v->lhs()->accept(this);
  v->rhs()->accept(this);

  const Expr* true_value = v->ret_val1();
  const Expr* false_value = v->ret_val2();

  auto enclosingScope = currentScope_;

  // Create scopes to hold downstream Loads. It's safe to put nullptr for the
  // Scope's Block as it is only used by Stmts, not Exprs.
  auto trueScope = std::make_shared<Scope>(nullptr, enclosingScope);
  auto falseScope = std::make_shared<Scope>(nullptr, enclosingScope);

  if (true_value) {
    currentScope_ = trueScope;
    true_value->accept(this);
  }

  if (false_value) {
    currentScope_ = falseScope;
    false_value->accept(this);
  }

  // This doesn't have the same issue as Cond where there could be false
  // positives from the enclosing scope since there are no Exprs which are
  // writes.

  // Merge both true and false branches into the parent, but don't close any
  // accesses.
  mergeScope(trueScope, enclosingScope, false);
  mergeScope(falseScope, enclosingScope, false);

  currentScope_ = enclosingScope;
}

// Inserts accesses for a map of buffers (ie. for inputs and outputs).
void MemDependencyChecker::insertBuffers(
    std::unordered_map<const Buf*, std::shared_ptr<AccessInfo>>& bufs,
    AccessType type) {
  for (auto& pair : bufs) {
    const Buf* b = pair.first;
    const Var* var = b->base_handle();
    IndexBounds bounds;
    for (auto* d : b->dims()) {
      bounds.push_back(
          {new IntImm(0), IRSimplifier::simplify(new Sub(d, new IntImm(1)))});
    }
    auto info =
        std::make_shared<AccessInfo>(nextAccess_++, type, nullptr, var, bounds);

    bufs[b] = info;

    auto& history = currentScope_->openWrites_[var];
    updateWriteHistory(history, info, info->id());
    currentScope_->accesses_.push_back(info);
  }
}

void MemDependencyChecker::visit(const Block* v) {
  auto prev_scope = currentScope_;

  // handle kernel inputs.
  if (prev_scope->block == nullptr) {
    insertBuffers(inputs_, AccessType::Input);
  }

  if (currentScope_->block != v) {
    currentScope_ = std::make_shared<Scope>((Block*)v, prev_scope);
  }

  for (auto* s : *v) {
    s->accept(this);
  }

  for (auto* v : currentScope_->localVars) {
    knownVarBounds_.erase(v);
  }
  for (auto& pair : currentScope_->shadowedVarBounds) {
    knownVarBounds_[pair.first] = pair.second;
  }

  scopeToAccesses_.emplace(v, currentScope_->accesses_);

  if (currentScope_ != prev_scope) {
    mergeScope(currentScope_, prev_scope, true);
    currentScope_ = prev_scope;
  }

  // handle kernel outputs.
  if (prev_scope->block == nullptr) {
    insertBuffers(outputs_, AccessType::Output);
  }
}

void MemDependencyChecker::visit(const Let* v) {
  const Stmt* last = lastStmt_;
  lastStmt_ = v;

  IRVisitor::visit(v);

  lastStmt_ = last;

  const Var* var = v->var();
  if (knownVarBounds_.count(var) != 0) {
    currentScope_->shadowedVarBounds[var] = knownVarBounds_[var];
  }

  currentScope_->localVars.insert(var);
  knownVarBounds_[var] = {v->value(), v->value()};
}

// Don't support AtomicAdd yet, it's a bit more complex since it's both a read
// and a write. It's only inserted during Cuda codegen so this should be okay.
void MemDependencyChecker::visit(const AtomicAdd* v) {
  throw std::runtime_error("MemDependencyChecker AtomicAdd unimplemented");
}

void MemDependencyChecker::visit(const Allocate* v) {
  const Stmt* last = lastStmt_;
  lastStmt_ = v;

  IRVisitor::visit(v);

  const Var* var = v->buffer_var();
  IndexBounds bounds;
  // TODO: remove the "buf_flat_size" process below and extend the buf bound
  // check to support N-d indices access and 1-d index access.
  // "Allocate" stmt is based on "Buf" which supports N-d indices access and 1-d
  // index access. Currently the write bound check in memory analysis cannot
  // identify 1-d index access for N-d bufs. Thus we flatten N-d bufs here to
  // avoid failing the bound check. But this is not the correct approach and
  // should be fixed.
  const Expr* flat_size = buf_flat_size(v->buf());
  flat_size = IRSimplifier::simplify(new Sub(flat_size, new IntImm(1)));
  bounds.push_back({new IntImm(0), flat_size});

  auto info = std::make_shared<AccessInfo>(
      nextAccess_++, AccessType::Alloc, nullptr, var, bounds);

  intermediates_[var] = info;

  auto& history = currentScope_->openWrites_[var];
  history.emplace_back(std::make_pair(info->bounds(), info));
  currentScope_->accesses_.push_back(info);

  lastStmt_ = last;
}

void MemDependencyChecker::visit(const Free* v) {
  const Stmt* last = lastStmt_;
  lastStmt_ = v;

  IRVisitor::visit(v);

  const Var* var = v->buffer_var();
  auto it = intermediates_.find(var);
  TORCH_INTERNAL_ASSERT(it != intermediates_.end());

  IndexBounds bounds = it->second->bounds();
  auto info = std::make_shared<AccessInfo>(
      nextAccess_++, AccessType::Free, nullptr, var, bounds);

  auto& history = currentScope_->openWrites_[var];
  updateWriteHistory(history, info, info->id());
  currentScope_->accesses_.push_back(info);

  lastStmt_ = last;
}

void MemDependencyChecker::updateWriteHistory(
    std::list<BoundRelationship>& writeHistory,
    const std::shared_ptr<AccessInfo>& info,
    size_t latestAccessToClose,
    bool closeOverlapped,
    bool insert) {
  bool isWrite = info->isWrite();

  for (auto it = writeHistory.begin(); it != writeHistory.end();) {
    auto& indexBounds = it->first;
    std::shared_ptr<AccessInfo> other = it->second;
    if (info->hasDependency(other)) {
      ++it;
      continue;
    }

    OverlapKind overlap = overlaps(indexBounds, info->bounds());

    if (overlap == NoOverlap) {
      ++it;
      continue;
    }

    // Only writes can close open accesses.
    if (!isWrite) {
      info->addDependency(other);
      other->addDependent(info);
      ++it;
      continue;
    }

    // If we're not closing accesses we can stop here.
    if (!closeOverlapped || other->id() > latestAccessToClose) {
      ++it;
      continue;
    }

    if (overlap == ContainedOrEqual) {
      // Total overlap is easy - the new access totally replaces the old.
      it = writeHistory.erase(it);
    } else {
      // The new write partially overlaps a previous write. We want to keep
      // both, but only track the unconvered part of the earlier write.

      // Determine the slices of the earlier bound not covered by info.
      auto newBounds =
          subtractIndicesBounds(indexBounds, info->bounds(), overlap);

      // Erase the old slice.
      it = writeHistory.erase(it);

      // Add all new slices.
      for (auto& b : newBounds) {
        it = writeHistory.insert(it, std::make_pair(b, other));
      }
      it++;
    }
  }

  if (insert && isWrite) {
    writeHistory.emplace_back(std::make_pair(info->bounds(), info));
  }
}

void MemDependencyChecker::mergeScope(
    const std::shared_ptr<Scope>& child,
    const std::shared_ptr<Scope>& parent,
    bool closeOverlapped) {
  if (child->accesses_.empty()) {
    return;
  }

  // Update dependencies, but don't add new open writes yet.
  for (auto& info : child->accesses_) {
    // Intentionally using operator[], we want it to be created if it does not
    // exist.
    auto& writeHistory = parent->openWrites_[info->var()];

    size_t latestAccessToClose = child->accesses_.front()->id();
    updateWriteHistory(
        writeHistory, info, latestAccessToClose, closeOverlapped, false);
  }

  // Copy open writes up.
  for (auto& pair : child->openWrites_) {
    const Var* var = pair.first;

    // Intentionally using operator[], we want it to be created if it does not
    // exist.
    auto& writeHistory = parent->openWrites_[var];

    for (auto& rel : pair.second) {
      writeHistory.push_back(rel);
    }
  }

  // the parent scope is responsible for holding all accesses now.
  parent->accesses_.insert(
      parent->accesses_.end(),
      std::make_move_iterator(child->accesses_.begin()),
      std::make_move_iterator(child->accesses_.end()));
}

// A visitor which applies known Bounds to symbolic expressions.
class VarBoundBinder : public IRVisitor {
 public:
  VarBoundBinder(const VarBoundMap& vars) : vars_(vars) {}

  Bound getBounds(const Expr* e) {
    min_ = e;
    max_ = e;
    e->accept(this);
    min_ = IRSimplifier::simplify(min_);
    max_ = IRSimplifier::simplify(max_);
    return {min_, max_};
  }

 private:
  void visit(const Var* v) override {
    auto it = vars_.find(v);
    if (it == vars_.end()) {
      return;
    }

    min_ = Substitute(min_, {{v, it->second.start}});
    max_ = Substitute(max_, {{v, it->second.end}});
  }

  const Expr* min_{nullptr};
  const Expr* max_{nullptr};
  const VarBoundMap& vars_;
};

std::vector<Bound> MemDependencyChecker::getIndicesBounds(
    const std::vector<const Expr*>& indices) {
  std::vector<Bound> bounds;
  bounds.reserve(indices.size());
  VarBoundBinder binder(knownVarBounds_);
  for (auto* s : indices) {
    bounds.push_back(binder.getBounds(s));
  }
  return bounds;
}

} // namespace analysis
} // namespace tensorexpr
} // namespace jit
} // namespace torch
