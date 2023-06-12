#include <torch/csrc/jit/tensorexpr/registerizer.h>

namespace torch::jit::tensorexpr {
namespace registerizer {

// AccessInfo

void AccessInfo::addStore(StorePtr store, const std::shared_ptr<Scope>& scope) {
  block_ =
      block_ ? Block::getSharedParent(block_, scope->block()) : scope->block();

  // If there is already a usage and it's this store, that means the same
  // access is present in the RHS.
  firstUsageOverlapped_ |= first_usage_ == store;
  first_usage_ = first_usage_ ? block_->getEnclosedRoot(first_usage_) : store;
  last_usage_ = store;

  store_cost_ =
      IRSimplifier::simplify(alloc<Add>(store_cost_, immLike(store_cost_, 1)));
  stores_.push_back(store);

  conditionId_ = scope->conditionId();
  hiddenAccess_.reset();
}

void AccessInfo::addLoad(
    LoadPtr load,
    const std::shared_ptr<Scope>& scope,
    StmtPtr usage) {
  block_ =
      block_ ? Block::getSharedParent(block_, scope->block()) : scope->block();
  first_usage_ = first_usage_ ? block_->getEnclosedRoot(first_usage_) : usage;
  last_usage_ = usage;

  load_cost_ =
      IRSimplifier::simplify(alloc<Add>(load_cost_, immLike(load_cost_, 1)));
  loads_.push_back(load);

  conditionId_ = scope->conditionId();
  hiddenAccess_.reset();
}

void AccessInfo::merge(const std::shared_ptr<AccessInfo>& other) {
  TORCH_INTERNAL_ASSERT(
      hash_ == other->hash(),
      buildErrorMessage(
          "Expected hashes to match in registerizer in the fuser."));
  TORCH_INTERNAL_ASSERT(
      indices_.size() == other->indices().size(),
      buildErrorMessage(
          "Expected ranks to match in registerizer in the fuser."));

  last_usage_ = other->last_usage();
  for (const auto& s : other->stores()) {
    stores_.push_back(s);
  }
  for (const auto& l : other->loads()) {
    loads_.push_back(l);
  }

  store_cost_ =
      IRSimplifier::simplify(alloc<Add>(store_cost_, other->store_cost()));
  load_cost_ =
      IRSimplifier::simplify(alloc<Add>(load_cost_, other->load_cost()));

  block_ = Block::getSharedParent(block_, other->block());
  // update first and last usage to be in the parent Block.
  first_usage_ = block_->getEnclosedRoot(first_usage_);
  last_usage_ = block_->getEnclosedRoot(last_usage_);
  hiddenAccess_.reset();
}

bool AccessInfo::overlaps(const std::shared_ptr<AccessInfo>& other) {
  // All accesses to a buf must have the same dimensionality.
  TORCH_INTERNAL_ASSERT(
      indices_.size() == other->indices().size(),
      buildErrorMessage(
          "Expected ranks to match in registerizer in the fuser."));

  auto& other_indices = other->indices();

  // They don't overlap if there is a guaranteed difference in any
  // dimension.
  bool overlap = true;
  for (size_t i = 0; i < indices_.size(); ++i) {
    ExprPtr diff = alloc<Sub>(indices_[i], other_indices[i]);
    diff = IRSimplifier::simplify(diff);

    if (diff->isConstant() && !immediateEquals(diff, 0)) {
      overlap = false;
      break;
    }
  }

  return overlap;
}

bool AccessInfo::dependsOnVar(VarPtr v) {
  VarFinder vf;
  for (const auto& i : indices_) {
    i->accept(&vf);
  }

  return vf.vars().count(v);
}

std::shared_ptr<AccessInfo> AccessInfo::cloneWithHiddenInfo(
    const std::shared_ptr<AccessInfo>& orig) {
  std::shared_ptr<AccessInfo> newInfo = std::make_shared<AccessInfo>(
      orig->hash(), orig->buf(), orig->indices(), orig->accessOrder());

  newInfo->block_ = orig->block_;
  newInfo->first_usage_ = orig->first_usage_;
  newInfo->last_usage_ = orig->last_usage_;
  newInfo->firstUsageOverlapped_ = orig->firstUsageOverlapped_;
  newInfo->store_cost_ = orig->store_cost_;
  newInfo->load_cost_ = orig->load_cost_;
  for (const auto& s : orig->stores_) {
    newInfo->stores_.push_back(s);
  }
  for (const auto& s : orig->loads_) {
    newInfo->loads_.push_back(s);
  }

  newInfo->conditionId_ = orig->conditionId_;
  newInfo->hiddenAccess_ = orig;
  return newInfo;
}

void AccessInfo::print() const {
  std::cout << "Access: " << *buf_ << "{";
  for (const auto& i : indices_) {
    std::cout << *i << " ";
  }
  std::cout << "} stores: " << stores_.size() << " (" << *store_cost_ << ") -";
  std::cout << " loads: " << loads_.size() << " (" << *load_cost_ << ")";
  if (conditionId_) {
    std::cout << " cond: " << conditionId_;
  }

  std::cout << "\n";
}

// Scope

void Scope::closeAccess(const std::shared_ptr<AccessInfo>& info) {
  closedAccesses_.push_back(info);
}

AccessHashMap& Scope::getAccessMapByBuf(BufPtr b) {
  auto it = openAccesses_.find(b);
  if (it == openAccesses_.end()) {
    // create and return
    return openAccesses_[b];
  }

  return it->second;
}

void Scope::filterClosed() {
  closedAccesses_.erase(
      std::remove_if(
          closedAccesses_.begin(),
          closedAccesses_.end(),
          [](auto info) {
            return info->store_cost()->isConstant() &&
                immediateAs<int>(info->store_cost()) <= 1 &&
                info->load_cost()->isConstant() &&
                immediateAs<int>(info->load_cost()) <= 1;
          }),
      closedAccesses_.end());
}

// RegisterizerAnalysis

void RegisterizerAnalysis::closeAccessIntoScope(
    const std::shared_ptr<AccessInfo>& info,
    const std::shared_ptr<Scope>& scope) {
  if (exprConditionals_.count(info->conditionId()) != 0) {
    return;
  }

  if (info->hiddenAccess()) {
    closeAccessIntoScope(info->hiddenAccess(), scope);
    return;
  }
  scope->closeAccess(info);
}

void RegisterizerAnalysis::visit(ForPtr v) {
  if (v->loop_options().is_gpu_block_index() ||
      v->loop_options().is_gpu_thread_index()) {
    throw malformed_input(
        "Registerization must occur after parallelism flattening");
  }

  auto parent = currentScope_;
  currentScope_ = std::make_shared<Scope>(v->body(), parent);

  currentScope_->addLocalVar(v->var());

  stmtStack_.push_front(v);
  v->body()->accept(this);
  stmtStack_.pop_front();

  ExprPtr loopExtent =
      IRSimplifier::simplify(alloc<Sub>(v->stop(), v->start()));

  // now we need to see which accesses we can hoist out of the for loop, their
  // costs should be multiplied by the loop extent.
  for (auto& pair : currentScope_->openAccesses()) {
    if (pair.second.empty()) {
      continue;
    }

    auto& childAccesses = pair.second;

    for (auto it = childAccesses.begin(); it != childAccesses.end();) {
      std::shared_ptr<AccessInfo>& candidate = it->second;

      // If the access is open, but conditional, then we have a problem. It's
      // possible that an access at a higher scope could "unhide" the
      // conditional access, in which case we need to hoist. If there is no
      // access to this element at a higher scope then we cannot safely hoist.
      // We cannot know at this level whether that will or wont occur.
      //
      // The solution we take here is to split the space-time continuum, and
      // keep both versions of the access handy. If the hoisted access is not
      // used above, we'll fall back to using the hidden, conditional
      // AccessInfo - if it is, we'll delete the copy.
      if (candidate->conditionId() != 0) {
        candidate = AccessInfo::cloneWithHiddenInfo(candidate);
      }

      bool closed = false;
      // If this access depends on a locally scoped variable, it cannot be
      // hosted out of the loop.
      for (const auto& v : currentScope_->localVars()) {
        if (candidate->dependsOnVar(v)) {
          closeAccessIntoScope(candidate, currentScope_);
          closed = true;
          break;
        }
      }
      if (closed) {
        it = childAccesses.erase(it);
        continue;
      }

      // hoist!
      // By hoisting we pull the reads and writes out of the loop, and so the
      // benefit of registerizing this access is multiplied by the loop extent.
      candidate->setEnclosingBlock(parent->block());
      candidate->hoistCosts(loopExtent);

      // in the parent block, this loop Stmt is the insertion point for the
      // initializer and finalizer.
      candidate->setUsageMarks(v, v);

      ++it;
    }
  }

  // If an access is closed within a loop then it cannot be merged into an
  // existing open access, but will still close that existing access. This is
  // somewhat different from the regular merge so we need to handle closed
  // accesses first.
  mergeHiddenScope(true);

  // having hoisted, now we can merge normally.
  mergeCurrentScopeIntoParent();
};

void RegisterizerAnalysis::visit(CondPtr v) {
  ExprPtr condition = v->condition();
  BlockPtr true_stmt = v->true_stmt();
  BlockPtr false_stmt = v->false_stmt();

  stmtStack_.push_front(v);

  // condition is in the enclosing scope.
  condition->accept(this);

  auto prev_scope = currentScope_;
  auto true_scope =
      std::make_shared<Scope>(true_stmt, prev_scope, ++conditionId_);
  auto false_scope =
      std::make_shared<Scope>(false_stmt, prev_scope, ++conditionId_);

  if (true_stmt) {
    currentScope_ = true_scope;
    true_stmt->accept(this);
    mergeHiddenScope(true);
    mergeCurrentScopeIntoParent();
  }
  if (false_stmt) {
    currentScope_ = false_scope;
    false_stmt->accept(this);
    mergeHiddenScope(true);
    mergeCurrentScopeIntoParent();
  }

  // TODO: even though both scopes are conditional, we can merge accesses if
  // they totally overlap in both branches, since we can guarantee one
  // definition will be hit. We might need a 3-way merge? Not as simple as
  // merging the true and false scopes together first.

  stmtStack_.pop_front();
}

// IfThenElses are just like Conds except they are not Stmts, which means no
// registerization can occur internally. However, the first reference to an
// access can occur within one if its visible outside the condition.
void RegisterizerAnalysis::visit(IfThenElsePtr v) {
  ExprPtr condition = v->condition();
  ExprPtr true_value = v->true_value();
  ExprPtr false_value = v->false_value();

  // condition is in enclosing scope.
  condition->accept(this);

  auto prev_scope = currentScope_;
  auto true_scope =
      std::make_shared<Scope>(prev_scope->block(), prev_scope, ++conditionId_);
  auto false_scope =
      std::make_shared<Scope>(prev_scope->block(), prev_scope, ++conditionId_);

  // We store IfThenElse scopes in a global map, which we use to prevent closing
  // any access that would require inserting statements in the values, which
  // cannot enclose Stmts.
  exprConditionals_.insert(true_scope->conditionId());
  exprConditionals_.insert(false_scope->conditionId());

  if (true_value) {
    currentScope_ = true_scope;
    true_value->accept(this);
    mergeHiddenScope(false);
    mergeCurrentScopeIntoParent();
  }

  if (false_value) {
    currentScope_ = false_scope;
    false_value->accept(this);
    mergeHiddenScope(false);
    mergeCurrentScopeIntoParent();
  }
}

void RegisterizerAnalysis::visit(LetPtr v) {
  currentScope_->addLocalVar(v->var());

  stmtStack_.push_front(v);
  v->value()->accept(this);
  stmtStack_.pop_front();
}

void RegisterizerAnalysis::visit(BlockPtr v) {
  auto prev_scope = currentScope_;
  if (currentScope_->block() != v) {
    currentScope_ = std::make_shared<Scope>(v, prev_scope);
  }

  stmtStack_.push_front(v);

  for (const auto& s : *v) {
    s->accept(this);
    if (currentScope_->block() != v) {
      // merge the inner block's accesses into this Block's accesses.
      mergeCurrentScopeIntoParent();
    }
  }

  stmtStack_.pop_front();

  if (prev_scope->block() == nullptr) {
    // close any open candidates.
    for (auto& p1 : currentScope_->openAccesses()) {
      for (auto& p2 : p1.second) {
        closeAccessIntoScope(p2.second, currentScope_);
      }
    }
  }
}

void RegisterizerAnalysis::visit(StorePtr v) {
  stmtStack_.push_front(v);
  v->value()->accept(this);
  stmtStack_.pop_front();

  if (v->indices().empty()) {
    // already a scalar.
    return;
  }

  // hash the Store:
  SimplifierHashType accessHash = hasher_.hash(v->buf());
  for (const auto& i : v->indices()) {
    accessHash = hasher_.hash_combine(accessHash, i);
  }

  auto& bufAccesses = currentScope_->getAccessMapByBuf(v->buf());
  auto candidateIt = bufAccesses.find(accessHash);

  // If an identical access already exists, add this Store to it.
  if (candidateIt != bufAccesses.end()) {
    candidateIt->second->addStore(v, currentScope_);
    return;
  }

  // Otherwise make a new AccessInfo and add this store.
  auto info = std::make_shared<AccessInfo>(
      accessHash, v->buf(), v->indices(), accessOrder_++);
  info->addStore(v, currentScope_);

  // This new access may overlap an existing open access, in which case we need
  // to close the older of the two.
  bool alreadyOverlapped = false;
  for (auto it = bufAccesses.begin(); it != bufAccesses.end();) {
    auto other = it->second;
    if (info->overlaps(other)) {
      if (other->last_usage() == v) {
        // we are already overlapped by an access in the RHS.
        alreadyOverlapped = true;
      }
      closeAccessIntoScope(other, currentScope_);
      it = bufAccesses.erase(it);
    } else {
      ++it;
    }
  }

  if (alreadyOverlapped) {
    closeAccessIntoScope(info, currentScope_);
  } else {
    bufAccesses.emplace(accessHash, info);
  }
}

void RegisterizerAnalysis::visit(LoadPtr v) {
  if (v->indices().empty()) {
    // already a scalar.
    return;
  }
  // hash the Load:
  SimplifierHashType accessHash = hasher_.hash(v->buf());
  for (const auto& i : v->indices()) {
    accessHash = hasher_.hash_combine(accessHash, i);
  }

  auto& bufAccesses = currentScope_->getAccessMapByBuf(v->buf());
  auto candidateIt = bufAccesses.find(accessHash);
  if (candidateIt != bufAccesses.end()) {
    // found the right access, can just insert.
    candidateIt->second->addLoad(v, currentScope_, stmtStack_.front());
    return;
  }

  std::shared_ptr<AccessInfo> info = std::make_shared<AccessInfo>(
      accessHash, v->buf(), v->indices(), accessOrder_++);
  info->addLoad(v, currentScope_, stmtStack_.front());

  bool alreadyOverlapped = false;
  // This new access may overlap an existing open access, in which case we need
  // to finalize the older of the two.
  for (auto it = bufAccesses.begin(); it != bufAccesses.end();) {
    auto other = it->second;
    if (info->overlaps(other)) {
      if (info->last_usage() == other->last_usage()) {
        // if these two accesses are from the same Stmt, they already overlap
        // each other.
        alreadyOverlapped = true;
      }
      closeAccessIntoScope(other, currentScope_);
      it = bufAccesses.erase(it);
    } else {
      ++it;
    }
  }

  if (alreadyOverlapped) {
    closeAccessIntoScope(info, currentScope_);
  } else {
    bufAccesses.emplace(accessHash, info);
  }
}

// Loop and Conditional scopes are different in that it may or may not be
// possible to hoist the initializer of a scalar variable outside the block
// depending on if we can tell that the Buffer access is valid outside. This is
// tricky because the access that demonstrates this may be later in the tree and
// we haven't encountered it yet.
// The allowClosed flag indicates whether we want to keep the closed accesses
// (For and Cond), or not (IfThenElse).
void RegisterizerAnalysis::mergeHiddenScope(bool allowClosed) {
  // The rule is that if any access is closed within the conditional block, any
  // accesses which overlap it must also be closed - since their initializer
  // cannot be hoisted out of the block.
  std::list<std::shared_ptr<AccessInfo>> newClosed;
  for (auto& info : currentScope_->closedAccesses()) {
    auto& candidates = currentScope_->getAccessMapByBuf(info->buf());
    for (auto it = candidates.begin(); it != candidates.end();) {
      std::shared_ptr<AccessInfo> candidate = it->second;

      if (info->hash() == candidate->hash() || info->overlaps(candidate)) {
        newClosed.push_back(candidate);
        it = candidates.erase(it);
      } else {
        ++it;
      }
    }
  }

  if (allowClosed) {
    for (auto& info : newClosed) {
      closeAccessIntoScope(info, currentScope_);
    }
  } else {
    currentScope_->closedAccesses().clear();
  }
}

// Merge currentScope_ into it's parent, and make parent the new currentScope_.
void RegisterizerAnalysis::mergeCurrentScopeIntoParent() {
  auto parent = currentScope_->parent();

  // copy across current closed accesses, merging / closing as necessary
  for (auto& candidate : currentScope_->closedAccesses()) {
    auto& parentAccesses = parent->getAccessMapByBuf(candidate->buf());

    auto parentIt = parentAccesses.find(candidate->hash());
    if (parentIt != parentAccesses.end()) {
      std::shared_ptr<AccessInfo> pCandidate = parentIt->second;

      // if the access is closed inside a condition, it can only be merged if
      // the parent is in the same condition.
      if (candidate->conditionId() &&
          pCandidate->conditionId() != candidate->conditionId()) {
        // the parent's access must be closed.
        closeAccessIntoScope(pCandidate, parent);
        parentAccesses.erase(parentIt);

        // the childs access inserted into the parent scope.
        closeAccessIntoScope(candidate, parent);
        continue;
      }

      // merge totally overlapping accesses.
      parentIt->second->merge(candidate);
      closeAccessIntoScope(parentIt->second, parent);
      parentAccesses.erase(parentIt);
      continue;
    }

    // we didn't find a perfect match, but we need to check all open accesses of
    // this buf for partial overlap.
    for (auto it = parentAccesses.begin(); it != parentAccesses.end();) {
      std::shared_ptr<AccessInfo> pCandidate = it->second;
      // Partial overlap of parent access: close parent access.
      if (candidate->overlaps(pCandidate)) {
        closeAccessIntoScope(pCandidate, parent);
        it = parentAccesses.erase(it);
        continue;
      }
      ++it;
    }

    // Insert the childs closed access into the parent scope.
    closeAccessIntoScope(candidate, parent);
  }

  // copy across current open accesses, merging as necessary.
  // for each Buf with an open access:
  for (auto& pair : currentScope_->openAccesses()) {
    BufPtr buf = pair.first;
    if (pair.second.empty()) {
      continue;
    }

    auto& parentAccesses = parent->getAccessMapByBuf(buf);

    // for each open access in the child scope for this Buf:
    for (auto& hpair : pair.second) {
      bool handled{false};
      std::shared_ptr<AccessInfo> candidate = hpair.second;

      for (auto it = parentAccesses.begin(); it != parentAccesses.end();) {
        std::shared_ptr<AccessInfo> pCandidate = it->second;

        // If it completely overlaps then merge.
        if (candidate->hash() == pCandidate->hash()) {
          // if both accesses are found in conditional blocks, they cannot be
          // merged, but the earlier must be closed.
          if (pCandidate->conditionId() != parent->conditionId() &&
              pCandidate->conditionId() != candidate->conditionId()) {
            closeAccessIntoScope(pCandidate, parent);
            it = parentAccesses.erase(it);
            continue;
          }
          pCandidate->merge(candidate);
          handled = true;
          ++it;
          continue;
        }

        // It can overlap an access in the parent: close the parent access.
        // The child access may still be open.
        if (candidate->overlaps(pCandidate)) {
          closeAccessIntoScope(pCandidate, parent);
          it = parentAccesses.erase(it);
          continue;
        }

        ++it;
      }

      // If this access depends on a locally scoped variable, it cannot be
      // lifted out of the loop.
      for (const auto& v : currentScope_->localVars()) {
        if (candidate->dependsOnVar(v)) {
          closeAccessIntoScope(candidate, parent);
          handled = true;
          break;
        }
      }

      if (!handled) {
        // If the inner scope was not conditional, but the outer scope is: all
        // current accesses are now conditional in the parent scope.
        if (candidate->conditionId() == 0) {
          candidate->setConditionId(parent->conditionId());
        }
        parentAccesses[candidate->hash()] = candidate;
      }
    }
  }

  currentScope_ = parent;
}

std::vector<std::shared_ptr<AccessInfo>> RegisterizerAnalysis::getCandidates() {
  currentScope_->filterClosed();
  std::sort(
      currentScope_->closedAccesses().begin(),
      currentScope_->closedAccesses().end(),
      [](auto i1, auto i2) { return i1->accessOrder() < i2->accessOrder(); });
  return currentScope_->closedAccesses();
}

// RegisterizerReplacer

ExprPtr RegisterizerReplacer::mutate(LoadPtr v) {
  auto it = loadToAccess_.find(v);
  if (it == loadToAccess_.end()) {
    // This access cannot be registerized.
    return v;
  }

  auto& info = it->second;

  return info->replacement().var;
}

StmtPtr RegisterizerReplacer::mutate(StorePtr v) {
  if (eliminatedIntializers_.count(v) != 0) {
    // This store is the initializer for a scalar var that is already inserted.
    return nullptr;
  }

  auto it = storeToAccess_.find(v);
  if (it == storeToAccess_.end()) {
    // This access cannot be registerized.
    return IRMutator::mutate(v);
  }

  auto& info = it->second;

  ExprPtr new_val = v->value()->accept_mutator(this);

  v->set_value(new_val);
  v->set_buf(info->replacement().var_wrapper);
  v->set_indices({});
  return v;
}

StmtPtr RegisterizerReplacer::mutate(BlockPtr v) {
  auto& scope = parentToAccesses_[v];

  std::vector<StmtPtr> stmts;
  for (const StmtPtr& stmt : v->stmts()) {
    {
      // Insert the initializer for any Scalars scoped to this block.
      auto it = scope.initializerPoints_.find(stmt);
      if (it != scope.initializerPoints_.end()) {
        for (auto& info : it->second) {
          StmtPtr initializer =
              info->replacement().initializer->accept_mutator(this);
          stmts.push_back(initializer);
        }
        scope.initializerPoints_.erase(it);
      }
    }

    StmtPtr stmt_new = stmt->accept_mutator(this);
    if (stmt_new) {
      if (stmt_new->get_parent()) {
        stmt_new = Stmt::clone(stmt_new);
      }
      stmts.push_back(stmt_new);
    }

    {
      // Insert the finalizer for any Scalars scoped to this block.
      auto it = scope.finalizePoints_.find(stmt);
      if (it != scope.finalizePoints_.end()) {
        for (auto& info : it->second) {
          StorePtr finalizer = alloc<Store>(
              info->buf(), info->indices(), info->replacement().var);
          stmts.push_back(finalizer);
        }
        scope.finalizePoints_.erase(it);
      }
    }
  }

  return alloc<Block>(stmts);
}

void RegisterizerReplacer::buildReplacements() {
  // Traverse the list of replacements, creating vars and updating our local
  // maps.
  for (auto& info : infoSet_) {
    VarPtr v = alloc<Var>(
        info->buf()->name_hint() + "_" +
            c10::to_string(getBufferAccessCount(info->buf())),
        info->buf()->dtype());

    info->replacement().var = v;

    // we need to wrap the Var in a Buf so we can Load or Store it.
    info->replacement().var_wrapper =
        alloc<Buf>(v, std::vector<ExprPtr>({}), info->buf()->dtype());

    bool first = true;
    for (const auto& s : info->stores()) {
      if (first && info->first_usage() == s && !info->firstUsageOverlapped()) {
        info->replacement().initializer = alloc<Let>(v, s->value());
        eliminatedIntializers_.insert(s);
      } else {
        storeToAccess_[s] = info;
      }

      first = false;
    }

    for (const auto& s : info->loads()) {
      loadToAccess_[s] = info;
    }

    auto& scope = parentToAccesses_[info->block()];
    scope.initializerPoints_[info->first_usage()].push_back(info);

    // Only finalize if the scalar is written.
    if (!info->stores().empty()) {
      // push front to finalize in reverse order of encounter.
      scope.finalizePoints_[info->last_usage()].push_front(info);
    }

    // create a default initializer by reading the access.
    if (info->replacement().initializer == nullptr) {
      info->replacement().initializer = alloc<Let>(
          v, alloc<Load>(info->buf()->dtype(), info->buf(), info->indices()));
    }
  }
}

} // namespace registerizer

// Apply scalar replacement to all accesses in s.
StmtPtr registerize(StmtPtr s) {
  s = IRSimplifier::simplify(s);

  // The outermost node must be a Block so we have somewhere to put outer scope
  // scalars.
  if (!to<Block>(s)) {
    s = alloc<Block>(std::vector<StmtPtr>({s}));
  }
  registerizer::RegisterizerAnalysis analysis;
  s->accept(&analysis);
  auto candidates = analysis.getCandidates();

  registerizer::RegisterizerReplacer replacer(candidates);
  s = s->accept_mutator(&replacer);
  return s;
}

} // namespace torch::jit::tensorexpr
