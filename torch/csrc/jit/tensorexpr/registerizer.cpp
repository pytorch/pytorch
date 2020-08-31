#include <torch/csrc/jit/tensorexpr/registerizer.h>

namespace torch {
namespace jit {
namespace tensorexpr {

void RegisterizerAnalysis::visit(const For* v) {
  if (v->loop_options().is_gpu_block_index() ||
      v->loop_options().is_gpu_thread_index()) {
    throw malformed_input(
        "Registerization must occur after parallelism flattening");
  }

  const Expr* old_loopCost = loopCost_;
  loopCost_ = IRSimplifier::simplify(
      new Mul(loopCost_, new Sub(v->stop(), v->start())));
  stmtStack_.push_front(v);
  v->body()->accept(this);
  stmtStack_.pop_front();

  loopCost_ = old_loopCost;
};

void RegisterizerAnalysis::visit(const Block* v) {
  const Block* last = enclosingBlock_;
  enclosingBlock_ = v;
  stmtStack_.push_front(v);
  costByBlock_[v] = loopCost_;
  IRVisitor::visit(v);
  stmtStack_.pop_front();
  enclosingBlock_ = last;
}

void RegisterizerAnalysis::visit(const Store* v) {
  // path into value first.
  stmtStack_.push_front(v);
  v->value()->accept(this);
  stmtStack_.pop_front();

  if (v->indices().empty()) {
    // already a scalar.
    return;
  }

  SimplifierHashType accessHash = hasher_.hash(v->buf());
  for (auto* i : v->indices()) {
    accessHash = hasher_.hash_combine(accessHash, i);
  }
  accessHash = hasher_.hash_combine(accessHash, v->mask());

  std::shared_ptr<AccessInfo> info;
  auto candidateIt = candidates_.find(accessHash);
  if (candidateIt != candidates_.end()) {
    info = candidateIt->second;
  } else {
    info = std::make_shared<AccessInfo>(v->buf(), v->indices());
    candidates_[accessHash] = info;
    encounterOrder_.push_back(info);
  }
  info->addStore(v, enclosingBlock_, loopCost_);
}

void RegisterizerAnalysis::visit(const Load* v) {
  if (v->indices().empty()) {
    // already a scalar.
    return;
  }

  SimplifierHashType accessHash = hasher_.hash(v->buf());
  for (auto* i : v->indices()) {
    accessHash = hasher_.hash_combine(accessHash, i);
  }
  accessHash = hasher_.hash_combine(accessHash, v->mask());

  std::shared_ptr<AccessInfo> info;
  auto candidateIt = candidates_.find(accessHash);
  if (candidateIt != candidates_.end()) {
    info = candidateIt->second;
  } else {
    info = std::make_shared<AccessInfo>(v->buf(), v->indices());
    candidates_[accessHash] = info;
    encounterOrder_.push_back(info);
  }

  info->addLoad(v, enclosingBlock_, loopCost_, stmtStack_.front());
}

std::vector<std::shared_ptr<AccessInfo>> RegisterizerAnalysis::getCandidates() {
  std::vector<std::shared_ptr<AccessInfo>> ret;

  // Group accesses by the base buffer they refer to, so it's easier to
  // determine which accesses may overlap.
  std::unordered_map<const Buf*, std::vector<std::shared_ptr<AccessInfo>>>
      access_by_buf;
  for (const auto& pair : candidates_) {
    std::shared_ptr<AccessInfo> info = pair.second;

    // We can "hoist" an access up the syntax tree if it's indices do not
    // depend on any loop vars.
    VarFinder vf;
    for (auto* i : info->indices) {
      i->accept(&vf);
    }

    const Stmt* ancestor = info->parent;
    const Stmt* target = nullptr;
    while (ancestor) {
      if (const For* f = dynamic_cast<const For*>(ancestor)) {
        if (vf.vars().count(f->var()) != 0) {
          break;
        }
        target = f->get_parent();
      }

      ancestor = ancestor->get_parent();
    }

    if (info->parent != target) {
      if (const Block* new_parent = dynamic_cast<const Block*>(target)) {
        info->parent = new_parent;
      }
    }

    // Now that analysis is complete we must normalize the costs by the
    // parent Block we plan to insert the scalar var into.
    info->store_cost = IRSimplifier::simplify(
        new Div(info->store_cost, costByBlock_[info->parent]));

    if (!info->loads.empty()) {
      info->load_cost = IRSimplifier::simplify(
          new Div(info->load_cost, costByBlock_[info->parent]));
    }

    access_by_buf[info->buf].push_back(info);
  }

  // For each buffer, for each access, determine if another access to the
  // buffer could possibly write to the same region.
  for (const auto& pair : access_by_buf) {
    const Buf* buf = pair.first;
    const std::vector<std::shared_ptr<AccessInfo>>& accesses = pair.second;
    for (const auto& info : accesses) {
      // Filter out low cost accesses.
      if (info->store_cost->isConstant() &&
          immediateAs<int>(info->store_cost) <= 1 &&
          info->load_cost->isConstant() &&
          immediateAs<int>(info->load_cost) <= 1) {
        info->invalid = true;
        continue;
      }

      // TODO: this is n^2 by the number of accesses to a single buffer
      // program wide, may be an issue in large programs.
      for (const auto& i2 : accesses) {
        if (info == i2) {
          continue;
        }

        // All accesses to a buf must have the same dimensionality.
        assert(info->indices.size() == i2->indices.size());

        // They don't overlap if there is a guaranteed difference in any
        // dimension.
        bool overlap = true;
        for (size_t i = 0; i < info->indices.size(); ++i) {
          const Expr* diff = new Sub(info->indices[i], i2->indices[i]);
          diff = IRSimplifier::simplify(diff);
          if (diff->isConstant() && !immediateEquals(diff, 0)) {
            overlap = false;
            break;
          }
        }

        if (overlap) {
          info->invalid = true;
          break;
        }
      }
    }
  }

  // Return valid access candidates in the order they were first seen.
  for (const auto& info : encounterOrder_) {
    if (!info->invalid) {
      ret.push_back(info);
    }
  }

  return ret;
}

const Expr* RegisterizerReplacer::mutate(const Load* v) {
  if (v->buf() != info_->buf) {
    return IRMutator::mutate(v);
  }

  initializerReady_ = false;

  // sanity check indices for the same buf must have the same dimensionality.
  assert(v->indices().size() == info_->indices.size());
  for (size_t i = 0; i < info_->indices.size(); ++i) {
    if (!exprEquals(v->indices()[i], info_->indices[i])) {
      return IRMutator::mutate(v);
    }
  }

  return var_;
}

Stmt* RegisterizerReplacer::mutate(const Store* v) {
  if (v->buf() != info_->buf) {
    return IRMutator::mutate(v);
  }

  if (initializerReady_ && info_->parent == v->get_parent()) {
    initializer_ = v;
    initializerReady_ = false;
    // This is the easiest way to return an empty statement;
    return new Block({});
  }

  initializerReady_ = false;

  // sanity check indices for the same buf must have the same dimensionality.
  assert(v->indices().size() == info_->indices.size());
  for (size_t i = 0; i < info_->indices.size(); ++i) {
    if (!exprEquals(v->indices()[i], info_->indices[i])) {
      return IRMutator::mutate(v);
    }
  }
  const Expr* new_val = v->value()->accept_mutator(this);

  Store* s = new Store(var_wrapper_, {}, new_val, v->mask());
  return s;
}

// Finds the Stmt in parent which contains stmt.
const Stmt* RegisterizerReplacer::findInsertionPoint(
    const Stmt* stmt,
    const Block* parent) {
  while (stmt) {
    if (stmt->get_parent() == parent) {
      return stmt;
    }
    stmt = stmt->get_parent();
  }
  return nullptr;
}

Stmt* RegisterizerReplacer::mutate(const Block* v) {
  // We need to mutate this block in place, rather than clone - since other
  // AccessInfo objects may hold a pointer to it.
  Block* v1 = const_cast<Block*>(v); // NOLINT
  assert(v1);

  Stmt* first_changed{nullptr};
  Stmt* last_changed{nullptr};
  std::list<Stmt*> stmts = v1->stmts();
  for (Stmt* stmt : stmts) {
    dirty_ = false;
    Stmt* stmt_new = stmt->accept_mutator(this);
    if (dirty_) {
      first_changed = first_changed ? first_changed : stmt_new;
      last_changed = stmt_new;
    }

    if (stmt_new == stmt) {
      continue;
    }
    v1->replace_stmt(stmt, stmt_new);
    first_changed = first_changed ? first_changed : stmt_new;
    last_changed = stmt_new;
  }

  dirty_ = first_changed != nullptr;

  if (v != info_->parent) {
    return v1;
  }

  Stmt* let;
  // If we didn't find an initial store: intialize with the original buffer.
  if (!initializer_) {
    let = new Let(
        var_,
        new Load(
            info_->buf->dtype(), info_->buf, info_->indices, new IntImm(1)));
  } else {
    let = new Let(var_, initializer_->value());
  }
  v1->insert_stmt_before(let, first_changed);

  // If it was written to the buffer, make sure we write it out.
  if (info_->stores.size() > 0) {
    v1->insert_stmt_after(
        new Store(info_->buf, info_->indices, var_, new IntImm(1)),
        last_changed);
  }
  return v1;
}

// Apply scalar replacement to all accesses in s.
Stmt* registerize(Stmt* s) {
  RegisterizerAnalysis analysis;
  s->accept(&analysis);
  auto candidates = analysis.getCandidates();
  for (const auto& info : candidates) {
    RegisterizerReplacer replacer(info);
    s = s->accept_mutator(&replacer);
  }
  return s;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
