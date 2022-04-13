#pragma once

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {
class HasRand : public IRVisitor {
 public:
  HasRand(StmtPtr stmt) : stmt_(stmt) {
    stmt_->accept(this);
  }

  bool has_rand() const {
    return has_rand_;
  }

 private:
  void visit(IntrinsicsPtr v) override {
    if (v->op_type() == IntrinsicsOp::kRand) {
      has_rand_ = true;
    } else {
      IRVisitor::visit(v);
    }
  }
  StmtPtr stmt_;
  bool has_rand_ = false;
};

template <typename Op>
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class NodeFinder : public IRVisitor {
 public:
  void visit(NodePtr<Op> v) override {
    nodes.push_back((NodePtr<Op>)v);
    IRVisitor::visit(v);
  }

  static std::vector<NodePtr<Op>> find(StmtPtr s) {
    NodeFinder<Op> nf;
    s->accept(&nf);
    return nf.nodes;
  }

  static std::vector<NodePtr<Op>> find(ExprPtr e) {
    NodeFinder<Op> nf;
    e->accept(&nf);
    return nf.nodes;
  }

  std::vector<NodePtr<Op>> nodes;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class VarFinder : public IRVisitor {
 public:
  void visit(VarPtr v) override {
    vars_.insert(v);
    IRVisitor::visit(v);
  }

  static std::unordered_set<VarPtr> find(StmtPtr s) {
    VarFinder nf;
    s->accept(&nf);
    return nf.vars();
  }

  static std::unordered_set<VarPtr> find(ExprPtr e) {
    VarFinder nf;
    e->accept(&nf);
    return nf.vars();
  }

  const std::unordered_set<VarPtr>& vars() {
    return vars_;
  }

 private:
  std::unordered_set<VarPtr> vars_;
};

class BufFinder : public IRVisitor {
 public:
  void visit(BufPtr v) override {
    bufs_.insert(v);
    IRVisitor::visit(v);
  }

  static std::unordered_set<BufPtr> find(StmtPtr s) {
    BufFinder nf;
    s->accept(&nf);
    return nf.bufs();
  }

  static std::unordered_set<BufPtr> find(ExprPtr e) {
    BufFinder nf;
    e->accept(&nf);
    return nf.bufs();
  }

  const std::unordered_set<BufPtr>& bufs() {
    return bufs_;
  }

 private:
  std::unordered_set<BufPtr> bufs_;
};

// Finds all kinds of write operations to the provided Buf.
class WritesToBuf : public IRVisitor {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  WritesToBuf(BufPtr target) : target_(target) {}

  std::vector<StmtPtr> writes() {
    return writes_;
  }

  static std::vector<StmtPtr> find(StmtPtr s, BufPtr b) {
    WritesToBuf finder(b);
    s->accept(&finder);
    return finder.writes();
  }

 private:
  void visit(StorePtr v) override {
    if (v->buf() == target_) {
      writes_.push_back(v);
    }
  }

  void visit(AtomicAddPtr v) override {
    if (v->buf() == target_) {
      writes_.push_back(v);
    }
  }

  BufPtr target_;
  std::vector<StmtPtr> writes_;
};

class StmtsReadingBuf : public IRVisitor {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  StmtsReadingBuf(BufPtr target) : target_(target) {}

  std::vector<StmtPtr> reads() {
    return reads_;
  }

  static std::vector<StmtPtr> find(StmtPtr s, BufPtr b) {
    StmtsReadingBuf finder(b);
    s->accept(&finder);
    return finder.reads();
  }

 private:
  bool readsBuffer(StmtPtr s) {
    auto loads = NodeFinder<Load>::find(s);
    for (auto l : loads) {
      if (l->buf() == target_) {
        return true;
      }
    }
    return false;
  }

  void visit(StorePtr v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  void visit(LetPtr v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  void visit(CondPtr v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  void visit(AtomicAddPtr v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  BufPtr target_;
  std::vector<StmtPtr> reads_;
};

class ExternalAllocBufFinder : public IRVisitor {
 public:
  void visit(ExternalCallWithAllocPtr v) override {
    const auto& bufs_out = v->buf_out_args();
    bufs_.insert(bufs_out.begin(), bufs_out.end());
    IRVisitor::visit(v);
  }

  static std::unordered_set<BufPtr> find(StmtPtr s) {
    ExternalAllocBufFinder f;
    s->accept(&f);
    return f.bufs();
  }

  static std::unordered_set<BufPtr> find(ExprPtr e) {
    ExternalAllocBufFinder f;
    e->accept(&f);
    return f.bufs();
  }

  const std::unordered_set<BufPtr>& bufs() {
    return bufs_;
  }

 private:
  std::unordered_set<BufPtr> bufs_;
};

// Traverses the IR to determine if a particular Var is modified within it.
class ModifiesVarChecker : public IRVisitor {
 public:
  ModifiesVarChecker(VarPtr v) : var_(v) {}

  static bool check(StmtPtr s, VarPtr v) {
    ModifiesVarChecker checker(v);
    s->accept(&checker);
    return checker.found();
  }

  bool found() {
    return found_;
  }

 private:
  void visit(StorePtr v) override {
    if (v->buf()->base_handle() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(AtomicAddPtr v) override {
    if (v->buf()->base_handle() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(LetPtr v) override {
    if (v->var() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(ForPtr v) override {
    if (v->var() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  VarPtr var_;
  bool found_{false};
};

// Traverse the Block stmt to identify the live range of the specified buf. The
// live range, indicated by a pair of integers, specifies the first and last
// stmt in block stmts that access to the buf.
class BufLiveRange : public IRVisitor {
 public:
  BufLiveRange(BufPtr b) : buf_(b) {}

  static std::tuple<int32_t, int32_t> liveRange(StmtPtr s, BufPtr b) {
    BlockPtr block = to<Block>(s);
    // We Only analze buffer live ranges for block stmts.
    if (!block) {
      return std::make_tuple(0, 0);
    }

    BufLiveRange analyzer(b);
    block->accept(&analyzer);
    return analyzer.getLiveRange();
  }

 private:
  std::tuple<int32_t, int32_t> getLiveRange() {
    return std::make_tuple(begin_, end_);
  }

  bool hasBufReads(StmtPtr s) {
    auto loads1 = NodeFinder<Load>::find(s);
    for (auto l : loads1) {
      if (l->buf() == buf_) {
        return true;
      }
    }
    auto loads2 = NodeFinder<ExternalCall>::find(s);
    for (auto l : loads2) {
      for (auto lb : l->buf_args()) {
        if (lb == buf_) {
          return true;
        }
      }
    }
    auto loads3 = NodeFinder<ExternalCallWithAlloc>::find(s);
    for (auto l : loads3) {
      for (auto lb : l->buf_args()) {
        if (lb == buf_) {
          return true;
        }
      }
    }
    return false;
  }

  bool hasBufWrites(StmtPtr s) {
    auto writes1 = NodeFinder<Store>::find(s);
    for (auto w : writes1) {
      if (w->buf() == buf_) {
        return true;
      }
    }
    auto writes2 = NodeFinder<ExternalCall>::find(s);
    for (auto w : writes2) {
      if (w->buf() == buf_) {
        return true;
      }
    }
    auto writes3 = NodeFinder<ExternalCallWithAlloc>::find(s);
    for (auto w : writes3) {
      for (auto wb : w->buf_out_args()) {
        if (wb == buf_) {
          return true;
        }
      }
    }
    return false;
  }

  void findAccAndUpdateLiveRange(StmtPtr s) {
    bool has_reads = hasBufReads(s), has_writes = hasBufWrites(s);
    if (has_reads || has_writes) {
      if (begin_ == -1) {
        begin_ = curr_index_;
      };
      end_ = curr_index_;
    }
  }

  void visit(BlockPtr v) {
    for (StmtPtr s : *v) {
      curr_index_ += 1;
      findAccAndUpdateLiveRange(s);
    }
  }

  BufPtr buf_;
  int32_t begin_ = -1;
  int32_t end_ = -1;
  int32_t curr_index_ = -1;
};

// A class that analyzes the given program relevant for Block backend
// It creates a map of multi dim buffers and their flat verions
class CreateBufferMap : public IRVisitor {
 public:
  const std::unordered_map<std::string, BufPtr>& getBufferMap() const {
    return map_input_to_tensor_bufs_;
  }

 private:
  void visit(StorePtr v) override {
    auto load_node = to<Load>(v->value());
    if (load_node) {
      auto t_buf = load_node->buf();
      map_input_to_tensor_bufs_.emplace(t_buf->name_hint(), v->buf());
    } else {
      auto add_node = to<Add>(v->value());
      auto mul_node = to<Mul>(v->value());
      // This means for now, v->value() can be Add or Mul
      TORCH_INTERNAL_ASSERT(add_node || mul_node, buildErrorMessage());
      map_input_to_tensor_bufs_.emplace(v->buf()->name_hint(), v->buf());
    }
    v->value()->accept(this);
  }
  std::unordered_map<std::string, BufPtr> map_input_to_tensor_bufs_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
