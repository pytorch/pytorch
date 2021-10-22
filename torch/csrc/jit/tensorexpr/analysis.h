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

// This indicates the path from root to a stmt A in AST. Specifically, it
// consists of a vector of integers: each integer represents the number of stmts
// before A's ancestor or A in the block. For example, "1:1:0" points to stmt
// "d[i] += e[i, j];" in the following code.
//
// c = a/b;
// for i
//  d[i] = 0;
//  for j
//    d[i] += e[i, j];
class StmtCounter {
 public:
  void append(int32_t id) {
    counter_.emplace_back(id);
  }
  void pop() {
    counter_.pop_back();
  }

  const std::vector<int32_t> getCounter() {
    return counter_;
  }

  bool operator==(StmtCounter& counter) {
    auto compare = counter.getCounter();
    if (counter_.size() != compare.size()) {
      return false;
    }
    int size = compare.size();
    for (int i = 0; i < size; i++) {
      if (counter_.at(i) != compare.at(i)) {
        return false;
      }
    }
    return true;
  }

  bool operator<(StmtCounter& counter) {
    auto compare = counter.getCounter();
    int size =
        counter_.size() < compare.size() ? counter_.size() : compare.size();
    for (int i = 0; i < size; i++) {
      if (counter_.at(i) > compare.at(i)) {
        return false;
      }
    }
    return !(*this == counter);
  }

  bool operator>(StmtCounter& counter) {
    auto compare = counter.getCounter();
    int size =
        counter_.size() < compare.size() ? counter_.size() : compare.size();
    for (int i = 0; i < size; i++) {
      if (counter_.at(i) < compare.at(i)) {
        return false;
      }
    }
    return !(*this == counter);
  }

  std::string getCounterString() {
    std::string cstr = "";
    for (int i = 0; i < counter_.size(); i++) {
      cstr += std::to_string(counter_.at(i));
      cstr += ":";
    }
    return cstr;
  }

 private:
  std::vector<int32_t> counter_;
};

// Visit Stmts and Record their PCs
class StmtRecorder : public IRVisitor {
 public:
  StmtCounter getStmtCounter() {
    return counter_;
  }

 private:
  void visit(BlockPtr v) {
    if (flatten_) {
      for (StmtPtr s : *v) {
        s->accept(this);
      }
      return;
    }

    int count = 0;
    for (StmtPtr s : *v) {
      counter_.append(count);
      s->accept(this);
      counter_.pop();
      count++;
    }
  }

  void visit(CondPtr v) {
    flatten_ = true;
    ExprPtr condition = v->condition();
    StmtPtr true_stmt = v->true_stmt();
    StmtPtr false_stmt = v->false_stmt();
    condition->accept(this);
    if (true_stmt) {
      true_stmt->accept(this);
    }
    if (false_stmt) {
      false_stmt->accept(this);
    }
    flatten_ = false;
  }

  StmtCounter counter_;
  bool flatten_ = false;
};

enum AccMode { READ, WRITE, REDUCE };

// Traverses the IR to identify all reads/writes to a buf, and their PCs
using BufAccessInfo = std::tuple<StmtPtr, AccMode, StmtCounter>;
class BufAccesses : public StmtRecorder {
 public:
  BufAccesses(BufPtr b) : buf_(b) {}

  std::vector<std::tuple<StmtPtr, AccMode, StmtCounter>> accesses() {
    return accesses_;
  }

  static std::vector<std::tuple<StmtPtr, AccMode, StmtCounter>> find(
      StmtPtr s,
      BufPtr b) {
    BufAccesses finder(b);
    s->accept(&finder);
    return finder.accesses();
  }

 private:
  bool readsBuffer(StmtPtr s) {
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
    return false;
  }

  bool writesBuffer(StmtPtr s) {
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
    return false;
  }

  void insertAccesses(StmtPtr s) {
    if (readsBuffer(s) && writesBuffer(s)) {
      auto acc = std::make_tuple(s, AccMode::REDUCE, getStmtCounter());
      accesses_.push_back(acc);
      return;
    }
    if (readsBuffer(s)) {
      auto acc = std::make_tuple(s, AccMode::READ, getStmtCounter());
      accesses_.push_back(acc);
      return;
    }
    if (writesBuffer(s)) {
      auto acc = std::make_tuple(s, AccMode::WRITE, getStmtCounter());
      accesses_.push_back(acc);
    }
  }

  void visit(StorePtr v) {
    insertAccesses(v);
  }

  void visit(LetPtr v) {
    insertAccesses(v);
  }

  void visit(CondPtr v) {
    insertAccesses(v);
  }

  void visit(AtomicAddPtr v) {
    insertAccesses(v);
  }

  void visit(ExternalCallPtr v) {
    insertAccesses(v);
  }

  BufPtr buf_;
  std::vector<BufAccessInfo> accesses_;
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
