#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <torch/csrc/jit/script/lexer.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>

namespace torch {
namespace jit {
namespace script {

// Tree's are used to represent all forms of TC IR, pre- and post- typechecking.
// Rather than have a full class hierarchy for all TC statements,
// Trees are a slight variation of Lisp S-expressions.
// for instance the expression a*b+1 is represented as:
// (+ (* (ident a) (ident b)) (const 1))
// Atoms like 'a', 'b', and '1' are represented by subclasses of Tree which
// define stringValue().
// Everything else is a Compound object, which has a 'kind' that is a token from
// Lexer.h's TokenKind enum, and contains a list of subtrees.
// Like TokenKind single-character operators like '+' are representing using the
// character itself, so add.kind() == '+'.
// Compound objects are also always associated with a SourceRange for
// reporting error message.

// Memory management of trees is done by arena allocation.
// Make sure to call Tree::freeArena when trees are no longer needed.

struct Tree;
using TreeRef = Tree*;
using TreeList = at::ArrayRef<TreeRef>;

static const TreeList empty_trees = {};

struct Tree {
  Tree(int kind_) : kind_(kind_) {}
  int kind() const {
    return kind_;
  }
  virtual bool isAtom() const {
    return true;
  }
  virtual const SourceRange& range() const {
    throw std::runtime_error("is an Atom");
  }
  virtual const std::string& stringValue() const {
    throw std::runtime_error("stringValue can only be called on TK_STRING");
  }
  virtual const TreeList& trees() const {
    return empty_trees;
  }
  const TreeRef& tree(size_t i) const {
    return trees().at(i);
  }
  virtual TreeRef map(const std::function<TreeRef(TreeRef)>& fn) {
    (void)fn;
    return this;
  }
  template <typename... Args>
  void match(int k, Args&... args) const {
    matchD(k, "unknown", 0, args...);
  }
  template <typename... Args>
  void matchD(int k, const char* filename, int lineno, Args&... args) const {
    std::initializer_list<TreeRef*> vars = {args...};
    matchNumSubtreesD(k, filename, lineno, vars.size(), true);
    size_t i = 0;
    for (TreeRef* v : vars) {
      *v = trees()[i++];
    }
  }
  void matchNumSubtrees(int k, size_t expected_subtrees) {
    return matchNumSubtreesD(k, "unknown", 0, expected_subtrees, false);
  }
  void matchNumSubtreesD(
      int k,
      const char* filename,
      int lineno,
      size_t expected_subtrees,
      bool allow_more) const {
    if (kind() != k) {
      std::stringstream ss;
      ss << filename << ":" << lineno << ": expecting kind '" << kindToString(k)
         << "' but found '" << kindToString(kind()) << "'\n";
      range().highlight(ss);
      throw std::runtime_error(ss.str());
    }
    if (trees().size() < expected_subtrees ||
        (!allow_more && trees().size() != expected_subtrees)) {
      std::stringstream ss;
      ss << filename << ":" << lineno << ": expected at least "
         << expected_subtrees << " subtrees, but found only " << trees().size()
         << "\n";
      range().highlight(ss);
      throw std::runtime_error(ss.str());
    }
  }

  void *operator new(size_t, void* p_) { return p_; }
  void *operator new(size_t size) {
    if (freelist_.empty() || size + slaboffset_ > ALLOC_SIZE) {
      auto slab = malloc(std::max(size, ALLOC_SIZE));
      slaboffset_ = size;
      freelist_.push_back(slab);
      return slab;
    }

    auto p = (char*)freelist_.back() + slaboffset_;
    slaboffset_ += size;
    return p;
  }

  void operator delete(void * p)  {}

  static void freeArena() {
    for (void* p : freelist_) {
      free(p);
    }
    freelist_.clear();
  }

 private:
  int kind_;

  static const size_t ALLOC_SIZE = 4096;
  TORCH_API static size_t slaboffset_;
  TORCH_API static std::vector<void*> freelist_;
};

struct String : public Tree {
  String(std::string value) : Tree(TK_STRING), value_(std::move(value)) {}
  const std::string& stringValue() const override {
    return value_;
  }
  template <typename... Args>
  static TreeRef create(Args&&... args) {
    return new String(std::forward<Args>(args)...);
  }

 private:
  std::string value_;
};

static SourceRange mergeRanges(SourceRange c, const TreeList& others) {
  for (const auto& t : others) {
    if (t->isAtom())
      continue;
    size_t s = std::min(c.start(), t->range().start());
    size_t e = std::max(c.end(), t->range().end());
    c = SourceRange(c.source(), s, e);
  }
  return c;
}

struct Compound : public Tree {
  Compound(int kind, SourceRange range)
      : Tree(kind), range_(std::move(range)) {}
  Compound(int kind, const SourceRange& range_, TreeList&& trees_)
      : Tree(kind),
        range_(mergeRanges(range_, trees_)),
        trees_(std::move(trees_)) {}
  const TreeList& trees() const override {
    return trees_;
  }
  static TreeRef create(
      int kind,
      const SourceRange& range_,
      TreeList&& trees_) {

    char* ptr = (char*)Tree::operator new(
      sizeof(Compound) + trees_.size() * sizeof(TreeRef));

    TreeRef* tree_ptr = (TreeRef*)(ptr + sizeof(Compound));
    for (size_t i = 0; i < trees_.size(); ++i) {
      tree_ptr[i] = trees_[i];
    }

    char* compound_ptr = ptr;
    new (compound_ptr) Compound(kind, range_, { tree_ptr, trees_.size() });

    return (Compound*)compound_ptr;
  }
  bool isAtom() const override {
    return false;
  }
  TreeRef map(const std::function<TreeRef(TreeRef)>& fn) override {
    at::SmallVector<TreeRef, 4> trees_;
    for (auto& t : trees()) {
      trees_.push_back(fn(t));
    }
    return Compound::create(kind(), range(), trees_);
  }

  const SourceRange& range() const override {
    return range_;
  }

 private:
  SourceRange range_;
  TreeList trees_;
};

// tree pretty printer
struct pretty_tree {
  pretty_tree(const TreeRef& tree, size_t col = 40) : tree(tree), col(col) {}
  const TreeRef& tree;
  size_t col;
  std::unordered_map<TreeRef, std::string> flat_strings;
  const std::string& get_flat(const TreeRef& t) {
    auto it = flat_strings.find(t);
    if (it != flat_strings.end())
      return it->second;

    std::stringstream out;
    switch (t->kind()) {
      case TK_STRING:
        out << t->stringValue();
        break;
      default:
        out << "(" << kindToString(t->kind());
        for (const auto& e : t->trees()) {
          out << " " << get_flat(e);
        }
        out << ")";
        break;
    }
    auto it_ = flat_strings.emplace(t, out.str());
    return it_.first->second;
  }
  void print(std::ostream& out, const TreeRef& t, int indent) {
    const std::string& s = get_flat(t);
    if (indent + s.size() < col || t->isAtom()) {
      out << s;
      return;
    }
    std::string k = kindToString(t->kind());
    out << "(" << k;
    for (const auto& e : t->trees()) {
      out << "\n" << std::string(indent + 2, ' ');
      print(out, e, indent + 2);
    }
    out << ")";
  }
};

static inline std::ostream& operator<<(std::ostream& out, pretty_tree t_) {
  t_.print(out, t_.tree, 0);
  return out << std::endl;
}

static inline std::ostream& operator<<(std::ostream& out, const TreeRef& t) {
  return out << pretty_tree(t);
}

} // namespace script
} // namespace jit
} // namespace torch
