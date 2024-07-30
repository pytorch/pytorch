#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <c10/util/SmallVector.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/jit/frontend/lexer.h>

namespace torch::jit {

// Trees are used to represent all forms of TC IR, pre- and post-typechecking.
// Rather than have a full class hierarchy for all TC statements, trees are a
// slight variation of Lisp s-expressions. For instance, the expression a*b+1
// is represented as:
// (+ (* (ident a) (ident b)) (const 1))
// Atoms like 'a', 'b', and '1' are represented by subclasses of Tree which
// define stringValue(). Everything else is a Compound object, which has a
// 'kind' that is a token from lexer.h's TokenKind enum. Single-character
// operators like '+' are represented using the character itself (so, add.kind()
// would be '+'). Each Compound object also contains a list of subtrees and is
// associated with a SourceRange for error reporting.
// Memory management of trees is done using intrusive_ptr.

struct Tree;
using TreeRef = c10::intrusive_ptr<Tree>;
using TreeList = at::SmallVector<TreeRef, 4>;

struct Tree : c10::intrusive_ptr_target {
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
    static const TreeList empty_trees = {};
    return empty_trees;
  }
  const TreeRef& tree(size_t i) const {
    return trees().at(i);
  }
  virtual TreeRef map(const std::function<TreeRef(TreeRef)>& fn) {
    (void)fn;
    c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                           // from a raw `this` pointer
                                           // so we need to bump the refcount
                                           // to account for this ownership
    return TreeRef::reclaim(this);
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
  ~Tree() override = default;

 private:
  int kind_;
};

struct String : public Tree {
  String(std::string value) : Tree(TK_STRING), value_(std::move(value)) {}
  const std::string& stringValue() const override {
    return value_;
  }
  template <typename... Args>
  static TreeRef create(Args&&... args) {
    return c10::make_intrusive<String>(std::forward<Args>(args)...);
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
    return c10::make_intrusive<Compound>(kind, range_, std::move(trees_));
  }
  bool isAtom() const override {
    return false;
  }
  TreeRef map(const std::function<TreeRef(TreeRef)>& fn) override {
    TreeList ret;
    for (auto& t : trees()) {
      ret.push_back(fn(t));
    }
    return Compound::create(kind(), range(), std::move(ret));
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
  return out << '\n';
}

static inline std::ostream& operator<<(std::ostream& out, const TreeRef& t) {
  return out << pretty_tree(t);
}

} // namespace torch::jit
