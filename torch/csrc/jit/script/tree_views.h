#pragma once
#include "error_report.h"
#include "tree.h"

#include <functional>

namespace torch {
namespace jit {
namespace script {

// TreeView provides a statically-typed way to traverse the tree, which should
// be formed according to the grammar below.
//
// A few notes on types and their aliases:
// - List<T> is really a Tree with kind TK_LIST and elements as subtrees
// - Maybe<T> is really either a Tree with kind TK_OPTION or T
// - Builtin types are: Ident (TK_IDENT), String (TK_STRING),
//                      Number (TK_NUMBER) and Bool (TK_BOOL)
//
// Type  = TensorType()                                                 TK_TENSOR_TYPE
// Param = Param(Type type, Ident name)                                 TK_PARAM
//
// -- TODO: change returns to be a list of expressions
// Def   = Def(Ident name, List<Param> params, List<Param> returns, List<Stmt> body) TK_DEF
//
// Stmt  = If(Expr cond, List<Stmt> true_body, List<Stmt> false_body)   TK_IF
//       | While(Expr cond, List<Stmt> body)                            TK_WHILE
//       | Global(List<Ident> idents)                                   TK_GLOBAL
//       | Assign(List<Ident> lhs, AssignType maybe_reduce, Expr rhs)   TK_ASSIGN
//       | ExprStmt(Expr expr)                                          TK_EXPR_STMT
//
// Expr  = TernaryIf(Expr cond, Expr true_expr, Expr false_expr)        TK_IF_EXPR
//       | BinOp(Expr lhs, Expr rhs)
//       |     And                                                      TK_AND
//       |     Or                                                       TK_OR
//       |     Lt                                                       '<'
//       |     Gt                                                       '>'
//       |     Eq                                                       TK_EQ
//       |     Le                                                       TK_LE
//       |     Ge                                                       TK_GE
//       |     Ne                                                       TK_NE
//       |     Add                                                      '+'
//       |     Sub                                                      '-'
//       |     Mul                                                      '*'
//       |     Div                                                      '/'
//       | UnaryOp(Expr expr)
//       |     Not                                                      TK_NOT
//       |     USub                                                     '-'
//       -- * is one of  |   Number  | Bool |
//       --              +-----------+------+
//       -- type is then | "i" | "f" |  "b" |
//       -- TODO: change this to a generic "Scalar" node that keep arbitrary precision values
//       | Const(* value, String type)                                  TK_CONST
//       | Cast(ScalarType type, Expr expr)                             TK_CAST
//       -- NB: x.name(y) is desugared into name(x, y)
//       | Apply(Ident name, List<Expr> args, List<Attribute> kwargs)   TK_APPLY
//       | Select(Expr base, Ident attr_name)                           '.'
//       | Slice(Expr value, Maybe<Expr> first, Maybe<Expr> second)     TK_SLICE
//       | Gather(Expr value, Expr indices)                             TK_GATHER
//       | Var(Ident name)                                              TK_VAR
//
// -- NB: only allowed expressions are Const or List(Const)
//        (List as a value, not type constructor)
// Attribute = Attribute(Ident name, Expr value)                        TK_ATTRIBUTE
//
// AssignKind = Regular()                                               '='
//            | Add()                                                   TK_PLUS_EQ
//            | Sub()                                                   TK_MINUS_EQ
//            | Mul()                                                   TK_TIMES_EQ
//            | Div()                                                   TK_DIV_EQ
//
// ScalarType = IntType()                                               TK_INT
//            | FloatType()                                             TK_FLOAT
//            | LongType()                                              TK_LONG
//            | DoubleType()                                            TK_DOUBLE

// Each subclass of TreeView should provide:
// 1. Constructor that takes a TreeRef, and checks that it's of the right type.
// 2. Accessors that get underlying information out of the object. If they
//    return subtrees, they should wrap them in appropriate views too.
// 3. Static method 'create' that creates the underlying TreeRef object
//    for every TreeRef kind that has a TreeView, the parser always uses
//    (e.g.) Ident::create rather than Compound::Create, this means that
//    changes to the structure of Ident are always made right here rather
//    than both in the parser and in this code.
// XXX: these structs should have no fields to prevent slicing when passing by value
struct TreeView {
  explicit TreeView(const TreeRef& tree_) : tree_(tree_) {}
  TreeRef tree() const {
    return tree_;
  }
  const SourceRange& range() const {
    return tree_->range();
  }
  operator TreeRef() const {
    return tree_;
  }
  const TreeRef& get() const {
    return tree_;
  }
  int kind() const {
    return tree_->kind();
  }

protected:
  const TreeRef& subtree(std::size_t i) const {
    return tree_->trees().at(i);
  }
  TreeRef tree_;
};

template <typename T>
struct List : public TreeView {
  struct Iterator {
    Iterator(TreeList::const_iterator it) : it(it) {}
    bool operator!=(const Iterator& rhs) const { return it != rhs.it; }
    T operator*() const { return T(*it); }
    void operator++() { ++it; }
    void operator--() { --it; }

  private:
    TreeList::const_iterator it;
  };
  typedef Iterator iterator;
  typedef Iterator const_iterator;

  List(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_LIST);
    // Iterate over list to temporarily instantiate Ts that will check the type
    for (const T& elem : *this) {
      (void) elem; //silence unused warning
    }
  }
  iterator begin() const {
    return iterator(tree_->trees().begin());
  }
  iterator end() const {
    return iterator(tree_->trees().end());
  }
  bool empty() const {
    return tree_->trees().begin() == tree_->trees().end();
  }
  T operator[](size_t i) const {
    return T(subtree(i));
  }
  TreeRef map(std::function<TreeRef(const T&)> fn) {
    return tree_->map([&](TreeRef v) { return fn(T(v)); });
  }
  static List create(const SourceRange& range, const std::vector<T>& subtrees) {
    TreeList type_erased_sub {subtrees.begin(), subtrees.end()};
    return List(Compound::create(TK_LIST, range, std::move(type_erased_sub)));
  }
  size_t size() const {
    return tree_->trees().size();
  }
};

template <typename T>
struct Maybe : public TreeView {
  explicit Maybe(const TreeRef& tree) : TreeView(tree) {
    if (tree->kind() != TK_OPTION) {
      std::cout << kindToString(tree->kind()) << std::endl;
      T{tree}; // invoke the constructor to check the type
    }
  }
  /* implicit */ Maybe(const T& tree) : TreeView(tree) {}
  bool present() const {
    return tree_->trees().size() > 0;
  }
  T get() const {
    return T(tree_->trees().at(0));
  }
  TreeRef map(std::function<TreeRef(const T&)> fn) {
    return tree_->map([&](TreeRef v) { return fn(T(v)); });
  }
};

struct Ident : public TreeView {
  explicit Ident(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_IDENT);
  }
  const std::string& name() const {
    return subtree(0)->stringValue();
  }
  static Ident create(const SourceRange& range, const std::string& name) {
    return Ident(Compound::create(TK_IDENT, range, {String::create(name)}));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Base types (production LHS)
////////////////////////////////////////////////////////////////////////////////

struct Type : public TreeView {
  explicit Type(const TreeRef& tree) : TreeView(tree) {
    switch (tree->kind()) {
      case TK_TENSOR_TYPE:
        return;
      default:
        throw ErrorReport(tree) << kindToString(tree->kind()) << " is not a valid Type";
    }
  }
};

struct Stmt : public TreeView {
  explicit Stmt(const TreeRef& tree) : TreeView(tree) {
    switch (tree->kind()) {
      case TK_IF:
      case TK_WHILE:
      case TK_GLOBAL:
      case TK_ASSIGN:
      case TK_EXPR_STMT:
        return;
      default:
        throw ErrorReport(tree) << kindToString(tree->kind()) << " is not a valid Stmt";
    }
  }
};

struct Expr : public TreeView {
  explicit Expr(const TreeRef& tree) : TreeView(tree) {
    switch (tree->kind()) {
      case TK_IF_EXPR:
      case TK_AND:
      case TK_OR:
      case '<':
      case '>':
      case TK_EQ:
      case TK_LE:
      case TK_GE:
      case TK_NE:
      case '+':
      case '-':
      case '*':
      case '/':
      case TK_NOT:
      /* case '-': - unary minus */
      case TK_CONST:
      case TK_CAST:
      case TK_APPLY:
      case '.':
      case TK_SLICE:
      case TK_GATHER:
      case TK_VAR:
        return;
      default:
        throw ErrorReport(tree) << kindToString(tree->kind()) << " is not a valid Expr";
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Helper nodes (mostly for function arguments)
////////////////////////////////////////////////////////////////////////////////

struct Attribute : public TreeView {
  explicit Attribute(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_ATTRIBUTE);
  }
  Ident name() const {
    return Ident(subtree(0));
  }
  Expr value() const {
    return Expr(subtree(1));
  }
  static Attribute create(const SourceRange& range, const Ident& name, const Expr& value) {
    return Attribute(Compound::create(TK_ATTRIBUTE, range, {name, value}));
  }
};


struct Param : public TreeView {
  explicit Param(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_PARAM);
  }
  static Param create(const SourceRange& range, const Ident& ident, const Type& type) {
    return Param(Compound::create(TK_PARAM, range, {ident, type}));
  }
  Ident ident() const {
    return Ident(subtree(0));
  }
  Type type() const {
    return Type(subtree(1));
  }
  template<typename T>
  T typeExpect() const {
    return T(type());
  }
};


////////////////////////////////////////////////////////////////////////////////
// Type
////////////////////////////////////////////////////////////////////////////////

struct TensorType : public Type {
  explicit TensorType(const TreeRef& tree) : Type(tree) {
    tree_->match(TK_TENSOR_TYPE);
  }
  static TensorType create(const SourceRange& range) {
    return TensorType(Compound::create(TK_TENSOR_TYPE, range, {}));
  }
};

struct ScalarType : public TreeView {
  explicit ScalarType(const TreeRef& tree) : TreeView(tree) {
    switch (tree->kind()) {
      case TK_INT:
      case TK_LONG:
      case TK_FLOAT:
      case TK_DOUBLE:
        return;
      default:
        throw ErrorReport(tree) << kindToString(tree->kind()) << " is not a valid ScalarType";
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Top level definitions
////////////////////////////////////////////////////////////////////////////////

struct Def : public TreeView {
  explicit Def(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_DEF);
  }
  Ident name() const {
    return Ident(subtree(0));
  }
  List<Param> params() const {
    return List<Param>(subtree(1));
  }
  List<Param> returns() const {
    return List<Param>(subtree(2));
  }
  List<Stmt> statements() const {
    return List<Stmt>(subtree(3));
  }
  static Def create(
      const SourceRange& range,
      const Ident& name,
      const List<Param>& params,
      const List<Param>& returns,
      const List<Stmt>& stmts) {
    return Def(Compound::create(
        TK_DEF, range, {name, params, returns, stmts}));
  }
};


////////////////////////////////////////////////////////////////////////////////
// Statements
////////////////////////////////////////////////////////////////////////////////

struct If : public Stmt {
  explicit If(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_IF);
  }
  Expr cond() const {
    return Expr(subtree(0));
  }
  List<Stmt> trueBranch() const {
    return List<Stmt>(subtree(1));
  }
  List<Stmt> falseBranch() const {
    return List<Stmt>(subtree(2));
  }
  static If create(
      const SourceRange& range,
      const Expr& cond,
      const List<Stmt>& true_branch,
      const List<Stmt>& false_branch) {
    return If(Compound::create(TK_IF, range, {cond, true_branch, false_branch}));
  }
};

struct While : public Stmt {
  explicit While(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_WHILE);
  }
  Expr cond() const {
    return Expr(subtree(0));
  }
  List<Stmt> body() const {
    return List<Stmt>(subtree(1));
  }
  static While create(const SourceRange& range, const Expr& cond, const List<Stmt>& body) {
    return While(Compound::create(TK_WHILE, range, {cond, body}));
  }
};

struct Global : public Stmt {
  explicit Global(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_GLOBAL);
  }
  List<Ident> names() {
    return List<Ident>(subtree(0));
  }
  static Global create(const SourceRange& range, const List<Ident>& names) {
    return Global(Compound::create(TK_GLOBAL, range, {names}));
  }
};

struct AssignKind : public TreeView {
  explicit AssignKind(const TreeRef& tree) : TreeView(tree) {
    switch (tree->kind()) {
      case '=':
      case '+':
      case '-':
      case '*':
      case '/':
        return;
      default:
        throw ErrorReport(tree) << "is not a valid AssignKind";
    }
  }
};

struct Assign : public Stmt {
  explicit Assign(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_ASSIGN);
  }
  static Assign create(
      const SourceRange& range,
      const List<Ident>& lhs,
      const AssignKind& reduction,
      const Expr& rhs) {
    return Assign(Compound::create(TK_ASSIGN, range, {lhs, reduction, rhs}));
  }
  List<Ident> lhs() const {
    return List<Ident>(subtree(0));
  }
  int reduction() const {
    return subtree(1)->kind();
  }
  Expr rhs() const {
    return Expr(subtree(2));
  }
};

struct ExprStmt : public Stmt {
  explicit ExprStmt(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_EXPR_STMT);
  }
  Expr expr() {
    return Expr(subtree(0));
  }
  static ExprStmt create(const SourceRange& range, const Expr& value) {
    return ExprStmt(Compound::create(TK_EXPR_STMT, range, {value}));
  }
};


////////////////////////////////////////////////////////////////////////////////
// Expressions
////////////////////////////////////////////////////////////////////////////////

struct BinOp : public Expr {
  explicit BinOp(const TreeRef& tree) : Expr(tree) {
    switch (tree->kind()) {
      case TK_AND:
      case TK_OR:
      case '<':
      case '>':
      case TK_EQ:
      case TK_LE:
      case TK_GE:
      case TK_NE:
      case '+':
      case '*':
      case '/':
      case '-':
        if (tree->trees().size() != 2)
          throw ErrorReport(tree) << "BinOp expected 2 subtrees, found " << tree->trees().size();
        return;
      default:
        throw ErrorReport(tree) << kindToString(tree->kind()) << " is not a valid BinOp";
    }
  }
  Expr lhs() const {
    return Expr(subtree(0));
  }
  Expr rhs() const {
    return Expr(subtree(1));
  }
  static BinOp create(const SourceRange& range, int kind, const Expr& lhs, const Expr& rhs) {
    return BinOp(Compound::create(kind, range, {lhs, rhs}));
  }
};

struct UnaryOp : public Expr {
  explicit UnaryOp(const TreeRef& tree) : Expr(tree) {
    switch (tree->kind()) {
      case '-':
      case TK_NOT:
        if (tree->trees().size() != 1)
          throw ErrorReport(tree) << "UnaryOp expected 1 subtree, found " << tree->trees().size();
        return;
      default:
        throw ErrorReport(tree) << kindToString(tree->kind()) << " is not a valid UnaryOp";
    }
  }
  static UnaryOp create(const SourceRange& range, int kind, const Expr& expr) {
    return UnaryOp(Compound::create(kind, range, {expr}));
  }
};

struct Cast : public Expr {
  explicit Cast(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_CAST);
  }
  ScalarType type() const {
    return ScalarType(subtree(0));
  }
  Expr input() const {
    return Expr(subtree(1));
  }
  static Cast create(const SourceRange& range, const Type& type, const Expr& input) {
    return Cast(Compound::create(TK_CAST, range, {type, input}));
  }
};

struct Apply : public Expr {
  explicit Apply(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_APPLY);
  }
  Ident name() const {
    return Ident(subtree(0));
  }
  List<Expr> inputs() const {
    return List<Expr>(subtree(1));
  }
  List<Attribute> attributes() const {
    return List<Attribute>(subtree(2));
  }
  static Apply create(
      const SourceRange& range,
      const Ident& name,
      const List<Expr>& inputs,
      const List<Attribute>& attributes) {
    return Apply(Compound::create(TK_APPLY, range, {name, inputs, attributes}));
  }
};

struct Select : public Expr {
  explicit Select(const TreeRef& tree) : Expr(tree) {
    tree_->match('.');
  }
  Expr value() const {
    return Expr(subtree(0));
  }
  Ident selector() const {
    return Ident(subtree(1));
  }
  static Select create(const SourceRange& range, const Expr& value, const Ident& selector) {
    return Select(Compound::create('.', range, {value, selector}));
  }
};

struct Slice : public Expr {
  explicit Slice(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_SLICE);
  }
  Expr value() const {
    return Expr(subtree(0));
  }
  Maybe<Expr> start() const {
    return Maybe<Expr>(subtree(1));
  }
  Maybe<Expr> end() const {
    return Maybe<Expr>(subtree(2));
  }
  Expr startOr(int alternative) const {
    const auto startOption = start();
    return startOption.present() ? startOption.get() : createInt(alternative);
  }
  Expr endOr(int alternative) const {
    const auto endOption = end();
    return endOption.present() ? endOption.get() : createInt(alternative);
  }
  static Slice create(
      const SourceRange& range,
      const Expr& value,
      const Maybe<Expr>& start,
      const Maybe<Expr>& end) {
    return Slice(Compound::create(TK_SLICE, range, {value, start, end}));
  }
private:
  Expr createInt(int value) const {
    return Expr(Compound::create(
        TK_CONST, range(), {Number::create(value), String::create("i")}));
  }
};

struct Gather : public Expr {
  explicit Gather(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_GATHER);
  }
  Expr value() const {
    return Expr(subtree(0));
  }
  Expr indices() const {
    return Expr(subtree(1));
  }
  static Gather create(const SourceRange& range, const Expr& value, const Expr& indices) {
    return Gather(Compound::create(TK_GATHER, range, {value, indices}));
  }
};

struct Var : public Expr {
  explicit Var(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_VAR);
  };
  Ident name() const {
    return Ident(subtree(0));
  }
  static Var create(const SourceRange& range, const Ident& name) {
    return Var(Compound::create(TK_VAR, range, {name}));
  }
};

struct TernaryIf : public Expr {
  explicit TernaryIf(const TreeRef& tree) : Expr(tree) {
    tree_->matchNumSubtrees(TK_IF_EXPR, 3);
  };
  Expr cond() const {
    return Expr(subtree(0));
  }
  Expr true_expr() const {
    return Expr(subtree(1));
  }
  Expr false_expr() const {
    return Expr(subtree(2));
  }
  static TernaryIf create(const SourceRange& range,
                          const Expr& cond,
                          const Expr& true_expr,
                          const Expr& false_expr) {
    return TernaryIf(Compound::create(TK_IF_EXPR, range, {cond, true_expr, false_expr}));
  };
};

} // namespace script
} // namespace jit
} // namespace torch
