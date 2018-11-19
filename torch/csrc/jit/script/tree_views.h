#pragma once
#include "error_report.h"
#include "tree.h"

#include <functional>
#include <string>

namespace torch {
namespace jit {
namespace script {

// TreeView provides a statically-typed way to traverse the tree, which should
// be formed according to the grammar below.
//
// A few notes on types and their aliases:
// - List<T> is really a Tree with kind TK_LIST and elements as subtrees
// - Maybe<T> is really a Tree with kind TK_OPTION that has 0 or 1 subtree of type T
// - Builtin types are: Ident (TK_IDENT), String (TK_STRING)
//
// Param = Param(Expr type, Ident name)                                 TK_PARAM
//
// Decl  = Decl(List<Param> params, Maybe<Expr> return_type)            TK_DECL
// Def   = Def(Ident name, Decl decl, List<Stmt> body)                  TK_DEF
//
// Stmt  = If(Expr cond, List<Stmt> true_body, List<Stmt> false_body)   TK_IF
//       | For(List<Expr> targets, List<Expr> iters, List<Stmt> body)   TK_FOR
//       | While(Expr cond, List<Stmt> body)                            TK_WHILE
//       | Global(List<Ident> idents)                                   TK_GLOBAL
//       -- NB: the only type of Expr's allowed on lhs are Var
//          Or a tuple containing Var with an optional terminating Starred
//       | Assign(Expr lhs, Expr rhs)                                  TK_ASSIGN
//       | AugAssign(Expr lhs, AugAssignKind aug_op, Expr rhs)          TK_AUG_ASSIGN
//       | Return(List<Expr> values)                                    TK_RETURN
//       | ExprStmt(List<Expr> expr)                                    TK_EXPR_STMT
//       | Raise(Expr expr)                                             TK_RAISE
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
//       |     Is                                                       TK_IS
//       |     IsNot                                                    TK_ISNOT
//       |     Add                                                      '+'
//       |     Sub                                                      '-'
//       |     Mul                                                      '*'
//       |     Div                                                      '/'
//       |     Mod                                                      '%'
//       |     MatMult                                                  '@'
//       |     Pow                                                      TK_POW
//       | UnaryOp(Expr expr)
//       |     Not                                                      TK_NOT
//       |     USub                                                     '-'
//       | Const(String value)                                          TK_CONST
//       -- NB: x.name(y) is desugared into name(x, y)
//       | Apply(Ident name, List<Expr> args, List<Attribute> kwargs)   TK_APPLY
//       | Select(Expr value, Ident selector)                           '.'
//       | Subscript(Expr value, List<Expr> subscript_exprs)            TK_SUBSCRIPT
//       | SliceExpr(Maybe<Expr> start, Maybe<Expr> end)                TK_SLICE_EXPR
//       | Var(Ident name)                                              TK_VAR
//       | ListLiteral(List<Expr> inputs)                               TK_LIST_LITERAL
//       | TupleLiteral(List<Expr> inputs)                              TK_TUPLE_LITERAL
//       | Starred(Expr expr)                                           TK_STARRED
//
// -- NB: only allowed expressions are Const or List(Const)
//        (List as a value, not type constructor)
// Attribute = Attribute(Ident name, Expr value)                        TK_ATTRIBUTE
//
// AugAssignKind =
//            | Add()                                                   TK_PLUS_EQ
//            | Sub()                                                   TK_MINUS_EQ
//            | Mul()                                                   TK_TIMES_EQ
//            | Div()                                                   TK_DIV_EQ
//

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
  explicit TreeView(TreeRef tree) : tree_(std::move(tree)) {}
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
  const TreeRef& subtree(size_t i) const {
    return tree_->trees().at(i);
  }
  TreeRef tree_;
};

template<typename T>
struct ListIterator {
  ListIterator(TreeList::const_iterator it) : it(it) {}
  bool operator!=(const ListIterator& rhs) const { return it != rhs.it; }
  bool operator==(const ListIterator& rhs) const { return it == rhs.it; }
  T operator*() const { return T(*it); }
  ListIterator& operator+=(std::ptrdiff_t n) { it += n; return *this; }
  ListIterator& operator++() { ++it; return *this; }
  ListIterator& operator--() { --it; return *this; }

private:
  TreeList::const_iterator it;
};

template <typename T>
struct List : public TreeView {
  using iterator = ListIterator<T>;
  using const_iterator = ListIterator<T>;

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
    tree_->match(TK_OPTION);
    if (tree_->trees().size() > 1)
      throw ErrorReport(tree) << "Maybe trees can have at most one subtree";
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
  static Maybe<T> create(const SourceRange& range) {
    return Maybe<T>(Compound::create(TK_OPTION, range, {}));
  }
  static Maybe<T> create(const SourceRange& range, const T& value) {
    return Maybe<T>(Compound::create(TK_OPTION, range, {value}));
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

struct Stmt : public TreeView {
  explicit Stmt(const TreeRef& tree) : TreeView(tree) {
    switch (tree->kind()) {
      case TK_IF:
      case TK_FOR:
      case TK_WHILE:
      case TK_GLOBAL:
      case TK_ASSIGN:
      case TK_AUG_ASSIGN:
      case TK_RETURN:
      case TK_EXPR_STMT:
      case TK_RAISE:
      case TK_ASSERT:
      case TK_PASS:
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
      case TK_IS:
      case TK_ISNOT:
      case TK_EQ:
      case TK_LE:
      case TK_GE:
      case TK_NE:
      case '+':
      case '-':
      case TK_UNARY_MINUS:
      case '*':
      case TK_STARRED:
      case '/':
      case '%':
      case TK_NOT:
      case TK_CONST:
      case TK_STRINGLITERAL:
      case TK_TRUE:
      case TK_FALSE:
      case TK_NONE:
      case TK_CAST:
      case TK_APPLY:
      case '.':
      case TK_SUBSCRIPT:
      case TK_SLICE_EXPR:
      case TK_VAR:
      case TK_LIST_LITERAL:
      case TK_TUPLE_LITERAL:
      case '@':
      case TK_POW:
      case TK_FLOOR_DIV:
      case '&':
      case '^':
      case '|':
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
  static Attribute create(const SourceRange& range, const Ident& name, const TreeRef& value) {
    return Attribute(Compound::create(TK_ATTRIBUTE, range, {name, value}));
  }
};


struct Param : public TreeView {
  explicit Param(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_PARAM);
  }
  static Param create(const SourceRange& range, const Ident& ident, const Expr& type) {
    return Param(Compound::create(TK_PARAM, range, {ident, type}));
  }
  Ident ident() const {
    return Ident(subtree(0));
  }
  Expr type() const {
    return Expr(subtree(1));
  }
  template<typename T>
  T typeExpect() const {
    return T(type());
  }
};

////////////////////////////////////////////////////////////////////////////////
// Top level definitions
////////////////////////////////////////////////////////////////////////////////

struct Decl : public TreeView {
  explicit Decl(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_DECL);
  }
  List<Param> params() const {
    return List<Param>(subtree(0));
  }
  Maybe<Expr> return_type() const {
    return Maybe<Expr>(subtree(1));
  }
  static Decl create(const SourceRange& range, const List<Param>& params, Maybe<Expr> return_type) {
    return Decl(Compound::create(TK_DECL, range, {params, return_type}));
  }
};

struct Def : public TreeView {
  explicit Def(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_DEF);
  }
  Def withName(std::string new_name) const {
    auto new_ident = Ident::create(name().range(), new_name);
    return create(range(), new_ident, decl(), statements());
  }
  Ident name() const {
    return Ident(subtree(0));
  }
  Decl decl() const {
    return Decl(subtree(1));
  }
  List<Stmt> statements() const {
    return List<Stmt>(subtree(2));
  }
  static Def create(
      const SourceRange& range,
      const Ident& name,
      const Decl& decl,
      const List<Stmt>& stmts) {
    return Def(Compound::create(
        TK_DEF, range, {name, decl, stmts}));
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

struct For : public Stmt {
  explicit For(const TreeRef& tree) : Stmt(tree) {
    tree->match(TK_FOR);
  }
  List<Expr> targets() const {
    return List<Expr>(subtree(0));
  }
  List<Expr> itrs() const {
    return List<Expr>(subtree(1));
  }
  List<Stmt> body() const {
    return List<Stmt>(subtree(2));
  }
  static For create(
      const SourceRange& range,
      const List<Expr>& targets,
      const List<Expr>& itrs,
      const List<Stmt>& body) {
    return For(Compound::create(TK_FOR, range, {targets, itrs, body}));
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

struct AugAssignKind : public TreeView {
  explicit AugAssignKind(const TreeRef& tree) : TreeView(tree) {
    switch (tree->kind()) {
      case '+':
      case '-':
      case '*':
      case '/':
        return;
      default:
        throw ErrorReport(tree) << "is not a valid AugAssignKind";
    }
  }
};

// Augmented assignment, like "foo += bar"
struct AugAssign : public Stmt {
  explicit AugAssign(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_AUG_ASSIGN);
  }
  static AugAssign create(
      const SourceRange& range,
      const Expr& lhs,
      const AugAssignKind& aug_op,
      const Expr& rhs) {
    return AugAssign(
        Compound::create(TK_AUG_ASSIGN, range, {lhs, aug_op, rhs}));
  }
  Expr lhs() const {
    return Expr(subtree(0));
  }
  int aug_op() const {
    return subtree(1)->kind();
  }
  Expr rhs() const {
    return Expr(subtree(2));
  }
};


struct Assign : public Stmt {
  explicit Assign(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_ASSIGN);
  }
  static Assign create(
      const SourceRange& range,
      const Expr& lhs,
      const Expr& rhs) {
    return Assign(Compound::create(TK_ASSIGN, range, {lhs, rhs}));
  }
  Expr lhs() const {
    return Expr(subtree(0));
  }
  Expr rhs() const {
    return Expr(subtree(1));
  }
};

struct Return : public Stmt {
  explicit Return(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_RETURN);
  }
  List<Expr> values() const {
    return List<Expr>(subtree(0));
  }
  static Return create(const SourceRange& range, const List<Expr>& values) {
    return Return(Compound::create(TK_RETURN, range, {values}));
  }
};

struct Raise : public Stmt {
  explicit Raise(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_RAISE);
  }
  Maybe<Expr> expr() const {
    return Maybe<Expr>(subtree(0));
  }
  static Raise create(const SourceRange& range, const Maybe<Expr>& expr) {
    return Raise(Compound::create(TK_RAISE, range, {expr}));
  }
};

struct Assert : public Stmt {
  explicit Assert(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_ASSERT);
  }
  Expr test() const {
    return Expr(subtree(0));
  }
  Maybe<Expr> msg() const {
    return Maybe<Expr>(subtree(1));
  }
  static Assert create(
      const SourceRange& range,
      const Expr& test,
      const Maybe<Expr>& msg) {
    return Assert(Compound::create(TK_ASSERT, range, {test, msg}));
  }
};

struct Pass : public Stmt {
  explicit Pass(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_PASS);
  }
  static Pass create(
      const SourceRange& range) {
    return Pass(Compound::create(TK_PASS, range, {}));
  }
};


struct ExprStmt : public Stmt {
  explicit ExprStmt(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_EXPR_STMT);
  }
  Expr expr() {
    return Expr(subtree(0));
  }
  static ExprStmt create(const SourceRange& range, const Expr list) {
    return ExprStmt(Compound::create(TK_EXPR_STMT, range, {list}));
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
      case TK_IS:
      case TK_ISNOT:
      case TK_EQ:
      case TK_LE:
      case TK_GE:
      case TK_NE:
      case '+':
      case '*':
      case '/':
      case '-':
      case '@':
      case TK_POW:
      case '%':
      case '&':
      case '^':
      case '|':
      case TK_FLOOR_DIV:
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
      case TK_UNARY_MINUS:
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

struct Const : public Expr {
  explicit Const(const TreeRef& tree) : Expr(tree) {
    tree_->matchNumSubtrees(TK_CONST, 1);
  }
  bool isFloatingPoint() const {
    return subtree(0)->stringValue().find_first_of(".eE") != std::string::npos;
  }
  bool isIntegral() const {
    return !isFloatingPoint();
  }
  int64_t asIntegral() const {
    return std::stoll(subtree(0)->stringValue());
  }
  double asFloatingPoint() const {
    return std::stod(subtree(0)->stringValue());
  }
  const std::string& text() const {
    return subtree(0)->stringValue();
  }
  static Const create(const SourceRange& range, const std::string& value) {
    return Const(Compound::create(TK_CONST, range, {String::create(value)}));
  }
};

struct StringLiteral : public Expr {
  explicit StringLiteral(const TreeRef& tree) : Expr(tree) {
    tree_->matchNumSubtrees(TK_STRINGLITERAL, 1);
  }
  const std::string& text() const {
    return subtree(0)->stringValue();
  }
  static StringLiteral create(const SourceRange& range, const std::string& value) {
    return StringLiteral(Compound::create(TK_STRINGLITERAL, range, {String::create(value)}));
  }
};

struct Apply : public Expr {
  explicit Apply(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_APPLY);
  }
  Expr callee() const {
    return Expr(subtree(0));
  }
  List<Expr> inputs() const {
    return List<Expr>(subtree(1));
  }
  List<Attribute> attributes() const {
    return List<Attribute>(subtree(2));
  }
  static Apply create(
      const SourceRange& range,
      const Expr& callee,
      const List<Expr>& inputs,
      const List<Attribute>& attributes) {
    return Apply(Compound::create(TK_APPLY, range, {callee, inputs, attributes}));
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

struct SliceExpr : public Expr {
  explicit SliceExpr(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_SLICE_EXPR);
  }
  Maybe<Expr> start() const {
    return Maybe<Expr>(subtree(0));
  }
  Maybe<Expr> end() const {
    return Maybe<Expr>(subtree(1));
  }
  Expr startOr(int alternative) const {
    const auto startOption = start();
    return startOption.present() ? startOption.get() : createInt(alternative);
  }
  Expr endOr(int alternative) const {
    const auto endOption = end();
    return endOption.present() ? endOption.get() : createInt(alternative);
  }
  static SliceExpr create(
      const SourceRange& range,
      const Maybe<Expr>& start,
      const Maybe<Expr>& end) {
    return SliceExpr(Compound::create(TK_SLICE_EXPR, range, {start, end}));
  }
private:
  Expr createInt(int value) const {
    return Expr(Const::create(range(), std::to_string(value)));
  }
};

struct Subscript : public Expr {
  explicit Subscript(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_SUBSCRIPT);
  }
  Expr value() const {
    return Expr(subtree(0));
  }
  List<Expr> subscript_exprs() const {
    return List<Expr>(subtree(1));
  }
  static Subscript create(
      const SourceRange& range,
      const Expr& value,
      const List<Expr>& subscript_exprs) {
    return Subscript(Compound::create(TK_SUBSCRIPT, range, {value, subscript_exprs}));
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


struct ListLiteral : public Expr {
  explicit ListLiteral(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_LIST_LITERAL);
  }
  List<Expr> inputs() const {
    return subtree(0);
  }
  static ListLiteral create(const SourceRange& range, const List<Expr>& inputs) {
    return ListLiteral(Compound::create(TK_LIST_LITERAL, range, {inputs}));
  }
};

struct TupleLiteral : public Expr {
  explicit TupleLiteral(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_TUPLE_LITERAL);
  }
  List<Expr> inputs() const {
    return subtree(0);
  }
  static TupleLiteral create(const SourceRange& range, const List<Expr>& inputs) {
    return TupleLiteral(Compound::create(TK_TUPLE_LITERAL, range, {inputs}));
  }
};

struct Starred : public Expr {
  explicit Starred(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_STARRED);
  }
  Expr expr() const {
    return Expr(subtree(0));
  }
  static Starred create(const SourceRange& range, const Expr& expr) {
    return Starred(Compound::create(TK_STARRED, range, {expr}));
  }
};

} // namespace script
} // namespace jit
} // namespace torch

namespace std {

template<typename T>
struct iterator_traits<torch::jit::script::ListIterator<T>>
  : std::iterator_traits<torch::jit::script::TreeList::const_iterator> {};

} // namespace std
