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
// Type  = TensorType()                                                 TK_TENSOR_TYPE
// Param = Param(Type type, Ident name)                                 TK_PARAM
//
// Def   = Def(Ident name, List<Param> params, List<Stmt> body)         TK_DEF
//
// Stmt  = If(Expr cond, List<Stmt> true_body, List<Stmt> false_body)   TK_IF
//       | For(List<Expr> targets, List<Expr> iters, List<Stmt> body)   TK_FOR
//       | While(Expr cond, List<Stmt> body)                            TK_WHILE
//       | Global(List<Ident> idents)                                   TK_GLOBAL
//       -- NB: the only type of Expr's allowed on lhs are Starred and Var
//       | Assign(List<Expr> lhs, AssignType maybe_reduce, Expr rhs)    TK_ASSIGN
//       | Return(List<Expr> values)                                    TK_RETURN
//       | ExprStmt(List<Expr> expr)                                    TK_EXPR_STMT
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
//       | Const(String value)                                          TK_CONST
//       | Cast(ScalarType type, Expr expr)                             TK_CAST
//       -- NB: x.name(y) is desugared into name(x, y)
//       | Apply(Ident name, List<Expr> args, List<Attribute> kwargs)   TK_APPLY
//       | Select(Expr base, Ident attr_name)                           '.'
//       | Slice(Expr value, Maybe<Expr> first, Maybe<Expr> second)     TK_SLICE
//       | Gather(Expr value, Expr indices)                             TK_GATHER
//       | Var(Ident name)                                              TK_VAR
//       | ListLiteral(List<Expr> inputs)                               TK_LIST_LITERAL
//       | Starred(Expr expr)                                           TK_STARRED
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
      case TK_FOR:
      case TK_WHILE:
      case TK_GLOBAL:
      case TK_ASSIGN:
      case TK_RETURN:
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
      case TK_UNARY_MINUS:
      case '*':
      case TK_STARRED:
      case '/':
      case TK_NOT:
      case TK_CONST:
      case TK_TRUE:
      case TK_FALSE:
      case TK_CAST:
      case TK_APPLY:
      case '.':
      case TK_SLICE:
      case TK_GATHER:
      case TK_VAR:
      case TK_LIST_LITERAL:
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
  List<Stmt> statements() const {
    return List<Stmt>(subtree(2));
  }
  static Def create(
      const SourceRange& range,
      const Ident& name,
      const List<Param>& params,
      const List<Stmt>& stmts) {
    return Def(Compound::create(
        TK_DEF, range, {name, params, stmts}));
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
      const List<Expr>& lhs,
      const AssignKind& reduction,
      const Expr& rhs) {
    return Assign(Compound::create(TK_ASSIGN, range, {lhs, reduction, rhs}));
  }
  List<Expr> lhs() const {
    return List<Expr>(subtree(0));
  }
  int reduction() const {
    return subtree(1)->kind();
  }
  Expr rhs() const {
    return Expr(subtree(2));
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

struct ExprStmt : public Stmt {
  explicit ExprStmt(const TreeRef& tree) : Stmt(tree) {
    tree_->match(TK_EXPR_STMT);
  }
  List<Expr> exprs() {
    return List<Expr>(subtree(0));
  }
  static ExprStmt create(const SourceRange& range, const List<Expr>& list) {
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
    return Expr(Const::create(range(), std::to_string(value)));
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
