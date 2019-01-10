#pragma once
#include "lexer.h"
#include "tree.h"
#include "tree_views.h"

namespace torch {
namespace jit {
namespace script {

struct Parser {
  explicit Parser(const std::string& str)
      : L(str), shared(sharedParserData()) {}

  Ident parseIdent() {
    auto t = L.expect(TK_IDENT);
    // whenever we parse something that has a TreeView type we always
    // use its create method so that the accessors and the constructor
    // of the Compound tree are in the same place.
    return Ident::create(t.range, t.text());
  }
  TreeRef createApply(Expr expr) {
    TreeList attributes;
    auto range = L.cur().range;
    TreeList inputs;
    parseOperatorArguments(inputs, attributes);
    return Apply::create(
        range,
        expr,
        List<Expr>(makeList(range, std::move(inputs))),
        List<Attribute>(makeList(range, std::move(attributes))));
  }
  // exp | expr, | expr, expr, ...
  TreeRef parseExpOrExpList(int end) {
    auto prefix = parseExp();
    if(L.cur().kind == ',') {
      std::vector<Expr> exprs = { prefix };
      while(L.cur().kind != end) {
        L.expect(',');
        exprs.push_back(parseExp());
      }
      auto list = List<Expr>::create(prefix.range(), exprs);
      prefix = ListLiteral::create(list.range(), list);
    }
    return prefix;
  }
  // things like a 1.0 or a(4) that are not unary/binary expressions
  // and have higher precedence than all of them
  TreeRef parseBaseExp() {
    TreeRef prefix;
    switch (L.cur().kind) {
      case TK_NUMBER: {
        prefix = parseConst();
      } break;
      case TK_TRUE:
      case TK_FALSE: {
        auto k = L.cur().kind;
        auto r = L.cur().range;
        prefix = c(k, r, {});
        L.next();
      } break;
      case '(': {
        L.next();
        prefix = parseExpOrExpList(')');
        L.expect(')');
      } break;
      case '[': {
        auto list = parseList('[', ',', ']', &Parser::parseExp);
        prefix = ListLiteral::create(list.range(), List<Expr>(list));
      } break;
      case TK_FLOAT:
      case TK_INT:
      case TK_LONG: {
        auto r = L.cur().range;
        auto type = c(L.next().kind, r, {});
        L.expect('(');
        auto exp = parseExp();
        L.expect(')');
        prefix = Cast::create(r, Type(type), Expr(exp));
      } break;
      default: {
        Ident name = parseIdent();
        prefix = Var::create(name.range(), name);
      } break;
    }
    while (true) {
      if (L.nextIf('.')) {
        const auto name = parseIdent();
        prefix = Select::create(name.range(), Expr(prefix), Ident(name));
      } else if (L.cur().kind == '(') {
        prefix = createApply(Expr(prefix));
      } else if (L.cur().kind == '[') {
        prefix = parseSliceOrGather(prefix);
      } else {
        break;
      }
    }
    return prefix;
  }
  TreeRef parseOptionalReduction() {
    auto r = L.cur().range;
    switch (L.cur().kind) {
      case TK_PLUS_EQ:
      case TK_MINUS_EQ:
      case TK_TIMES_EQ:
      case TK_DIV_EQ: {
        int modifier = L.next().text()[0];
        return c(modifier, r, {});
      } break;
      default: {
        L.expect('=');
        return c('=', r, {}); // no reduction
      } break;
    }
  }
  TreeRef
  parseTrinary(TreeRef true_branch, const SourceRange& range, int binary_prec) {
    auto cond = parseExp();
    L.expect(TK_ELSE);
    auto false_branch = parseExp(binary_prec);
    return c(TK_IF_EXPR, range, {cond, true_branch, false_branch});
  }
  // parse the longest expression whose binary operators have
  // precedence strictly greater than 'precedence'
  // precedence == 0 will parse _all_ expressions
  // this is the core loop of 'top-down precedence parsing'
  Expr parseExp() { return parseExp(0); }
  Expr parseExp(int precedence) {
    TreeRef prefix = nullptr;
    int unary_prec;
    if (shared.isUnary(L.cur().kind, &unary_prec)) {
      auto kind = L.cur().kind;
      auto pos = L.cur().range;
      L.next();
      auto unary_kind = kind == '*' ? TK_STARRED :
                        kind == '-' ? TK_UNARY_MINUS :
                                      kind;
      auto subexp = parseExp(unary_prec);
      // fold '-' into constant numbers, so that attributes can accept
      // things like -1
      if(unary_kind == TK_UNARY_MINUS && subexp.kind() == TK_CONST) {
        prefix = Const::create(subexp.range(), "-" + Const(subexp).text());
      } else {
        prefix = c(unary_kind, pos, {subexp});
      }
    } else {
      prefix = parseBaseExp();
    }
    int binary_prec;
    while (shared.isBinary(L.cur().kind, &binary_prec)) {
      if (binary_prec <= precedence) // not allowed to parse something which is
        // not greater than 'precedenc'
        break;

      int kind = L.cur().kind;
      auto pos = L.cur().range;
      L.next();
      if (shared.isRightAssociative(kind))
        binary_prec--;

      // special case for trinary operator
      if (kind == TK_IF) {
        prefix = parseTrinary(prefix, pos, binary_prec);
        continue;
      }

      prefix = c(kind, pos, {prefix, parseExp(binary_prec)});
    }
    return Expr(prefix);
  }
  template<typename T>
  List<T> parseList(int begin, int sep, int end, T (Parser::*parse)()) {
    auto r = L.cur().range;
    if (begin != TK_NOTHING)
      L.expect(begin);
    std::vector<T> elements;
    if (L.cur().kind != end) {
      do {
        elements.push_back((this->*parse)());
      } while (L.nextIf(sep));
    }
    if (end != TK_NOTHING)
      L.expect(end);
    return List<T>::create(r, elements);
  }
  Const parseConst() {
    auto range = L.cur().range;
    auto t = L.expect(TK_NUMBER);
    return Const::create(t.range, t.text());
  }
  Expr parseAttributeValue() {
    return parseExp();
  }
  void parseOperatorArguments(TreeList& inputs, TreeList& attributes) {
    L.expect('(');
    if (L.cur().kind != ')') {
      do {
        if (L.cur().kind == TK_IDENT && L.lookahead().kind == '=') {
          auto ident = parseIdent();
          L.expect('=');
          auto v = parseAttributeValue();
          attributes.push_back(Attribute::create(ident.range(), Ident(ident), v));
        } else {
          inputs.push_back(parseExp());
        }
      } while (L.nextIf(','));
    }
    L.expect(')');
  }

  // OK: [a] (gather), [a:], [:a], [a:b], [:] (slice)
  // Not OK: []
  TreeRef parseSliceOrGather(TreeRef value) {
    const auto range = L.cur().range;
    L.expect('[');

    // `first` will either be the gather indices, or the start of the slice.
    TreeRef first, second;

    // Here we can either have a colon (which starts a slice), or an expression.
    // If an expression, we don't know yet if it will be a slice or a gather.
    if (L.cur().kind != ':') {
      first = parseExp();
      if (L.nextIf(']')) {
        return Gather::create(range, Expr(value), Expr(first));
      } else {
        first = c(TK_OPTION, range, {first});
      }
    } else {
      first = c(TK_OPTION, range, {});
    }
    L.expect(':');
    // Now we *may* have an expression.
    if (L.cur().kind != ']') {
      second = c(TK_OPTION, range, {parseExp()});
    } else {
      second = c(TK_OPTION, range, {});
    }
    L.expect(']');

    return Slice::create(range, Expr(value), Maybe<Expr>(first), Maybe<Expr>(second));
  }
  TreeRef parseParam() {
    auto typ = parseType();
    if (L.cur().kind != TK_IDENT && typ->trees()[0]->kind() == TK_IDENT) {
      // oops, it wasn't a type but just a param without any type specified
      return Param::create(
          typ->range(), Ident(typ->trees()[0]), Type(c(TK_INFERRED, typ->range(), {})));
    }
    auto ident = parseIdent();
    return Param::create(typ->range(), Ident(ident), Type(typ));
  }

  // 'first' has already been parsed since expressions can exist
  // alone on a line:
  // first[,other,lhs] = rhs
  Assign parseAssign(List<Expr> list) {
    auto red = parseOptionalReduction();
    auto rhs = parseExpOrExpList(TK_NEWLINE);
    L.expect(TK_NEWLINE);
    return Assign::create(list.range(), list, AssignKind(red), Expr(rhs));
  }
  TreeRef parseStmt() {
    switch (L.cur().kind) {
      case TK_IF:
        return parseIf();
      case TK_WHILE:
        return parseWhile();
      case TK_FOR:
        return parseFor();
      case TK_GLOBAL: {
        auto range = L.next().range;
        auto idents = parseList(TK_NOTHING, ',', TK_NOTHING, &Parser::parseIdent);
        L.expect(TK_NEWLINE);
        return Global::create(range, idents);
      }
      case TK_RETURN: {
        auto range = L.next().range;
        // XXX: TK_NEWLINE makes it accept an empty list
        auto values = parseList(TK_NOTHING, ',', TK_NEWLINE, &Parser::parseExp);
        return Return::create(range, values);
      }
      default: {
        List<Expr> exprs = parseList(TK_NOTHING, ',', TK_NOTHING, &Parser::parseExp);
        if (L.cur().kind != TK_NEWLINE) {
          return parseAssign(exprs);
        } else {
          L.expect(TK_NEWLINE);
          return ExprStmt::create(exprs[0].range(), exprs);
        }
      }
    }
  }
  TreeRef parseScalarType() {
    switch (L.cur().kind) {
      case TK_INT:
      case TK_FLOAT:
      case TK_LONG:
      case TK_DOUBLE: {
        auto t = L.next();
        return c(t.kind, t.range, {});
      }
      default:
        return parseIdent();
    }
  }
  TreeRef parseOptionalIdentList() {
    TreeRef list = nullptr;
    if (L.cur().kind == '(') {
      list = parseList('(', ',', ')', &Parser::parseIdent);
    } else {
      list = c(TK_LIST, L.cur().range, {});
    }
    return list;
  }
  TreeRef parseType() {
    return TensorType::create(SourceRange(std::make_shared<std::string>(""), 0, 0));
  }
  TreeRef parseIf() {
    auto r = L.cur().range;
    L.expect(TK_IF);
    auto cond = parseExp();
    L.expect(':');
    auto true_branch = parseStatements();
    auto false_branch = makeList(L.cur().range, {});
    if (L.nextIf(TK_ELSE)) {
      L.expect(':');
      false_branch = parseStatements();
    }
    return If::create(r, Expr(cond), List<Stmt>(true_branch), List<Stmt>(false_branch));
  }
  TreeRef parseWhile() {
    auto r = L.cur().range;
    L.expect(TK_WHILE);
    auto cond = parseExp();
    L.expect(':');
    auto body = parseStatements();
    return While::create(r, Expr(cond), List<Stmt>(body));
  }
  TreeRef parseFor() {
    auto r = L.cur().range;
    L.expect(TK_FOR);
    auto targets = parseList(TK_NOTHING, ',', TK_NOTHING, &Parser::parseExp);
    L.expect(TK_IN);
    auto itrs = parseList(TK_NOTHING, ',', TK_NOTHING, &Parser::parseExp);
    L.expect(':');
    auto body = parseStatements();
    return For::create(r, targets, itrs, body);
  }
  TreeRef parseStatements() {
    auto r = L.cur().range;
    L.expect(TK_INDENT);
    TreeList stmts;
    while (true) {
      stmts.push_back(parseStmt());
      if (L.nextIf(TK_DEDENT))
        break;
    }
    return c(TK_LIST, r, std::move(stmts));
  }
  TreeRef parseFunction() {
    L.expect(TK_DEF);
    auto name = parseIdent();
    auto paramlist = parseList('(', ',', ')', &Parser::parseParam);
    L.expect(':');
    auto stmts_list = parseStatements();
    return Def::create(name.range(), Ident(name), List<Param>(paramlist),
                       List<Stmt>(stmts_list));
  }
  Lexer& lexer() {
    return L;
  }

 private:
  // short helpers to create nodes
  TreeRef c(int kind, const SourceRange& range, TreeList&& trees) {
    return Compound::create(kind, range, std::move(trees));
  }
  TreeRef makeList(const SourceRange& range, TreeList&& trees) {
    return c(TK_LIST, range, std::move(trees));
  }
  Lexer L;
  SharedParserData& shared;
};
} // namespace script
} // namespace jit
} // namespace torch
