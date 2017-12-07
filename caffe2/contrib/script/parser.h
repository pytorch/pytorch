#pragma once
#include "lexer.h"
#include "tree.h"
#include "tree_views.h"

namespace caffe2 {
namespace script {

struct Parser {
  explicit Parser(const std::string& str)
      : L(str), shared(sharedParserData()) {}

  TreeRef parseIdent() {
    auto t = L.expect(TK_IDENT);
    // whenever we parse something that has a TreeView type we always
    // use its create method so that the accessors and the constructor
    // of the Compound tree are in the same place.
    return Ident::create(t.range, t.text());
  }
  TreeRef createApply(TreeRef ident, TreeList& inputs) {
    TreeList attributes;
    auto range = L.cur().range;
    parseOperatorArguments(inputs, attributes);
    return Apply::create(
        range,
        ident,
        List(range, std::move(inputs)),
        List(range, std::move(attributes)));
  }
  // things like a 1.0 or a(4) that are not unary/binary expressions
  // and have higher precedence than all of them
  TreeRef parseBaseExp() {
    TreeRef prefix;
    switch (L.cur().kind) {
      case TK_NUMBER:
      case TK_TRUE:
      case TK_FALSE: {
        prefix = parseConst();
      } break;
      case '(': {
        L.next();
        prefix = parseExp();
        L.expect(')');
      } break;
      case TK_FLOAT:
      case TK_INT:
      case TK_LONG: {
        auto r = L.cur().range;
        auto type = c(L.next().kind, r, {});
        L.expect('(');
        auto exp = parseExp();
        L.expect(')');
        prefix = Cast::create(r, type, exp);
      } break;
      default: {
        prefix = parseIdent();
        if (L.cur().kind == '(') {
          TreeList inputs;
          prefix = createApply(prefix, inputs);
        }
      } break;
    }
    while (L.nextIf('.')) {
      auto name = parseIdent();
      TreeList inputs = {prefix};
      prefix = createApply(name, inputs);
    }
    return prefix;
  }
  TreeRef parseOptionalReduction() {
    auto r = L.cur().range;
    switch (L.cur().kind) {
      case '+':
      case '*':
      case TK_MIN:
      case TK_MAX:
        return c(L.next().kind, r, {});
      default:
        return c('=', r, {}); // no reduction
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
  TreeRef parseExp(int precedence = 0) {
    TreeRef prefix = nullptr;
    int unary_prec;
    if (shared.isUnary(L.cur().kind, &unary_prec)) {
      auto kind = L.cur().kind;
      auto pos = L.cur().range;
      L.next();
      prefix = c(kind, pos, {parseExp(unary_prec)});
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
    return prefix;
  }
  TreeRef
  parseList(int begin, int sep, int end, std::function<TreeRef(int)> parse) {
    auto r = L.cur().range;
    L.expect(begin);
    TreeList elements;
    if (L.cur().kind != end) {
      int i = 0;
      do {
        elements.push_back(parse(i++));
      } while (L.nextIf(sep));
    }
    L.expect(end);
    return c(TK_LIST, r, std::move(elements));
  }
  TreeRef parseNonEmptyList(int sep, std::function<TreeRef(int)> parse) {
    TreeList elements;
    int i = 0;
    do {
      elements.push_back(parse(i++));
    } while (L.nextIf(sep));
    return c(TK_LIST, elements[0]->range(), std::move(elements));
  }
  TreeRef parseExpList() {
    return parseList('(', ',', ')', [&](int i) { return parseExp(); });
  }
  TreeRef parseConst() {
    // 'b' - boolean
    // 'LL' 64-bit integer
    // 'f' single-precision float
    // 'i' 32-bit integer
    // 'f' is default if '.' appears in the number
    auto range = L.cur().range;
    if (L.nextIf(TK_TRUE)) {
      return c(TK_CONST, range, {d(1), s("b")});
    } else if (L.nextIf(TK_FALSE)) {
      return c(TK_CONST, range, {d(0), s("b")});
    }
    float mult = 1.0f;
    while (L.nextIf('-')) {
      mult *= -1.0f;
    }
    auto t = L.expect(TK_NUMBER);
    std::string type_ident =
        (t.text().find('.') == std::string::npos) ? "i" : "f";
    if (L.cur().kind == TK_IDENT) {
      Token type_ident_tok = L.expect(TK_IDENT);
      type_ident = type_ident_tok.text();
      if (type_ident != "LL" && type_ident != "f") {
        throw ErrorReport(type_ident_tok)
            << "expected 'f' or 'LL' "
            << "as numeric type identifier but found '" << type_ident << "'";
      }
    }
    return c(TK_CONST, t.range, {d(mult * t.doubleValue()), s(type_ident)});
  }
  TreeRef parseAttributeValue() {
    int kind = L.cur().kind;
    switch (kind) {
      case '[':
        return parseList('[', ',', ']', [&](int i) { return parseConst(); });
      default:
        return parseConst();
    }
  }
  void parseOperatorArguments(TreeList& inputs, TreeList& attributes) {
    L.expect('(');
    if (L.cur().kind != ')') {
      do {
        if (L.cur().kind == TK_IDENT && L.lookahead().kind == '=') {
          auto ident = parseIdent();
          L.expect('=');
          auto v = parseAttributeValue();
          attributes.push_back(Attribute::create(ident->range(), ident, v));
        } else {
          inputs.push_back(parseExp());
        }
      } while (L.nextIf(','));
    }
    L.expect(')');
  }
  TreeRef parseIdentList() {
    return parseList('(', ',', ')', [&](int i) { return parseIdent(); });
  }
  TreeRef parseParam() {
    auto typ = parseType();
    if (L.cur().kind != TK_IDENT && typ->trees()[0]->kind() == TK_IDENT) {
      // oops, it wasn't a type but just a param without any type specified
      return Param::create(
          typ->range(), typ->trees()[0], c(TK_INFERRED, typ->range(), {}));
    }
    auto ident = parseIdent();
    return Param::create(typ->range(), ident, typ);
  }
  void parseEndOfLine() {
    // TODO: make sure parser always generates newline before dedent
    if (L.cur().kind != TK_DEDENT)
      L.expect(TK_NEWLINE);
  }
  TreeRef parseAssign() {
    TreeRef list = parseOneOrMoreIdent();
    auto red = parseOptionalReduction();
    L.expect('=');
    auto rhs = parseExp();
    parseEndOfLine();
    return Assign::create(list->range(), list, red, rhs);
  }
  TreeRef parseStmt() {
    switch (L.cur().kind) {
      case TK_IF:
        return parseIf();
      case TK_WHILE:
        return parseWhile();
      default: {
        // TODO: when we have type-checking/semantic analysis,
        // we should just parse <exp>, <exp> = <exp>, <exp>
        // rather than use lookahead here.
        // but for now allow statements of the form foo(...)
        // and a.foo(...) to appear on lines of their own.
        int lookahead = L.lookahead().kind;
        if (lookahead == '(' || lookahead == '.') {
          auto r = parseExp();
          parseEndOfLine();
          return r;
        } else {
          return parseAssign();
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
      list = parseIdentList();
    } else {
      list = c(TK_LIST, L.cur().range, {});
    }
    return list;
  }
  TreeRef parseType() {
    auto st = parseScalarType();
    auto list = parseOptionalIdentList();
    return TensorType::create(st->range(), st, list);
  }
  TreeRef parseOneOrMoreIdent() {
    TreeList list;
    do {
      list.push_back(parseIdent());
    } while (L.nextIf(','));
    return List(list.back()->range(), std::move(list));
  }
  TreeRef parseIf() {
    auto r = L.cur().range;
    L.expect(TK_IF);
    auto cond = parseExp();
    L.expect(':');
    auto true_branch = parseStatements();
    auto false_branch = List(L.cur().range, {});
    if (L.nextIf(TK_ELSE)) {
      L.expect(':');
      false_branch = parseStatements();
    }
    return If::create(r, cond, true_branch, false_branch);
  }
  TreeRef parseWhile() {
    auto r = L.cur().range;
    L.expect(TK_WHILE);
    auto cond = parseExp();
    L.expect(':');
    auto body = parseStatements();
    return While::create(r, cond, body);
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
    auto paramlist =
        parseList('(', ',', ')', [&](int i) { return parseParam(); });
    L.expect(TK_ARROW);
    auto retlist =
        parseList('(', ',', ')', [&](int i) { return parseParam(); });
    L.expect(':');
    auto stmts_list = parseStatements();
    return Def::create(name->range(), name, paramlist, retlist, stmts_list);
  }
  Lexer& lexer() {
    return L;
  }

 private:
  // short helpers to create nodes
  TreeRef d(double v) {
    return Number::create(v);
  }
  TreeRef s(const std::string& s) {
    return String::create(s);
  }
  TreeRef c(int kind, const SourceRange& range, TreeList&& trees) {
    return Compound::create(kind, range, std::move(trees));
  }
  TreeRef List(const SourceRange& range, TreeList&& trees) {
    return c(TK_LIST, range, std::move(trees));
  }
  Lexer L;
  SharedParserData& shared;
};
} // namespace script
} // namespace caffe2
