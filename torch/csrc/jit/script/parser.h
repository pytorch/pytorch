#pragma once
#include "lexer.h"
#include "tree.h"
#include "tree_views.h"
#include "c10/util/Optional.h"

namespace torch {
namespace jit {
namespace script {


inline Decl mergeTypesFromTypeComment(Decl decl, Decl type_annotation_decl, bool is_method) {
  auto expected_num_annotations = decl.params().size();
  if (is_method) {
    // `self` argument
    expected_num_annotations -= 1;
  }
  if (expected_num_annotations != type_annotation_decl.params().size()) {
    throw ErrorReport(type_annotation_decl.range()) << "Number of type annotations ("
      << type_annotation_decl.params().size() << ") did not match the number of "
      << "function parameters (" << expected_num_annotations << ")";
  }
  auto old = decl.params();
  auto _new = type_annotation_decl.params();
  // Merge signature idents and ranges with annotation types

  std::vector<Param> new_params;
  size_t i = is_method ? 1 : 0;
  size_t j = 0;
  if (is_method) {
    new_params.push_back(old[0]);
  }
  for (; i < decl.params().size(); ++i, ++j) {
    new_params.push_back(Param::create(old[i].range(), old[i].ident(), _new[j].type()));
  }
  return Decl::create(decl.range(), List<Param>::create(decl.range(), new_params), type_annotation_decl.return_type());
}

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

  static bool followsTuple(int kind) {
    switch(kind) {
      case TK_PLUS_EQ:
      case TK_MINUS_EQ:
      case TK_TIMES_EQ:
      case TK_DIV_EQ:
      case TK_NEWLINE:
      case '=':
      case ')':
        return true;
      default:
        return false;
    }
  }

  // exp | expr, | expr, expr, ...
  Expr parseExpOrExpTuple() {
    auto prefix = parseExp();
    if(L.cur().kind == ',') {
      std::vector<Expr> exprs = { prefix };
      while(L.nextIf(',')) {
        if (followsTuple(L.cur().kind))
          break;
        exprs.push_back(parseExp());
      }
      auto list = List<Expr>::create(prefix.range(), exprs);
      prefix = TupleLiteral::create(list.range(), list);
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
      case TK_FALSE:
      case TK_NONE: {
        auto k = L.cur().kind;
        auto r = L.cur().range;
        prefix = c(k, r, {});
        L.next();
      } break;
      case '(': {
        L.next();
        if (L.nextIf(')')) {
          /// here we have the empty tuple case
          std::vector<Expr> vecExpr;
          List<Expr> listExpr = List<Expr>::create(L.cur().range, vecExpr);
          prefix = TupleLiteral::create(L.cur().range, listExpr);
          break;
        }
        prefix = parseExpOrExpTuple();
        L.expect(')');
      } break;
      case '[': {
        auto list = parseList('[', ',', ']', &Parser::parseExp);
        prefix = ListLiteral::create(list.range(), List<Expr>(list));
      } break;
      case TK_STRINGLITERAL: {
        prefix = parseStringLiteral();
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
        prefix = parseSubscript(prefix);
      } else {
        break;
      }
    }
    return prefix;
  }
  TreeRef parseAssignmentOp() {
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
        // not greater than 'precedence'
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

  bool isCharCount(char c, const std::string& str, size_t start, int len) {
    //count checks from [start, start + len)
    return start + len <= str.size() && std::count(str.begin() + start, str.begin() + start + len, c) == len;
  }

  static bool isOctal(char c) {
    return c >= '0' && c < '8';
  }

  c10::optional<char> parseOctal(const std::string& str, size_t pos) {
    if (pos + 3 >= str.size())
      return c10::nullopt;
    size_t c = 0;
    for(size_t i = 0, b = 64; i < 3; ++i, b /= 8) {
      c += b * (str[pos + i] - '0');
    }
    if(c >= 256)
      return c10::nullopt;
    return c;
  }
  std::string parseString(const SourceRange& range, const std::string &str) {
    int quote_len = isCharCount(str[0], str, 0, 3) ? 3 : 1;
    auto ret_str = str.substr(quote_len, str.size() - quote_len * 2);
    size_t pos = ret_str.find('\\');
    while(pos != std::string::npos) {
      //invariant: pos has to escape a character because it is a valid string
      char c = ret_str[pos + 1];
      size_t to_erase = 2;
      switch (ret_str[pos + 1]) {
        case '\\':
        case '\'':
        case '\"':
        case '\n':
          break;
        case 'a':
          c = '\a';
          break;
        case 'b':
          c = '\b';
          break;
        case 'f':
          c = '\f';
          break;
        case 'n':
          c = '\n';
          break;
        case 'v':
          c = '\v';
          break;
        case 'h':
          throw ErrorReport(range)
              << "unsupported hex specifier";
        default:
          // \0NN
          if (auto v = parseOctal(str, pos)) {
            to_erase = 4;
            c = *v;
          } else {
            throw ErrorReport(range)
                << " ill formed octal specifier";
          }
      }
      ret_str.replace(pos, to_erase, /* num copies */ 1, c);
      pos = ret_str.find('\\', pos + 1);
    }
    return ret_str;
  }

  StringLiteral parseStringLiteral() {
    auto range = L.cur().range;
    std::stringstream ss;
    while(L.cur().kind == TK_STRINGLITERAL) {
      auto literal_range = L.cur().range;
      ss << parseString(literal_range, L.next().text());
    }
    return StringLiteral::create(range, ss.str());
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

  // Parse expr's of the form [a:], [:b], [a:b], [:]
  Expr parseSubscriptExp() {
    TreeRef first, second;
    auto range = L.cur().range;
    if (L.cur().kind != ':') {
      first = parseExp();
    }
    if (L.nextIf(':')) {
      if (L.cur().kind != ',' && L.cur().kind != ']') {
        second = parseExp();
      }
      auto maybe_first = first ? Maybe<Expr>::create(range, Expr(first)) : Maybe<Expr>::create(range);
      auto maybe_second = second ? Maybe<Expr>::create(range, Expr(second)) : Maybe<Expr>::create(range);
      return SliceExpr::create(range, maybe_first, maybe_second);
    } else {
      return Expr(first);
    }
  }

  TreeRef parseSubscript(TreeRef value) {
    const auto range = L.cur().range;

    auto subscript_exprs = parseList('[', ',', ']', &Parser::parseSubscriptExp);
    return Subscript::create(range, Expr(value), subscript_exprs);
  }

  TreeRef parseParam() {
    auto ident = parseIdent();
    TreeRef type;
    if (L.nextIf(':')) {
      type = parseExp();
    } else {
      type = Var::create(L.cur().range, Ident::create(L.cur().range, "Tensor"));
    }
    return Param::create(type->range(), Ident(ident), Expr(type));
  }

  Param parseBareTypeAnnotation() {
    auto type = parseExp();
    return Param::create(type.range(), Ident::create(type.range(), ""), type);
  }

  TreeRef parseTypeComment(bool parse_full_line=false) {
    auto range = L.cur().range;
    if (parse_full_line) {
      L.expect(TK_TYPE_COMMENT);
    }
    auto param_types = parseList('(', ',', ')', &Parser::parseBareTypeAnnotation);
    TreeRef return_type;
    if (L.nextIf(TK_ARROW)) {
      auto return_type_range = L.cur().range;
      return_type = Maybe<Expr>::create(return_type_range, parseExp());
    } else {
      return_type = Maybe<Expr>::create(L.cur().range);
    }
    if (!parse_full_line)
      L.expect(TK_NEWLINE);
    return Decl::create(range, param_types, Maybe<Expr>(return_type));
  }

  // 'first' has already been parsed since expressions can exist
  // alone on a line:
  // first[,other,lhs] = rhs
  TreeRef parseAssign(Expr lhs) {
    auto op = parseAssignmentOp();
    auto rhs = parseExpOrExpTuple();
    L.expect(TK_NEWLINE);
    if (op->kind() == '=') {
      return Assign::create(lhs.range(), lhs, Expr(rhs));
    } else {
      // this is an augmented assignment
      if (lhs.kind() == TK_TUPLE_LITERAL) {
        throw ErrorReport(lhs.range())
            << " augmented assignment can only have one LHS expression";
      }
      return AugAssign::create(
          lhs.range(), lhs, AugAssignKind(op), Expr(rhs));
    }
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
      case TK_RAISE: {
        auto range = L.next().range;
        auto expr = parseExp();
        L.expect(TK_NEWLINE);
        return Raise::create(range, expr);
      }
      case TK_ASSERT: {
        auto range = L.next().range;
        auto cond = parseExp();
        Maybe<Expr> maybe_first = Maybe<Expr>::create(range);
        if (L.nextIf(','))  {
          auto msg = parseExp();
          maybe_first = Maybe<Expr>::create(range, Expr(msg));
        }
        L.expect(TK_NEWLINE);
        return Assert::create(range, cond, maybe_first);
      }
      case TK_PASS: {
        auto range = L.next().range;
        L.expect(TK_NEWLINE);
        return Pass::create(range);
      }
      default: {
        auto lhs = parseExpOrExpTuple();
        if (L.cur().kind != TK_NEWLINE) {
          return parseAssign(lhs);
        } else {
          L.expect(TK_NEWLINE);
          return ExprStmt::create(lhs.range(), lhs);
        }
      }
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
  TreeRef parseIf(bool expect_if=true) {
    auto r = L.cur().range;
    if (expect_if)
      L.expect(TK_IF);
    auto cond = parseExp();
    L.expect(':');
    auto true_branch = parseStatements();
    auto false_branch = makeList(L.cur().range, {});
    if (L.nextIf(TK_ELSE)) {
      L.expect(':');
      false_branch = parseStatements();
    } else if (L.nextIf(TK_ELIF)) {
      // NB: this needs to be a separate statement, since the call to parseIf
      // mutates the lexer state, and thus causes a heap-use-after-free in
      // compilers which evaluate argument expressions LTR
      auto range = L.cur().range;
      false_branch = makeList(range, {parseIf(false)});
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

  TreeRef parseStatements(bool expect_indent=true) {
    auto r = L.cur().range;
    if (expect_indent)
      L.expect(TK_INDENT);
    TreeList stmts;
    for (size_t i=0; ; ++i) {
      auto stmt = parseStmt();
      stmts.push_back(stmt);
      if (L.nextIf(TK_DEDENT))
        break;
    }
    return c(TK_LIST, r, std::move(stmts));
  }
  Decl parseDecl() {
    auto paramlist = parseList('(', ',', ')', &Parser::parseParam);
    // Parse return type annotation
    TreeRef return_type;
    if (L.nextIf(TK_ARROW)) {
      // Exactly one expression for return type annotation
      auto return_type_range = L.cur().range;
      return_type = Maybe<Expr>::create(return_type_range, parseExp());
    } else {
      // Default to returning single tensor. TODO: better sentinel value?
      return_type = Maybe<Expr>::create(L.cur().range);
    }
    L.expect(':');
    return Decl::create(paramlist.range(), List<Param>(paramlist), Maybe<Expr>(return_type));
  }

  TreeRef parseFunction(bool is_method) {
    L.expect(TK_DEF);
    auto name = parseIdent();
    auto decl = parseDecl();

    // Handle type annotations specified in a type comment as the first line of
    // the function.
    L.expect(TK_INDENT);
    if (L.nextIf(TK_TYPE_COMMENT)) {
      auto type_annotation_decl = Decl(parseTypeComment());
      decl = mergeTypesFromTypeComment(decl, type_annotation_decl, is_method);
    }

    auto stmts_list = parseStatements(false);
    return Def::create(name.range(), Ident(name), Decl(decl),
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
