#include <torch/csrc/jit/script/parser.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/script/lexer.h>
#include <torch/csrc/jit/script/parse_string_literal.h>
#include <torch/csrc/jit/script/tree.h>
#include <torch/csrc/jit/script/tree_views.h>

namespace torch {
namespace jit {
namespace script {

Decl mergeTypesFromTypeComment(
    const Decl& decl,
    const Decl& type_annotation_decl,
    bool is_method) {
  auto expected_num_annotations = decl.params().size();
  if (is_method) {
    // `self` argument
    expected_num_annotations -= 1;
  }
  if (expected_num_annotations != type_annotation_decl.params().size()) {
    throw ErrorReport(type_annotation_decl.range())
        << "Number of type annotations ("
        << type_annotation_decl.params().size()
        << ") did not match the number of "
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
    new_params.emplace_back(old[i].withType(_new[j].type()));
  }
  return Decl::create(
      decl.range(),
      List<Param>::create(decl.range(), new_params),
      type_annotation_decl.return_type());
}

struct ParserImpl {
  explicit ParserImpl(const std::string& str)
      : L(str), shared(sharedParserData()) {}

  Ident parseIdent() {
    auto t = L.expect(TK_IDENT);
    // whenever we parse something that has a TreeView type we always
    // use its create method so that the accessors and the constructor
    // of the Compound tree are in the same place.
    return Ident::create(t.range, t.text());
  }
  TreeRef createApply(const Expr& expr) {
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
    switch (kind) {
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
    if (L.cur().kind == ',') {
      std::vector<Expr> exprs = {prefix};
      while (L.nextIf(',')) {
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
        auto list = parseList('[', ',', ']', &ParserImpl::parseExp);
        prefix = ListLiteral::create(list.range(), List<Expr>(list));
      } break;
      case '{': {
        L.next();
        std::vector<Expr> keys;
        std::vector<Expr> values;
        auto range = L.cur().range;
        if (L.cur().kind != '}') {
          do {
            keys.push_back(parseExp());
            L.expect(':');
            values.push_back(parseExp());
          } while (L.nextIf(','));
        }
        L.expect('}');
        prefix = DictLiteral::create(
            range,
            List<Expr>::create(range, keys),
            List<Expr>::create(range, values));
      } break;
      case TK_STRINGLITERAL: {
        prefix = parseConcatenatedStringLiterals();
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
  TreeRef parseTrinary(
      TreeRef true_branch,
      const SourceRange& range,
      int binary_prec) {
    auto cond = parseExp();
    L.expect(TK_ELSE);
    auto false_branch = parseExp(binary_prec);
    return c(TK_IF_EXPR, range, {cond, std::move(true_branch), false_branch});
  }
  // parse the longest expression whose binary operators have
  // precedence strictly greater than 'precedence'
  // precedence == 0 will parse _all_ expressions
  // this is the core loop of 'top-down precedence parsing'
  Expr parseExp() {
    return parseExp(0);
  }
  Expr parseExp(int precedence) {
    TreeRef prefix = nullptr;
    int unary_prec;
    if (shared.isUnary(L.cur().kind, &unary_prec)) {
      auto kind = L.cur().kind;
      auto pos = L.cur().range;
      L.next();
      auto unary_kind =
          kind == '*' ? TK_STARRED : kind == '-' ? TK_UNARY_MINUS : kind;
      auto subexp = parseExp(unary_prec);
      // fold '-' into constant numbers, so that attributes can accept
      // things like -1
      if (unary_kind == TK_UNARY_MINUS && subexp.kind() == TK_CONST) {
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
  void parseSequence(
      int begin,
      int sep,
      int end,
      const std::function<void()>& parse) {
    if (begin != TK_NOTHING)
      L.expect(begin);
    if (L.cur().kind != end) {
      do {
        parse();
      } while (L.nextIf(sep));
    }
    if (end != TK_NOTHING)
      L.expect(end);
  }
  template <typename T>
  List<T> parseList(int begin, int sep, int end, T (ParserImpl::*parse)()) {
    auto r = L.cur().range;
    std::vector<T> elements;
    parseSequence(
        begin, sep, end, [&] { elements.emplace_back((this->*parse)()); });
    return List<T>::create(r, elements);
  }

  Const parseConst() {
    auto range = L.cur().range;
    auto t = L.expect(TK_NUMBER);
    return Const::create(t.range, t.text());
  }

  StringLiteral parseConcatenatedStringLiterals() {
    auto range = L.cur().range;
    std::stringstream ss;
    while (L.cur().kind == TK_STRINGLITERAL) {
      auto literal_range = L.cur().range;
      ss << parseStringLiteral(literal_range, L.next().text());
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
          attributes.push_back(
              Attribute::create(ident.range(), Ident(ident), v));
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
      auto maybe_first = first ? Maybe<Expr>::create(range, Expr(first))
                               : Maybe<Expr>::create(range);
      auto maybe_second = second ? Maybe<Expr>::create(range, Expr(second))
                                 : Maybe<Expr>::create(range);
      return SliceExpr::create(range, maybe_first, maybe_second);
    } else {
      return Expr(first);
    }
  }

  TreeRef parseSubscript(const TreeRef& value) {
    const auto range = L.cur().range;

    auto subscript_exprs =
        parseList('[', ',', ']', &ParserImpl::parseSubscriptExp);
    return Subscript::create(range, Expr(value), subscript_exprs);
  }

  TreeRef parseParam(bool kwarg_only) {
    auto ident = parseIdent();
    TreeRef type;
    if (L.nextIf(':')) {
      type = parseExp();
    } else {
      type = Var::create(L.cur().range, Ident::create(L.cur().range, "Tensor"));
    }
    TreeRef def;
    if (L.nextIf('=')) {
      def = Maybe<Expr>::create(L.cur().range, parseExp());
    } else {
      def = Maybe<Expr>::create(L.cur().range);
    }
    return Param::create(
        type->range(), Ident(ident), Expr(type), Maybe<Expr>(def), kwarg_only);
  }

  Param parseBareTypeAnnotation() {
    auto type = parseExp();
    return Param::create(
        type.range(),
        Ident::create(type.range(), ""),
        type,
        Maybe<Expr>::create(type.range()),
        /*kwarg_only=*/false);
  }

  Decl parseTypeComment() {
    auto range = L.cur().range;
    L.expect(TK_TYPE_COMMENT);
    auto param_types =
        parseList('(', ',', ')', &ParserImpl::parseBareTypeAnnotation);
    TreeRef return_type;
    if (L.nextIf(TK_ARROW)) {
      auto return_type_range = L.cur().range;
      return_type = Maybe<Expr>::create(return_type_range, parseExp());
    } else {
      return_type = Maybe<Expr>::create(L.cur().range);
    }
    return Decl::create(range, param_types, Maybe<Expr>(return_type));
  }

  // 'first' has already been parsed since expressions can exist
  // alone on a line:
  // first[,other,lhs] = rhs
  TreeRef parseAssign(const Expr& lhs) {
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
      return AugAssign::create(lhs.range(), lhs, AugAssignKind(op), Expr(rhs));
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
        auto idents =
            parseList(TK_NOTHING, ',', TK_NOTHING, &ParserImpl::parseIdent);
        L.expect(TK_NEWLINE);
        return Global::create(range, idents);
      }
      case TK_RETURN: {
        auto range = L.next().range;
        Expr value = L.cur().kind != TK_NEWLINE ? parseExpOrExpTuple()
                                                : Expr(c(TK_NONE, range, {}));
        L.expect(TK_NEWLINE);
        return Return::create(range, value);
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
        if (L.nextIf(',')) {
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
      case TK_DEF: {
        return parseFunction(/*is_method=*/false);
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
      list = parseList('(', ',', ')', &ParserImpl::parseIdent);
    } else {
      list = c(TK_LIST, L.cur().range, {});
    }
    return list;
  }
  TreeRef parseIf(bool expect_if = true) {
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
    return If::create(
        r, Expr(cond), List<Stmt>(true_branch), List<Stmt>(false_branch));
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
    auto targets =
        parseList(TK_NOTHING, ',', TK_NOTHING, &ParserImpl::parseExp);
    L.expect(TK_IN);
    auto itrs = parseList(TK_NOTHING, ',', TK_NOTHING, &ParserImpl::parseExp);
    L.expect(':');
    auto body = parseStatements();
    return For::create(r, targets, itrs, body);
  }

  TreeRef parseStatements(bool expect_indent = true) {
    auto r = L.cur().range;
    if (expect_indent) {
      L.expect(TK_INDENT);
    }
    TreeList stmts;
    do {
      stmts.push_back(parseStmt());
    } while (!L.nextIf(TK_DEDENT));
    return c(TK_LIST, r, std::move(stmts));
  }

  Maybe<Expr> parseReturnAnnotation() {
    if (L.nextIf(TK_ARROW)) {
      // Exactly one expression for return type annotation
      auto return_type_range = L.cur().range;
      return Maybe<Expr>::create(return_type_range, parseExp());
    } else {
      return Maybe<Expr>::create(L.cur().range);
    }
  }

  List<Param> parseParams() {
    auto r = L.cur().range;
    std::vector<Param> params;
    bool kwarg_only = false;
    parseSequence('(', ',', ')', [&] {
      if (!kwarg_only && L.nextIf('*')) {
        kwarg_only = true;
      } else {
        params.emplace_back(parseParam(kwarg_only));
      }
    });
    return List<Param>::create(r, params);
  }
  Decl parseDecl() {
    // Parse return type annotation
    List<Param> paramlist = parseParams();
    TreeRef return_type;
    Maybe<Expr> return_annotation = parseReturnAnnotation();
    L.expect(':');
    return Decl::create(
        paramlist.range(), List<Param>(paramlist), return_annotation);
  }

  TreeRef parseClass() {
    L.expect(TK_CLASS_DEF);
    const auto name = parseIdent();
    if (L.nextIf('(')) {
      // The parser only supports py3 syntax, so classes are new-style when
      // they don't inherit from anything.
      L.reportError(
          "Inheritance is not yet supported for TorchScript classes yet.");
    }
    L.expect(':');

    L.expect(TK_INDENT);
    std::vector<Def> methods;
    while (L.cur().kind != TK_DEDENT) {
      methods.push_back(Def(parseFunction(/*is_method=*/true)));
    }
    L.expect(TK_DEDENT);

    return ClassDef::create(
        name.range(), name, List<Def>::create(name.range(), methods));
  }

  TreeRef parseFunction(bool is_method) {
    L.expect(TK_DEF);
    auto name = parseIdent();
    auto decl = parseDecl();

    // Handle type annotations specified in a type comment as the first line of
    // the function.
    L.expect(TK_INDENT);
    if (L.cur().kind == TK_TYPE_COMMENT) {
      auto type_annotation_decl = Decl(parseTypeComment());
      L.expect(TK_NEWLINE);
      decl = mergeTypesFromTypeComment(decl, type_annotation_decl, is_method);
    }

    auto stmts_list = parseStatements(false);
    return Def::create(
        name.range(), Ident(name), Decl(decl), List<Stmt>(stmts_list));
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

Parser::Parser(const std::string& src) : pImpl(new ParserImpl(src)) {}

Parser::~Parser() = default;

TreeRef Parser::parseFunction(bool is_method) {
  return pImpl->parseFunction(is_method);
}
TreeRef Parser::parseClass() {
  return pImpl->parseClass();
}
Lexer& Parser::lexer() {
  return pImpl->lexer();
}
Decl Parser::parseTypeComment() {
  return pImpl->parseTypeComment();
}
Expr Parser::parseExp() {
  return pImpl->parseExp();
}

} // namespace script
} // namespace jit
} // namespace torch
