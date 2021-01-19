#include <torch/csrc/jit/frontend/parser.h>

#include <c10/util/Optional.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parse_string_literal.h>
#include <torch/csrc/jit/frontend/tree.h>
#include <torch/csrc/jit/frontend/tree_views.h>

namespace torch {
namespace jit {

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
    throw ErrorReport(decl.range())
        << "Number of type annotations ("
        << type_annotation_decl.params().size()
        << ") did not match the number of "
        << (is_method ? "method" : "function") << " parameters ("
        << expected_num_annotations << ")";
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
  explicit ParserImpl(const std::shared_ptr<Source>& source)
      : L(source), shared(sharedParserData()) {}

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
    parseArguments(inputs, attributes);
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
      case TK_MOD_EQ:
      case TK_BIT_OR_EQ:
      case TK_BIT_AND_EQ:
      case TK_BIT_XOR_EQ:
      case TK_LSHIFT_EQ:
      case TK_RSHIFT_EQ:
      case TK_POW_EQ:
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
        prefix = create_compound(k, r, {});
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

        if (list.size() == 1 && (*list.begin()).kind() == TK_LIST_COMP) {
          prefix = *list.begin();
        } else {
          for (auto se : list) {
            if (se.kind() == TK_LIST_COMP) {
              throw ErrorReport(list.range())
                  << " expected a single list comprehension within '[' , ']'";
            }
          }
          prefix = ListLiteral::create(list.range(), List<Expr>(list));
        }

      } break;
      case '{': {
        L.next();
        // If we have a dict literal, `keys` and `values` will store the keys
        // and values used in the object's construction. EDGE CASE: We have a
        // dict comprehension, so we'll get the first element of the dict
        // comprehension in `keys` and a list comprehension in `values`.
        // For example, `{i : chr(i + 65) for i in range(4)}` would give us
        // `i` in `keys` and `chr(i + 65) for i in range(4)` in `values`.
        // The optimal way of handling this case is to simply splice the new
        // dict comprehension together from the existing list comprehension.
        // Splicing prevents breaking changes to our API and does not require
        // the use of global variables.
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
        if (keys.size() == 1 && (*values.begin()).kind() == TK_LIST_COMP) {
          ListComp lc(*values.begin());
          prefix = DictComp::create(
              range, *keys.begin(), lc.elt(), lc.target(), lc.iter());
        } else {
          prefix = DictLiteral::create(
              range,
              List<Expr>::create(range, keys),
              List<Expr>::create(range, values));
        }
      } break;
      case TK_STRINGLITERAL: {
        prefix = parseConcatenatedStringLiterals();
      } break;
      case TK_ELLIPSIS:
      case TK_DOTS: {
        prefix = Dots::create(L.cur().range);
        L.next();
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
  c10::optional<TreeRef> maybeParseAssignmentOp() {
    auto r = L.cur().range;
    switch (L.cur().kind) {
      case TK_PLUS_EQ:
      case TK_MINUS_EQ:
      case TK_TIMES_EQ:
      case TK_DIV_EQ:
      case TK_BIT_OR_EQ:
      case TK_BIT_AND_EQ:
      case TK_BIT_XOR_EQ:
      case TK_MOD_EQ: {
        int modifier = L.next().text()[0];
        return create_compound(modifier, r, {});
      } break;
      case TK_LSHIFT_EQ: {
        L.next();
        return create_compound(TK_LSHIFT, r, {});
      } break;
      case TK_RSHIFT_EQ: {
        L.next();
        return create_compound(TK_RSHIFT, r, {});
      } break;
      case TK_POW_EQ: {
        L.next();
        return create_compound(TK_POW, r, {});
      } break;
      case '=': {
        L.next();
        return create_compound('=', r, {}); // no reduction
      } break;
      default:
        return c10::nullopt;
    }
  }
  TreeRef parseTrinary(
      TreeRef true_branch,
      const SourceRange& range,
      int binary_prec) {
    auto cond = parseExp();
    L.expect(TK_ELSE);
    auto false_branch = parseExp(binary_prec);
    return create_compound(
        TK_IF_EXPR, range, {cond, std::move(true_branch), false_branch});
  }
  // parse the longest expression whose binary operators have
  // precedence strictly greater than 'precedence'
  // precedence == 0 will parse _all_ expressions
  // this is the core loop of 'top-down precedence parsing'
  Expr parseExp() {
    return parseExp(0);
  }
  Expr parseExp(int precedence) {
    TreeRef prefix;
    int unary_prec;
    if (shared.isUnary(L.cur().kind, &unary_prec)) {
      auto kind = L.cur().kind;
      auto pos = L.cur().range;
      L.next();
      auto unary_kind = kind == '*' ? TK_STARRED
          : kind == '-'             ? TK_UNARY_MINUS
                                    : kind;
      auto subexp = parseExp(unary_prec);
      // fold '-' into constant numbers, so that attributes can accept
      // things like -1
      if (unary_kind == TK_UNARY_MINUS && subexp.kind() == TK_CONST) {
        prefix = Const::create(subexp.range(), "-" + Const(subexp).text());
      } else {
        prefix = create_compound(unary_kind, pos, {subexp});
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

      if (kind == TK_NOTIN) {
        // NB: `not in` is just `not( in )`, so we don't introduce new tree view
        // but just make it a nested call in our tree view structure
        prefix = create_compound(TK_IN, pos, {prefix, parseExp(binary_prec)});
        prefix = create_compound(TK_NOT, pos, {prefix});
        continue;
      }

      // special case for trinary operator
      if (kind == TK_IF) {
        prefix = parseTrinary(prefix, pos, binary_prec);
        continue;
      }

      if (kind == TK_FOR) {
        // TK_FOR targets should only parse exprs prec greater than 4, which
        // only includes subset of Exprs that suppose to be on the LHS according
        // to the python grammar
        // https://docs.python.org/3/reference/grammar.html
        auto target = parseLHSExp();
        L.expect(TK_IN);
        auto iter = parseExp();
        prefix = ListComp::create(pos, Expr(prefix), target, iter);
        continue;
      }

      prefix = create_compound(kind, pos, {prefix, parseExp(binary_prec)});
    }
    return Expr(prefix);
  }

  void parseSequence(
      int begin,
      int sep,
      int end,
      const std::function<void()>& parse) {
    if (begin != TK_NOTHING) {
      L.expect(begin);
    }
    while (end != L.cur().kind) {
      parse();
      if (!L.nextIf(sep)) {
        if (end != TK_NOTHING) {
          L.expect(end);
        }
        return;
      }
    }
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
    std::string ss;
    while (L.cur().kind == TK_STRINGLITERAL) {
      auto literal_range = L.cur().range;
      ss.append(parseStringLiteral(literal_range, L.next().text()));
    }
    return StringLiteral::create(range, ss);
  }

  Expr parseAttributeValue() {
    return parseExp();
  }

  void parseArguments(TreeList& inputs, TreeList& attributes) {
    parseSequence('(', ',', ')', [&] {
      if (L.cur().kind == TK_IDENT && L.lookahead().kind == '=') {
        auto ident = parseIdent();
        L.expect('=');
        auto v = parseAttributeValue();
        attributes.push_back(Attribute::create(ident.range(), Ident(ident), v));
      } else {
        inputs.push_back(parseExp());
      }
    });
  }

  // parse LHS acceptable exprs, which only includes subset of Exprs that prec
  // is greater than 4 according to the python grammar
  Expr parseLHSExp() {
    return parseExp(4);
  }

  // Parse expr's of the form [a:], [:b], [a:b], [:] and all variations with
  // "::"
  Expr parseSubscriptExp() {
    TreeRef first, second, third;
    auto range = L.cur().range;
    if (L.cur().kind != ':') {
      first = parseExp();
    }
    if (L.nextIf(':')) {
      if (L.cur().kind != ',' && L.cur().kind != ']' && L.cur().kind != ':') {
        second = parseExp();
      }
      if (L.nextIf(':')) {
        if (L.cur().kind != ',' && L.cur().kind != ']') {
          third = parseExp();
        }
      }
      auto maybe_first = first ? Maybe<Expr>::create(range, Expr(first))
                               : Maybe<Expr>::create(range);
      auto maybe_second = second ? Maybe<Expr>::create(range, Expr(second))
                                 : Maybe<Expr>::create(range);
      auto maybe_third = third ? Maybe<Expr>::create(range, Expr(third))
                               : Maybe<Expr>::create(range);
      return SliceExpr::create(range, maybe_first, maybe_second, maybe_third);
    } else {
      return Expr(first);
    }
  }

  TreeRef parseSubscript(const TreeRef& value) {
    const auto range = L.cur().range;

    auto subscript_exprs =
        parseList('[', ',', ']', &ParserImpl::parseSubscriptExp);

    const auto whole_range =
        SourceRange(range.source(), range.start(), L.cur().range.start());
    return Subscript::create(whole_range, Expr(value), subscript_exprs);
  }

  Maybe<Expr> maybeParseTypeAnnotation() {
    if (L.nextIf(':')) {
      // NB: parseExp must not be called inline, since argument evaluation order
      // changes when L.cur().range is mutated with respect to the parseExp()
      // call.
      auto expr = parseExp();
      return Maybe<Expr>::create(expr.range(), expr);
    } else {
      return Maybe<Expr>::create(L.cur().range);
    }
  }

  TreeRef parseFormalParam(bool kwarg_only) {
    auto ident = parseIdent();
    TreeRef type = maybeParseTypeAnnotation();
    TreeRef def;
    if (L.nextIf('=')) {
      // NB: parseExp must not be called inline, since argument evaluation order
      // changes when L.cur().range is mutated with respect to the parseExp()
      // call.
      auto expr = parseExp();
      def = Maybe<Expr>::create(expr.range(), expr);
    } else {
      def = Maybe<Expr>::create(L.cur().range);
    }
    return Param::create(
        type->range(),
        Ident(ident),
        Maybe<Expr>(type),
        Maybe<Expr>(def),
        kwarg_only);
  }

  Param parseBareTypeAnnotation() {
    auto type = parseExp();
    return Param::create(
        type.range(),
        Ident::create(type.range(), ""),
        Maybe<Expr>::create(type.range(), type),
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
    auto type = maybeParseTypeAnnotation();
    auto maybeOp = maybeParseAssignmentOp();
    if (maybeOp) {
      // There is an assignment operator, parse the RHS and generate the
      // assignment.
      auto rhs = parseExpOrExpTuple();
      if (maybeOp.value()->kind() == '=') {
        std::vector<Expr> lhs_list = {lhs};
        while (L.nextIf('=')) {
          lhs_list.push_back(rhs);
          rhs = parseExpOrExpTuple();
        }
        if (type.present() && lhs_list.size() > 1) {
          throw ErrorReport(type.range())
              << "Annotated multiple assignment is not supported in python";
        }
        L.expect(TK_NEWLINE);
        return Assign::create(
            lhs.range(),
            List<Expr>::create(lhs_list[0].range(), lhs_list),
            Maybe<Expr>::create(rhs.range(), rhs),
            type);
      } else {
        L.expect(TK_NEWLINE);
        // this is an augmented assignment
        if (lhs.kind() == TK_TUPLE_LITERAL) {
          throw ErrorReport(lhs.range())
              << " augmented assignment can only have one LHS expression";
        }
        return AugAssign::create(
            lhs.range(), lhs, AugAssignKind(*maybeOp), Expr(rhs));
      }
    } else {
      // There is no assignment operator, so this is of the form `lhs : <type>`
      TORCH_INTERNAL_ASSERT(type.present());
      L.expect(TK_NEWLINE);
      return Assign::create(
          lhs.range(),
          List<Expr>::create(lhs.range(), {lhs}),
          Maybe<Expr>::create(lhs.range()),
          type);
    }
  }

  TreeRef parseStmt(bool in_class = false) {
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
        Expr value = L.cur().kind != TK_NEWLINE
            ? parseExpOrExpTuple()
            : Expr(create_compound(TK_NONE, range, {}));
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
      case TK_BREAK: {
        auto range = L.next().range;
        L.expect(TK_NEWLINE);
        return Break::create(range);
      }
      case TK_CONTINUE: {
        auto range = L.next().range;
        L.expect(TK_NEWLINE);
        return Continue::create(range);
      }
      case TK_PASS: {
        auto range = L.next().range;
        L.expect(TK_NEWLINE);
        return Pass::create(range);
      }
      case TK_DEF: {
        return parseFunction(/*is_method=*/in_class);
      }
      case TK_DELETE: {
        auto range = L.next().range;
        auto targets =
            parseList(TK_NOTHING, ',', TK_NOTHING, &ParserImpl::parseExp);
        L.expect(TK_NEWLINE);
        return Delete::create(range, targets);
      }
      case TK_WITH: {
        return parseWith();
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

  WithItem parseWithItem() {
    auto target = parseExp();

    if (L.cur().kind == TK_AS) {
      // If the current token is TK_AS, this with item is of the form
      // "expression as target".
      auto token = L.expect(TK_AS);
      Ident ident = parseIdent();
      auto var = Var::create(ident.range(), ident);
      return WithItem::create(
          token.range, target, Maybe<Var>::create(ident.range(), var));
    } else {
      // If not, this with item is of the form "expression".
      return WithItem::create(
          target.range(), target, Maybe<Var>::create(target.range()));
    }
  }

  TreeRef parseIf(bool expect_if = true) {
    auto r = L.cur().range;
    if (expect_if)
      L.expect(TK_IF);
    auto cond = parseExp();
    L.expect(':');
    auto true_branch = parseStatements(/*expect_indent=*/true);
    auto false_branch = makeList(L.cur().range, {});
    if (L.nextIf(TK_ELSE)) {
      L.expect(':');
      false_branch = parseStatements(/*expect_indent=*/true);
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
    auto body = parseStatements(/*expect_indent=*/true);
    return While::create(r, Expr(cond), List<Stmt>(body));
  }

  TreeRef parseFor() {
    auto r = L.cur().range;
    L.expect(TK_FOR);
    auto targets = parseList(TK_NOTHING, ',', TK_IN, &ParserImpl::parseLHSExp);
    auto itrs = parseList(TK_NOTHING, ',', ':', &ParserImpl::parseExp);
    auto body = parseStatements(/*expect_indent=*/true);
    return For::create(r, targets, itrs, body);
  }

  TreeRef parseWith() {
    auto r = L.cur().range;
    // Parse "with expression [as target][, expression [as target]]*:".
    L.expect(TK_WITH);
    auto targets = parseList(TK_NOTHING, ',', ':', &ParserImpl::parseWithItem);
    // Parse the body.
    auto body = parseStatements(/*expect_indent=*/true);
    return With::create(r, targets, body);
  }

  TreeRef parseStatements(bool expect_indent, bool in_class = false) {
    auto r = L.cur().range;
    if (expect_indent) {
      L.expect(TK_INDENT);
    }
    TreeList stmts;
    do {
      stmts.push_back(parseStmt(in_class));
    } while (!L.nextIf(TK_DEDENT));
    return create_compound(TK_LIST, r, std::move(stmts));
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

  List<Param> parseFormalParams() {
    auto r = L.cur().range;
    std::vector<Param> params;
    bool kwarg_only = false;
    parseSequence('(', ',', ')', [&] {
      if (!kwarg_only && L.nextIf('*')) {
        kwarg_only = true;
      } else {
        params.emplace_back(parseFormalParam(kwarg_only));
      }
    });
    return List<Param>::create(r, params);
  }
  Decl parseDecl() {
    // Parse return type annotation
    List<Param> paramlist = parseFormalParams();
    TreeRef return_type;
    Maybe<Expr> return_annotation = parseReturnAnnotation();
    L.expect(':');
    return Decl::create(
        paramlist.range(), List<Param>(paramlist), return_annotation);
  }

  TreeRef parseClass() {
    L.expect(TK_CLASS_DEF);
    const auto name = parseIdent();
    Maybe<Expr> superclass = Maybe<Expr>::create(name.range());
    if (L.nextIf('(')) {
      // Only support inheriting from NamedTuple right now.
      auto id = parseExp();
      superclass = Maybe<Expr>::create(id.range(), id);
      L.expect(')');
    }
    L.expect(':');
    const auto statements =
        parseStatements(/*expect_indent=*/true, /*in_class=*/true);
    return ClassDef::create(
        name.range(), name, superclass, List<Stmt>(statements));
  }

  TreeRef parseFunction(bool is_method) {
    L.expect(TK_DEF);
    auto name = parseIdent();
    auto decl = parseDecl();

    TreeRef stmts_list;
    if (L.nextIf(TK_INDENT)) {
      // Handle type annotations specified in a type comment as the first line
      // of the function.
      if (L.cur().kind == TK_TYPE_COMMENT) {
        auto type_annotation_decl = Decl(parseTypeComment());
        L.expect(TK_NEWLINE);
        decl = mergeTypesFromTypeComment(decl, type_annotation_decl, is_method);
      }

      stmts_list = parseStatements(false);
    } else {
      // Special case: the Python grammar allows one-line functions with a
      // single statement.
      if (L.cur().kind == TK_TYPE_COMMENT) {
        auto type_annotation_decl = Decl(parseTypeComment());
        decl = mergeTypesFromTypeComment(decl, type_annotation_decl, is_method);
      }

      TreeList stmts;
      stmts.push_back(parseStmt(is_method));
      stmts_list = create_compound(TK_LIST, L.cur().range, std::move(stmts));
    }

    return Def::create(
        name.range(), Ident(name), Decl(decl), List<Stmt>(stmts_list));
  }
  Lexer& lexer() {
    return L;
  }

 private:
  // short helpers to create nodes
  TreeRef create_compound(
      int kind,
      const SourceRange& range,
      TreeList&& trees) {
    return Compound::create(kind, range, std::move(trees));
  }
  TreeRef makeList(const SourceRange& range, TreeList&& trees) {
    return create_compound(TK_LIST, range, std::move(trees));
  }
  Lexer L;
  SharedParserData& shared;
};

Parser::Parser(const std::shared_ptr<Source>& src)
    : pImpl(new ParserImpl(src)) {}

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

} // namespace jit
} // namespace torch
