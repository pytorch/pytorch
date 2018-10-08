#pragma once
#include "lexer.h"
#include "tree.h"
#include "tree_views.h"

namespace torch {
namespace jit {
namespace script {

Decl mergeTypesFromTypeComment(
    Decl decl,
    Decl type_annotation_decl,
    bool is_method);

class Parser {
 public:
  explicit Parser(const std::string& str)
      : L(str), shared(sharedParserData()) {}

  Lexer& lexer() {
    return L;
  }

  TreeRef parseFunction(bool is_method);
  TreeRef parseTypeComment(bool parse_full_line = false);

 private:
  template <typename T>
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

  Ident parseIdent();
  TreeRef createApply(Expr expr);
  TreeRef parseExpOrExpTuple(int end);
  TreeRef parseBaseExp();
  TreeRef parseOptionalReduction();
  TreeRef parseTrinary(
      TreeRef true_branch,
      const SourceRange& range,
      int binary_prec);
  Expr parseExp() {
    return parseExp(0);
  }
  Expr parseExp(int precedence);
  Const parseConst();
  std::string parseString(const SourceRange& range, const std::string& str);
  StringLiteral parseStringLiteral();
  Expr parseAttributeValue();
  void parseOperatorArguments(TreeList& inputs, TreeList& attributes);
  Expr parseSubscriptExp();
  TreeRef parseSubscript(TreeRef value);
  TreeRef parseParam();
  Param parseBareTypeAnnotation();
  Assign parseAssign(List<Expr> list);
  TreeRef parseStmt();
  TreeRef parseOptionalIdentList();
  TreeRef parseIf(bool expect_if = true);
  TreeRef parseWhile();
  TreeRef parseFor();

  TreeRef parseStatements(bool expect_indent = true);
  Decl parseDecl();

  bool isCharCount(char c, const std::string& str, size_t start, int len);

  // short helpers to create nodes
  TreeRef c(int kind, const SourceRange& range, TreeList&& trees);
  TreeRef makeList(const SourceRange& range, TreeList&& trees);

  Lexer L;
  SharedParserData& shared;
};
} // namespace script
} // namespace jit
} // namespace torch
