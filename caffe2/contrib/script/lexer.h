#pragma once
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "caffe2/core/common.h"

namespace caffe2 {
namespace script {

// single character tokens are just the character itself '+'
// multi-character tokens need an entry here
// if the third entry is not the empty string, it is used
// in the lexer to match this token.

// These kinds are also used in Tree.h as the kind of the AST node.
// Some kinds TK_APPLY, TK_LIST are only used in the AST and are not seen in the
// lexer.

#define TC_FORALL_TOKEN_KINDS(_)                 \
  _(TK_EOF, "eof", "")                           \
  _(TK_WHITESPACE, "whitespace", "")             \
  _(TK_NUMBER, "number", "")                     \
  _(TK_NEWLINE, "newline", "")                   \
  _(TK_INDENT, "indent", "")                     \
  _(TK_DEDENT, "dedent", "")                     \
  _(TK_WHERE, "where", "where")                  \
  _(TK_FLOAT, "float", "float")                  \
  _(TK_DOUBLE, "double", "double")               \
  _(TK_LONG, "long", "long")                     \
  _(TK_INT, "int", "int")                        \
  _(TK_DEF, "def", "def")                        \
  _(TK_ARROW, "arrow", "->")                     \
  _(TK_EQUIVALENT, "equivalent", "<=>")          \
  _(TK_IDENT, "ident", "")                       \
  _(TK_STRING, "string", "")                     \
  _(TK_CONST, "const", "")                       \
  _(TK_LIST, "list", "")                         \
  _(TK_OPTION, "option", "")                     \
  _(TK_APPLY, "apply", "")                       \
  _(TK_COMPREHENSION, "comprehension", "")       \
  _(TK_TENSOR_TYPE, "tensor_type", "")           \
  _(TK_RANGE_CONSTRAINT, "range_constraint", "") \
  _(TK_PARAM, "param", "")                       \
  _(TK_INFERRED, "inferred", "")                 \
  _(TK_BOOL, "bool", "")                         \
  _(TK_ACCESS, "access", "")                     \
  _(TK_ASSIGN, "assign", "")                     \
  _(TK_ATTRIBUTE, "attribute", "")               \
  _(TK_IF, "if", "if")                           \
  _(TK_ELSE, "else", "else")                     \
  _(TK_ELIF, "elif", "elif")                     \
  _(TK_WHILE, "while", "while")                  \
  _(TK_NE, "ne", "!=")                           \
  _(TK_EQ, "eq", "==")                           \
  _(TK_LE, "le", "<=")                           \
  _(TK_GE, "ge", ">=")                           \
  _(TK_IF_EXPR, "if", "")                        \
  _(TK_TRUE, "True", "True")                     \
  _(TK_FALSE, "False", "False")                  \
  _(TK_AND, "and", "and")                        \
  _(TK_OR, "or", "or")                           \
  _(TK_NOT, "not", "not")                        \
  _(TK_CAST, "cast", "")                         \
  _(TK_PLUS_EQ, "+=", "+=")                      \
  _(TK_MINUS_EQ, "-=", "-=")                     \
  _(TK_TIMES_EQ, "*=", "*=")                     \
  _(TK_DIV_EQ, "/=", "/=")                       \
  _(TK_GLOBAL, "global", "global")               \
  _(TK_BUILT_IN, "built-in", "")                 \
  _(TK_SLICE, "slice", "")                       \
  _(TK_GATHER, "gather", "")
static const char* valid_single_char_tokens = "+-*/()[]:,={}><.";

enum TokenKind {
  // we use characters to represent themselves so skip all valid characters
  // before
  // assigning enum values to multi-char tokens.
  TK_DUMMY_START = 256,
#define DEFINE_TOKEN(tok, _, _2) tok,
  TC_FORALL_TOKEN_KINDS(DEFINE_TOKEN)
#undef DEFINE_TOKEN
};

std::string kindToString(int kind);

// nested hash tables that indicate char-by-char what is a valid token.
struct TokenTrie;
using TokenTrieRef = std::unique_ptr<TokenTrie>;
struct TokenTrie {
  TokenTrie() : kind(0) {}
  void insert(const char* str, int tok) {
    if (*str == '\0') {
      assert(kind == 0);
      kind = tok;
      return;
    }
    auto& entry = children[*str];
    if (entry == nullptr) {
      entry.reset(new TokenTrie());
    }
    entry->insert(str + 1, tok);
  }
  int kind; // 0 == invalid token
  std::unordered_map<char, TokenTrieRef> children;
};

// stuff that is shared against all TC lexers/parsers and is initialized only
// once.
struct SharedParserData {
  SharedParserData() : head(new TokenTrie()) {
    // listed in increasing order of precedence
    std::vector<std::vector<int>> binary_ops = {
        {TK_IF},
        {TK_AND, TK_OR},
        {}, // reserve a level for unary not
        {'<', '>', TK_EQ, TK_LE, TK_GE, TK_NE},
        {'+', '-'},
        {'*', '/'},
    };
    std::vector<std::vector<int>> unary_ops = {
        {'-'},
    };

    std::stringstream ss;
    for (const char* c = valid_single_char_tokens; *c; c++) {
      const char str[] = {*c, '\0'};
      head->insert(str, *c);
    }

#define ADD_CASE(tok, _, tokstring) \
  if (*tokstring != '\0') {         \
    head->insert(tokstring, tok);   \
  }
    TC_FORALL_TOKEN_KINDS(ADD_CASE)
#undef ADD_CASE

    // precedence starts at 1 so that there is always a 0 precedence
    // less than any other precedence
    int prec = 1;
    for (auto& group : binary_ops) {
      for (auto& element : group) {
        binary_prec[element] = prec;
      }
      prec++;
    }
    // unary ops
    for (auto& group : unary_ops) {
      for (auto& element : group) {
        unary_prec[element] = prec;
      }
      prec++;
    }
    // add unary not separately because it slots into the precedence of
    // binary operators
    unary_prec[TK_NOT] = binary_prec[TK_AND] + 1;
  }
  // 1. skip whitespace
  // 2. handle comment or newline
  //
  bool isNumber(const std::string& str, size_t start, size_t* len) {
    char first = str[start];
    // strtod allows numbers to start with + or - or nan or inf
    // http://en.cppreference.com/w/cpp/string/byte/strtof
    // but we want only the number part, otherwise 1+3 will turn into two
    // adjacent numbers in the lexer
    if (first == '-' || first == '+' || isalpha(first))
      return false;
    const char* startptr = str.c_str() + start;
    char* endptr;
    std::strtod(startptr, &endptr);
    *len = endptr - startptr;
    return *len > 0;
  }
  bool isblank(int n) {
    return isspace(n) && n != '\n';
  }
  // find the longest match of str.substring(pos) against a token, return true
  // if successful
  // filling in kind, start,and len
  bool match(
      const std::string& str,
      size_t pos,
      bool continuation, // are we inside a scope where newlines don't count
                         // (e.g. inside parens)
      bool whitespace_token, // should we treat whitespace as a token
      int* kind,
      size_t* start,
      size_t* len) {
    *start = pos;
    // skip whitespace
    while (pos < str.size() && isblank(str[pos]))
      pos++;

    // special handling
    if (pos < str.size()) {
      if (str[pos] == '#') {
        // skip comments
        while (pos < str.size() && str[pos] != '\n')
          pos++;
        // tail call, handle whitespace and more comments
        return match(
            str, pos, continuation, whitespace_token, kind, start, len);
      }
      if (str[pos] == '\\' && pos + 1 < str.size() && str[pos + 1] == '\n' &&
          !whitespace_token) {
        return match(str, pos + 2, continuation, false, kind, start, len);
      }
      if (str[pos] == '\n') {
        return match(
            str, pos + 1, continuation, !continuation, kind, start, len);
      }
    }
    if (pos == str.size()) {
      *kind = TK_EOF;
      *start = pos;
      *len = 0;
      return true;
    }
    // invariant: the next token is not whitespace or newline
    if (whitespace_token) {
      *kind = TK_WHITESPACE;
      *len = pos - *start;
      return true;
    }
    *start = pos;
    // check for a valid number
    if (isNumber(str, pos, len)) {
      *kind = TK_NUMBER;
      return true;
    }
    // check for either an ident or a token
    // ident tracks whether what we have scanned so far could be an identifier
    // matched indicates if we have found any match.
    bool matched = false;
    bool ident = true;
    TokenTrie* cur = head.get();
    for (size_t i = 0; pos + i < str.size() && (ident || cur != nullptr); i++) {
      ident = ident && validIdent(i, str[pos + i]);
      if (ident) {
        matched = true;
        *len = i + 1;
        *kind = TK_IDENT;
      }
      // check for token second, so that e.g. 'max' matches the token TK_MAX
      // rather the
      // identifier 'max'
      if (cur) {
        auto it = cur->children.find(str[pos + i]);
        cur = (it == cur->children.end()) ? nullptr : it->second.get();
        if (cur && cur->kind != 0) {
          matched = true;
          *len = i + 1;
          *kind = cur->kind;
        }
      }
    }
    return matched;
  }
  bool isUnary(int kind, int* prec) {
    auto it = unary_prec.find(kind);
    if (it != unary_prec.end()) {
      *prec = it->second;
      return true;
    }
    return false;
  }
  bool isBinary(int kind, int* prec) {
    auto it = binary_prec.find(kind);
    if (it != binary_prec.end()) {
      *prec = it->second;
      return true;
    }
    return false;
  }
  bool isRightAssociative(int kind) {
    switch (kind) {
      case '?':
        return true;
      default:
        return false;
    }
  }

 private:
  bool validIdent(size_t i, char n) {
    return isalpha(n) || n == '_' || (i > 0 && isdigit(n));
  }
  TokenTrieRef head;
  std::unordered_map<int, int>
      unary_prec; // map from token to its unary precedence
  std::unordered_map<int, int>
      binary_prec; // map from token to its binary precedence
};

SharedParserData& sharedParserData();

// a range of a shared string 'file_' with functions to help debug by highlight
// that
// range.
struct SourceRange {
  SourceRange(
      const std::shared_ptr<std::string>& file_,
      size_t start_,
      size_t end_)
      : file_(file_), start_(start_), end_(end_) {}
  const std::string text() const {
    return file().substr(start(), end() - start());
  }
  size_t size() const {
    return end() - start();
  }
  void highlight(std::ostream& out) const {
    const std::string& str = file();
    size_t begin = start();
    size_t end = start();
    while (begin > 0 && str[begin - 1] != '\n')
      --begin;
    while (end < str.size() && str[end] != '\n')
      ++end;
    out << str.substr(0, end) << "\n";
    out << std::string(start() - begin, ' ');
    size_t len = std::min(size(), end - start());
    out << std::string(len, '~')
        << (len < size() ? "...  <--- HERE" : " <--- HERE");
    out << str.substr(end);
    if (str.size() > 0 && str.back() != '\n')
      out << "\n";
  }
  const std::string& file() const {
    return *file_;
  }
  const std::shared_ptr<std::string>& file_ptr() const {
    return file_;
  }
  size_t start() const {
    return start_;
  }
  size_t end() const {
    return end_;
  }

 private:
  std::shared_ptr<std::string> file_;
  size_t start_;
  size_t end_;
};

struct Token {
  int kind;
  SourceRange range;
  Token(int kind, const SourceRange& range) : kind(kind), range(range) {}
  double doubleValue() {
    assert(TK_NUMBER == kind);
    size_t idx;
    double r = stod(text(), &idx);
    assert(idx == range.size());
    return r;
  }
  std::string text() {
    return range.text();
  }
  std::string kindString() const {
    return kindToString(kind);
  }
};

struct Lookahead {
  Lookahead(const Token& t) : t(t) {}
  Token t;
  bool valid = false;
  size_t repeat = 0;
};

struct Lexer {
  std::shared_ptr<std::string> file;
  explicit Lexer(const std::string& str)
      : file(std::make_shared<std::string>(str)),
        pos(0),
        cur_(TK_EOF, SourceRange(file, 0, 0)),
        lookahead_(cur_),
        repeat(0),
        nesting(0),
        shared(sharedParserData()) {
    auto first_indent = lexRaw(true);
    indent_stack.push_back(first_indent.range.size());
    next();
  }
  Token next() {
    Token r = cur_;
    if (repeat > 0) {
      repeat--;
    } else if (lookahead_.valid) {
      lookahead_.valid = false;
      repeat = lookahead_.repeat;
      cur_ = lookahead_.t;
    } else {
      std::tie(cur_, repeat) = lex();
    }
    return r;
  }
  bool nextIf(int kind) {
    if (cur_.kind != kind)
      return false;
    next();
    return true;
  }

  [[noreturn]] void reportError(const std::string& what) {
    reportError(what, cur_);
  }
  [[noreturn]] void reportError(const std::string& what, const Token& t) {
    std::stringstream ss;
    ss << what << ":\n";
    t.range.highlight(ss);
    throw std::runtime_error(ss.str());
  }
  [[noreturn]] void expected(const std::string& what, const Token& t) {
    std::stringstream ss;
    ss << "expected " << what << " but found '" << t.kindString()
       << "' here:\n";
    t.range.highlight(ss);
    throw std::runtime_error(ss.str());
  }
  [[noreturn]] void expected(const std::string& what) {
    expected(what, cur_);
  }
  Token expect(int kind) {
    if (cur_.kind != kind) {
      expected(kindToString(kind));
    }
    return next();
  }
  Token& lookahead() {
    if (!lookahead_.valid) {
      lookahead_.valid = true;
      std::tie(lookahead_.t, lookahead_.repeat) = lex();
    }
    return lookahead_.t;
  }
  Token& cur() {
    return cur_;
  }

 private:
  // token, number of times to repeat it
  std::pair<Token, int> lex() {
    auto r = lexRaw();
    int repeat = 0;
    switch (r.kind) {
      case '(':
      case '[':
      case '{':
        nesting++;
        break;
      case ')':
      case ']':
      case '}':
        nesting--;
        break;
      case TK_WHITESPACE: {
        size_t depth = r.range.size();
        if (depth > indent_stack.back()) {
          indent_stack.push_back(depth);
          r.kind = TK_INDENT;
        } else if (depth == indent_stack.back()) {
          r.kind = TK_NEWLINE;
        } else {
          while (indent_stack.back() != depth) {
            indent_stack.pop_back();
            repeat++;
            if (indent_stack.size() == 0) {
              reportError("invalid ident level", r);
            }
          }
          repeat--; // first repeat is this return
          r.kind = TK_DEDENT;
        }
      } break;
      case TK_EOF:
        if (indent_stack.size() > 1) {
          r.kind = TK_DEDENT;
          indent_stack.pop_back();
        }
        break;
      default:
        break;
    }
    return std::make_pair(r, repeat);
  }
  Token lexRaw(bool whitespace_token = false) {
    int kind;
    size_t start;
    size_t length;
    assert(file);
    if (!shared.match(
            *file,
            pos,
            nesting > 0,
            whitespace_token,
            &kind,
            &start,
            &length)) {
      expected(
          "a valid token",
          Token((*file)[start], SourceRange(file, start, start + 1)));
    }
    auto t = Token(kind, SourceRange(file, start, start + length));
    pos = start + length;
    return t;
  }
  size_t pos;
  Token cur_;
  Lookahead lookahead_;
  size_t repeat; // how many times to repeat the current token until we continue

  size_t nesting; // depth of ( [ { nesting...
  std::vector<int> indent_stack; // stack of identation level of blocks
  SharedParserData& shared;
};
} // namespace script
} // namespace caffe2
