#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/parser_constants.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/strtod.h>
#include <algorithm>
#include <clocale>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace torch::jit {

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
  _(TK_WHITESPACE_EOF, "whitespace_eof", "")     \
  _(TK_NUMBER, "number", "")                     \
  _(TK_NEWLINE, "newline", "")                   \
  _(TK_INDENT, "indent", "")                     \
  _(TK_DEDENT, "dedent", "")                     \
  _(TK_DEF, "def", "def")                        \
  _(TK_EQUIVALENT, "equivalent", "<=>")          \
  _(TK_IDENT, "ident", "")                       \
  _(TK_STRING, "string", "")                     \
  _(TK_STRINGLITERAL, "string_literal", "")      \
  _(TK_CONST, "const", "")                       \
  _(TK_LIST, "list", "")                         \
  _(TK_DICT, "dict", "")                         \
  _(TK_OPTION, "option", "")                     \
  _(TK_APPLY, "apply", "")                       \
  _(TK_COMPREHENSION, "comprehension", "")       \
  _(TK_RANGE_CONSTRAINT, "range_constraint", "") \
  _(TK_PARAM, "param", "")                       \
  _(TK_INFERRED, "inferred", "")                 \
  _(TK_ACCESS, "access", "")                     \
  _(TK_ASSIGN, "assign", "")                     \
  _(TK_AUG_ASSIGN, "aug_assign", "")             \
  _(TK_ATTRIBUTE, "attribute", "")               \
  _(TK_IF, "if", "if")                           \
  _(TK_ELSE, "else", "else")                     \
  _(TK_ELIF, "elif", "elif")                     \
  _(TK_WHILE, "while", "while")                  \
  _(TK_EXPR_STMT, "expression statement", "")    \
  _(TK_RETURN, "return", "return")               \
  _(TK_IS, "is", "is")                           \
  _(TK_ISNOT, "is not", "is not")                \
  _(TK_NE, "ne", "!=")                           \
  _(TK_EQ, "eq", "==")                           \
  _(TK_LE, "le", "<=")                           \
  _(TK_GE, "ge", ">=")                           \
  _(TK_FLOOR_DIV, "floordiv", "//")              \
  _(TK_IF_EXPR, "if", "")                        \
  _(TK_TRUE, "True", "True")                     \
  _(TK_FALSE, "False", "False")                  \
  _(TK_NONE, "None", "None")                     \
  _(TK_AND, "and", "and")                        \
  _(TK_OR, "or", "or")                           \
  _(TK_NOT, "not", "not")                        \
  _(TK_LSHIFT, "<<", "<<")                       \
  _(TK_RSHIFT, ">>", ">>")                       \
  _(TK_CAST, "cast", "")                         \
  _(TK_PLUS_EQ, "+=", "+=")                      \
  _(TK_MINUS_EQ, "-=", "-=")                     \
  _(TK_TIMES_EQ, "*=", "*=")                     \
  _(TK_DIV_EQ, "/=", "/=")                       \
  _(TK_MOD_EQ, "%=", "%=")                       \
  _(TK_BIT_OR_EQ, "|=", "|=")                    \
  _(TK_BIT_AND_EQ, "&=", "&=")                   \
  _(TK_BIT_XOR_EQ, "^=", "^=")                   \
  _(TK_LSHIFT_EQ, "<<=", "<<=")                  \
  _(TK_RSHIFT_EQ, ">>=", ">>=")                  \
  _(TK_POW_EQ, "**=", "**=")                     \
  _(TK_GLOBAL, "global", "global")               \
  _(TK_BUILT_IN, "built-in", "")                 \
  _(TK_SUBSCRIPT, "subscript", "")               \
  _(TK_VAR, "variable", "")                      \
  _(TK_NOTHING, "nothing", "")                   \
  _(TK_DICT_LITERAL, "dict-literal", "")         \
  _(TK_LIST_LITERAL, "list-literal", "")         \
  _(TK_TUPLE_LITERAL, "tuple-literal", "")       \
  _(TK_FOR, "for", "for")                        \
  _(TK_IN, "in", "in")                           \
  _(TK_NOTIN, "not in", "not in")                \
  _(TK_STARRED, "starred", "")                   \
  _(TK_UNARY_MINUS, "unary minus", "")           \
  _(TK_POW, "pow operator", "**")                \
  _(TK_ARROW, "arrow", "->")                     \
  _(TK_DECL, "decl", "")                         \
  _(TK_SLICE_EXPR, "slice expr", "")             \
  _(TK_TYPE_COMMENT, "type comment", "# type:")  \
  _(TK_RAISE, "raise", "raise")                  \
  _(TK_ASSERT, "assert", "assert")               \
  _(TK_DOTS, "dots", "...")                      \
  _(TK_LIST_COMP, "list comprehension", "")      \
  _(TK_DICT_COMP, "dict comprehension", "")      \
  _(TK_BREAK, "break", "break")                  \
  _(TK_CONTINUE, "continue", "continue")         \
  _(TK_DELETE, "del", "del")                     \
  _(TK_PASS, "pass", "pass")                     \
  _(TK_CLASS_DEF, "class", "class")              \
  _(TK_IMPORT, "import", "import")               \
  _(TK_WITH, "with", "with")                     \
  _(TK_WITH_ITEM, "withitem", "")                \
  _(TK_AS, "as", "as")                           \
  _(TK_PROP, "property", "")                     \
  _(TK_ELLIPSIS, "Ellipsis", "Ellipsis")         \
  _(TK_NONE_TYPE, "NoneType", "NoneType")

enum TokenKind {
  // we use characters to represent themselves so skip all valid characters
  // before
  // assigning enum values to multi-char tokens.
  TK_DUMMY_START = 256,
#define DEFINE_TOKEN(tok, _, _2) tok,
  TC_FORALL_TOKEN_KINDS(DEFINE_TOKEN)
#undef DEFINE_TOKEN
};

TORCH_API std::string kindToString(int kind);
TORCH_API int stringToKind(const std::string& str);

// nested hash tables that indicate char-by-char what is a valid token.
struct TokenTrie;
using TokenTrieRef = std::unique_ptr<TokenTrie>;
struct TokenTrie {
  TokenTrie() = default;
  void insert(const char* str, int tok) {
    if (*str == '\0') {
      AT_ASSERT(kind == 0);
      kind = tok;
      return;
    }

    for (size_t i = 0, e = child_chars.size(); i < e; ++i) {
      if (child_chars[i] == *str) {
        child_tries[i]->insert(str + 1, tok);
        return;
      }
    }

    child_chars.emplace_back(*str);
    child_tries.emplace_back(std::make_unique<TokenTrie>());
    child_tries.back()->insert(str + 1, tok);
  }
  int kind{0}; // 0 == invalid token

  std::vector<char> child_chars;
  std::vector<TokenTrieRef> child_tries;
};

// stuff that is shared against all TC lexers/parsers and is initialized only
// once.
struct TORCH_API SharedParserData {
  SharedParserData() : head(new TokenTrie()) {
    for (const char* c = valid_single_char_tokens; *c; c++) {
      std::string str(1, *c);
      head->insert(str.c_str(), *c);
    }

#define ADD_CASE(tok, _, tokstring)   \
  if (*(tokstring) != '\0') {         \
    head->insert((tokstring), (tok)); \
  }
    TC_FORALL_TOKEN_KINDS(ADD_CASE)
#undef ADD_CASE
  }

  bool match(
      StringCordView::Iterator pos,
      bool continuation, // are we inside a scope where newlines don't count
                         // (e.g. inside parens)
      bool whitespace_token, // should we treat whitespace as a token
      int* kind,
      StringCordView::Iterator* start,
      StringCordView::Iterator* end) {
    *start = pos;
    // skip whitespace
    while (pos.has_next() && isblank(*pos)) {
      ++pos;
    }

    // special handling
    if (pos.has_next()) {
      if (*pos == '#' && !isTypeComment(pos)) {
        // skip comments
        while (pos.has_next() && *pos != '\n')
          ++pos;
        // tail call, handle whitespace and more comments
        return match(pos, continuation, whitespace_token, kind, start, end);
      }
      if (*pos == '\\') {
        auto newiter = pos;
        ++newiter;
        if (newiter.has_next() && *newiter == '\n' && !whitespace_token) {
          ++newiter;
          return match(newiter, continuation, false, kind, start, end);
        }
      }
      if (*pos == '\n') {
        return match(++pos, continuation, !continuation, kind, start, end);
      }
    }
    // we handle white space before EOF because in the case we have something
    // like the following where we need to generate the dedent token if foo:
    //   ...
    // else:
    //   pass
    if (whitespace_token) {
      *kind = !pos.has_next() ? TK_WHITESPACE_EOF : TK_WHITESPACE;
      *end = pos;
      return true;
    }
    if (!pos.has_next()) {
      *kind = TK_EOF;
      *start = pos;
      *end = *start;
      return true;
    }
    // invariant: the next token is not whitespace or newline
    *start = pos;
    // check for a valid number
    size_t len = 0;
    if (isNumber(pos.rest_line(), 0, &len)) {
      *end = *start;
      *end += len;
      *kind = TK_NUMBER;
      return true;
    }
    // check for string
    if (isString(pos.rest_line(), 0, &len)) {
      *kind = TK_STRINGLITERAL;
      *end = *start;
      *end += len;
      return true;
    }

    // check for either an ident or a token
    // ident tracks whether what we have scanned so far could be an identifier
    // matched indicates if we have found any match.
    bool matched = false;
    bool ident = true;
    TokenTrie* cur = head.get();
    // for (size_t i = 0; pos + i < str.size() && (ident || cur != nullptr);
    // i++)
    for (size_t i = 0; pos.has_next() && (ident || cur != nullptr);
         ++pos, ++i) {
      ident = ident && validIdent(i, *pos);
      if (ident) {
        matched = true;
        *end = pos.next_iter();
        *kind = TK_IDENT;
      }
      // check for token second, so that e.g. 'max' matches the token TK_MAX
      // rather the
      // identifier 'max'
      if (cur) {
        const auto begin_it = cur->child_chars.begin();
        const auto end_it = cur->child_chars.end();
        const auto ch_it = std::find(begin_it, end_it, *pos);

        cur = (ch_it == end_it) ? nullptr
                                : cur->child_tries[ch_it - begin_it].get();

        if (cur && cur->kind != 0) {
          matched = true;
          *end = pos.next_iter();
          *kind = cur->kind;
        }
      }
    }
    return matched;
  }

  bool isUnary(int kind, int* prec);
  bool isBinary(int kind, int* prec);
  bool isRightAssociative(int kind) {
    switch (kind) {
      case '?':
      case TK_POW:
      case TK_IF:
        return true;
      default:
        return false;
    }
  }

 private:
  bool validIdent(size_t i, char n) {
    return isalpha(n) || n == '_' || (i > 0 && isdigit(n));
  }

  // 1. skip whitespace
  // 2. handle comment or newline
  //
  bool isNumber(std::string_view str, size_t start, size_t* len) {
    char first = str[start];
    // strtod allows numbers to start with + or - or nan or inf
    // http://en.cppreference.com/w/cpp/string/byte/strtof
    // but we want only the number part, otherwise 1+3 will turn into two
    // adjacent numbers in the lexer
    if (first == '-' || first == '+' || isalpha(first))
      return false;
    const char* startptr = str.data() + start;
    char* endptr = nullptr;
    torch::jit::strtod_c(startptr, &endptr);
    *len = endptr - startptr;
    // check if the number is complex valued
    // access is safe because string is assumed to be null terminated
    if (endptr != nullptr && *endptr == 'j') {
      *len += 1;
    }
    return *len > 0;
  }

  bool isCharCount(char c, std::string_view str, size_t start, int len) {
    // count checks from [start, start + len)
    return start + len <= str.size() &&
        std::count(str.begin() + start, str.begin() + start + len, c) == len;
  }

  // python concatenates all adjacent strings "a" "b" == "ab"
  // strings can be enclosed with 1 or 3 single or double quotes
  // if enclosed with 3 quotes newlines are valid
  // as elsewhere, backslash and new line should be ignored
  bool isString(std::string_view str, size_t start, size_t* len) {
    char quote = str[start];
    if (quote != '\"' && quote != '\'')
      return false;
    int quote_len = isCharCount(quote, str, start, 3) ? 3 : 1;

    // end is now set past the opening quotation marks
    size_t end = start + quote_len;
    while (end < str.size() && !isCharCount(quote, str, end, quote_len)) {
      if (str[end] == '\n' && quote_len != 3) {
        return false;
      }
      // handle escaped characters. advances past escaped quotation marks,
      // escaped newlines and escaped backslashes
      // multi-char escapes like \x1A are handled fine here because the
      // remainder of the escape are valid string characters anyway
      if (str[end] == '\\') {
        end++;
      }
      end++;
    }
    // set length equal to the complete string including quotations
    *len = end - start + quote_len;
    // if end finished without going past the last character of the string than
    // there is a match
    return end < str.size();
  }

  bool isblank(int n) {
    return isspace(n) && n != '\n';
  }

  bool isTypeComment(StringCordView::Iterator str_iter) {
    std::string_view rest_line = str_iter.rest_line();
    const std::string type_string = "# type:";
    if (rest_line.size() < type_string.length()) {
      return false;
    }
    auto match_string = rest_line.substr(0, type_string.size());
    return match_string == type_string;
  }

  // Make an exception ignoring comments for type annotation comments
  bool isTypeComment(const StringCordView& str, size_t pos) {
    const std::string type_string = "# type:";
    if (str.size() < pos + type_string.length()) {
      return false;
    }
    auto match_string = str.substr(pos, type_string.size());
    return match_string == type_string;
  }

  TokenTrieRef head;
};

TORCH_API SharedParserData& sharedParserData();

struct Token {
  int kind;
  SourceRange range;
  Token(int kind, SourceRange range) : kind(kind), range(std::move(range)) {}
  std::string text() {
    return std::string(range.token_text());
  }
  std::string kindString() const {
    return kindToString(kind);
  }
};

struct Lexer {
  explicit Lexer(std::shared_ptr<Source> source)
      : source(std::move(source)),

        indent_stack(),
        next_tokens(),
        shared(sharedParserData()) {
    auto first_indent = lexRaw(true);
    indent_stack.push_back(first_indent.range.size());
    lex();
  }
  // Return the current token, and then move to the next one
  Token next() {
    if (next_tokens.empty())
      reportError("Lexer invariant violated: empty token queue");
    Token r = std::move(next_tokens.front());
    next_tokens.erase(next_tokens.begin());
    if (next_tokens.empty()) {
      lex();
    }
    return r;
  }
  // Skip the current token if it matches the given kind
  bool nextIf(int kind) {
    if (cur().kind != kind)
      return false;
    next();
    return true;
  }

  [[noreturn]] void reportError(const std::string& what) {
    reportError(what, cur());
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
    expected(what, cur());
  }
  // Check that the current token has a given kind, return the current token,
  // and advance to the next one.
  Token expect(int kind) {
    if (cur().kind != kind) {
      expected(kindToString(kind));
    }
    return next();
  }
  Token& lookahead() {
    if (next_tokens.size() < 2) {
      lex();
    }
    return next_tokens[1];
  }
  Token& cur() {
    return next_tokens.front();
  }

 private:
  void lex() {
    auto r = lexRaw();
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
      case TK_WHITESPACE:
      case TK_WHITESPACE_EOF: {
        const auto depth =
            r.kind == TK_WHITESPACE_EOF ? indent_stack.front() : r.range.size();
        // note: TK_WHITESPACE_EOF is whitespace right before the EOF token
        // just like we allow the code to be indented to a particular initial
        // indent level, we allow the final indent to be anything and set
        // it back to the initial indent level. This allows the code to be
        // put into string literals inside code without worrying about final
        // whitespace
        if (depth > indent_stack.back()) {
          indent_stack.push_back(depth);
          r.kind = TK_INDENT;
        } else if (depth == indent_stack.back()) {
          r.kind = TK_NEWLINE;
        } else {
          next_tokens.emplace_back(TK_NEWLINE, r.range);
          while (indent_stack.back() != depth) {
            indent_stack.pop_back();
            next_tokens.emplace_back(TK_DEDENT, r.range);
            if (indent_stack.empty()) {
              reportError("invalid indent level " + std::to_string(depth), r);
            }
          }
          return; // We've already queued the tokens
        }
      } break;
      default:
        break;
    }
    next_tokens.push_back(std::move(r));
  }
  Token lexRaw(bool whitespace_token = false) {
    AT_ASSERT(source);
    if (current == nullptr) {
      AT_ASSERT(pos == 0);
      current = std::make_unique<StringCordView::Iterator>(
          source->text_str().begin());
    }

    StringCordView::Iterator start_iter = *current;
    StringCordView::Iterator end_iter = *current;
    int kind = 0;
    if (!shared.match(
            *current,
            nesting > 0,
            whitespace_token,
            &kind,
            &start_iter,
            &end_iter)) {
      expected(
          "a valid token",
          Token(
              **current,
              SourceRange(source, start_iter, start_iter.pos() + 1)));
    }

    auto t = Token(kind, SourceRange(source, start_iter, end_iter.pos()));
    pos = end_iter.pos();
    *current = end_iter;
    return t;
  }

  std::shared_ptr<Source> source;
  std::unique_ptr<StringCordView::Iterator> current;
  size_t pos{0};
  size_t nesting{0}; // depth of ( [ { nesting...
  std::vector<size_t> indent_stack; // stack of indentation level of blocks
  // Invariant: this should always contain at least a single element
  std::vector<Token> next_tokens;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  SharedParserData& shared;
};
} // namespace torch::jit
