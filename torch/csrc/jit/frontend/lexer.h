#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/parser_constants.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/strtod.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <clocale>
#include <cstdlib>
#include <cstring>
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

// stuff that is shared against all TC lexers/parsers and is initialized only
// once.
struct TORCH_API SharedParserData {
  SharedParserData() = default;

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

    if (std::isalpha(*pos) || *pos == '_') {
      matchIdentOrKeyword(pos, kind, end);
      return true;
    }

    // Hand-coded DFA matching for tokens that cannot be confused with
    // identifiers. We could use a lexer generator toolkit like Flex
    // or re2c instead, but that would add another dependency, and I
    // expect this component to change infrequently given that PyTorch
    // 2.0 is years old already. Note that the tests in text_lexer.cpp
    // should guarantee that we don't forget to update this when we
    // update TC_FORALL_TOKEN_KINDS.
    const auto next_pos = pos.next_iter();
    switch (*pos) {
      case '+': {
        if (pos.has_next() && *next_pos == '=') {
          *end = next_pos.next_iter();
          *kind = TK_PLUS_EQ;
          return true;
        }
        goto single_char_token;
      }
      case '-':
        if (pos.has_next()) {
          if (*next_pos == '=') {
            *end = next_pos.next_iter();
            *kind = TK_MINUS_EQ;
            return true;
          }
          if (*next_pos == '>') {
            *end = next_pos.next_iter();
            *kind = TK_ARROW;
            return true;
          }
        }
        goto single_char_token;
      case '*':
        if (pos.has_next()) {
          if (*next_pos == '*') {
            if (next_pos.has_next() && *next_pos.next_iter() == '=') {
              *end = next_pos.next_iter().next_iter();
              *kind = TK_POW_EQ;
              return true;
            }
            *end = next_pos.next_iter();
            *kind = TK_POW;
            return true;
          }
          if (*next_pos == '=') {
            *end = next_pos.next_iter();
            *kind = TK_TIMES_EQ;
            return true;
          }
        }
        goto single_char_token;
      case '/':
        if (pos.has_next()) {
          if (*next_pos == '/') {
            *end = next_pos.next_iter();
            *kind = TK_FLOOR_DIV;
            return true;
          }
          if (*next_pos == '=') {
            *end = next_pos.next_iter();
            *kind = TK_DIV_EQ;
            return true;
          }
        }
        goto single_char_token;
      case '%':
        if (pos.has_next()) {
          if (*next_pos == '=') {
            *end = next_pos.next_iter();
            *kind = TK_MOD_EQ;
            return true;
          }
        }
        goto single_char_token;
      case '=':
        if (pos.has_next()) {
          if (*next_pos == '=') {
            *end = next_pos.next_iter();
            *kind = TK_EQ;
            return true;
          }
        }
        goto single_char_token;
      case '>':
        if (pos.has_next()) {
          if (*next_pos == '=') {
            *end = next_pos.next_iter();
            *kind = TK_GE;
            return true;
          }
          if (*next_pos == '>') {
            if (next_pos.has_next() && *next_pos.next_iter() == '=') {
              *end = next_pos.next_iter().next_iter();
              *kind = TK_RSHIFT_EQ;
              return true;
            }
            *end = next_pos.next_iter();
            *kind = TK_RSHIFT;
            return true;
          }
        }
        goto single_char_token;
      case '<':
        if (pos.has_next()) {
          if (*next_pos == '=') {
            if (next_pos.has_next() && *next_pos.next_iter() == '>') {
              *end = next_pos.next_iter().next_iter();
              *kind = TK_EQUIVALENT;
              return true;
            }
            *end = next_pos.next_iter();
            *kind = TK_LE;
            return true;
          }
          if (*next_pos == '<') {
            if (next_pos.has_next() && *next_pos.next_iter() == '=') {
              *end = next_pos.next_iter().next_iter();
              *kind = TK_LSHIFT_EQ;
              return true;
            }
            *end = next_pos.next_iter();
            *kind = TK_LSHIFT;
            return true;
          }
        }
        goto single_char_token;
      case '.':
        if (pos.has_next()) {
          if (*next_pos == '.' && next_pos.has_next() &&
              *next_pos.next_iter() == '.') {
            *end = next_pos.next_iter().next_iter();
            *kind = TK_DOTS;
            return true;
          }
        }
        goto single_char_token;
      case '!':
        if (pos.has_next()) {
          if (*next_pos == '=') {
            *end = next_pos.next_iter();
            *kind = TK_NE;
            return true;
          }
        }
        goto single_char_token;
      case '&':
        if (pos.has_next()) {
          if (*next_pos == '=') {
            *end = next_pos.next_iter();
            *kind = TK_BIT_AND_EQ;
            return true;
          }
        }
        goto single_char_token;
      case '^':
        if (pos.has_next()) {
          if (*next_pos == '=') {
            *end = next_pos.next_iter();
            *kind = TK_BIT_XOR_EQ;
            return true;
          }
        }
        goto single_char_token;
      case '|':
        if (pos.has_next()) {
          if (*next_pos == '=') {
            *end = next_pos.next_iter();
            *kind = TK_BIT_OR_EQ;
            return true;
          }
        }
        goto single_char_token;
      case '#':
        *end = pos + std::strlen("# type:");
        *kind = TK_TYPE_COMMENT;
        return true;
      case '@':
      case '(':
      case ')':
      case '[':
      case ']':
      case ':':
      case ',':
      case '{':
      case '}':
      case '?':
      case '~':
      single_char_token:
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            std::strchr(valid_single_char_tokens, *pos) != nullptr,
            "Did you forget to add the character `",
            *pos,
            "` to valid_single_char_tokens?");
        *end = next_pos;
        *kind = *pos;
        return true;
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        std::strchr(valid_single_char_tokens, *pos) == nullptr,
        "Did you forget to add the character `",
        *pos,
        "` to the above switch statement?");
    return false;
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
  void matchIdentOrKeyword(
      StringCordView::Iterator pos,
      int* kind,
      StringCordView::Iterator* end) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(pos.has_next());
    static constexpr char kIsNot[] = "is not";
    static constexpr char kNotIn[] = "not in";
    constexpr char kMaybeIsNot = 'i';
    constexpr char kMaybeNotIn = 'n';
    constexpr int kIsNotSpaceIndex = 2;
    constexpr int kNotInSpaceIndex = 3;
    auto start = pos;
    char possible_special_token = *pos;
    // The longest tokens are 8 chars.
    std::array<char, 8> token_chars;
    token_chars.fill('\0');
    token_chars[0] = possible_special_token;
    ++pos;
    size_t i;
    auto valid_ident_char = [](const char ch) {
      return std::isalpha(ch) || ch == '_' || std::isdigit(ch);
    };
    for (i = 1; pos.has_next(); ++pos, ++i) {
      auto ch = *pos;
      if (possible_special_token == kMaybeIsNot) {
        if (ch != kIsNot[i]) {
          if (i >= kIsNotSpaceIndex + 1) {
            // Kick out to the after-loop flow, which will correctly
            // record that we found TK_IS.
            break;
          }
          possible_special_token = '\0';
        } else if (ch == ' ') {
          continue;
        }
        if (possible_special_token && i == sizeof(kIsNot) - 2 &&
            (!pos.has_next() || !valid_ident_char(*(pos + 1)))) {
          *kind = TK_ISNOT;
          *end = pos.next_iter();
          return;
        }
      } else if (possible_special_token == kMaybeNotIn) {
        if (ch != kNotIn[i]) {
          if (i >= kNotInSpaceIndex + 1) {
            // Kick out to the after-loop flow, which will correctly
            // record that we found TK_NOT.
            break;
          }
          possible_special_token = '\0';
        } else if (ch == ' ') {
          continue;
        }

        if (possible_special_token && i == sizeof(kNotIn) - 2 &&
            (!pos.has_next() || !valid_ident_char(*(pos + 1)))) {
          *kind = TK_NOTIN;
          *end = pos.next_iter();
          return;
        }
      }
      if (valid_ident_char(ch)) {
        if (i < token_chars.size()) {
          token_chars[i] = ch;
        }
        continue;
      }
      break;
    }

    // These two possible_special_token checks have to be after the
    // loop and not in the loop because we might see end-of-input
    // (e.g., the entire input `not p`).
    if (possible_special_token == kMaybeIsNot) {
      if (i >= kIsNotSpaceIndex) {
        *kind = TK_IS;
        *end = start + kIsNotSpaceIndex;
        return;
      }
    } else if (possible_special_token == kMaybeNotIn) {
      if (i >= kNotInSpaceIndex) {
        *kind = TK_NOT;
        *end = start + kNotInSpaceIndex;
        return;
      }
    }

    *end = pos;
    *kind = identTokenKind(token_chars, i);
  }

  template <size_t N>
  static constexpr uint64_t stringToUint64(const char (&str)[N]) {
    static_assert(N <= 9);
    uint64_t result = 0;
    for (auto i : c10::irange(N)) {
      if (!str[i]) {
        return result;
      }
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      result |= static_cast<uint64_t>(str[i]) << (8 * i);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      result |= static_cast<uint64_t>(str[i]) << (56 - 8 * i);
#else
#error "Unexpected or undefined value of __BYTE_ORDER__"
#endif
    }
    return result;
  }

  static int identTokenKind(
      const std::array<char, 8>& token_chars,
      size_t token_length) {
    if (token_length > token_chars.size()) {
      return TK_IDENT;
    }
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    static_assert(stringToUint64("and") == 0x646e61);
    static_assert(stringToUint64("Ellipsis") == 0x73697370696c6c45);
#else
    static_assert(stringToUint64("and") == 0x616e640000000000);
    static_assert(stringToUint64("Ellipsis") == 0x456c6c6970736973);
#endif

    std::uint64_t token = 0;
    std::memcpy(&token, token_chars.data(), token_chars.size());
    // FWIW, based on checking Godbolt this probably compiles down to
    // binary or linear search over the integers representing our
    // strings. I tried an alternate version that switched on the
    // first character of the token, but it doesn't seem to matter for
    // performance.
    switch (token) {
      case stringToUint64("Ellipsis"):
        return TK_ELLIPSIS;
      case stringToUint64("False"):
        return TK_FALSE;
      case stringToUint64("None"):
        return TK_NONE;
      case stringToUint64("NoneType"):
        return TK_NONE_TYPE;
      case stringToUint64("True"):
        return TK_TRUE;
      case stringToUint64("and"):
        return TK_AND;
      case stringToUint64("as"):
        return TK_AS;
      case stringToUint64("assert"):
        return TK_ASSERT;
      case stringToUint64("break"):
        return TK_BREAK;
      case stringToUint64("class"):
        return TK_CLASS_DEF;
      case stringToUint64("continue"):
        return TK_CONTINUE;
      case stringToUint64("def"):
        return TK_DEF;
      case stringToUint64("del"):
        return TK_DELETE;
      case stringToUint64("elif"):
        return TK_ELIF;
      case stringToUint64("else"):
        return TK_ELSE;
      case stringToUint64("for"):
        return TK_FOR;
      case stringToUint64("global"):
        return TK_GLOBAL;
      case stringToUint64("if"):
        return TK_IF;
      case stringToUint64("import"):
        return TK_IMPORT;
      case stringToUint64("in"):
        return TK_IN;
      case stringToUint64("is"):
        return TK_IS;
      case stringToUint64("not"):
        return TK_NOT;
      case stringToUint64("or"):
        return TK_OR;
      case stringToUint64("pass"):
        return TK_PASS;
      case stringToUint64("raise"):
        return TK_RAISE;
      case stringToUint64("return"):
        return TK_RETURN;
      case stringToUint64("while"):
        return TK_WHILE;
      case stringToUint64("with"):
        return TK_WITH;
      default:
        return TK_IDENT;
    }
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
    if (first == '-' || first == '+' || std::isalpha(first))
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
};

TORCH_API SharedParserData& sharedParserData();

struct Token {
  int kind;
  SourceRange range;
  Token(int kind, SourceRange range) : kind(kind), range(std::move(range)) {}
  std::string text() const {
    return std::string(range.token_text());
  }

  std::string_view text_view() const {
    return range.token_text();
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
