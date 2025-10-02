#include <gtest/gtest.h>

#include <torch/csrc/jit/frontend/lexer.h>

namespace torch::jit {

TEST(LexerTest, AllTokens) {
  std::vector<std::pair<int /* TokenKind */, std::string>> tokens;
  for (const char* ch = valid_single_char_tokens; *ch; ch++) {
    tokens.emplace_back(*ch, std::string(1, *ch));
  }
#define ADD_TOKEN(tok, _, tokstring)     \
  if (*tokstring) {                      \
    tokens.emplace_back(tok, tokstring); \
  }
  TC_FORALL_TOKEN_KINDS(ADD_TOKEN);
#undef ADD_TOKEN

  for (const auto& [kind, token] : tokens) {
    Lexer l(std::make_shared<Source>(token));
    const auto& tok = l.cur();
    EXPECT_EQ(kind, tok.kind) << tok.range.text().str();
    EXPECT_EQ(token, tok.range.text().str()) << tok.range.text().str();
    l.next();
    EXPECT_EQ(l.cur().kind, TK_EOF);
  }
}

TEST(LexerTest, SlightlyOffIsNot) {
  std::vector<std::string> suffixes = {"", " ", "**"};
  for (const auto& suffix : suffixes) {
    std::vector<std::string> extras = {"n", "no", "no3"};
    for (const auto& extra : extras) {
      std::string s = "is " + extra + suffix;
      Lexer l(std::make_shared<Source>(s));
      const auto& is_tok = l.next();
      EXPECT_EQ(is_tok.kind, TK_IS) << is_tok.range.text().str();
      const auto& no_tok = l.cur();
      EXPECT_EQ(no_tok.kind, TK_IDENT) << no_tok.range.text().str();
      EXPECT_EQ(no_tok.range.text().str(), extra) << no_tok.range.text().str();
    }
  }
}

TEST(LexerTest, SlightlyOffNotIn) {
  std::vector<std::string> suffixes = {"", " ", "**"};
  for (const auto& suffix : suffixes) {
    std::vector<std::string> extras = {"i", "i3"};
    for (const auto& extra : extras) {
      std::string s = "not " + extra + suffix;
      Lexer l(std::make_shared<Source>(s));
      const auto& not_tok = l.next();
      EXPECT_EQ(not_tok.kind, TK_NOT) << not_tok.range.text().str();
      const auto& in_tok = l.cur();
      EXPECT_EQ(in_tok.kind, TK_IDENT) << in_tok.range.text().str();
      EXPECT_EQ(in_tok.range.text().str(), extra) << in_tok.range.text().str();
    }
  }
}

TEST(LexerTest, IsNoteBug) {
  // The code string `is note` is lexed as TK_ISNOT followed by a
  // TK_IDENT that is an e. This is not how it works in Python, but
  // presumably we need to maintain this behavior.
  Lexer l(std::make_shared<Source>("is note"));
  const auto is_not_tok = l.next();
  EXPECT_EQ(is_not_tok.kind, TK_ISNOT);
  const auto e_tok = l.next();
  EXPECT_EQ(e_tok.kind, TK_IDENT);
  EXPECT_EQ(e_tok.range.text(), "e");
  const auto eof_tok = l.next();
  EXPECT_EQ(eof_tok.kind, TK_EOF);
}

TEST(LexerTest, NotInpBug) {
  // Another manifestation of the above IsNoteBug; `not inp` is lexed
  // as TK_NOT_IN followed by a TK_IDENT that is a p. Again, not how
  // it works in Python.
  Lexer l(std::make_shared<Source>("not inp"));
  const auto not_in_tok = l.next();
  EXPECT_EQ(not_in_tok.kind, TK_NOTIN);
  const auto p_tok = l.next();
  EXPECT_EQ(p_tok.kind, TK_IDENT);
  EXPECT_EQ(p_tok.range.text(), "p");
  const auto eof_tok = l.next();
  EXPECT_EQ(eof_tok.kind, TK_EOF);
}
} // namespace torch::jit
