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
    std::vector<std::string> extras = {"n", "no", "no3", "note"};
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
    std::vector<std::string> extras = {"i", "i3", "inn"};
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
} // namespace torch::jit
