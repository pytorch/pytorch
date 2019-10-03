#include <string>
#include <pcrecpp.h>
#include <vector>

#include <ATen/core/op_registration/op_registration.h>

namespace {

// https://www.pcre.org/original/doc/html/pcresyntax.html
const pcrecpp::RE kPattern(
  "(*UCP)(\\'s|\\'t|\\'re|\\'ve|\\'m|\\'ll|\\'d| ?\\pL+|"
  " ?\\pN+| ?[^\\s\\pL\\pN]+|\\s+(?!\\S)|\\s+)",
  pcrecpp::UTF8()
);

std::vector<std::string> gpt2_bpe_tokenizer(std::string input) {
  pcrecpp::StringPiece line(input);
  std::string token;
  std::vector<std::string> tokens;

  while(kPattern.FindAndConsume(&line, &token)) {
    tokens.push_back(token);
  }
  return tokens;
}

static auto reg = torch::RegisterOperators().op(
    "internal::gpt2_bpe_tokenizer",
    &gpt2_bpe_tokenizer);

} //namespace
