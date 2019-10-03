#include <locale>
#include <string>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/TensorTypeId.h>
#include <folly/FixedString.h>
#include <folly/String.h>

namespace {

constexpr auto kUnkToken = folly::makeFixedString("[UNK]");
constexpr auto kBeginningOfSentenceCharacter = folly::makeFixedString("[CLS]");
constexpr auto kEndOfSentenceCharacter = folly::makeFixedString("[SEP]");
constexpr auto kPadToken = folly::makeFixedString("[PAD]");
const std::locale kLocPunctProcessing = std::locale("en_US.UTF-8");

std::vector<std::string> SplitOnWhitespace(const std::string& line) {
  folly::StringPiece clean_line =
      folly::ltrimWhitespace(folly::StringPiece(line));
  clean_line = folly::rtrimWhitespace(clean_line);
  std::vector<std::string> split_line;
  folly::split(' ', clean_line, split_line);
  return split_line;
}

bool IsPunct(char ch) {
  return std::ispunct(ch, kLocPunctProcessing);
}

void TruncateSeqPair(
    std::vector<std::string>& tokens_a,
    std::vector<std::string>& tokens_b,
    int64_t max_seq_length) {

  // Truncates a sequence pair in place to the maximum length.
  // This is a simple heuristic which will always truncate the longer sequence
  // one token at a time. This makes more sense than truncating an equal percent
  // of tokens from each, since if one sequence is very short then each token
  // that's truncated likely contains more information than a longer sequence.

  int64_t total_length = tokens_a.size() + tokens_b.size();
  while (total_length > max_seq_length) {
    if (tokens_a.size() > tokens_b.size()) {
      tokens_a.pop_back();
    } else {
      tokens_b.pop_back();
    }
    total_length = tokens_a.size() + tokens_b.size();
  }
}

std::vector<std::string> SplitOnPunct(
    const std::vector<std::string>& split_line) {
  std::vector<std::string> result;
  for (const std::string& word : split_line) {
    std::string cur;
    for (char ch : word) {
      if (IsPunct(ch)) {
        if (!cur.empty()) {
          result.emplace_back(std::move(cur));
        }
        result.emplace_back(1, ch);
      } else {
        cur.push_back(ch);
      }
    }
    if (!cur.empty()) {
      result.emplace_back(std::move(cur));
    }
  }
  return result;
}

int64_t GreedyBPE(
    int64_t max_seq_length,
    int64_t unk_index,
    const c10::Dict<std::string, int64_t>& dict,
    const std::vector<std::string>& split_line,
    int64_t* tokens,
    int64_t* token_mask,
    int64_t curr_line_size = 1) {

  // greedy subword algorithm
  for (const std::string& word : split_line) {
    size_t char_start = 0;
    bool word_is_unk = false;
    while (char_start < word.size() && curr_line_size < max_seq_length - 1) {
      int64_t index = -1;
      size_t char_end = word.size();
      for (; char_start < char_end; --char_end) {
        const std::string cand_substr = char_start > 0
            ? "##" + word.substr(char_start, char_end - char_start)
            : word.substr(char_start, char_end - char_start);
        const auto it = dict.find(cand_substr);
        if (it != dict.end()) {
          index = it->value();
          break;
        }
      }
      if (index == -1) {
        word_is_unk = true;
        break;
      }
      tokens[curr_line_size] = index;
      token_mask[curr_line_size] = 1;
      ++curr_line_size;
      char_start = char_end;
    }
    if (word_is_unk) {
      tokens[curr_line_size] = unk_index;
      token_mask[curr_line_size] = 1;
      ++curr_line_size;
    }
  }
  return curr_line_size;
}

std::vector<at::Tensor> wordpiece_tokenizer(
    c10::List<std::string> input_text,
    c10::Dict<std::string, int64_t> vocab,
    int64_t max_seq_len) {
  DCHECK(vocab.contains(kUnkToken));
  DCHECK(vocab.contains(kBeginningOfSentenceCharacter));
  DCHECK(vocab.contains(kEndOfSentenceCharacter));
  DCHECK(vocab.contains(kPadToken));

  // initialize tokenized with padIndex
  const int64_t N = input_text.size();
  const int64_t pad_index = vocab.find(kPadToken)->value();
  const int64_t beg_index = vocab.find(kBeginningOfSentenceCharacter)->value();
  const int64_t end_index = vocab.find(kEndOfSentenceCharacter)->value();
  const int64_t unk_index = vocab.find(kUnkToken)->value();
  const auto kInt64TensorOpt = c10::TensorOptions().dtype(at::kLong);
  at::Tensor tokenized =
      at::empty({N, max_seq_len}, kInt64TensorOpt).fill_(pad_index);
  at::Tensor token_masks = at::zeros({N, max_seq_len}, kInt64TensorOpt);
  at::Tensor segment_ids = at::ones({N, max_seq_len}, kInt64TensorOpt);

  int64_t* tokens_data = tokenized.data_ptr<int64_t>();
  int64_t* token_mask_data = token_masks.data_ptr<int64_t>();
  // run bpe
  for (int64_t i = 0; i < N; ++i) {
    int64_t* tokens_ptr = tokens_data + i * max_seq_len;
    int64_t* token_mask_ptr = token_mask_data + i * max_seq_len;
    const std::string& line = input_text[i];
    std::vector<std::string> split_line = SplitOnWhitespace(line);
    std::vector<std::string> split_punct_line = SplitOnPunct(split_line);

    // first add beginningOfSentenceCharacter
    tokens_ptr[0] = beg_index;
    token_mask_ptr[0] = 1;

    auto curr_line_size = GreedyBPE(
        max_seq_len,
        unk_index,
        vocab,
        split_punct_line,
        tokens_ptr,
        token_mask_ptr);

    // end of sentence character
    tokens_ptr[curr_line_size] = end_index;
    token_mask_ptr[curr_line_size] = 1;
  }

  return {tokenized, token_masks, segment_ids};
}

std::vector<at::Tensor> wordpiece_pairwise_classification_tokenizer(
    c10::List<std::string> input_text_a,
    c10::List<std::string> input_text_b,
    c10::Dict<std::string, int64_t> vocab,
    int64_t max_seq_len) {
  DCHECK(vocab.contains(kUnkToken));
  DCHECK(vocab.contains(kBeginningOfSentenceCharacter));
  DCHECK(vocab.contains(kEndOfSentenceCharacter));
  DCHECK(vocab.contains(kPadToken));

  // initialize tokenized with padIndex
  const int64_t N = input_text_a.size();
  const int64_t pad_index = vocab.find(kPadToken)->value();
  const int64_t beg_index = vocab.find(kBeginningOfSentenceCharacter)->value();
  const int64_t end_index = vocab.find(kEndOfSentenceCharacter)->value();
  const int64_t unk_index = vocab.find(kUnkToken)->value();
  const auto kInt64TensorOpt = c10::TensorOptions().dtype(at::kLong);
  at::Tensor tokenized =
      at::empty({N, max_seq_len}, kInt64TensorOpt).fill_(pad_index);
  at::Tensor token_masks = at::zeros({N, max_seq_len}, kInt64TensorOpt);
  at::Tensor segment_ids = at::zeros({N, max_seq_len}, kInt64TensorOpt);

  int64_t* tokens_data = tokenized.data_ptr<int64_t>();
  int64_t* token_mask_data = token_masks.data_ptr<int64_t>();
  int64_t* segment_ids_data = segment_ids.data_ptr<int64_t>();

  // run bpe
  for (int64_t i = 0; i < N; ++i) {
    int64_t* tokens_ptr = tokens_data + i * max_seq_len;
    int64_t* token_mask_ptr = token_mask_data + i * max_seq_len;
    int64_t* segment_ids_ptr = segment_ids_data + i * max_seq_len;

    auto line_a = input_text_a[i];
    auto line_b = input_text_b[i];

    std::vector<std::string> split_line_a = SplitOnWhitespace(line_a);
    std::vector<std::string> split_punct_line_a = SplitOnPunct(split_line_a);

    std::vector<std::string> split_line_b = SplitOnWhitespace(line_b);
    std::vector<std::string> split_punct_line_b = SplitOnPunct(split_line_b);

    TruncateSeqPair(split_punct_line_a, split_punct_line_b, max_seq_len);

    // first add beginningOfSentenceCharacter
    tokens_ptr[0] = beg_index;
    token_mask_ptr[0] = 1;

    int64_t curr_line_size_after_first_sentence = GreedyBPE(
        max_seq_len,
        unk_index,
        vocab,
        split_punct_line_a,
        tokens_ptr,
        token_mask_ptr);

    // end of sentence character before second sentence
    tokens_ptr[curr_line_size_after_first_sentence] = end_index;
    token_mask_ptr[curr_line_size_after_first_sentence] = 1;
    curr_line_size_after_first_sentence++;

    int64_t curr_line_size_after_second_sentence = GreedyBPE(
        max_seq_len,
        unk_index,
        vocab,
        split_punct_line_b,
        tokens_ptr,
        token_mask_ptr,
        curr_line_size_after_first_sentence);

    // end of sentence character after second sentence
    tokens_ptr[curr_line_size_after_second_sentence] = end_index;
    token_mask_ptr[curr_line_size_after_second_sentence] = 1;

    // fill segment_ids with 1's only for second sentence
    for (int64_t y = curr_line_size_after_first_sentence;
         y < curr_line_size_after_second_sentence + 1;
         y++) {
      segment_ids_ptr[y] = 1;
    }
  }

  return {tokenized, token_masks, segment_ids};
}

// Register operator
static auto reg1 = torch::RegisterOperators().op(
    "internal::wordpiece_tokenizer",
    &wordpiece_tokenizer);

static auto reg2 = torch::RegisterOperators().op(
    "internal::wordpiece_pairwise_classification_tokenizer",
    &wordpiece_pairwise_classification_tokenizer);

} // namespace
