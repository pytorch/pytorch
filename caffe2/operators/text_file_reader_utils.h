#ifndef CAFFE2_OPERATORS_TEXT_FILE_READER_UTILS_H
#define CAFFE2_OPERATORS_TEXT_FILE_READER_UTILS_H

#include <memory>
#include <string>
#include <vector>

#include "caffe2/core/common.h"

namespace caffe2 {

struct Token {
  int startDelimId;
  const char* start;
  const char* end;
};

class TokenizedString {
  // holder for strings that have been modified
  std::vector<std::unique_ptr<std::string>> modifiedStrings_;
  std::vector<Token> tokens_;
  int lastDelim_;

 public:
  const std::vector<Token>& tokens() const {
    return tokens_;
  }
  int lastDelim() const {
    return lastDelim_;
  }
  friend class Tokenizer;
};

class Tokenizer {
 private:
  int startDelimId_;
  // state of the tokenizer
  std::string leftover_;
  // if we need to skip the first characters of the next batch because
  // e.g. an escape char that was the last character of the last batch.
  int toBeSkipped_;
  int delimTable_[256];
  const char escape_;

 public:
  Tokenizer(const std::vector<char>& delimiters, char escape);
  void reset();
  void next(char* start, char* end, TokenizedString& tokenized);
};

struct CharRange {
  char* start;
  char* end;
};

struct StringProvider {
  virtual void operator()(CharRange&) = 0;
  virtual void reset() = 0;
  virtual ~StringProvider() {}
};

class BufferedTokenizer {
 public:
  BufferedTokenizer(const Tokenizer& t, StringProvider* p, int numPasses = 1)
      : provider_(p), tokenizer_(t), tokenIndex_(0), numPasses_(numPasses) {}

  bool next(Token& token) {
    CharRange range;
    while (tokenIndex_ >= tokenized_.tokens().size()) {
      range.start = nullptr;
      while (range.start == nullptr && pass_ < numPasses_) {
        (*provider_)(range);
        if (range.start == nullptr) {
          ++pass_;
          if (pass_ < numPasses_) {
            provider_->reset();
            tokenizer_.reset();
          }
        }
      }
      if (range.start == nullptr) {
        return false;
      }
      tokenizer_.next(range.start, range.end, tokenized_);
      tokenIndex_ = 0;
    }
    token = tokenized_.tokens()[tokenIndex_++];
    return true;
  };

  int endDelim() const {
    if (tokenIndex_ + 1 < tokenized_.tokens().size()) {
      return tokenized_.tokens()[tokenIndex_ + 1].startDelimId;
    }
    return tokenized_.lastDelim();
  }

 private:
  StringProvider* provider_;
  Tokenizer tokenizer_;
  TokenizedString tokenized_;
  int tokenIndex_;
  int numPasses_;
  int pass_{0};
};

class FileReader : public StringProvider {
 public:
  explicit FileReader(const std::string& path, size_t bufferSize = 65536);
  ~FileReader();
  void operator()(CharRange& range) override;
  void reset() override;

 private:
  const size_t bufferSize_;
  int fd_;
  std::unique_ptr<char[]> buffer_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TEXT_FILE_READER_UTILS_H
