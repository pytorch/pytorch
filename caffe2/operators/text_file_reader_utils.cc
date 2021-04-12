#include "caffe2/operators/text_file_reader_utils.h"

#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <sstream>

#include <c10/util/irange.h>

namespace caffe2 {

Tokenizer::Tokenizer(const std::vector<char>& delims, char escape)
    : escape_(escape) {
  reset();
  std::memset(delimTable_, 0, sizeof(delimTable_));
  for (const auto i : c10::irange(delims.size())) {
    delimTable_[(unsigned char)delims.at(i)] = i + 1;
  }
}

void Tokenizer::reset() {
  toBeSkipped_ = 0;
  startDelimId_ = 0;
  leftover_.clear();
}

void Tokenizer::next(char* start, char* end, TokenizedString& tokenized) {
  tokenized.modifiedStrings_.clear();
  tokenized.tokens_.clear();

  char* currentStart = start;
  std::string* copied = nullptr;
  if (!leftover_.empty()) {
    tokenized.modifiedStrings_.emplace_back(std::make_shared<std::string>());
    copied = tokenized.modifiedStrings_.back().get();
    *copied = std::move(leftover_);
  }

  char* ch;
  for (ch = start + toBeSkipped_; ch < end; ++ch) {
    if (*ch == escape_) {
      if (!copied) {
        tokenized.modifiedStrings_.emplace_back(std::make_shared<std::string>());
        copied = tokenized.modifiedStrings_.back().get();
      }
      copied->append(currentStart, ch);
      currentStart = ch + 1;
      // skip next character, since it's escaped
      ++ch;
      continue;
    }
    int newDelimId = delimTable_[(unsigned char)*ch];
    if (newDelimId > 0) {
      // found delimiter
      tokenized.tokens_.emplace_back();
      auto& token = tokenized.tokens_.back();
      token.startDelimId = startDelimId_;
      if (copied) {
        copied->append(currentStart, ch);
        const char* c_str = copied->data();
        token.start = c_str;
        token.end = c_str + copied->size();
      } else {
        token.start = currentStart;
        token.end = ch;
      }
      currentStart = ch + 1;
      copied = nullptr;
      startDelimId_ = newDelimId - 1;
    }
  }
  tokenized.lastDelim_ = startDelimId_;

  toBeSkipped_ = ch - end;
  if (copied) {
    copied->append(currentStart, end);
    leftover_ = std::move(*copied);
  } else {
    leftover_.assign(currentStart, end);
  }
}

FileReader::FileReader(const std::string& path, size_t bufferSize)
    : bufferSize_(bufferSize), buffer_(new char[bufferSize]) {
  fd_ = open(path.c_str(), O_RDONLY, 0777);
  if (fd_ < 0) {
    throw std::runtime_error(
        "Error opening file for reading: " + std::string(std::strerror(errno)) +
        " Path=" + path);
  }
}

void FileReader::reset() {
  if (lseek(fd_, 0, SEEK_SET) == -1) {
    throw std::runtime_error(
        "Error reseting file cursor: " + std::string(std::strerror(errno)));
  }
}

FileReader::~FileReader() {
  if (fd_ >= 0) {
    close(fd_);
  }
}

void FileReader::operator()(CharRange& range) {
  char* buffer = buffer_.get();
  auto numRead = read(fd_, buffer, bufferSize_);
  if (numRead == -1) {
    throw std::runtime_error(
        "Error reading file: " + std::string(std::strerror(errno)));
  }
  if (numRead == 0) {
    range.start = nullptr;
    range.end = nullptr;
    return;
  }
  range.start = buffer;
  range.end = buffer + numRead;
}
}
