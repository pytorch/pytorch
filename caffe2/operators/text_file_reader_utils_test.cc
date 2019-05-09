#include <fstream>
#include "caffe2/core/blob.h"
#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/math.h"
#include <gtest/gtest.h>

#include "caffe2/operators/text_file_reader_utils.h"
#include "caffe2/utils/string_utils.h"

#include <cstdio>
#include <cstdlib>

namespace caffe2 {

TEST(TextFileReaderUtilsTest, TokenizeTest) {
  TokenizedString tokenized;
  std::string ch =
      "label\1text\xc3\xbf\nlabel2\\\nTest\1tex\\\\t2\n"
      "Two\\\\Escapes\\\1\1Second\n";
  std::vector<char> seps = {'\n', '\1'};
  Tokenizer tokenizer(seps, '\\');
  tokenizer.next(&ch.front(), &ch.back() + 1, tokenized);

  std::vector<std::pair<int, std::string>> expected = {{0, "label"},
                                                       {1, "text\xc3\xbf"},
                                                       {0, "label2\nTest"},
                                                       {1, "tex\\t2"},
                                                       {0, "Two\\Escapes\1"},
                                                       {1, "Second"}};

  EXPECT_EQ(expected.size(), tokenized.tokens().size());
  for (int i = 0; i < expected.size(); ++i) {
    const auto& token = tokenized.tokens().at(i);
    EXPECT_EQ(expected.at(i).first, token.startDelimId);
    EXPECT_EQ(expected.at(i).second, std::string(token.start, token.end));
  }

  // try each of the subsplits
  for (int i = 0; i < ch.size() - 1; ++i) {
    tokenizer.reset();
    char* mid = &ch.front() + i;

    tokenizer.next(&ch.front(), mid, tokenized);
    EXPECT_GE(expected.size(), tokenized.tokens().size());
    for (int j = 0; j < tokenized.tokens().size(); ++j) {
      const auto& token = tokenized.tokens().at(j);
      EXPECT_EQ(expected.at(j).first, token.startDelimId);
      EXPECT_EQ(expected.at(j).second, std::string(token.start, token.end));
    }
    int s1 = tokenized.tokens().size();

    tokenizer.next(mid, &ch.back() + 1, tokenized);
    EXPECT_EQ(expected.size(), s1 + tokenized.tokens().size());
    for (int j = 0; j < tokenized.tokens().size(); ++j) {
      const auto& token = tokenized.tokens().at(j);
      EXPECT_EQ(expected.at(j + s1).first, token.startDelimId);
      EXPECT_EQ(
          expected.at(j + s1).second, std::string(token.start, token.end));
    }
    EXPECT_EQ(0, tokenized.lastDelim());
  }

  struct ChunkProvider : public StringProvider {
    ChunkProvider(const std::string& str) : ch(str) {}
    std::string ch;
    size_t charIdx{0};
    void operator()(CharRange& range) override {
      if (charIdx >= ch.size()) {
        range.start = nullptr;
        range.end = nullptr;
      } else {
        size_t endIdx = std::min(charIdx + 10, ch.size());
        range.start = &ch.front() + charIdx;
        range.end = &ch.front() + endIdx;
        charIdx = endIdx;
      }
    };
    void reset() override {
      charIdx = 0;
    }
  };

  for (int numPasses = 1; numPasses <= 2; ++numPasses) {
    ChunkProvider chunkProvider(ch);
    BufferedTokenizer bt(tokenizer, &chunkProvider, numPasses);
    Token token;
    int i = 0;
    for (i = 0; bt.next(token); ++i) {
      EXPECT_GT(expected.size() * numPasses, i);
      const auto& expectedToken = expected.at(i % expected.size());
      EXPECT_EQ(expectedToken.first, token.startDelimId);
      EXPECT_EQ(expectedToken.second, std::string(token.start, token.end));
    }
    EXPECT_EQ(expected.size() * numPasses, i);
    EXPECT_EQ(0, bt.endDelim());
  }

  char* tmpname = std::tmpnam(nullptr);
  std::ofstream outFile;
  outFile.open(tmpname);
  outFile << ch;
  outFile.close();
  for (int numPasses = 1; numPasses <= 2; ++numPasses) {
    FileReader fr(tmpname, 5);
    BufferedTokenizer fileTokenizer(tokenizer, &fr, numPasses);
    Token token;
    int i;
    for (i = 0; fileTokenizer.next(token); ++i) {
      EXPECT_GT(expected.size() * numPasses, i);
      const auto& expectedToken = expected.at(i % expected.size());
      EXPECT_EQ(expectedToken.first, token.startDelimId);
      EXPECT_EQ(expectedToken.second, std::string(token.start, token.end));
    }
    EXPECT_EQ(expected.size() * numPasses, i);
    EXPECT_EQ(0, fileTokenizer.endDelim());
  }
  std::remove(tmpname);
}

} // namespace caffe2
