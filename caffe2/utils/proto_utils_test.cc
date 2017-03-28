#include "caffe2/utils/proto_utils.h"
#include <gtest/gtest.h>

namespace caffe2 {

TEST(ProtoUtilsTest, SimpleReadWrite) {
  string content("The quick brown fox jumps over the lazy dog.");
  string name = std::tmpnam(nullptr);
  EXPECT_TRUE(WriteStringToFile(content, name.c_str()));
  string read_back;
  EXPECT_TRUE(ReadStringFromFile(name.c_str(), &read_back));
  EXPECT_EQ(content, read_back);
}

}  // namespace caffe2
