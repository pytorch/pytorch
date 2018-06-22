// Unit tests for ProfilingInfo.
#include "caffe2/contrib/prof/profiling_info.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "caffe2/utils/proto_utils.h"

namespace caffe2 {
namespace contrib {
namespace prof {
namespace {

const char* kTestProfile = R"(
  stats {
    name: "op1"
    mean: 0
    stddev: 0

    execution_time {
      mean: 5
      stddev: 7
      count: 11
    }

    output_profile {
      name: "var1"
      bytes_used {
        mean: 13
        stddev: 17
        count: 10
      }
    }
    output_profile {
      name: "var12"
      bytes_used {
        mean: 43
        stddev: 47
        count: 53
      }
    }
  }
  stats {
    name: ""
    mean: 0
    stddev: 0

    execution_time {
      mean: 19
      stddev: 23
      count: 29
    }

    output_profile {
      name: "var2"

      bytes_used {
        mean: 31
        stddev: 37
        count: 41
      }
    }
  }
)";

const char* kTestNetDefCorrect = R"(
  op {
    name: "op1"
    output: "var1"
    output: "var12"
  }
  op {
    name: ""
    output: "var2"
  }
)";

const char* kTestNetDefPartial = R"(
  op {
    name: "op1"
    output: "var1"
  }
  op {
    name: ""
    output: "var_NET_WRONG"
  }
  op {
    name: "op_NET_WRONG"
  }
)";

TEST(ProfilingInfo, CorrectParse) {
  NetDef net_def;
  ProfDAGProtos profile;
  ASSERT_TRUE(TextFormat::ParseFromString(string(kTestProfile), &profile));
  ASSERT_TRUE(
      TextFormat::ParseFromString(string(kTestNetDefCorrect), &net_def));
  ProfilingInfo info;
  EXPECT_TRUE(info.Init(net_def, profile));
  EXPECT_EQ(3, info.getBlobMap().size());
  EXPECT_EQ(2, info.getOperatorMap().size());

  auto it = info.getBlobMap().find("var1");
  ASSERT_NE(info.getBlobMap().end(), it);
  EXPECT_FLOAT_EQ(13, it->second.getUsedBytes().getMean());

  it = info.getBlobMap().find("var12");
  ASSERT_NE(info.getBlobMap().end(), it);
  EXPECT_FLOAT_EQ(47, it->second.getUsedBytes().getStddev());

  it = info.getBlobMap().find("var2");
  ASSERT_NE(info.getBlobMap().end(), it);
  EXPECT_FLOAT_EQ(31, it->second.getUsedBytes().getMean());

  auto it2 = info.getOperatorMap().find(0);
  ASSERT_NE(info.getOperatorMap().end(), it2);
  EXPECT_FLOAT_EQ(5, it2->second.getExecutionTimeMs().getMean());

  it2 = info.getOperatorMap().find(1);
  ASSERT_NE(info.getOperatorMap().end(), it2);
  EXPECT_FLOAT_EQ(23, it2->second.getExecutionTimeMs().getStddev());
}

TEST(ProfilingInfo, PartialParse) {
  NetDef net_def;
  ProfDAGProtos profile;
  ASSERT_TRUE(TextFormat::ParseFromString(string(kTestProfile), &profile));
  ASSERT_TRUE(
      TextFormat::ParseFromString(string(kTestNetDefPartial), &net_def));
  ProfilingInfo info;
  EXPECT_FALSE(info.Init(net_def, profile));
  EXPECT_EQ(1, info.getBlobMap().size());
  EXPECT_EQ(2, info.getOperatorMap().size());

  auto it = info.getBlobMap().find("var1");
  ASSERT_NE(info.getBlobMap().end(), it);
  EXPECT_FLOAT_EQ(13, it->second.getUsedBytes().getMean());

  auto it2 = info.getOperatorMap().find(0);
  ASSERT_NE(info.getOperatorMap().end(), it2);
  EXPECT_FLOAT_EQ(5, it2->second.getExecutionTimeMs().getMean());

  it2 = info.getOperatorMap().find(1);
  ASSERT_NE(info.getOperatorMap().end(), it2);
  EXPECT_FLOAT_EQ(23, it2->second.getExecutionTimeMs().getStddev());
}

} // namespace
} // namespace prof
} // namespace contrib
} // namespace caffe2
