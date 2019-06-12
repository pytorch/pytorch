// Unit tests for ProfilingInfo.
#include "caffe2/contrib/prof/profiling_info.h"

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>

#include "caffe2/utils/proto_utils.h"

namespace caffe2 {
namespace contrib {
namespace prof {
namespace {

const char* kTestProfile = R"(
  stats {
    name: "op1"
    mean: 5
    stddev: 7

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
    mean: 19
    stddev: 23

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
  net_name: "example_net"
)";

const char* kTestNetDefCorrect = R"(
  name: "example_net"
  op {
    name: "op1"
    type: "add"
    output: "var1"
    output: "var12"
  }
  op {
    name: ""
    type: "mult"
    output: "var2"
  }
)";

const char* kTestNetDefPartial = R"(
  name: "example_net"
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

TEST(ProfilingInfoTest, CorrectParse) {
  NetDef net_def;
  ProfDAGProtos profile;
  ASSERT_TRUE(TextFormat::ParseFromString(string(kTestProfile), &profile));
  ASSERT_TRUE(
      TextFormat::ParseFromString(string(kTestNetDefCorrect), &net_def));
  ProfilingInfo info;
  EXPECT_TRUE(info.Restore(net_def, profile));
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

TEST(ProfilingInfoTest, PartialParse) {
  NetDef net_def;
  ProfDAGProtos profile;
  ASSERT_TRUE(TextFormat::ParseFromString(string(kTestProfile), &profile));
  ASSERT_TRUE(
      TextFormat::ParseFromString(string(kTestNetDefPartial), &net_def));
  ProfilingInfo info;
  EXPECT_FALSE(info.Restore(net_def, profile));
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

TEST(ProfilingInfoTest, InitAndAddStats) {
  NetDef net_def;
  ProfilingInfo info;
  ASSERT_TRUE(
      TextFormat::ParseFromString(string(kTestNetDefCorrect), &net_def));
  info.Init(net_def);
  // Expect correct number of items.
  EXPECT_EQ(3, info.getBlobMap().size());
  EXPECT_EQ(2, info.getOperatorMap().size());

  // Expect correct key contents.
  auto mutable_blob_it = info.getMutableBlobMap()->find("var1");
  ASSERT_NE(info.getMutableBlobMap()->end(), mutable_blob_it);
  // Example profiling call:
  mutable_blob_it->second.getMutableUsedBytes()->addPoint(3);
  auto const_blob_it = info.getBlobMap().find("var12");
  EXPECT_NE(info.getBlobMap().end(), const_blob_it);
  const_blob_it = info.getBlobMap().find("var2");
  EXPECT_NE(info.getBlobMap().end(), const_blob_it);

  auto mutable_op_it = info.getMutableOperatorMap()->find(0);
  ASSERT_NE(info.getMutableOperatorMap()->end(), mutable_op_it);
  // Example profiling call:
  mutable_op_it->second.getMutableExecutionTimeMs()->addPoint(3);
  auto const_op_it = info.getOperatorMap().find(1);
  EXPECT_NE(info.getOperatorMap().end(), const_op_it);
}

const char* kExpectedProto = R"(
  stats {
    name: "op1"
    mean: 5
    stddev: 0

    execution_time {
      mean: 5
      stddev: 0
      count: 1
    }

    output_profile {
      name: "var1"
      bytes_used {
        mean: 3
        stddev: 0
        count: 1
      }
    }
    output_profile {
      name: "var12"
      bytes_used {
        mean: 0
        stddev: 0
        count: 0
      }
    }
  }
  stats {
    name: ""
    mean: 0
    stddev: 0

    execution_time {
      mean: 0
      stddev: 0
      count: 0
    }

    output_profile {
      name: "var2"

      bytes_used {
        mean: 0
        stddev: 0
        count: 0
      }
    }
  }
  net_name: "example_net"
)";

TEST(ProfilingInfoTest, InitAddStatsAndGetOperatorStatsProto) {
  NetDef net_def;
  ProfilingInfo info;
  ASSERT_TRUE(
      TextFormat::ParseFromString(string(kTestNetDefCorrect), &net_def));
  ProfDAGProtos expected;
  ASSERT_TRUE(TextFormat::ParseFromString(string(kExpectedProto), &expected));

  info.Init(net_def);
  // Add stats.
  auto op_it = info.getMutableOperatorMap()->find(0);
  op_it->second.getMutableExecutionTimeMs()->addPoint(5);
  auto blob_it = info.getMutableBlobMap()->find("var1");
  blob_it->second.getMutableUsedBytes()->addPoint(3);

  // Export to proto.
  ProfDAGProtos generated;
  ASSERT_TRUE(info.GetOperatorAndDataStats(net_def, false, &generated));
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::ApproximatelyEquals(
      expected, generated))
      << generated.DebugString();
}

TEST(ProfilingInfoTest, NewFormatSymmetry) {
  NetDef net_def;
  ASSERT_TRUE(
      TextFormat::ParseFromString(string(kTestNetDefCorrect), &net_def));
  ProfDAGProtos profile;
  ASSERT_TRUE(TextFormat::ParseFromString(string(kTestProfile), &profile));
  ProfilingInfo info;
  info.Restore(net_def, profile);
  ProfDAGProtos generated;
  ASSERT_TRUE(info.GetOperatorAndDataStats(net_def, false, &generated));
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::ApproximatelyEquals(
      profile, generated))
      << generated.DebugString();
}

const char* kExpectedProtoOldFormat = R"(
  stats {
    name: "example_net___0___add"
    mean: 5
    stddev: 0
  }
  stats {
    name: "example_net___1___mult"
    mean: 0
    stddev: 0
  }
)";

TEST(ProfilingInfoTest, InitAddStatsAndGetOperatorStatsProtoOldFormat) {
  NetDef net_def;
  ProfilingInfo info;
  ASSERT_TRUE(
      TextFormat::ParseFromString(string(kTestNetDefCorrect), &net_def));
  ProfDAGProtos expected;
  ASSERT_TRUE(
      TextFormat::ParseFromString(string(kExpectedProtoOldFormat), &expected));

  info.Init(net_def);
  // Add stats.
  auto op_it = info.getMutableOperatorMap()->find(0);
  op_it->second.getMutableExecutionTimeMs()->addPoint(5);
  auto blob_it = info.getMutableBlobMap()->find("var1");
  blob_it->second.getMutableUsedBytes()->addPoint(3);

  // Export to proto.
  ProfDAGProtos generated;
  ASSERT_TRUE(info.GetOperatorAndDataStats(net_def, true, &generated));
  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::ApproximatelyEquivalent(
          expected, generated))
      << generated.DebugString();
}

const char* kTestNetDefMultiOpMultiType = R"(
  name: "example_net"
  op {
    name: "op1"
    type: "add"
    output: "var1"
    output: "var12"
  }
  op {
    name: "op2"
    type: "add"
    output: "var12"
  }
  op {
    name: "op3"
    type: "mult"
    output: "var2"
  }
)";

const char* kExpectedProtoByType = R"(
  stats {
    name: "add"
    mean: 6
    stddev: 1
  }
  stats {
    name: "mult"
    mean: 11
    stddev: 0
  }
)";

TEST(ProfilingInfoTest, InitAddStatsAndGetOperatorTypeStatsProto) {
  NetDef net_def;
  ProfilingInfo info;
  ASSERT_TRUE(TextFormat::ParseFromString(
      string(kTestNetDefMultiOpMultiType), &net_def));
  ProfDAGProtos expected;
  ASSERT_TRUE(
      TextFormat::ParseFromString(string(kExpectedProtoByType), &expected));

  info.Init(net_def);
  // Add stats.
  auto op_it = info.getMutableOperatorMap()->find(0);
  op_it->second.getMutableExecutionTimeMs()->addPoint(5);
  op_it = info.getMutableOperatorMap()->find(1);
  op_it->second.getMutableExecutionTimeMs()->addPoint(7);
  op_it = info.getMutableOperatorMap()->find(2);
  op_it->second.getMutableExecutionTimeMs()->addPoint(11);

  // Export to proto.
  ProfDAGProtos generated;
  ASSERT_TRUE(info.GetOperatorTypeStats(net_def, &generated));
  // Since op types are unordered, it is OK to match either.
  ProfDAGProtos generated_swapped = generated;
  std::swap(
      *generated_swapped.mutable_stats(0), *generated_swapped.mutable_stats(1));
  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::ApproximatelyEquivalent(
          expected, generated) ||
      google::protobuf::util::MessageDifferencer::ApproximatelyEquivalent(
          expected, generated_swapped))
      << generated.DebugString();
}

} // namespace
} // namespace prof
} // namespace contrib
} // namespace caffe2
