#include <gtest/gtest.h>

#include <c10/util/ThreadLocalDebugInfo.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/profiler/util.h>

using torch::ParamCommsDebugInfo;
using namespace torch::profiler::impl;

namespace {

std::shared_ptr<ParamCommsDebugInfo> makeDebugInfo(
    const std::string& pgName,
    const std::string& pgDesc,
    int rank,
    std::string collName,
    int worldSize,
    int64_t seqNumber,
    bool isP2P,
    int globalRankStart = 0,
    int globalRankStride = 1) {
  auto info = std::make_shared<ParamCommsDebugInfo>(
      std::make_tuple(pgName, pgDesc),
      rank,
      std::move(collName),
      /*inNelems=*/1024,
      /*outNelems=*/1024,
      /*dType=*/at::kFloat,
      /*inSplitSizes=*/std::vector<int64_t>{},
      /*outSplitSizes=*/std::vector<int64_t>{},
      globalRankStart,
      globalRankStride,
      worldSize);
  info->setSequenceInfo(seqNumber, isP2P);
  return info;
}

// Helper to get comms_id from saveNcclMeta for a given ParamCommsDebugInfo.
std::string getCommsIdViaSaveNcclMeta(
    const std::shared_ptr<ParamCommsDebugInfo>& debugInfo) {
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PARAM_COMMS_INFO, debugInfo);
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);
  fn._setAsync();
  auto meta = saveNcclMeta(fn);
  if (meta.count(kCommsId) == 0) {
    return "";
  }
  return meta.at(kCommsId);
}

} // namespace

TEST(CommsIdTest, ParamCommsDebugInfoStoresSeqNumberAndIsP2P) {
  auto info =
      makeDebugInfo("pg_uid_123", "default_pg", 0, "allreduce", 8, 42, false);

  EXPECT_EQ(info->getSequenceNumber(), 42);
  EXPECT_FALSE(info->getIsP2P());
  EXPECT_EQ(info->getProcessGroupName(), "pg_uid_123");
  EXPECT_EQ(info->getCollectiveName(), "allreduce");
  EXPECT_EQ(info->getWorldSize(), 8);
}

TEST(CommsIdTest, ParamCommsDebugInfoP2PFlag) {
  auto info = makeDebugInfo("pg_uid_456", "custom_pg", 3, "send", 4, 7, true);

  EXPECT_EQ(info->getSequenceNumber(), 7);
  EXPECT_TRUE(info->getIsP2P());
}

TEST(CommsIdTest, ParamCommsDebugInfoDefaultSeqNumberAndIsP2P) {
  auto info = std::make_shared<ParamCommsDebugInfo>(
      std::make_tuple(std::string("pg_uid_789"), std::string("default_pg")),
      /*rank=*/1,
      /*collName=*/std::string("allgather"),
      /*inNelems=*/256,
      /*outNelems=*/512,
      /*dType=*/at::kFloat,
      /*inSplitSizes=*/std::vector<int64_t>{},
      /*outSplitSizes=*/std::vector<int64_t>{},
      /*globalRankStart=*/0,
      /*globalRankStride=*/1,
      /*worldSize=*/2);

  EXPECT_EQ(info->getSequenceNumber(), -1);
  EXPECT_FALSE(info->getIsP2P());
}

TEST(CommsIdTest, SaveNcclMetaEmitsCommsId) {
  auto debugInfo =
      makeDebugInfo("pg_uid_123", "default_pg", 0, "allreduce", 8, 42, false);

  auto commsId = getCommsIdViaSaveNcclMeta(debugInfo);
  EXPECT_FALSE(commsId.empty());

  // Verify determinism: same input produces the same comms_id
  auto commsId2 = getCommsIdViaSaveNcclMeta(debugInfo);
  EXPECT_EQ(commsId, commsId2);
}

TEST(CommsIdTest, SaveNcclMetaOmitsCommsIdWhenSeqNotSet) {
  auto debugInfo = std::make_shared<ParamCommsDebugInfo>(
      std::make_tuple(std::string("pg_uid_no_seq"), std::string("default_pg")),
      /*rank=*/0,
      /*collName=*/std::string("allreduce"),
      /*inNelems=*/1024,
      /*outNelems=*/1024,
      /*dType=*/at::kFloat,
      /*inSplitSizes=*/std::vector<int64_t>{},
      /*outSplitSizes=*/std::vector<int64_t>{},
      /*globalRankStart=*/0,
      /*globalRankStride=*/1,
      /*worldSize=*/8);

  auto commsId = getCommsIdViaSaveNcclMeta(debugInfo);
  EXPECT_TRUE(commsId.empty());
}

TEST(CommsIdTest, CommsIdDiffersForDifferentSeqNumbers) {
  auto id1 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 8, 1, false));
  auto id2 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 8, 2, false));
  EXPECT_NE(id1, id2);
}

TEST(CommsIdTest, CommsIdDiffersForDifferentPGNames) {
  auto id1 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg_A", "desc", 0, "allreduce", 8, 42, false));
  auto id2 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg_B", "desc", 0, "allreduce", 8, 42, false));
  EXPECT_NE(id1, id2);
}

TEST(CommsIdTest, CommsIdDiffersForP2PvsCollective) {
  auto id_collective = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 8, 42, false));
  auto id_p2p = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "send", 8, 42, true));
  EXPECT_NE(id_collective, id_p2p);
}

TEST(CommsIdTest, CommsIdDiffersForDifferentTopology) {
  auto id1 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 8, 42, false, 0, 1));
  auto id2 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 4, 42, false, 0, 1));
  EXPECT_NE(id1, id2);

  auto id3 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 4, 42, false, 0, 2));
  auto id4 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 4, 42, false, 1, 2));
  EXPECT_NE(id3, id4);
}
