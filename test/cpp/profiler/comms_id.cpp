#include <gtest/gtest.h>

#include <c10/util/ThreadLocalDebugInfo.h>
#include <c10/util/hash.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/profiler/util.h>

using torch::ParamCommsDebugInfo;
using namespace torch::profiler::impl;

namespace {

ParamCommsDebugInfo makeDebugInfo(
    const std::string& pgName,
    const std::string& pgDesc,
    int rank,
    std::string collName,
    int worldSize,
    int64_t seqNumber,
    bool isP2P,
    int globalRankStart = 0,
    int globalRankStride = 1) {
  return ParamCommsDebugInfo(
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
      worldSize,
      /*isAsynchronizedOp=*/true,
      seqNumber,
      isP2P);
}

size_t computeExpectedCommsId(
    const std::string& pgName,
    int64_t seqNumber,
    bool isP2P,
    int globalRankStart = 0,
    int globalRankStride = 1,
    int worldSize = 8) {
  return c10::get_hash(
      pgName, seqNumber, isP2P, globalRankStart, globalRankStride, worldSize);
}

} // namespace

TEST(CommsIdTest, ParamCommsDebugInfoStoresSeqNumberAndIsP2P) {
  auto info =
      makeDebugInfo("pg_uid_123", "default_pg", 0, "allreduce", 8, 42, false);

  EXPECT_EQ(info.getSeqNumber(), 42);
  EXPECT_FALSE(info.isP2P());
  EXPECT_EQ(info.getProcessGroupName(), "pg_uid_123");
  EXPECT_EQ(info.getCollectiveName(), "allreduce");
  EXPECT_EQ(info.getWorldSize(), 8);
}

TEST(CommsIdTest, ParamCommsDebugInfoP2PFlag) {
  auto info = makeDebugInfo("pg_uid_456", "custom_pg", 3, "send", 4, 7, true);

  EXPECT_EQ(info.getSeqNumber(), 7);
  EXPECT_TRUE(info.isP2P());
}

TEST(CommsIdTest, ParamCommsDebugInfoDefaultSeqNumberAndIsP2P) {
  auto info = ParamCommsDebugInfo(
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

  EXPECT_EQ(info.getSeqNumber(), 0);
  EXPECT_FALSE(info.isP2P());
}

TEST(CommsIdTest, SaveNcclMetaEmitsCommsId) {
  auto debugInfo = std::make_shared<ParamCommsDebugInfo>(
      std::make_tuple(std::string("pg_uid_123"), std::string("default_pg")),
      /*rank=*/0,
      /*collName=*/std::string("allreduce"),
      /*inNelems=*/1024,
      /*outNelems=*/1024,
      /*dType=*/at::kFloat,
      /*inSplitSizes=*/std::vector<int64_t>{},
      /*outSplitSizes=*/std::vector<int64_t>{},
      /*globalRankStart=*/0,
      /*globalRankStride=*/1,
      /*worldSize=*/8,
      /*isAsynchronizedOp=*/true,
      /*seqNumber=*/int64_t(42),
      /*isP2P=*/false);

  c10::DebugInfoGuard guard(c10::DebugInfoKind::PARAM_COMMS_INFO, debugInfo);

  // Create a dummy RecordFunction to pass to saveNcclMeta
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);
  fn._setAsync();

  auto meta = saveNcclMeta(fn);

  // Verify comms_id is in the metadata
  ASSERT_TRUE(meta.count(kCommsId) > 0);

  // Verify the comms_id value matches
  // hash(pg_name, seqNumber, isP2P, globalRankStart, globalRankStride,
  // worldSize)
  size_t expected_comms_id = computeExpectedCommsId(
      "pg_uid_123",
      int64_t(42),
      false,
      /*globalRankStart=*/0,
      /*globalRankStride=*/1,
      /*worldSize=*/8);
  EXPECT_EQ(meta.at(kCommsId), std::to_string(expected_comms_id));
}

TEST(CommsIdTest, CommsIdDeterministicAcrossInstances) {
  std::string pg_name = "pg_uid_shared";
  int64_t seq = 100;

  size_t comms_id_1 = computeExpectedCommsId(pg_name, seq, false);
  size_t comms_id_2 = computeExpectedCommsId(pg_name, seq, false);

  EXPECT_EQ(comms_id_1, comms_id_2);
}

TEST(CommsIdTest, CommsIdDiffersForDifferentSeqNumbers) {
  std::string pg_name = "pg_uid_shared";

  size_t comms_id_seq1 = computeExpectedCommsId(pg_name, int64_t(1), false);
  size_t comms_id_seq2 = computeExpectedCommsId(pg_name, int64_t(2), false);

  EXPECT_NE(comms_id_seq1, comms_id_seq2);
}

TEST(CommsIdTest, CommsIdDiffersForDifferentPGNames) {
  int64_t seq = 42;

  size_t comms_id_pg1 = computeExpectedCommsId("pg_uid_A", seq, false);
  size_t comms_id_pg2 = computeExpectedCommsId("pg_uid_B", seq, false);

  EXPECT_NE(comms_id_pg1, comms_id_pg2);
}

TEST(CommsIdTest, CommsIdDiffersForP2PvsCollective) {
  std::string pg_name = "pg_uid_shared";
  int64_t seq = 42;

  size_t comms_id_collective = computeExpectedCommsId(pg_name, seq, false);
  size_t comms_id_p2p = computeExpectedCommsId(pg_name, seq, true);

  EXPECT_NE(comms_id_collective, comms_id_p2p);
}

TEST(CommsIdTest, CommsIdDiffersForDifferentCommunicatorTopology) {
  std::string pg_name = "pg_uid_shared";
  int64_t seq = 42;

  // Same PG name and seq, but different communicator topology
  // (e.g., after comm split producing different rank subsets)
  size_t comms_id_full = computeExpectedCommsId(pg_name, seq, false, 0, 1, 8);
  size_t comms_id_split = computeExpectedCommsId(pg_name, seq, false, 0, 1, 4);

  EXPECT_NE(comms_id_full, comms_id_split);

  // Different globalRankStart (different subset of ranks)
  size_t comms_id_start0 = computeExpectedCommsId(pg_name, seq, false, 0, 2, 4);
  size_t comms_id_start1 = computeExpectedCommsId(pg_name, seq, false, 1, 2, 4);

  EXPECT_NE(comms_id_start0, comms_id_start1);
}
