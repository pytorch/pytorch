#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>

#include <ATen/cuda/tunable/Tunable.h>

// Manager-level coverage for LookupNearest / RecordBorrowed. Unlike
// cuda_tunable_ld_rules_test.cpp (header-only, no ATen), this drives the real
// TuningResultsManager through getTuningContext(). The manager logic is CPU
// logic -- no GEMM is launched -- but the test links against torch_cuda like
// every other ATen CUDA test.
//
// Each test uses a fresh TuningResultsManager and saves/restores the global
// context flags it touches, so tests do not leak flag state into each other.

using at::cuda::tunable::getTuningContext;
using at::cuda::tunable::ResultEntry;
using at::cuda::tunable::TuningResultsManager;

namespace {

constexpr const char* kOp = "gemm_nn_f32";

// A specialized (non-Default) donor entry, as produced by tuning.
ResultEntry Donor(const char* key) {
  return ResultEntry(std::string(key), /*time=*/1.0);
}

// Tight m=32 and m=64 NN GEMMs (k=256, n=512): the 32-shape borrows the 64.
constexpr const char* kQSig = "NN_32_512_256_ld_32_256_32";
constexpr const char* kCSig = "NN_64_512_256_ld_64_256_64";

struct FlagGuard {
  FlagGuard() {
    auto* ctx = getTuningContext();
    tuning_ = ctx->IsTuningEnabled();
    nearest_ = ctx->IsNearestMatchEnabled();
    borrowed_ = ctx->IsRecordBorrowedEnabled();
  }
  ~FlagGuard() {
    auto* ctx = getTuningContext();
    ctx->EnableTuning(tuning_);
    ctx->EnableNearestMatch(nearest_);
    ctx->EnableRecordBorrowed(borrowed_);
  }
  bool tuning_, nearest_, borrowed_;
};

} // namespace

TEST(TunableNearest, ReturnsNullWhenFlagOff) {
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  ctx->EnableNearestMatch(false);
  TuningResultsManager mgr;
  mgr.Add(kOp, kCSig, Donor("k1"));
  EXPECT_EQ(mgr.LookupNearest(kOp, kQSig), ResultEntry::Null());
}

TEST(TunableNearest, ReturnsNullWhileTuning) {
  // A tuning pass must tune the real shape, never short-circuit to a neighbour.
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(true);
  ctx->EnableNearestMatch(true);
  TuningResultsManager mgr;
  mgr.Add(kOp, kCSig, Donor("k1"));
  EXPECT_EQ(mgr.LookupNearest(kOp, kQSig), ResultEntry::Null());
}

TEST(TunableNearest, BorrowsClosestDonor) {
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  ctx->EnableNearestMatch(true);
  TuningResultsManager mgr;
  mgr.Add(kOp, kCSig, Donor("k64"));
  mgr.Add(kOp, "NN_128_512_256_ld_128_256_128", Donor("k128"));
  const ResultEntry got = mgr.LookupNearest(kOp, kQSig);
  EXPECT_EQ(got, Donor("k64"));
}

TEST(TunableNearest, SkipsDefaultDonors) {
  // A near Default must not mask a farther specialized donor: the validator
  // would benchmark Default against Default and persist it.
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  ctx->EnableNearestMatch(true);
  TuningResultsManager mgr;
  mgr.Add(kOp, "NN_48_512_256_ld_48_256_48", ResultEntry::Default());
  mgr.Add(kOp, kCSig, Donor("k64"));
  EXPECT_EQ(mgr.LookupNearest(kOp, kQSig), Donor("k64"));
}

TEST(TunableNearest, TieBreakIsDeterministic) {
  // Equal absolute deltas in different dims must not depend on unordered_map
  // iteration order: lexicographically smallest signature wins.
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  ctx->EnableNearestMatch(true);
  TuningResultsManager mgr;
  // M-gap 32 (64-32) vs N-gap 32 (544-512): tie on delta. ldb tracks K=256
  // for transb='N', so it stays 256 in both; ldc tracks M and matches.
  mgr.Add(kOp, "NN_32_544_256_ld_32_256_32", Donor("n544"));
  mgr.Add(kOp, kCSig, Donor("k64"));
  const ResultEntry got = mgr.LookupNearest(kOp, kQSig);
  // "NN_32_544_..." < "NN_64_512_..." lexicographically.
  EXPECT_EQ(got, Donor("n544"));
}

TEST(TunableNearest, ResultIsNotMemoized) {
  // The borrow is unmeasured for this shape; the caller validates and persists
  // instead. A second lookup must still report the donor, not an exact hit.
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  ctx->EnableNearestMatch(true);
  TuningResultsManager mgr;
  mgr.Add(kOp, kCSig, Donor("k64"));
  EXPECT_EQ(mgr.LookupNearest(kOp, kQSig), Donor("k64"));
  EXPECT_EQ(mgr.Lookup(kOp, kQSig), ResultEntry::Null());
  EXPECT_EQ(mgr.LookupNearest(kOp, kQSig), Donor("k64"));
}

TEST(TunableNearest, RecordBorrowedBecomesExactHit) {
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  ctx->EnableNearestMatch(true);
  ctx->EnableRecordBorrowed(true);
  TuningResultsManager mgr;
  mgr.Add(kOp, kCSig, Donor("k64"));
  EXPECT_EQ(mgr.Lookup(kOp, kQSig), ResultEntry::Null());
  mgr.RecordBorrowed(kOp, kQSig, Donor("k64"));
  // The recorded borrow is an ordinary exact hit from then on.
  EXPECT_EQ(mgr.Lookup(kOp, kQSig), Donor("k64"));
}

TEST(TunableNearest, RecordBorrowedDisabledWritesNothing) {
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  ctx->EnableNearestMatch(true);
  ctx->EnableRecordBorrowed(false);
  TuningResultsManager mgr;
  // Warns and ignores; the shape stays a miss.
  mgr.RecordBorrowed(kOp, kQSig, Donor("k64"));
  EXPECT_EQ(mgr.Lookup(kOp, kQSig), ResultEntry::Null());
}

namespace {

// Read an entire file into a string for content assertions.
std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  std::string content((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
  return content;
}

} // namespace

TEST(TunableNearest, RecordBorrowedPersistsToFile) {
  // The persist predicate in Add() is the reason RecordBorrowed exists: with
  // tuning off and record-borrowed on, the entry must reach the results file,
  // not just the in-memory map. Uses a standalone manager but points the
  // global context at a tmp file (no global manager init required).
  FlagGuard g;
  auto* ctx = getTuningContext();
  const std::string prev_file = ctx->GetFilename();
  const std::string tmp = std::tmpnam(nullptr);
  ctx->SetFilename(tmp, /*insert_device_ordinal=*/false);
  ctx->EnableTuning(false);
  ctx->EnableNearestMatch(true);
  ctx->EnableRecordBorrowed(true);
  {
    TuningResultsManager mgr;
    mgr.Add(kOp, kCSig, Donor("k64"));
    mgr.RecordBorrowed(kOp, kQSig, Donor("k64"));
    EXPECT_EQ(mgr.Lookup(kOp, kQSig), Donor("k64"));
  }
  const std::string content = ReadFile(tmp);
  EXPECT_NE(content.find(kQSig), std::string::npos);
  std::remove(tmp.c_str());
  ctx->SetFilename(prev_file, /*insert_device_ordinal=*/false);
}

TEST(TunableNearest, NoPersistWithoutBorrowOrTuning) {
  // Tuning off and record-borrowed off writes nothing: the verdict lives only
  // in the calling manager's map.
  FlagGuard g;
  auto* ctx = getTuningContext();
  const std::string prev_file = ctx->GetFilename();
  const std::string tmp = std::tmpnam(nullptr);
  ctx->SetFilename(tmp, /*insert_device_ordinal=*/false);
  ctx->EnableTuning(false);
  ctx->EnableNearestMatch(true);
  ctx->EnableRecordBorrowed(false);
  {
    TuningResultsManager mgr;
    mgr.Add(kOp, kCSig, Donor("k64"));
    // Disabled path warns and ignores.
    mgr.RecordBorrowed(kOp, kQSig, Donor("k64"));
    EXPECT_EQ(mgr.Lookup(kOp, kQSig), ResultEntry::Null());
  }
  std::ifstream in(tmp);
  EXPECT_FALSE(in.good());
  ctx->SetFilename(prev_file, /*insert_device_ordinal=*/false);
}
