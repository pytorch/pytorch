#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/tunable/GemmCommon.h>
#include <ATen/cuda/tunable/GemmLdRules.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <c10/util/env.h>
#include <c10/util/tempfile.h>

// Manager-level coverage for LookupNearest / RecordBorrowed. Unlike
// cuda_tunable_ld_rules_test.cpp (header-only, no ATen), this drives the real
// TuningResultsManager through getTuningContext(). No GEMM is launched -- the
// logic under test is all CPU -- but reaching the context at all needs a
// device, so those tests skip without one. See TunableNearestTest below.
//
// Each test uses a fresh TuningResultsManager and saves/restores the global
// context flags it touches, so tests do not leak flag state into each other.

using at::cuda::tunable::getTuningContext;
using at::cuda::tunable::ResultEntry;
using at::cuda::tunable::TuningResultsManager;

namespace {

constexpr const char* kOp = "gemm_nn_f32";

// Compare entries by key rather than with EXPECT_EQ on the entries themselves.
// ResultEntry::operator== already compares only the key, but passing entries to
// EXPECT_EQ makes gtest print them on failure via the operator<< declared as a
// friend in Tunable.h. That friend declaration does not pick up the class's
// TORCH_CUDA_CPP_API, so the symbol is not exported and the test fails to link.
std::string Key(const ResultEntry& entry) {
  return entry.GetKey();
}

// A specialized (non-Default) donor entry, as produced by tuning. The time is
// the donor shape's, measured when the donor was tuned.
ResultEntry Donor(const char* key, double time_ms = 1.0) {
  return ResultEntry(std::string(key), time_ms);
}

// Tight m=32 and m=64 NN GEMMs (k=256, n=512): the 32-shape borrows the 64.
constexpr const char* kQSig = "NN_32_512_256_ld_32_256_32";
constexpr const char* kCSig = "NN_64_512_256_ld_64_256_64";
constexpr const char* kQBlasSig =
    "- { function: matmul, M: 32, N: 512, K: 256 }";

// What the reuse pass hands to RecordBorrowed: the winning key, with the time
// and BLAS signature measured on the BORROWER's params. Recording the donor's
// entry instead would file the donor's numbers under the borrower's signature.
ResultEntry Validated(const char* key, double measured_ms) {
  return ResultEntry(std::string(key), measured_ms, std::string(kQBlasSig));
}

struct FlagGuard {
  FlagGuard() {
    auto* ctx = getTuningContext();
    tuning_ = ctx->IsTuningEnabled();
    untuned_ = ctx->IsRecordUntunedEnabled();
    nearest_ = ctx->IsNearestMatchEnabled();
    borrowed_ = ctx->IsRecordBorrowedEnabled();
  }
  ~FlagGuard() {
    auto* ctx = getTuningContext();
    ctx->EnableTuning(tuning_);
    ctx->EnableRecordUntuned(untuned_);
    ctx->EnableNearestMatch(nearest_);
    ctx->EnableRecordBorrowed(borrowed_);
  }
  bool tuning_, untuned_, nearest_, borrowed_;
};

// Read an entire file into a string for content assertions.
std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  std::string content((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
  return content;
}

// Point the global context's results file at a tmp file for the duration of a
// test. Any test that might write has to own the file rather than append to
// whatever the process default is. No global manager init is required.
class TempResultsFile {
 public:
  explicit TempResultsFile(const char* prefix)
      : tmp_(c10::make_tempfile(prefix)),
        prev_(getTuningContext()->GetFilename()) {
    getTuningContext()->SetFilename(tmp_.name, /*insert_device_ordinal=*/false);
  }
  ~TempResultsFile() {
    getTuningContext()->SetFilename(prev_, /*insert_device_ordinal=*/false);
  }
  std::string Contents() const { return ReadFile(tmp_.name); }

 private:
  c10::TempFile tmp_;
  std::string prev_;
};

// Seed a manager's map without taking the write path, which needs a device.
void SeedDonor(TuningResultsManager& mgr) {
  getTuningContext()->EnableTuning(false);
  mgr.Add(kOp, kCSig, Donor("k64"));
}

} // namespace

// Constructing the TuningContext builds the results validator, which queries
// the current device. Without a usable one that is a segfault inside
// c10::cuda::get_cuda_error_help rather than a clean error, so skip instead of
// crashing the binary. Every test below reaches getTuningContext(); the
// TunableNearestParse suite does not and stays unguarded.
class TunableNearestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!at::cuda::is_available()) {
      GTEST_SKIP() << "needs a device to construct the TuningContext";
    }
  }
};

TEST_F(TunableNearestTest, LookupNearestIgnoresPolicyFlags) {
  // Calling LookupNearest is itself the opt-in: dispatch never reaches it, so
  // it answers the question asked rather than re-deciding from the pass's
  // mutable policy flags. The env kill switch IS still honoured here -- see
  // EnvKillSwitchVetoesLookupAndRecord below.
  FlagGuard g;
  auto* ctx = getTuningContext();
  TuningResultsManager mgr;
  SeedDonor(mgr);

  ctx->EnableNearestMatch(false);
  EXPECT_EQ(Key(mgr.LookupNearest(kOp, kQSig)), Key(Donor("k64")));

  ctx->EnableTuning(true);
  ctx->EnableNearestMatch(true);
  EXPECT_EQ(Key(mgr.LookupNearest(kOp, kQSig)), Key(Donor("k64")));
}

TEST_F(TunableNearestTest, BorrowsClosestDonor) {
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  TuningResultsManager mgr;
  mgr.Add(kOp, kCSig, Donor("k64"));
  mgr.Add(kOp, "NN_128_512_256_ld_128_256_128", Donor("k128"));
  const ResultEntry got = mgr.LookupNearest(kOp, kQSig);
  EXPECT_EQ(Key(got), Key(Donor("k64")));
}

TEST_F(TunableNearestTest, SkipsDefaultDonors) {
  // A near Default must not mask a farther specialized donor: the validator
  // would benchmark Default against Default and persist it.
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  TuningResultsManager mgr;
  mgr.Add(kOp, "NN_48_512_256_ld_48_256_48", ResultEntry::Default());
  mgr.Add(kOp, kCSig, Donor("k64"));
  EXPECT_EQ(Key(mgr.LookupNearest(kOp, kQSig)), Key(Donor("k64")));
}

TEST_F(TunableNearestTest, TieBreakIsDeterministic) {
  // Equal absolute deltas in different dims must not depend on unordered_map
  // iteration order: lexicographically smallest signature wins.
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  TuningResultsManager mgr;
  // M-gap 32 (64-32) vs N-gap 32 (544-512): tie on delta. ldb tracks K=256
  // for transb='N', so it stays 256 in both; ldc tracks M and matches.
  mgr.Add(kOp, "NN_32_544_256_ld_32_256_32", Donor("n544"));
  mgr.Add(kOp, kCSig, Donor("k64"));
  const ResultEntry got = mgr.LookupNearest(kOp, kQSig);
  // "NN_32_544_..." < "NN_64_512_..." lexicographically.
  EXPECT_EQ(Key(got), Key(Donor("n544")));
}

TEST_F(TunableNearestTest, ResultIsNotMemoized) {
  // The borrow is unmeasured for this shape; the caller validates and persists
  // instead. A second lookup must still report the donor, not an exact hit.
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  TuningResultsManager mgr;
  mgr.Add(kOp, kCSig, Donor("k64"));
  EXPECT_EQ(Key(mgr.LookupNearest(kOp, kQSig)), Key(Donor("k64")));
  EXPECT_EQ(Key(mgr.Lookup(kOp, kQSig)), Key(ResultEntry::Null()));
  EXPECT_EQ(Key(mgr.LookupNearest(kOp, kQSig)), Key(Donor("k64")));
}

TEST_F(TunableNearestTest, RecordBorrowedBecomesExactHit) {
  FlagGuard g;
  TempResultsFile results("tunableop-exact-hit-");
  TuningResultsManager mgr;
  SeedDonor(mgr);
  EXPECT_EQ(Key(mgr.Lookup(kOp, kQSig)), Key(ResultEntry::Null()));
  mgr.RecordBorrowed(kOp, kQSig, Validated("k64", /*measured_ms=*/2.5));
  // The recorded borrow is an ordinary exact hit from then on.
  EXPECT_EQ(Key(mgr.Lookup(kOp, kQSig)), Key(Donor("k64")));
}

TEST_F(TunableNearestTest, RecordBorrowedIgnoresPolicyFlags) {
  // Same opt-in-by-calling contract as LookupNearest: the reuse pass decides
  // whether to record, and RecordBorrowed records what it is handed. As
  // above, this covers the mutable policy flag only -- the env kill switch
  // does suppress the write.
  FlagGuard g;
  auto* ctx = getTuningContext();
  TempResultsFile results("tunableop-flags-");
  ctx->EnableTuning(false);
  ctx->EnableRecordBorrowed(false);
  TuningResultsManager mgr;
  mgr.RecordBorrowed(kOp, kQSig, Validated("k64", /*measured_ms=*/2.5));
  EXPECT_EQ(Key(mgr.Lookup(kOp, kQSig)), Key(Donor("k64")));
}

TEST_F(TunableNearestTest, RecordBorrowedPersistsWhileTuningIsOff) {
  // Why RecordBorrowed exists: Add() only writes during tuning, and nothing
  // bulk-writes results_ at exit, so the reuse pass needs an entry point that
  // reaches the results file with tuning off.
  FlagGuard g;
  TempResultsFile results("tunableop-borrowed-");
  getTuningContext()->EnableTuning(false);
  {
    TuningResultsManager mgr;
    mgr.RecordBorrowed(kOp, kQSig, Validated("k64", /*measured_ms=*/2.5));
  }
  EXPECT_NE(results.Contents().find(kQSig), std::string::npos);
}

TEST_F(TunableNearestTest, RecordBorrowedPersistsTheBorrowersMeasurement) {
  // The line written under the borrower's signature carries the borrower's own
  // measurement: the key may come from the donor, the time and BLAS signature
  // never do. Otherwise an offline tool replaying that line would benchmark the
  // donor's shape believing it was the borrower's.
  FlagGuard g;
  TempResultsFile results("tunableop-measurement-");
  getTuningContext()->EnableTuning(false);
  {
    TuningResultsManager mgr;
    // Donor tuned at 9.0 ms on its own (larger) shape.
    mgr.Add(kOp, kCSig, Donor("k64", /*time_ms=*/9.0));
    // Same kernel re-measured at 2.5 ms on the borrower's shape.
    mgr.RecordBorrowed(kOp, kQSig, Validated("k64", /*measured_ms=*/2.5));
  }
  const std::string content = results.Contents();
  EXPECT_NE(content.find("k64,2.5"), std::string::npos) << content;
  EXPECT_EQ(content.find("k64,9"), std::string::npos) << content;
}

TEST_F(TunableNearestTest, RecordBorrowedFirstWriteWins) {
  // A recorded verdict is final. AddImpl try_emplaces, and the realtime append
  // only fires for a key that was new, so a second RecordBorrowed for the same
  // shape -- or one for a shape Add() already populated -- must change neither
  // the map nor the file. This is what makes the reuse pass idempotent, and
  // what makes a recorded Default irreversible.
  FlagGuard g;
  TempResultsFile results("tunableop-first-write-");
  auto* ctx = getTuningContext();
  ctx->EnableTuning(false);
  TuningResultsManager mgr;

  mgr.RecordBorrowed(kOp, kQSig, Validated("first", /*measured_ms=*/2.5));
  mgr.RecordBorrowed(kOp, kQSig, Validated("second", /*measured_ms=*/3.5));
  EXPECT_EQ(Key(mgr.Lookup(kOp, kQSig)), Key(Donor("first")));

  mgr.Add(kOp, kCSig, Donor("k64"));
  mgr.RecordBorrowed(kOp, kCSig, Validated("usurper", /*measured_ms=*/4.5));
  EXPECT_EQ(Key(mgr.Lookup(kOp, kCSig)), Key(Donor("k64")));

  const std::string content = results.Contents();
  EXPECT_NE(content.find("first"), std::string::npos);
  EXPECT_EQ(content.find("second"), std::string::npos);
  EXPECT_EQ(content.find("usurper"), std::string::npos);
}

TEST_F(TunableNearestTest, RecordBorrowedDoesNotWidenAddPersistence) {
  // RecordBorrowed overrides the write rule for itself only. Add() is on every
  // GEMM's path on every thread, so its rule -- write during tuning unless the
  // run is only collecting untuned shapes -- must hold whatever the reuse-pass
  // flags say. Neither combination below writes anything.
  FlagGuard g;
  auto* ctx = getTuningContext();
  ctx->EnableNearestMatch(true);
  ctx->EnableRecordBorrowed(true);

  {  // Tuning off: not a tuning run, so Add() has nothing to record.
    TempResultsFile results("tunableop-add-notuning-");
    ctx->EnableTuning(false);
    {
      TuningResultsManager mgr;
      mgr.Add(kOp, kCSig, Donor("k64"));
    }
    EXPECT_TRUE(results.Contents().empty());
  }

  {  // Tuning on but record-untuned on: a collection run writes the untuned
     // file, not the results file.
    TempResultsFile results("tunableop-add-untuned-");
    ctx->EnableTuning(true);
    ctx->EnableRecordUntuned(true);
    {
      TuningResultsManager mgr;
      mgr.Add(kOp, kCSig, Donor("k64"));
    }
    EXPECT_TRUE(results.Contents().empty());
  }
}

TEST_F(TunableNearestTest, EnvKillSwitchVetoesLookupAndRecord) {
  // The env is read once into a function-local static, so one process only
  // ever sees one value and the rest of this binary needs it unset. run_tests.sh
  // starts a second invocation with both set to 0 and filters to this case.
  using c10::utils::check_env;
  if (check_env("PYTORCH_TUNABLEOP_NEAREST_MATCH_ENABLED") != false ||
      check_env("PYTORCH_TUNABLEOP_RECORD_BORROWED_ENABLED") != false) {
    GTEST_SKIP() << "needs both kill switches set to 0 in the environment";
  }

  // Policy on, so only the env can be suppressing anything below.
  FlagGuard g;
  auto* ctx = getTuningContext();
  TempResultsFile results("tunableop-killswitch-");
  ctx->EnableNearestMatch(true);
  ctx->EnableRecordBorrowed(true);

  TuningResultsManager mgr;
  SeedDonor(mgr);
  // BorrowsClosestDonor covers this donor being returned when the env is unset.
  EXPECT_EQ(Key(mgr.LookupNearest(kOp, kQSig)), Key(ResultEntry::Null()));

  // Dropped outright, map as well as file -- see RecordBorrowed's contract.
  mgr.RecordBorrowed(kOp, kQSig, Validated("k64", /*measured_ms=*/2.5));
  EXPECT_EQ(Key(mgr.Lookup(kOp, kQSig)), Key(ResultEntry::Null()));
  EXPECT_EQ(results.Contents().find(kQSig), std::string::npos);
}

// --- The parser against the real signature builders ---
//
// ParseGemmShape only works because it agrees with the Signature() builders in
// GemmCommon.h, and nothing but these tests ties the two together. Drift is
// silent: every parse would return !ok, LookupNearest would return Null for
// every shape, and borrowing would just stop happening. So parse what the
// builders actually emit, never a hand-written string.

using at::cuda::tunable::GemmShape;
using at::cuda::tunable::NearestLargerMatch;
using at::cuda::tunable::ParseGemmShape;
using at::cuda::tunable::SplitUnderscore;

namespace {

GemmShape Parse(const std::string& sig) {
  return ParseGemmShape(SplitUnderscore(sig));
}

// Reassemble a parsed shape into a signature. Equality with the builder's
// output proves the parser accounted for every token, not merely that it found
// the ones it was looking for.
std::string Rebuild(const GemmShape& s) {
  std::string out = s.transab + "_" + std::to_string(s.m) + "_" +
      std::to_string(s.n) + "_" + std::to_string(s.k);
  if (s.has_b) {
    out += "_B_" + std::to_string(s.b);
  }
  out += "_ld_" + std::to_string(s.lda) + "_" + std::to_string(s.ldb) + "_" +
      std::to_string(s.ldc);
  return out + s.suffix;
}

// A scaled GEMM with tight leading dims, so only M separates two of them.
std::string ScaledSig(int64_t m, at::ScalarType bias_dtype) {
  const double bias = 0.0;  // never dereferenced; only nullness is read
  at::cuda::tunable::ScaledGemmParams<float> p;
  p.transa = 'N';
  p.transb = 'N';
  p.m = m;
  p.n = 512;
  p.k = 256;
  p.lda = m;
  p.ldb = 256;
  p.ldc = m;
  p.a_dtype = at::ScalarType::Float8_e4m3fn;
  p.b_dtype = at::ScalarType::Float8_e4m3fn;
  p.c_dtype = at::ScalarType::BFloat16;
  p.a_scale_dtype = at::ScalarType::Float;
  p.b_scale_dtype = at::ScalarType::Float;
  p.bias_ptr = &bias;
  p.bias_dtype = bias_dtype;
  return p.Signature();
}

} // namespace

TEST(TunableNearestParse, ParsesRealGemmParamsSignature) {
  at::cuda::tunable::GemmParams<double> p;
  p.transa = 'N';
  p.transb = 'T';
  p.m = 1024;
  p.n = 512;
  p.k = 256;
  p.lda = 1024;
  p.ldb = 512;
  p.ldc = 1024;

  const std::string sig = p.Signature();
  const GemmShape s = Parse(sig);
  ASSERT_TRUE(s.ok) << sig;
  EXPECT_EQ(s.transab, "NT");
  EXPECT_EQ(s.m, 1024);
  EXPECT_EQ(s.n, 512);
  EXPECT_EQ(s.k, 256);
  EXPECT_FALSE(s.has_b);
  EXPECT_EQ(s.lda, 1024);
  EXPECT_EQ(s.ldb, 512);
  EXPECT_EQ(s.ldc, 1024);
  EXPECT_EQ(Rebuild(s), sig);
}

TEST(TunableNearestParse, ParsesRealStridedBatchedSignature) {
  at::cuda::tunable::GemmStridedBatchedParams<double> p;
  p.transa = 'N';
  p.transb = 'N';
  p.m = 8;
  p.n = 512;
  p.k = 256;
  p.batch = 4;
  p.lda = 8;
  p.ldb = 256;
  p.ldc = 8;

  const std::string sig = p.Signature();
  const GemmShape s = Parse(sig);
  ASSERT_TRUE(s.ok) << sig;
  EXPECT_TRUE(s.has_b);
  EXPECT_EQ(s.b, 4);
  EXPECT_EQ(s.m, 8);
  EXPECT_EQ(s.lda, 8);
  EXPECT_EQ(Rebuild(s), sig);
}

TEST(TunableNearestParse, ParsesRealScaledGemmSignature) {
  const std::string sig = ScaledSig(32, at::ScalarType::BFloat16);
  const GemmShape s = Parse(sig);
  ASSERT_TRUE(s.ok) << sig;
  EXPECT_EQ(s.m, 32);
  EXPECT_EQ(s.n, 512);
  EXPECT_EQ(s.k, 256);
  EXPECT_EQ(s.lda, 32);
  EXPECT_EQ(s.ldc, 32);
  // Both branches of ScaledGemmParams::Signature must survive the
  // split/rejoin, and they differ: ROCm emits "_rw_<0|1>_bias_<dtype>"
  // (plus "_swz_<a>_<b>" when swizzled), while CUDA spells out every dtype,
  // so its suffix carries names that themselves contain underscores
  // (Float8_e4m3fn) -- the case most likely to confuse a token-based parser.
  EXPECT_FALSE(s.suffix.empty());
  EXPECT_EQ(Rebuild(s), sig);
}

TEST(TunableNearestParse, RealScaledGemmBorrowsAcrossM) {
  int64_t delta = -1;
  const GemmShape q = Parse(ScaledSig(32, at::ScalarType::BFloat16));
  const GemmShape c = Parse(ScaledSig(64, at::ScalarType::BFloat16));
  ASSERT_TRUE(q.ok);
  ASSERT_TRUE(c.ok);
  EXPECT_TRUE(NearestLargerMatch(q, c, delta));
  EXPECT_EQ(delta, 32);
}

TEST(TunableNearestParse, RealScaledGemmRejectsDifferingSuffix) {
  // bias dtype appears in the suffix on both the ROCm and CUDA forms, so this
  // donor is rejected on either platform.
  int64_t delta = -1;
  const GemmShape q = Parse(ScaledSig(32, at::ScalarType::BFloat16));
  const GemmShape c = Parse(ScaledSig(64, at::ScalarType::Half));
  ASSERT_TRUE(q.ok);
  ASSERT_TRUE(c.ok);
  EXPECT_FALSE(NearestLargerMatch(q, c, delta));
}
