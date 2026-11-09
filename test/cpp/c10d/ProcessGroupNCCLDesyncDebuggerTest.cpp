#include <chrono>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace {

// Records store operations and retains values for desync report reads.
class CountingStore : public c10d::Store {
 public:
  using Write = std::pair<std::string, std::vector<uint8_t>>;

  c10::intrusive_ptr<c10d::Store> clone() override {
    return c10::make_intrusive<CountingStore>();
  }

  void set(const std::string& key, const std::vector<uint8_t>& value) override {
    values_[key] = value;
    writes_.emplace_back(key, value);
    operations_.push_back("set:" + key);
  }

  std::vector<uint8_t> get(const std::string& key) override {
    operations_.push_back("get:" + key);
    return values_.at(key);
  }
  int64_t add(const std::string& /*key*/, int64_t /*value*/) override {
    return 0;
  }
  bool deleteKey(const std::string& key) override {
    return values_.erase(key) > 0;
  }
  bool check(const std::vector<std::string>& keys) override {
    bool found = true;
    for (const auto& key : keys) {
      operations_.push_back("check:" + key);
      found = found && values_.count(key) != 0;
    }
    return found;
  }
  int64_t getNumKeys() override {
    return static_cast<int64_t>(values_.size());
  }
  void wait(const std::vector<std::string>& /*keys*/) override {}
  void wait(
      const std::vector<std::string>& /*keys*/,
      const std::chrono::milliseconds& /*timeout*/) override {}

  const std::vector<Write>& writes() const {
    return writes_;
  }

  const std::vector<std::string>& operations() const {
    return operations_;
  }

 private:
  std::map<std::string, std::vector<uint8_t>> values_;
  std::vector<Write> writes_;
  std::vector<std::string> operations_;
};

// traceUpdate() packs the seq as raw bytes ahead of the op name.
uint64_t seqOf(const std::vector<uint8_t>& value) {
  uint64_t seq = 0;
  std::memcpy(&seq, value.data(), sizeof(seq));
  return seq;
}

// Representative of one watchdog pass: ~7 new works plus ~7 completing ones.
constexpr uint64_t kWorksPerPass = 7;

class DesyncDebuggerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    store_ = c10::make_intrusive<CountingStore>();
    debugger_.init(
        /*rank=*/0, /*size=*/1, /*globalRank=*/0, /*pgId=*/0, store_);
  }

  void logWork(uint64_t seq, bool ended) {
    at::Device device(at::kCUDA, 0);
    c10d::ProcessGroupNCCL::WorkNCCL work(
        "0", "default_pg", device, /*rank=*/0, c10d::OpType::ALLREDUCE, seq);
    debugger_.logWorkStart(work);
    if (ended) {
      debugger_.logWorkEnd(work);
    }
  }

  void logOnePass() {
    for (uint64_t seq = 1; seq <= kWorksPerPass; ++seq) {
      logWork(seq, /*ended=*/true);
    }
  }

  c10::intrusive_ptr<CountingStore> store_;
  c10d::ProcessGroupNCCL::DesyncDebugger debugger_;
};

// Buffering must not write to the store.
TEST_F(DesyncDebuggerTest, BufferingUnderTheLockWritesNothing) {
  logOnePass();

  EXPECT_TRUE(store_->writes().empty());
}

// A pass is coalesced to one write per trace key.
TEST_F(DesyncDebuggerTest, FlushCoalescesAPassIntoTwoWrites) {
  logOnePass();
  debugger_.flush();

  ASSERT_EQ(store_->writes().size(), 2);
  EXPECT_NE(store_->writes()[0].first, store_->writes()[1].first);
}

// Each trace key retains the pass's last sequence number.
TEST_F(DesyncDebuggerTest, FlushPublishesTheLastSeqOfThePass) {
  logOnePass();
  debugger_.flush();

  ASSERT_EQ(store_->writes().size(), 2);
  EXPECT_EQ(seqOf(store_->writes()[0].second), kWorksPerPass);
  EXPECT_EQ(seqOf(store_->writes()[1].second), kWorksPerPass);
}

// An idle pass does not write to the store.
TEST_F(DesyncDebuggerTest, FlushWithNothingBufferedIsANoOp) {
  debugger_.flush();
  EXPECT_TRUE(store_->writes().empty());

  logOnePass();
  debugger_.flush();
  ASSERT_EQ(store_->writes().size(), 2);

  debugger_.flush();
  EXPECT_EQ(store_->writes().size(), 2);
}

TEST_F(DesyncDebuggerTest, RunPublishesOwnStateBeforeReadingTheReport) {
  logWork(kWorksPerPass, /*ended=*/true);
  logWork(kWorksPerPass + 1, /*ended=*/false);

  debugger_.run();

  const std::vector<std::string> expectedWrites = {
      "set:NCCL_0_trace_start",
      "set:NCCL_0_trace_end",
  };
  ASSERT_GT(store_->operations().size(), expectedWrites.size());
  const std::vector<std::string> actualWrites(
      store_->operations().begin(),
      store_->operations().begin() + expectedWrites.size());
  EXPECT_EQ(actualWrites, expectedWrites);

  const auto writesBefore = store_->writes().size();
  logWork(kWorksPerPass + 2, /*ended=*/false);
  debugger_.flush();
  EXPECT_EQ(store_->writes().size(), writesBefore + 1);
}

} // namespace
