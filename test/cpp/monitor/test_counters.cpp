#include <gtest/gtest.h>

#include <torch/csrc/monitor/counters.h>

using namespace torch::monitor;

TEST(MonitorTest, CounterDouble) {
  Stat<double> a{
      "a",
      {MEAN, COUNT},
  };
  a.add(5.0);
  ASSERT_EQ(a.count(), 1);
  a.add(6.0);
  ASSERT_EQ(a.count(), 2);
  a.closeWindow();
  auto stats = a.get();
  ASSERT_EQ(a.count(), 0);

  std::vector<std::pair<Aggregation, double>> want = {
      {MEAN, 5.5},
      {COUNT, 2.0},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64Sum) {
  Stat<int64_t> a{
      "a",
      {SUM},
  };
  a.add(5);
  a.add(6);
  a.closeWindow();
  auto stats = a.get();
  std::vector<std::pair<Aggregation, int64_t>> want = {
      {SUM, 11},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64Value) {
  Stat<int64_t> a{
      "a",
      {VALUE},
  };
  a.add(5);
  a.add(6);
  a.closeWindow();
  auto stats = a.get();
  std::vector<std::pair<Aggregation, int64_t>> want = {
      {VALUE, 6},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64Mean) {
  Stat<int64_t> a{
      "a",
      {MEAN},
  };
  a.add(0);
  a.add(10);

  {
    a.closeWindow();
    auto stats = a.get();
    std::vector<std::pair<Aggregation, int64_t>> want = {
        {MEAN, 5},
    };
    ASSERT_EQ(stats, want);
  }

  {
    // zero samples case
    a.closeWindow();
    auto stats = a.get();
    std::vector<std::pair<Aggregation, int64_t>> want = {
        {MEAN, 0},
    };
    ASSERT_EQ(stats, want);
  }
}

TEST(MonitorTest, CounterInt64Count) {
  Stat<int64_t> a{
      "a",
      {COUNT},
  };
  ASSERT_EQ(a.count(), 0);
  a.add(0);
  ASSERT_EQ(a.count(), 1);
  a.add(10);
  ASSERT_EQ(a.count(), 2);
  a.closeWindow();
  auto stats = a.get();
  ASSERT_EQ(a.count(), 0);
  std::vector<std::pair<Aggregation, int64_t>> want = {
      {COUNT, 2},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64MinMax) {
  Stat<int64_t> a{
      "a",
      {MIN, MAX},
  };
  {
    a.closeWindow();
    auto stats = a.get();
    std::vector<std::pair<Aggregation, int64_t>> want = {
        {MAX, 0},
        {MIN, 0},
    };
    ASSERT_EQ(stats, want);
  }
  a.add(0);
  a.add(5);
  a.add(-5);
  a.add(-6);
  a.add(9);
  a.add(2);
  {
    a.closeWindow();
    auto stats = a.get();
    std::vector<std::pair<Aggregation, int64_t>> want = {
        {MAX, 9},
        {MIN, -6},
    };
    ASSERT_EQ(stats, want);
  }
}

TEST(MonitorTest, CounterInt64WindowSize) {
  Stat<int64_t> a{
      "a",
      {COUNT, SUM},
      /*windowSize=*/3,
  };
  a.add(1);
  a.add(2);
  ASSERT_EQ(a.count(), 2);
  a.add(3);
  ASSERT_EQ(a.count(), 0);

  a.closeWindow();
  auto stats = a.get();
  std::vector<std::pair<Aggregation, int64_t>> want = {
      {COUNT, 3},
      {SUM, 6},
  };
  ASSERT_EQ(stats, want);
  a.closeWindow();
  ASSERT_EQ(stats, a.get());
}

TEST(MonitorTest, CloseAndGetStats) {
  Stat<int64_t> a{
      "a",
      {COUNT, SUM},
      /*windowSize=*/3,
  };
  Stat<double> b{
      "b",
      {MIN, MAX},
      2,
  };

  a.add(1);
  b.add(1);

  {
    auto out = closeAndGetStats();
    std::pair<
        std::unordered_map<std::string, double>,
        std::unordered_map<std::string, int64_t>>
        want = {
            {{"a.count", 1}, {"a.sum", 1}},
            {{"b.min", 0}, {"b.max", 0}},
        };
  }

  a.add(2);
  b.add(2);

  {
    auto out = closeAndGetStats();
    std::pair<
        std::unordered_map<std::string, double>,
        std::unordered_map<std::string, int64_t>>
        want = {
            {{"a.count", 1}, {"a.sum", 2}},
            {{"b.min", 1}, {"b.max", 2}},
        };
  }
}
