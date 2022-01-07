#include <gtest/gtest.h>

#include <thread>

#include <torch/csrc/monitor/counters.h>
#include <torch/csrc/monitor/events.h>

using namespace torch::monitor;

TEST(MonitorTest, CounterDouble) {
  FixedCountStat<double> a{
      "a",
      {MEAN, COUNT},
      2,
  };
  a.add(5.0);
  ASSERT_EQ(a.count(), 1);
  a.add(6.0);
  ASSERT_EQ(a.count(), 0);

  auto stats = a.get();
  std::unordered_map<Aggregation, double> want = {
      {MEAN, 5.5},
      {COUNT, 2.0},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64Sum) {
  FixedCountStat<int64_t> a{
      "a",
      {SUM},
      2,
  };
  a.add(5);
  a.add(6);
  auto stats = a.get();
  std::unordered_map<Aggregation, int64_t> want = {
      {SUM, 11},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64Value) {
  FixedCountStat<int64_t> a{
      "a",
      {VALUE},
      2,
  };
  a.add(5);
  a.add(6);
  auto stats = a.get();
  std::unordered_map<Aggregation, int64_t> want = {
      {VALUE, 6},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64Mean) {
  FixedCountStat<int64_t> a{
      "a",
      {MEAN},
      2,
  };
  {
    // zero samples case
    auto stats = a.get();
    std::unordered_map<Aggregation, int64_t> want = {
        {MEAN, 0},
    };
    ASSERT_EQ(stats, want);
  }

  a.add(0);
  a.add(10);

  {
    auto stats = a.get();
    std::unordered_map<Aggregation, int64_t> want = {
        {MEAN, 5},
    };
    ASSERT_EQ(stats, want);
  }
}

TEST(MonitorTest, CounterInt64Count) {
  FixedCountStat<int64_t> a{
      "a",
      {COUNT},
      2,
  };
  ASSERT_EQ(a.count(), 0);
  a.add(0);
  ASSERT_EQ(a.count(), 1);
  a.add(10);
  ASSERT_EQ(a.count(), 0);

  auto stats = a.get();
  std::unordered_map<Aggregation, int64_t> want = {
      {COUNT, 2},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64MinMax) {
  FixedCountStat<int64_t> a{
      "a",
      {MIN, MAX},
      6,
  };
  {
    auto stats = a.get();
    std::unordered_map<Aggregation, int64_t> want = {
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
    auto stats = a.get();
    std::unordered_map<Aggregation, int64_t> want = {
        {MAX, 9},
        {MIN, -6},
    };
    ASSERT_EQ(stats, want);
  }
}

TEST(MonitorTest, CounterInt64WindowSize) {
  FixedCountStat<int64_t> a{
      "a",
      {COUNT, SUM},
      /*windowSize=*/3,
  };
  a.add(1);
  a.add(2);
  ASSERT_EQ(a.count(), 2);
  a.add(3);
  ASSERT_EQ(a.count(), 0);

  a.add(4);
  ASSERT_EQ(a.count(), 1);

  auto stats = a.get();
  std::unordered_map<Aggregation, int64_t> want = {
      {COUNT, 3},
      {SUM, 6},
  };
  ASSERT_EQ(stats, want);
}

template <typename T>
struct TestIntervalStat : public IntervalStat<T> {
  uint64_t mockWindowId{0};

  TestIntervalStat(
      std::string name,
      std::initializer_list<Aggregation> aggregations,
      std::chrono::milliseconds windowSize)
      : IntervalStat<T>(name, aggregations, windowSize) {}

  uint64_t currentWindowId() const override {
    return mockWindowId;
  }
};

struct AggregatingEventHandler : public EventHandler {
  std::vector<Event> events;

  void handle(const Event& e) override {
    events.emplace_back(e);
  }
};

template <typename T>
struct HandlerGuard {
  std::shared_ptr<T> handler;

  HandlerGuard() : handler(std::make_shared<T>()) {
    registerEventHandler(handler);
  }

  ~HandlerGuard() {
    unregisterEventHandler(handler);
  }
};

TEST(MonitorTest, IntervalStat) {
  HandlerGuard<AggregatingEventHandler> guard;

  IntervalStat<int64_t> a{
      "a",
      {COUNT, SUM},
      std::chrono::milliseconds(1),
  };
  ASSERT_EQ(guard.handler->events.size(), 0);

  a.add(1);
  ASSERT_LE(a.count(), 1);

  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  a.add(2);
  ASSERT_LE(a.count(), 1);

  ASSERT_GE(guard.handler->events.size(), 1);
  ASSERT_LE(guard.handler->events.size(), 2);
}

TEST(MonitorTest, IntervalStatEvent) {
  HandlerGuard<AggregatingEventHandler> guard;

  TestIntervalStat<int64_t> a{
      "a",
      {COUNT, SUM},
      std::chrono::milliseconds(1),
  };
  ASSERT_EQ(guard.handler->events.size(), 0);

  a.add(1);
  ASSERT_EQ(a.count(), 1);
  a.add(2);
  ASSERT_EQ(a.count(), 2);
  ASSERT_EQ(guard.handler->events.size(), 0);

  a.mockWindowId = 100;

  a.add(3);
  ASSERT_LE(a.count(), 1);

  ASSERT_EQ(guard.handler->events.size(), 1);
  Event e = guard.handler->events.at(0);
  ASSERT_EQ(e.type, "torch.monitor.Stat");
  ASSERT_EQ(e.message, "a");
  ASSERT_NE(e.timestamp, std::chrono::system_clock::time_point{});
  std::unordered_map<std::string, metadata_value_t> metadata{
      {"a.sum", 3L},
      {"a.count", 2L},
  };
  ASSERT_EQ(e.metadata, metadata);
}

TEST(MonitorTest, IntervalStatEventDestruction) {
  HandlerGuard<AggregatingEventHandler> guard;

  {
    TestIntervalStat<int64_t> a{
        "a",
        {COUNT, SUM},
        std::chrono::hours(10),
    };
    a.add(1);
    ASSERT_EQ(a.count(), 1);
    ASSERT_EQ(guard.handler->events.size(), 0);
  }
  ASSERT_EQ(guard.handler->events.size(), 1);

  Event e = guard.handler->events.at(0);
  ASSERT_EQ(e.type, "torch.monitor.Stat");
  ASSERT_EQ(e.message, "a");
  ASSERT_NE(e.timestamp, std::chrono::system_clock::time_point{});
  std::unordered_map<std::string, metadata_value_t> metadata{
      {"a.sum", 1L},
      {"a.count", 1L},
  };
  ASSERT_EQ(e.metadata, metadata);
}

TEST(MonitorTest, FixedCountStatEvent) {
  HandlerGuard<AggregatingEventHandler> guard;

  FixedCountStat<int64_t> a{
      "a",
      {COUNT, SUM},
      3,
  };
  ASSERT_EQ(guard.handler->events.size(), 0);

  a.add(1);
  ASSERT_EQ(a.count(), 1);
  a.add(2);
  ASSERT_EQ(a.count(), 2);
  ASSERT_EQ(guard.handler->events.size(), 0);

  a.add(1);
  ASSERT_EQ(a.count(), 0);
  ASSERT_EQ(guard.handler->events.size(), 1);

  Event e = guard.handler->events.at(0);
  ASSERT_EQ(e.type, "torch.monitor.Stat");
  ASSERT_EQ(e.message, "a");
  ASSERT_NE(e.timestamp, std::chrono::system_clock::time_point{});
  std::unordered_map<std::string, metadata_value_t> metadata{
      {"a.sum", 4L},
      {"a.count", 3L},
  };
  ASSERT_EQ(e.metadata, metadata);
}

TEST(MonitorTest, FixedCountStatEventDestruction) {
  HandlerGuard<AggregatingEventHandler> guard;

  {
    FixedCountStat<int64_t> a{
        "a",
        {COUNT, SUM},
        3,
    };
    ASSERT_EQ(guard.handler->events.size(), 0);
    a.add(1);
    ASSERT_EQ(a.count(), 1);
    ASSERT_EQ(guard.handler->events.size(), 0);
  }
  ASSERT_EQ(guard.handler->events.size(), 1);

  Event e = guard.handler->events.at(0);
  ASSERT_EQ(e.type, "torch.monitor.Stat");
  ASSERT_EQ(e.message, "a");
  ASSERT_NE(e.timestamp, std::chrono::system_clock::time_point{});
  std::unordered_map<std::string, metadata_value_t> metadata{
      {"a.sum", 1L},
      {"a.count", 1L},
  };
  ASSERT_EQ(e.metadata, metadata);
}
