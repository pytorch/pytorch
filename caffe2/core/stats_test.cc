#include <chrono>
#include <iostream>
#include <thread>

#include <gtest/gtest.h>
#include "caffe2/core/stats.h"

namespace caffe2 {
namespace {

struct MyCaffeClass {
  explicit MyCaffeClass(const std::string& name) : stats_(name) {}

  void tryRun(int) {}

  void run(int numRuns) {
    try {
      CAFFE_EVENT(stats_, num_runs, numRuns);
      tryRun(numRuns);
      CAFFE_EVENT(stats_, num_successes);
    } catch (std::exception& e) {
      CAFFE_EVENT(stats_, num_failures, 1, "arg_to_usdt", e.what());
    }
    CAFFE_EVENT(stats_, usdt_only, 1, "arg_to_usdt");
  }

 private:
  struct MyStats {
    CAFFE_STAT_CTOR(MyStats);
    CAFFE_EXPORTED_STAT(num_runs);
    CAFFE_EXPORTED_STAT(num_successes);
    CAFFE_EXPORTED_STAT(num_failures);
    CAFFE_STAT(usdt_only);
  } stats_;
};

ExportedStatMap filterMap(
    const ExportedStatMap& map,
    const ExportedStatMap& keys) {
  ExportedStatMap filtered;
  for (const auto& kv : map) {
    if (keys.count(kv.first) > 0) {
      filtered.insert(kv);
    }
  }
  return filtered;
}

#define EXPECT_SUBSET(map, sub) EXPECT_EQ(filterMap((map), (sub)), (sub))

TEST(StatsTest, StatsTestClass) {
  MyCaffeClass a("first");
  MyCaffeClass b("second");
  for (int i = 0; i < 10; ++i) {
    a.run(10);
    b.run(5);
  }
  EXPECT_SUBSET(
      ExportedStatMap({
          {"first/num_runs", 100},
          {"first/num_successes", 10},
          {"first/num_failures", 0},
          {"second/num_runs", 50},
          {"second/num_successes", 10},
          {"second/num_failures", 0},
      }),
      toMap(StatRegistry::get().publish()));
}

TEST(StatsTest, StatsTestDuration) {
  struct TestStats {
    CAFFE_STAT_CTOR(TestStats);
    CAFFE_STAT(count);
    CAFFE_AVG_EXPORTED_STAT(time_ns);
  };
  TestStats stats("stats");
  CAFFE_DURATION(stats, time_ns) {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }

  ExportedStatList data;
  StatRegistry::get().publish(data);
  auto map = toMap(data);
  auto countIt = map.find("stats/time_ns/count");
  auto sumIt = map.find("stats/time_ns/sum");
  EXPECT_TRUE(countIt != map.end() && sumIt != map.end());
  EXPECT_EQ(countIt->second, 1);
  EXPECT_GT(sumIt->second, 0);
}

TEST(StatsTest, StatsTestSimple) {
  struct TestStats {
    CAFFE_STAT_CTOR(TestStats);
    CAFFE_STAT(s1);
    CAFFE_STAT(s2);
    CAFFE_EXPORTED_STAT(s3);
  };
  TestStats i1("i1");
  TestStats i2("i2");
  CAFFE_EVENT(i1, s1);
  CAFFE_EVENT(i1, s2);
  CAFFE_EVENT(i1, s3, 1);
  CAFFE_EVENT(i1, s3, -1);
  CAFFE_EVENT(i2, s3, 2);

  ExportedStatList data;
  StatRegistry::get().publish(data);
  EXPECT_SUBSET(toMap(data), ExportedStatMap({{"i1/s3", 0}, {"i2/s3", 2}}));

  StatRegistry reg2;
  reg2.update(data);
  reg2.update(data);

  EXPECT_SUBSET(
      toMap(reg2.publish(true)), ExportedStatMap({{"i1/s3", 0}, {"i2/s3", 4}}));
  EXPECT_SUBSET(
      toMap(reg2.publish()), ExportedStatMap({{"i1/s3", 0}, {"i2/s3", 0}}));
}

TEST(StatsTest, StatsTestStatic) {
  struct TestStats {
    CAFFE_STAT_CTOR(TestStats);
    CAFFE_STATIC_STAT(cpuUsage);
    CAFFE_STATIC_STAT(memUsage);
  };
  TestStats i1("i1");
  TestStats i2("i2");
  CAFFE_EVENT(i1, cpuUsage, 95);
  CAFFE_EVENT(i2, memUsage, 80);

  ExportedStatList data;
  StatRegistry::get().publish(data);
  EXPECT_SUBSET(
      toMap(data), ExportedStatMap({{"i1/cpuUsage", 95}, {"i2/memUsage", 80}}));

  CAFFE_EVENT(i1, cpuUsage, 80);
  CAFFE_EVENT(i1, memUsage, 50);
  CAFFE_EVENT(i2, memUsage, 90);

  StatRegistry::get().publish(data);
  EXPECT_SUBSET(
      toMap(data),
      ExportedStatMap(
          {{"i1/cpuUsage", 80}, {"i1/memUsage", 50}, {"i2/memUsage", 90}}));
}

static std::atomic<int64_t> svCount{0};
static std::atomic<int64_t> svValue{0};

struct CustomStatValue : public StatValue {
  int64_t increment(int64_t inc) override {
    svCount += 1;
    return svValue += inc;
  }

  int64_t reset(int64_t) override {
    return 0;
  }

  int64_t get() const override {
    return 0;
  }
};

TEST(StatsTest, StatValueCreator) {
  auto& registry = StatRegistry::get();
  std::string name = "";
  registry.setStatValueCreator([&](const std::string& n) {
    name = n;
    return caffe2::make_unique<CustomStatValue>();
  });

  struct TestStats {
    CAFFE_STAT_CTOR(TestStats);
    CAFFE_EXPORTED_STAT(bar);
  };
  TestStats foo("foo");
  CAFFE_EVENT(foo, bar, 5);
  CAFFE_EVENT(foo, bar, 10);

  ASSERT_EQ(name, "foo/bar");
  ASSERT_EQ(svCount.load(), 2);
  ASSERT_EQ(svValue.load(), 15);
}
} // namespace
} // namespace caffe2
