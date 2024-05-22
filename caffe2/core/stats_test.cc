#include <chrono>
#include <iostream>
#include <thread>

#include "caffe2/core/stats.h"
#include <gtest/gtest.h>

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
    // NOLINTNEXTLINE(modernize-pass-by-value)
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
    // NOLINTNEXTLINE(modernize-pass-by-value)
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
    // NOLINTNEXTLINE(modernize-pass-by-value)
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
    // NOLINTNEXTLINE(modernize-pass-by-value)
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
} // namespace
} // namespace caffe2
