#include <c10/util/irange.h>
#include "StoreTestCommon.hpp"

#include <iostream>
#include <thread>

#include <torch/csrc/distributed/c10d/Backoff.hpp>

TEST(BackoffTest, exponentialBackoffDefaults) {
  c10d::ExponentialBackoffWithJitter backoff;
  EXPECT_EQ(backoff.initialInterval, std::chrono::milliseconds(500));
  EXPECT_EQ(backoff.maxInterval, std::chrono::milliseconds(60000));
  EXPECT_EQ(backoff.multiplier, 1.5);
  EXPECT_EQ(backoff.randomizationFactor, 0.5);
}

TEST(BackoffTest, exponentialBackoff) {
  c10d::ExponentialBackoffWithJitter backoff;
  backoff.randomizationFactor = 0.0;
  backoff.multiplier = 2.0;
  backoff.maxInterval = std::chrono::milliseconds(5000);

  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(500));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(1000));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(2000));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(4000));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(5000));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(5000));

  backoff.reset();
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(500));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(1000));
}

TEST(BackoffTest, expontentialBackoffRandomization) {
  c10d::ExponentialBackoffWithJitter backoff;
  backoff.initialInterval = std::chrono::milliseconds(1000);
  backoff.randomizationFactor = 0.5;
  backoff.multiplier = 1.0;
  backoff.maxInterval = std::chrono::milliseconds(5000);

  for (int i = 0; i < 100; i++) {
    auto backoffDur = backoff.nextBackoff();
    EXPECT_GE(backoffDur, std::chrono::milliseconds(500));
    EXPECT_LE(backoffDur, std::chrono::milliseconds(1500));
  }
}

TEST(BackoffTest, fixedBackoff) {
  c10d::FixedBackoff backoff{std::chrono::milliseconds(1000)};

  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(1000));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(1000));
  backoff.reset();
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(1000));
}

TEST(BackoffTest, sleep) {
  std::chrono::milliseconds sleepTime{10};
  c10d::FixedBackoff backoff{sleepTime};

  EXPECT_EQ(backoff.nextBackoff(), sleepTime);

  auto start = std::chrono::high_resolution_clock::now();
  backoff.sleepBackoff();
  auto dur = std::chrono::high_resolution_clock::now() - start;
  EXPECT_GE(dur, sleepTime);
}
