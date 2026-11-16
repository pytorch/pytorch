#include <c10/util/irange.h>
#include "StoreTestCommon.hpp"

#include <unistd.h>

#include <thread>

#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <utility>

constexpr int64_t kShortStoreTimeoutMillis = 100;

void testGetSet(const std::string& prefix = "") {
  // Basic set/get
  {
    auto hashStore = c10::make_intrusive<c10d::HashStore>();
    c10d::PrefixStore store(prefix, hashStore);
    c10d::test::set(store, "key0", "value0");
    c10d::test::set(store, "key1", "value1");
    c10d::test::set(store, "key2", "value2");
    c10d::test::check(store, "key0", "value0");
    c10d::test::check(store, "key1", "value1");
    c10d::test::check(store, "key2", "value2");

    // Check compareSet, does not check return value
    c10d::test::compareSet(store, "key0", "wrongExpectedValue", "newValue");
    c10d::test::check(store, "key0", "value0");
    c10d::test::compareSet(store, "key0", "value0", "newValue");
    c10d::test::check(store, "key0", "newValue");

    auto numKeys = store.getNumKeys();
    EXPECT_EQ(numKeys, 3);
    auto delSuccess = store.deleteKey("key0");
    EXPECT_TRUE(delSuccess);
    numKeys = store.getNumKeys();
    EXPECT_EQ(numKeys, 2);
    auto delFailure = store.deleteKey("badKeyName");
    EXPECT_FALSE(delFailure);
    auto timeout = std::chrono::milliseconds(kShortStoreTimeoutMillis);
    store.setTimeout(timeout);
    EXPECT_THROW(store.get("key0"), c10::DistStoreError);
  }

  // get() waits up to timeout_.
  {
    auto hashStore = c10::make_intrusive<c10d::HashStore>();
    c10d::PrefixStore store(prefix, hashStore);
    std::thread th([&]() { c10d::test::set(store, "key0", "value0"); });
    c10d::test::check(store, "key0", "value0");
    th.join();
  }
}

void stressTestStore(std::string prefix = "") {
  // Hammer on HashStore::add
  const auto numThreads = 4;
  const auto numIterations = 100;

  std::vector<std::thread> threads;
  c10d::test::Semaphore sem1, sem2;
  auto hashStore = c10::make_intrusive<c10d::HashStore>();
  c10d::PrefixStore store(std::move(prefix), hashStore);

  for ([[maybe_unused]] const auto i : c10::irange(numThreads)) {
    threads.emplace_back([&] {
      sem1.post();
      sem2.wait();
      for ([[maybe_unused]] const auto j : c10::irange(numIterations)) {
        store.add("counter", 1);
      }
    });
  }

  sem1.wait(numThreads);
  sem2.post(numThreads);

  for (auto& thread : threads) {
    thread.join();
  }
  std::string expected = std::to_string(numThreads * numIterations);
  c10d::test::check(store, "counter", expected);
}

TEST(HashStoreTest, testGetAndSet) {
  testGetSet();
}

TEST(HashStoreTest, testGetAndSetWithPrefix) {
  testGetSet("testPrefix");
}

TEST(HashStoreTest, testStressStore) {
  stressTestStore();
}

TEST(HashStoreTest, testStressStoreWithPrefix) {
  stressTestStore("testPrefix");
}
