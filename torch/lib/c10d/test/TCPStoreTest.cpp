#include <c10d/test/StoreTestCommon.hpp>

#include <cstdlib>
#include <iostream>
#include <thread>

#include <gtest/gtest.h>

#include <c10d/PrefixStore.hpp>
#include <c10d/TCPStore.hpp>

constexpr int64_t kShortStoreTimeoutMillis = 100;

// Different ports for different tests.
void testHelper(const std::string& prefix = "") {
  const auto numThreads = 16;
  const auto numWorkers = numThreads + 1;

  auto serverTCPStore = c10::make_intrusive<c10d::TCPStore>(
      "127.0.0.1",
      0,
      numWorkers,
      true,
      std::chrono::seconds(30),
      /* wait */ false);

  auto serverStore =
      c10::make_intrusive<c10d::PrefixStore>(prefix, serverTCPStore);
  // server store
  auto serverThread = std::thread([&serverStore, &serverTCPStore] {
    // Wait for all workers to join.
    serverTCPStore->waitForWorkers();

    // Basic set/get on the server store
    c10d::test::set(*serverStore, "key0", "value0");
    c10d::test::set(*serverStore, "key1", "value1");
    c10d::test::set(*serverStore, "key2", "value2");
    c10d::test::check(*serverStore, "key0", "value0");
    c10d::test::check(*serverStore, "key1", "value1");
    c10d::test::check(*serverStore, "key2", "value2");
    serverStore->add("counter", 1);
    auto numKeys = serverStore->getNumKeys();
    // We expect 5 keys since 3 are added above, 'counter' is added by the
    // helper thread, and the init key to coordinate workers.
    EXPECT_EQ(numKeys, 5);

    // Check compareSet, does not check return value
    c10d::test::compareSet(*serverStore, "key0", "wrongCurrentValue", "newValue");
    c10d::test::check(*serverStore, "key0", "value0");
    c10d::test::compareSet(*serverStore, "key0", "value0", "newValue");
    c10d::test::check(*serverStore, "key0", "newValue");

    auto delSuccess = serverStore->deleteKey("key0");
    // Ensure that the key was successfully deleted
    EXPECT_TRUE(delSuccess);
    auto delFailure = serverStore->deleteKey("badKeyName");
    // The key was not in the store so the delete operation should have failed
    // and returned false.
    EXPECT_FALSE(delFailure);
    numKeys = serverStore->getNumKeys();
    EXPECT_EQ(numKeys, 4);
    auto timeout = std::chrono::milliseconds(kShortStoreTimeoutMillis);
    serverStore->setTimeout(timeout);
    EXPECT_THROW(serverStore->get("key0"), std::runtime_error);
  });

  // Hammer on TCPStore
  std::vector<std::thread> threads;
  const auto numIterations = 1000;
  c10d::test::Semaphore sem1, sem2;

  // Each thread will have a client store to send/recv data
  std::vector<c10::intrusive_ptr<c10d::TCPStore>> clientTCPStores;
  std::vector<c10::intrusive_ptr<c10d::PrefixStore>> clientStores;
  for (auto i = 0; i < numThreads; i++) {
    clientTCPStores.push_back(c10::make_intrusive<c10d::TCPStore>(
        "127.0.0.1", serverTCPStore->getPort(), numWorkers, false));
    clientStores.push_back(
        c10::make_intrusive<c10d::PrefixStore>(prefix, clientTCPStores[i]));
  }

  std::string expectedCounterRes = std::to_string(numThreads * numIterations + 1);

  for (auto i = 0; i < numThreads; i++) {
    threads.push_back(
        std::thread([&sem1, &sem2, &clientStores, i, &expectedCounterRes, &numIterations, &numThreads] {
          for (auto j = 0; j < numIterations; j++) {
            clientStores[i]->add("counter", 1);
          }
          // Let each thread set and get key on its client store
          std::string key = "thread_" + std::to_string(i);
          for (auto j = 0; j < numIterations; j++) {
            std::string val = "thread_val_" + std::to_string(j);
            c10d::test::set(*clientStores[i], key, val);
            c10d::test::check(*clientStores[i], key, val);
          }

          sem1.post();
          sem2.wait();
          // Check the counter results
          c10d::test::check(*clientStores[i], "counter", expectedCounterRes);
          // Now check other threads' written data
          for (auto j = 0; j < numThreads; j++) {
            if (j == i) {
              continue;
            }
            std::string key = "thread_" + std::to_string(i);
            std::string val = "thread_val_" + std::to_string(numIterations - 1);
            c10d::test::check(*clientStores[i], key, val);
          }
        }));
  }

  sem1.wait(numThreads);
  sem2.post(numThreads);

  for (auto& thread : threads) {
    thread.join();
  }

  serverThread.join();

  // Clear the store to test that client disconnect won't shutdown the store
  clientStores.clear();
  clientTCPStores.clear();

  // Check that the counter has the expected value
  c10d::test::check(*serverStore, "counter", expectedCounterRes);

  // Check that each threads' written data from the main thread
  for (auto i = 0; i < numThreads; i++) {
    std::string key = "thread_" + std::to_string(i);
    std::string val = "thread_val_" + std::to_string(numIterations - 1);
    c10d::test::check(*serverStore, key, val);
  }
}

TEST(TCPStoreTest, testHelper) {
  testHelper();
}

TEST(TCPStoreTest, testHelperPrefix) {
  testHelper("testPrefix");
}
