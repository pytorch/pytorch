#include <c10/util/irange.h>
#include "StoreTestCommon.hpp"

#include <cstdlib>
#include <future>
#include <iostream>
#include <system_error>
#include <thread>

#include <gtest/gtest.h>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

constexpr int64_t kShortStoreTimeoutMillis = 100;
constexpr int64_t kStoreCallbackTimeoutMillis = 5000;
constexpr int defaultTimeout = 20;

c10::intrusive_ptr<c10d::TCPStore> _createServer(
    int numWorkers = 1,
    int timeout = defaultTimeout) {
  return c10::make_intrusive<c10d::TCPStore>(
      "127.0.0.1",
      c10d::TCPStoreOptions{
          /* port */ 0,
          /* isServer */ true,
          numWorkers,
          /* waitWorkers */ false,
          /* timeout */ std::chrono::seconds(timeout)});
}

// Different ports for different tests.
void testHelper(const std::string& prefix = "") {
  constexpr auto numThreads = 16;
  constexpr auto numWorkers = numThreads + 1;

  auto serverTCPStore = _createServer(numWorkers);

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
    c10d::test::compareSet(
        *serverStore, "key0", "wrongExpectedValue", "newValue");
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
    EXPECT_THROW(serverStore->get("key0"), c10::Error);
  });

  // Hammer on TCPStore
  std::vector<std::thread> threads;
  constexpr auto numIterations = 1000;
  c10d::test::Semaphore sem1, sem2;

  c10d::TCPStoreOptions opts{};
  opts.port = serverTCPStore->getPort();
  opts.numWorkers = numWorkers;

  // Each thread will have a client store to send/recv data
  std::vector<c10::intrusive_ptr<c10d::TCPStore>> clientTCPStores;
  std::vector<c10::intrusive_ptr<c10d::PrefixStore>> clientStores;
  for (const auto i : c10::irange(numThreads)) {
    clientTCPStores.push_back(
        c10::make_intrusive<c10d::TCPStore>("127.0.0.1", opts));
    clientStores.push_back(
        c10::make_intrusive<c10d::PrefixStore>(prefix, clientTCPStores[i]));
  }

  std::string expectedCounterRes =
      std::to_string(numThreads * numIterations + 1);

  for (const auto i : c10::irange(numThreads)) {
    threads.emplace_back(
        std::thread([=, &sem1, &sem2, &clientStores, &expectedCounterRes] {
          for (C10_UNUSED const auto j : c10::irange(numIterations)) {
            clientStores[i]->add("counter", 1);
          }
          // Let each thread set and get key on its client store
          std::string key = "thread_" + std::to_string(i);
          for (const auto j : c10::irange(numIterations)) {
            std::string val = "thread_val_" + std::to_string(j);
            c10d::test::set(*clientStores[i], key, val);
            c10d::test::check(*clientStores[i], key, val);
          }

          sem1.post();
          sem2.wait();
          // Check the counter results
          c10d::test::check(*clientStores[i], "counter", expectedCounterRes);
          // Now check other threads' written data
          for (const auto j : c10::irange(numThreads)) {
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
  for (const auto i : c10::irange(numThreads)) {
    std::string key = "thread_" + std::to_string(i);
    std::string val = "thread_val_" + std::to_string(numIterations - 1);
    c10d::test::check(*serverStore, key, val);
  }
}

void testWatchKeyCallback(const std::string& prefix = "") {
  // Callback function increments counter of the total number of callbacks that
  // were run
  std::promise<int> numCallbacksExecutedPromise;
  std::atomic<int> numCallbacksExecuted{0};
  constexpr int numThreads = 16;
  constexpr int keyChangeOperation = 3;
  c10d::WatchKeyCallback callback =
      [=, &numCallbacksExecuted, &numCallbacksExecutedPromise](
          c10::optional<std::string> /* unused */,
          c10::optional<std::string> /* unused */) {
        numCallbacksExecuted++;
        if (numCallbacksExecuted == numThreads * keyChangeOperation * 2) {
          numCallbacksExecutedPromise.set_value(numCallbacksExecuted);
        }
      };

  const int numWorkers = numThreads + 1;
  auto serverTCPStore = _createServer(numWorkers);
  auto serverStore =
      c10::make_intrusive<c10d::PrefixStore>(prefix, serverTCPStore);

  c10d::TCPStoreOptions opts{};
  opts.port = serverTCPStore->getPort();
  opts.numWorkers = numWorkers;

  // Each thread will have a client store to send/recv data
  std::vector<c10::intrusive_ptr<c10d::TCPStore>> clientTCPStores;
  std::vector<c10::intrusive_ptr<c10d::PrefixStore>> clientStores;
  for (const auto i : c10::irange(numThreads)) {
    clientTCPStores.push_back(
        c10::make_intrusive<c10d::TCPStore>("127.0.0.1", opts));
    clientStores.push_back(
        c10::make_intrusive<c10d::PrefixStore>(prefix, clientTCPStores[i]));
  }

  // Start watching key on server and client stores
  std::string internalKey = "internalKey";
  std::string internalKeyCount = "internalKeyCount";
  for (const auto i : c10::irange(numThreads)) {
    serverStore->watchKey(internalKey + std::to_string(i), callback);
    serverStore->watchKey(internalKeyCount + std::to_string(i), callback);
    clientStores[i]->watchKey(internalKey + std::to_string(i), callback);
    clientStores[i]->watchKey(internalKeyCount + std::to_string(i), callback);
  }

  std::vector<std::thread> threads;
  std::atomic<int> keyChangeOperationCount{0};
  for (const auto i : c10::irange(numThreads)) {
    threads.emplace_back(std::thread([=,
                                      &clientStores,
                                      &internalKey,
                                      &internalKeyCount,
                                      &keyChangeOperationCount] {
      // Let each thread set and get key on its client store
      std::string key = internalKey + std::to_string(i);
      std::string keyCounter = internalKeyCount + std::to_string(i);
      std::string val = "thread_val_" + std::to_string(i);
      // The set, compareSet, add methods count as key change operations
      c10d::test::set(*clientStores[i], key, val);
      c10d::test::compareSet(*clientStores[i], key, val, "newValue");
      clientStores[i]->add(keyCounter, i);
      keyChangeOperationCount += keyChangeOperation * 2;
      c10d::test::check(*clientStores[i], key, "newValue");
      c10d::test::check(*clientStores[i], keyCounter, std::to_string(i));
    }));
  }

  // Ensures that internal_key has been "set" and "get"
  for (auto& thread : threads) {
    thread.join();
  }

  std::future<int> numCallbacksExecutedFuture =
      numCallbacksExecutedPromise.get_future();
  std::chrono::milliseconds span(kStoreCallbackTimeoutMillis);
  if (numCallbacksExecutedFuture.wait_for(span) == std::future_status::timeout)
    TORCH_CHECK(false, "Callback execution timed out.");

  // Check number of callbacks executed equal to number of key change operations
  // Wait for all callbacks to be triggered
  EXPECT_EQ(keyChangeOperationCount, numCallbacksExecutedFuture.get());
}

TEST(TCPStoreTest, testHelper) {
  testHelper();
}

TEST(TCPStoreTest, testHelperPrefix) {
  testHelper("testPrefix");
}

TEST(TCPStoreTest, testWatchKeyCallback) {
  testWatchKeyCallback();
}

TEST(TCPStoreTest, testWatchKeyCallbackWithPrefix) {
  testWatchKeyCallback("testPrefix");
}

// Helper function to create a key on the store, watch it, and run the callback
void testKeyChangeHelper(
    c10d::Store& store,
    std::string key,
    const c10::optional<std::string>& expectedOldValue,
    const c10::optional<std::string>& expectedNewValue) {
  std::exception_ptr eptr = nullptr;
  std::promise<bool> callbackPromise;

  // Test the correctness of new_value and old_value
  c10d::WatchKeyCallback callback = [expectedOldValue,
                                     expectedNewValue,
                                     &callbackPromise,
                                     &eptr](
                                        c10::optional<std::string> oldValue,
                                        c10::optional<std::string> newValue) {
    try {
      EXPECT_EQ(expectedOldValue.value_or("NONE"), oldValue.value_or("NONE"));
      EXPECT_EQ(expectedNewValue.value_or("NONE"), newValue.value_or("NONE"));
    } catch (...) {
      eptr = std::current_exception();
    }
    callbackPromise.set_value(true);
  };
  store.watchKey(key, callback);

  // Perform the specified update according to key
  if (key == "testEmptyKeyValue" || key == "testRegularKeyValue" ||
      key == "testWatchKeyCreate") {
    c10d::test::set(store, key, expectedNewValue.value());
  } else if (key == "testWatchKeyAdd") {
    store.add(key, std::stoi(expectedNewValue.value()));
  } else if (key == "testWatchKeyDelete") {
    store.deleteKey(key);
  }

  // Test that the callback is fired and the expected values are correct
  std::future<bool> callbackFuture = callbackPromise.get_future();
  std::chrono::milliseconds span(kStoreCallbackTimeoutMillis);
  if (callbackFuture.wait_for(span) == std::future_status::timeout)
    TORCH_CHECK(false, "Callback execution timed out.");

  // Any exceptions raised from asserts should be rethrown
  if (eptr)
    std::rethrow_exception(eptr);
}

TEST(TCPStoreTest, testKeyEmptyUpdate) {
  auto store = _createServer();

  std::string key = "testEmptyKeyValue";
  c10d::test::set(*store, key, "");
  store->get(key);
  testKeyChangeHelper(*store, key, "", "2");
}

TEST(TCPStoreTest, testKeyUpdate) {
  auto store = _createServer();

  std::string key = "testRegularKeyValue";
  c10d::test::set(*store, key, "1");
  store->get(key);
  testKeyChangeHelper(*store, key, "1", "2");
}

TEST(TCPStoreTest, testKeyCreate) {
  auto store = _createServer();

  std::string key = "testWatchKeyCreate";
  testKeyChangeHelper(*store, key, c10::nullopt, "2");
}

TEST(TCPStoreTest, testKeyAdd) {
  auto store = _createServer();

  std::string key = "testWatchKeyAdd";
  testKeyChangeHelper(*store, key, c10::nullopt, "2");
}

TEST(TCPStoreTest, testKeyDelete) {
  auto store = _createServer();

  std::string key = "testWatchKeyDelete";
  c10d::test::set(*store, key, "1");
  store->get(key);
  testKeyChangeHelper(*store, key, "1", c10::nullopt);
}

TEST(TCPStoreTest, testCleanShutdown) {
  int numWorkers = 2;

  auto serverTCPStore = std::make_unique<c10d::TCPStore>(
      "127.0.0.1",
      0,
      numWorkers,
      true,
      std::chrono::seconds(defaultTimeout),
      /* wait */ false);
  c10d::test::set(*serverTCPStore, "key", "val");

  auto clientTCPStore = c10::make_intrusive<c10d::TCPStore>(
      "127.0.0.1",
      c10d::TCPStoreOptions{
          /* port */ serverTCPStore->getPort(),
          /* isServer */ false,
          numWorkers,
          /* waitWorkers */ false,
          /* timeout */ std::chrono::seconds(defaultTimeout)});
  clientTCPStore->get("key");

  auto clientThread = std::thread([&clientTCPStore] {
    EXPECT_THROW(clientTCPStore->get("invalid_key"), std::system_error);
  });

  // start server shutdown during a client request
  serverTCPStore = nullptr;

  clientThread.join();
}

TEST(TCPStoreTest, testMultiTenantStores) {
  c10d::TCPStoreOptions opts{};
  opts.isServer = true;
  opts.multiTenant = true;

  // Construct two server stores on the same port.
  auto store1 = c10::make_intrusive<c10d::TCPStore>("localhost", opts);
  auto store2 = c10::make_intrusive<c10d::TCPStore>("localhost", opts);

  // Assert that the two stores share the same server.
  c10d::test::set(*store1, "key0", "value0");
  c10d::test::check(*store2, "key0", "value0");

  // Dispose the second instance and assert that the server is still alive.
  store2.reset();

  c10d::test::set(*store1, "key0", "value0");
  c10d::test::check(*store1, "key0", "value0");
}
