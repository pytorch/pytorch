#include "StoreTestCommon.hpp"
#include "TCPStore.hpp"

#include <thread>
#include <iostream>
#include <cstdlib>


int main(int argc, char** argv) {

  // server store
  c10d::TCPStore serverStore("127.0.0.1", 29500, true);

  // Basic set/get on the server store
  c10d::test::set(serverStore, "key0", "value0");
  c10d::test::set(serverStore, "key1", "value1");
  c10d::test::set(serverStore, "key2", "value2");
  c10d::test::check(serverStore, "key0", "value0");
  c10d::test::check(serverStore, "key1", "value1");
  c10d::test::check(serverStore, "key2", "value2");

  // Hammer on TCPStore
  std::vector<std::thread> threads;
  const auto numThreads = 16;
  const auto numIterations = 1000;
  c10d::test::Semaphore sem1, sem2;

  // Each thread will have a client store to send/recv data
  std::vector<std::unique_ptr<c10d::TCPStore>> clientStores;
  for (auto i = 0; i < numThreads; i++) {
    clientStores.push_back(std::unique_ptr<c10d::TCPStore>(new
      c10d::TCPStore("127.0.0.1", 29500, false)));
  }

  std::string expectedCounterRes = std::to_string(numThreads * numIterations);

  for (auto i = 0; i < numThreads; i++) {
    threads.push_back(std::move(std::thread([&sem1,
                                             &sem2,
                                             &clientStores,
                                             i,
                                             &expectedCounterRes] {
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
              std::string val = "thread_val_" +
                  std::to_string(numIterations - 1);
              c10d::test::check(*clientStores[i], key, val);
            }

          })));
  }

  sem1.wait(numThreads);
  sem2.post(numThreads);

  for (auto& thread : threads) {
    thread.join();
  }

  // Clear the store to test that client disconnect won't shutdown the store
  clientStores.clear();

  // Check that the counter has the expected value
  c10d::test::check(serverStore, "counter", expectedCounterRes);

  // Check that each threads' written data from the main thread
  for (auto i = 0; i < numThreads; i++) {
    std::string key = "thread_" + std::to_string(i);
    std::string val = "thread_val_" + std::to_string(numIterations - 1);
    c10d::test::check(serverStore, key, val);
  }

  std::cout << "Test succeeded" << std::endl;
  return EXIT_SUCCESS;
}
