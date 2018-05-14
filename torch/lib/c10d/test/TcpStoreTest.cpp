#include "StoreTestCommon.hpp"
#include "TcpStore.hpp"

#include <thread>
#include <iostream>
#include <cstdlib>


int main(int argc, char** argv) {

  // Master store
  c10d::TcpStore masterStore("127.0.0.1", 29500, true);

  // Basic set/get on the master store
  c10d::test::set(masterStore, "key0", "value0");
  c10d::test::set(masterStore, "key1", "value1");
  c10d::test::set(masterStore, "key2", "value2");
  c10d::test::check(masterStore, "key0", "value0");
  c10d::test::check(masterStore, "key1", "value1");
  c10d::test::check(masterStore, "key2", "value2");

  // Hammer on TcpStore
  std::vector<std::thread> threads;
  const auto numThreads = 16;
  const auto numIterations = 1000;
  c10d::test::Semaphore sem1, sem2;

  // Each thread will have a slave store to send/recv data
  std::vector<std::unique_ptr<c10d::TcpStore>> slaveStores;
  for (auto i = 0; i < numThreads; i++) {
    slaveStores.push_back(std::unique_ptr<c10d::TcpStore>(new
      c10d::TcpStore("127.0.0.1", 29500, false)));
  }

  for (auto i = 0; i < numThreads; i++) {
    threads.push_back(std::move(std::thread([&sem1, &sem2, &slaveStores, i] {
            sem1.post();
            sem2.wait();
            for (auto j = 0; j < numIterations; j++) {
              slaveStores[i]->add("counter", 1);
            }
            // Let each thread set and get key on its slave store
            std::string key = "thread_" + std::to_string(i);
            std::string val = "thread_val_" + std::to_string(i);
            for (auto j = 0; j < numIterations; j++) {
              c10d::test::set(*slaveStores[i], key, val);
              c10d::test::check(*slaveStores[i], key, val);
            }
          })));
  }
  sem1.wait(numThreads);
  sem2.post(numThreads);
  for (auto& thread : threads) {
    thread.join();
  }

  // Check that the counter has the expected value
  std::string expected = std::to_string(numThreads * numIterations);
  c10d::test::check(masterStore, "counter", expected);

  std::cout << "Test succeeded" << std::endl;
  return EXIT_SUCCESS;
}
