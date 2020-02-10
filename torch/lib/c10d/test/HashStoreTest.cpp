#include <c10d/test/StoreTestCommon.hpp>

#include <unistd.h>

#include <iostream>
#include <thread>

#include <c10d/HashStore.hpp>
#include <c10d/PrefixStore.hpp>

void testHelper(std::string prefix = "") {
  // Basic set/get
  {
    auto hashStore = std::make_shared<c10d::HashStore>();
    c10d::PrefixStore store(prefix, hashStore);
    c10d::test::set(store, "key0", "value0");
    c10d::test::set(store, "key1", "value1");
    c10d::test::set(store, "key2", "value2");
    c10d::test::check(store, "key0", "value0");
    c10d::test::check(store, "key1", "value1");
    c10d::test::check(store, "key2", "value2");
  }

  // get() waits up to timeout_.
  {
    auto hashStore = std::make_shared<c10d::HashStore>();
    c10d::PrefixStore store(prefix, hashStore);
    std::thread th([&]() { c10d::test::set(store, "key0", "value0"); });
    c10d::test::check(store, "key0", "value0");
    th.join();
  }

  // Hammer on HashStore#add
  std::vector<std::thread> threads;
  const auto numThreads = 4;
  const auto numIterations = 100;
  c10d::test::Semaphore sem1, sem2;
  auto hashStore = std::make_shared<c10d::HashStore>();
  c10d::PrefixStore store(prefix, hashStore);
  for (auto i = 0; i < numThreads; i++) {
    threads.push_back(std::thread([&] {
      sem1.post();
      sem2.wait();
      for (auto j = 0; j < numIterations; j++) {
        store.add("counter", 1);
      }
    }));
  }
  sem1.wait(numThreads);
  sem2.post(numThreads);
  for (auto& thread : threads) {
    thread.join();
  }
  std::string expected = std::to_string(numThreads * numIterations);
  c10d::test::check(store, "counter", expected);
}

int main(int /* unused */, char** /* unused */) {
  testHelper();
  testHelper("testPrefix");
  std::cout << "Test succeeded" << std::endl;
}
