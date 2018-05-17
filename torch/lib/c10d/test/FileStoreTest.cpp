#include "StoreTestCommon.hpp"
#include "FileStore.hpp"

#include <unistd.h>
#include <iostream>
#include <thread>

std::string tmppath() {
  const char* tmpdir = getenv("TMPDIR");
  if (tmpdir == nullptr) {
    tmpdir = "/tmp";
  }

  // Create template
  std::vector<char> tmp(256);
  auto len = snprintf(tmp.data(), tmp.size(), "%s/testXXXXXX", tmpdir);
  tmp.resize(len);

  // Create temporary file
  auto fd = mkstemp(&tmp[0]);
  if (fd == -1) {
    throw std::system_error(errno, std::system_category());
  }
  close(fd);
  return std::string(tmp.data(), tmp.size());
}

int main(int argc, char** argv) {
  auto path = tmppath();
  std::cout << "Using temporary file: " << path << std::endl;

  // Basic set/get
  {
    c10d::FileStore store(path);
    c10d::test::set(store, "key0", "value0");
    c10d::test::set(store, "key1", "value1");
    c10d::test::set(store, "key2", "value2");
    c10d::test::check(store, "key0", "value0");
    c10d::test::check(store, "key1", "value1");
    c10d::test::check(store, "key2", "value2");
  }

  // Perform get on new instance
  {
    c10d::FileStore store(path);
    c10d::test::check(store, "key0", "value0");
  }

  // Hammer on FileStore#add
  std::vector<std::thread> threads;
  const auto numThreads = 4;
  const auto numIterations = 100;
  c10d::test::Semaphore sem1, sem2;
  for (auto i = 0; i < numThreads; i++) {
    threads.push_back(std::move(std::thread([&] {
            c10d::FileStore store(path);
            sem1.post();
            sem2.wait();
            for (auto j = 0; j < numIterations; j++) {
              store.add("counter", 1);
            }
          })));
  }
  sem1.wait(numThreads);
  sem2.post(numThreads);
  for (auto& thread : threads) {
    thread.join();
  }

  // Check that the counter has the expected value
  {
    c10d::FileStore store(path);
    std::string expected = std::to_string(numThreads * numIterations);
    c10d::test::check(store, "counter", expected);
  }

  unlink(path.c_str());
  std::cout << "Test succeeded" << std::endl;
  return 0;
}
