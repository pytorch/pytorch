#include <unistd.h>

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include "FileStore.hpp"

using namespace c10d;

class Semaphore {
 public:
  void post(int n = 1) {
    std::unique_lock<std::mutex> lock(m_);
    n_ += n;
    cv_.notify_all();
  }

  void wait(int n = 1) {
    std::unique_lock<std::mutex> lock(m_);
    while (n_ < n) {
      cv_.wait(lock);
    }
    n_ -= n;
  }

 protected:
  int n_ = 0;
  std::mutex m_;
  std::condition_variable cv_;
};

void set(Store& store, const std::string& key, const std::string& value) {
  std::vector<uint8_t> data(value.begin(), value.end());
  store.set(key, data);
}

void check(Store& store, const std::string& key, const std::string& expected) {
  auto tmp = store.get(key);
  auto actual = std::string((const char*) tmp.data(), tmp.size());
  if (actual != expected) {
    throw std::runtime_error("Expected " + expected + ", got " + actual);
  }
}

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
    FileStore store(path);
    set(store, "key0", "value0");
    set(store, "key1", "value1");
    set(store, "key2", "value2");
    check(store, "key0", "value0");
    check(store, "key1", "value1");
    check(store, "key2", "value2");
  }

  // Perform get on new instance
  {
    FileStore store(path);
    check(store, "key0", "value0");
  }

  // Hammer on FileStore#add
  std::vector<std::thread> threads;
  const auto numThreads = 4;
  const auto numIterations = 100;
  Semaphore sem1, sem2;
  for (auto i = 0; i < numThreads; i++) {
    threads.push_back(std::move(std::thread([&] {
            FileStore store(path);
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
    FileStore store(path);
    std::stringstream ss;
    ss << (numThreads * numIterations);
    check(store, "counter", ss.str());
  }

  unlink(path.c_str());
  return 0;
}
