#include <c10d/test/StoreTestCommon.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <iostream>
#include <thread>

#include <gtest/gtest.h>

#include <c10d/FileStore.hpp>
#include <c10d/PrefixStore.hpp>

#ifdef _WIN32
std::string tmppath() {
  return c10d::test::autoGenerateTmpFilePath();
}
#else
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
#endif

void testGetSet(std::string path, std::string prefix = "") {
  // Basic Set/Get on File Store
  {
    auto fileStore = c10::make_intrusive<c10d::FileStore>(path, 2);
    c10d::PrefixStore store(prefix, fileStore);
    c10d::test::set(store, "key0", "value0");
    c10d::test::set(store, "key1", "value1");
    c10d::test::set(store, "key2", "value2");
    c10d::test::check(store, "key0", "value0");
    c10d::test::check(store, "key1", "value1");
    c10d::test::check(store, "key2", "value2");
    auto numKeys = fileStore->getNumKeys();
    EXPECT_EQ(numKeys, 3);

    // Check compareSet, does not check return value
    c10d::test::compareSet(store, "key0", "wrongExpectedValue", "newValue");
    c10d::test::check(store, "key0", "value0");
    c10d::test::compareSet(store, "key0", "value0", "newValue");
    c10d::test::check(store, "key0", "newValue");
  }

  // Perform get on new instance
  {
    auto fileStore = c10::make_intrusive<c10d::FileStore>(path, 2);
    c10d::PrefixStore store(prefix, fileStore);
    c10d::test::check(store, "key0", "newValue");
    auto numKeys = fileStore->getNumKeys();
    // There will be 4 keys since we still use the same underlying file as the
    // other store above.
    EXPECT_EQ(numKeys, 4);
  }
}

void stressTestStore(std::string path, std::string prefix = "") {
  // Hammer on FileStore::add
  const auto numThreads = 4;
  const auto numIterations = 100;

  std::vector<std::thread> threads;
  c10d::test::Semaphore sem1, sem2;

  for (auto i = 0; i < numThreads; i++) {
    threads.push_back(std::thread([&] {
      auto fileStore =
          c10::make_intrusive<c10d::FileStore>(path, numThreads + 1);
      c10d::PrefixStore store(prefix, fileStore);
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

  // Check that the counter has the expected value
  {
    auto fileStore = c10::make_intrusive<c10d::FileStore>(path, numThreads + 1);
    c10d::PrefixStore store(prefix, fileStore);
    std::string expected = std::to_string(numThreads * numIterations);
    c10d::test::check(store, "counter", expected);
  }
}

class FileStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = tmppath();
  }

  void TearDown() override {
    unlink(path_.c_str());
  }

  std::string path_;
};

TEST_F(FileStoreTest, testGetAndSet) {
  testGetSet(path_);
}

TEST_F(FileStoreTest, testGetAndSetWithPrefix) {
  testGetSet(path_, "testPrefix");
}

TEST_F(FileStoreTest, testStressStore) {
  stressTestStore(path_);
}

TEST_F(FileStoreTest, testStressStoreWithPrefix) {
  stressTestStore(path_, "testPrefix");
}
