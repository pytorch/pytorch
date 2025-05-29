#include <c10/core/impl/COW.h>
#include <c10/core/impl/COWDeleter.h>

#include <c10/core/CPUAllocator.h>
#include <c10/core/StorageImpl.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <thread>
#include <unordered_set>

// NOLINTBEGIN(clang-analyzer-cplusplus*)
namespace c10::impl {
namespace {

TEST(storage_id_test, performance) {
  std::time_t startTime = std::time(nullptr);

  const int numThreads = 64;
  std::vector<std::thread> threads(numThreads);
  for (int i = 0; i < numThreads; ++i) {
    threads[i] = std::thread([]() {
      // size_t n = 1000000000;
      size_t n = 100000000;
      for (size_t j = 0; j < n; j++) {
        StorageImpl original_storage({}, 6, GetCPUAllocator(), false);
      }
    });
  }
  // Wait for all threads to finish
  for (auto& thread : threads) {
    thread.join();
  }

  std::time_t endTime = std::time(nullptr);
  std::time_t diffTime = endTime - startTime;
  std::cout << "Time taken: " << diffTime << " seconds"
            << "\n";
}

TEST(storage_id_test, uniqueness) {
  std::unordered_set<size_t> storage_ids;

  const int numThreads = 64;
  std::vector<std::thread> threads(numThreads);
  bool found_duplicate = false;
  std::mutex m;
  for (int i = 0; i < numThreads; ++i) {
    threads[i] = std::thread([&found_duplicate, &storage_ids, &m]() {
      // size_t n = 100000;
      size_t n = 10000;
      for (size_t j = 0; j < n; j++) {
        StorageImpl original_storage({}, 6, GetCPUAllocator(), false);
        size_t id = original_storage.get_id();
        {
          std::lock_guard<std::mutex> lock(m);
          if (storage_ids.find(id) != storage_ids.end()) {
            found_duplicate = true;
            break;
          }
          storage_ids.insert(id);
        }
      }
    });
  }
  // Wait for all threads to finish
  for (auto& thread : threads) {
    thread.join();
  }

  ASSERT_FALSE(found_duplicate);
}

} // namespace
} // namespace c10::impl
// NOLINTEND(clang-analyzer-cplusplus*)
