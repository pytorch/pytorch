#include <atomic>
#include <thread>

#include <gtest/gtest.h>

#include <torch/nativert/detail/MPMCQueue.h>

using torch::nativert::detail::MPMCQueue;

TEST(MPMCQueueTest, EmptyQueue) {
  MPMCQueue<int> queue(5);
  int out = 0;
  EXPECT_FALSE(queue.readIfNotEmpty(out));
}

TEST(MPMCQueueTest, SingleElement) {
  MPMCQueue<int> queue(5);
  EXPECT_TRUE(queue.writeIfNotFull(10));
  int out = 0;
  EXPECT_TRUE(queue.readIfNotEmpty(out));
  EXPECT_EQ(out, 10);
}

TEST(MPMCQueueTest, MultipleElements) {
  MPMCQueue<int> queue(5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(queue.writeIfNotFull(i));
  }
  for (int i = 0; i < 5; ++i) {
    int out = 0;
    EXPECT_TRUE(queue.readIfNotEmpty(out));
    EXPECT_EQ(out, i);
  }
}

TEST(MPMCQueueTest, FullQueue) {
  MPMCQueue<int> queue(5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(queue.writeIfNotFull(i));
  }
  EXPECT_FALSE(queue.writeIfNotFull(10));
}

TEST(MPMCQueueTest, ConcurrentAccess) {
  MPMCQueue<int> queue(10);
  std::thread writer([&queue]() {
    for (int i = 0; i < 5; ++i) {
      queue.writeIfNotFull(i);
    }
  });
  std::thread reader([&queue]() {
    for (int i = 0; i < 5; ++i) {
      int out = 0;
      while (!queue.readIfNotEmpty(out)) {
        // Wait until an element is available
        // TODO We could provide a blocking version of read() instead of
        // looping here. We only provide a non blocking wait API because
        // for now the queue is paired with a semaphore in executor.
        std::this_thread::yield();
      }
      EXPECT_LT(out, 5);
    }
  });
  writer.join();
  reader.join();
}

TEST(MPMCQueueTest, MPMCConcurrentAccess) {
  const size_t queueCapacity = 100000;
  const size_t numWriters = 5;
  const size_t numReaders = 5;
  const size_t numElementsPerWriter = 10000;
  MPMCQueue<int> queue(queueCapacity);
  // Writer threads
  std::vector<std::thread> writers;
  writers.reserve(numWriters);
  for (size_t i = 0; i < numWriters; ++i) {
    writers.emplace_back([&]() {
      for (size_t j = 0; j < numElementsPerWriter; ++j) {
        size_t value = i * numElementsPerWriter + j;
        while (!queue.writeIfNotFull(static_cast<int>(value))) {
          // Retry until the queue has space
          // TODO We could provide a blocking version of read() instead of
          // looping here. We only provide a non blocking wait API because
          // for now the queue is paired with a semaphore in executor.
          std::this_thread::yield();
        }
      }
    });
  }
  // Reader threads
  std::vector<std::thread> readers;
  std::atomic<size_t> totalReadCount{0};
  readers.reserve(numReaders);
  for (size_t i = 0; i < numReaders; ++i) {
    readers.emplace_back([&]() {
      int value = 0;
      while (totalReadCount < numWriters * numElementsPerWriter) {
        if (queue.readIfNotEmpty(value)) {
          ++totalReadCount;
        } else {
          // TODO We could provide a blocking version of read() instead of
          // looping here. We only provide a non blocking wait API because
          // for now the queue is paired with a semaphore in executor.
          std::this_thread::yield();
        }
      }
    });
  }
  // Join all threads
  for (auto& writer : writers) {
    writer.join();
  }
  for (auto& reader : readers) {
    reader.join();
  }
  // Verify that all elements were read
  EXPECT_EQ(totalReadCount, numWriters * numElementsPerWriter);
}

TEST(MPMCQueueTest, MoveOnlyType) {
  struct MoveOnly {
    MoveOnly() = default;
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;
    MoveOnly(MoveOnly&&) = default;
    MoveOnly& operator=(MoveOnly&&) = default;
    ~MoveOnly() = default;
  };
  MPMCQueue<MoveOnly> queue(5);
  EXPECT_TRUE(queue.writeIfNotFull(MoveOnly()));
  MoveOnly out;
  EXPECT_TRUE(queue.readIfNotEmpty(out));
}
