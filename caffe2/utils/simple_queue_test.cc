#include <thread>  // NOLINT

#include "caffe2/utils/simple_queue.h"
#include <gtest/gtest.h>

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::unique_ptr<SimpleQueue<int> > gQueue;

static void ConsumerFunction(int thread_idx) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int value;
  while (true) {
    if (!gQueue->Pop(&value)) return;
    VLOG(1) << "Emitting " << value << " from thread " << thread_idx;
  }
}

static void ProducerFunction(int thread_idx, int start, int count) {
  for (int i = 0; i < count; ++i) {
    VLOG(1) << "Pushing " << i + start << " from thread " << thread_idx;
    gQueue->Push(i + start);
  }
}


// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SimpleQueueTest, SingleProducerSingleConsumer) {
  // NOLINTNEXTLINE(modernize-make-unique)
  gQueue.reset(new SimpleQueue<int>());
  std::thread consumer(ConsumerFunction, 0);
  for (int i = 0; i < 10; ++i) {
    gQueue->Push(i);
  }
  gQueue->NoMoreJobs();
  consumer.join();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SimpleQueueTest, SingleProducerDoubleConsumer) {
  // NOLINTNEXTLINE(modernize-make-unique)
  gQueue.reset(new SimpleQueue<int>());
  std::thread consumer0(ConsumerFunction, 0);
  std::thread consumer1(ConsumerFunction, 1);
  for (int i = 0; i < 10; ++i) {
    gQueue->Push(i);
  }
  gQueue->NoMoreJobs();
  consumer0.join();
  consumer1.join();
}


// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SimpleQueueTest, DoubleProducerDoubleConsumer) {
  // NOLINTNEXTLINE(modernize-make-unique)
  gQueue.reset(new SimpleQueue<int>());
  std::thread producer0(ProducerFunction, 0, 0, 10);
  std::thread producer1(ProducerFunction, 0, 10, 10);
  std::thread consumer0(ConsumerFunction, 2);
  std::thread consumer1(ConsumerFunction, 3);
  producer0.join();
  producer1.join();
  gQueue->NoMoreJobs();
  consumer0.join();
  consumer1.join();
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SimpleQueueDeathTest, CannotAddAfterQueueFinished) {
  // NOLINTNEXTLINE(modernize-make-unique)
  gQueue.reset(new SimpleQueue<int>());
  gQueue->Push(0);
  gQueue->NoMoreJobs();
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(gQueue->Push(0), EnforceNotMet);
}


}  // namespace caffe2
