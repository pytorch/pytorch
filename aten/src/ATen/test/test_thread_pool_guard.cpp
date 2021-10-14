#include <gtest/gtest.h>

#include <caffe2/utils/threadpool/thread_pool_guard.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

TEST(TestThreadPoolGuard, TestThreadPoolGuard) {
  auto threadpool_ptr = caffe2::pthreadpool_();

  ASSERT_NE(threadpool_ptr, nullptr);
  {
    caffe2::_NoPThreadPoolGuard g1;
    auto threadpool_ptr1 = caffe2::pthreadpool_();
    ASSERT_EQ(threadpool_ptr1, nullptr);

    {
      caffe2::_NoPThreadPoolGuard g2;
      auto threadpool_ptr2 = caffe2::pthreadpool_();
      ASSERT_EQ(threadpool_ptr2, nullptr);
    }

    // Guard should restore prev value (nullptr)
    auto threadpool_ptr3 = caffe2::pthreadpool_();
    ASSERT_EQ(threadpool_ptr3, nullptr);
  }

  // Guard should restore prev value (pthreadpool_)
  auto threadpool_ptr4 = caffe2::pthreadpool_();
  ASSERT_NE(threadpool_ptr4, nullptr);
  ASSERT_EQ(threadpool_ptr4, threadpool_ptr);
}

TEST(TestThreadPoolGuard, TestRunWithGuard) {
  const std::vector<int64_t> array = {1, 2, 3};

  auto pool = caffe2::pthreadpool();
  int64_t inner = 0;
  {
    // Run on same thread
    caffe2::_NoPThreadPoolGuard g1;
    auto fn = [&array, &inner](const size_t task_id) {
      inner += array[task_id];
    };
    pool->run(fn, 3);

    // confirm the guard is on
    auto threadpool_ptr = caffe2::pthreadpool_();
    ASSERT_EQ(threadpool_ptr, nullptr);
  }
  ASSERT_EQ(inner, 6);
}
