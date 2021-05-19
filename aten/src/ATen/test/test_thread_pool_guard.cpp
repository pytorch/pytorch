#include <gtest/gtest.h>

#include <caffe2/utils/threadpool/thread_pool_guard.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>


// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
