#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/Parallel.h>
#include <ATen/ParallelFuture.h>

#include <iostream>
// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <string.h>
#include <sstream>
#if AT_MKL_ENABLED()
#include <mkl.h>
#include <thread>
#endif

struct NumThreadsGuard {
  int old_num_threads_;
  NumThreadsGuard(int nthreads) {
    old_num_threads_ = at::get_num_threads();
    at::set_num_threads(nthreads);
  }

  ~NumThreadsGuard() {
    at::set_num_threads(old_num_threads_);
  }
};

using namespace at;

TEST(TestParallel, TestParallel) {
  manual_seed(123);
  NumThreadsGuard guard(1);

  Tensor a = rand({1, 3});
  a[0][0] = 1;
  a[0][1] = 0;
  a[0][2] = 0;
  Tensor as = rand({3});
  as[0] = 1;
  as[1] = 0;
  as[2] = 0;
  ASSERT_TRUE(a.sum(0).equal(as));
}

TEST(TestParallel, NestedParallel) {
  Tensor a = ones({1024, 1024});
  auto expected = a.sum();
  // check that calling sum() from within a parallel block computes the same result
  at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
    if (begin == 0) {
      ASSERT_TRUE(a.sum().equal(expected));
    }
  });
}

#ifdef TH_BLAS_MKL
TEST(TestParallel, LocalMKLThreadNumber) {
  auto master_thread_num = mkl_get_max_threads();
  auto f = [](int nthreads){
    set_num_threads(nthreads);
  };
  std::thread t(f, 1);
  t.join();
  ASSERT_EQ(master_thread_num, mkl_get_max_threads());
}
#endif

TEST(TestParallel, NestedParallelThreadId) {
  // check that thread id within a nested parallel block is accurate
  at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
    at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
      // Nested parallel regions execute on a single thread
      ASSERT_EQ(begin, 0);
      ASSERT_EQ(end, 10);

      // Thread id reflects inner parallel region
      ASSERT_EQ(at::get_thread_num(), 0);
    });
  });

  at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
    auto num_threads =
      at::parallel_reduce(0, 10, 1, 0, [&](int64_t begin, int64_t end, int ident) {
        // Thread id + 1 should always be 1
        return at::get_thread_num() + 1;
      }, std::plus<>{});
    ASSERT_EQ(num_threads, 1);
  });
}

TEST(TestParallel, Exceptions) {
  // parallel case
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(
    at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
      throw std::runtime_error("exception");
    }),
    std::runtime_error);

  // non-parallel case
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(
    at::parallel_for(0, 1, 1000, [&](int64_t begin, int64_t end) {
      throw std::runtime_error("exception");
    }),
    std::runtime_error);
}

TEST(TestParallel, IntraOpLaunchFuture) {
  int v1 = 0;
  int v2 = 0;

  auto fut1 = at::intraop_launch_future([&v1](){
    v1 = 1;
  });

  auto fut2 = at::intraop_launch_future([&v2](){
    v2 = 2;
  });

  fut1->wait();
  fut2->wait();

  ASSERT_TRUE(v1 == 1 && v2 == 2);
}
