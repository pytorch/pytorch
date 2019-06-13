#include <ATen/ATen.h>
#include "ATen/Parallel.h"

#include "c10/util/Flags.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <ctime>

C10_DEFINE_int(iter, 100, "Number of iterations (tasks)");
C10_DEFINE_int(sub_iter, 100, "Number of subtasks")
C10_DEFINE_int(warmup_iter, 10, "Number of warmup iterations")
C10_DEFINE_int(inter_op_threads, 0, "Number of inter-op iterations");
C10_DEFINE_int(intra_op_threads, 0, "Number of intra-op threads");
C10_DEFINE_int(tensor_dim, 2000, "Tensor dim");
C10_DEFINE_int(benchmark_iter, 3, "Number of times to run benchmark")

namespace {
std::atomic<int> counter{0};
int overall_tasks = 0;
std::condition_variable cv;
std::mutex mutex;
}

void launch_task(at::Tensor& left, at::Tensor& right) {
  at::launch([&left, &right]() {
    at::parallel_for(0, FLAGS_sub_iter, 1,
        [&left, &right](int64_t begin, int64_t end) {
      for (auto k = begin; k < end; ++k) {
        auto result = left.add(right);
        auto cur_ctr = ++counter;
        if (cur_ctr == overall_tasks) {
          std::unique_lock<std::mutex> lk(mutex);
          cv.notify_one();
        }
      }
    });
  });
}

void wait() {
  std::unique_lock<std::mutex> lk(mutex);
  while (counter < overall_tasks) {
    cv.wait(lk);
  }
}

void launch_tasks_and_wait(at::Tensor& left, at::Tensor& right, int tasks_num) {
  overall_tasks = tasks_num * FLAGS_sub_iter;
  counter = 0;
  for (auto idx = 0; idx < tasks_num; ++idx) {
    launch_task(left, right);
  }
  wait();
}

int main(int argc, char** argv) {
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cout << "Failed to parse command line flags" << std::endl;
    return -1;
  }
  at::init_num_threads();

  if (FLAGS_inter_op_threads > 0) {
    at::set_num_interop_threads(FLAGS_inter_op_threads);
  }
  if (FLAGS_intra_op_threads > 0) {
    at::set_num_threads(FLAGS_intra_op_threads);
  }

  auto left = at::ones({FLAGS_tensor_dim, FLAGS_tensor_dim}, at::kFloat);
  auto right = at::ones({FLAGS_tensor_dim, FLAGS_tensor_dim}, at::kFloat);

  std::cout << "Launching " << FLAGS_warmup_iter << " warmup tasks" << std::endl;

  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::milliseconds ms;

  std::chrono::time_point<clock> start_time = clock::now();
  launch_tasks_and_wait(left, right, FLAGS_warmup_iter);
  auto duration = static_cast<float>(
      std::chrono::duration_cast<ms>(clock::now() - start_time).count());

  std::cout << "Warmup time: " << duration << " ms." << std::endl;

  std::cout << "Launching " << FLAGS_iter << " tasks with "
            << FLAGS_sub_iter << " subtasks each, using "
            << at::get_num_interop_threads() << " inter-op threads and "
            << at::get_num_threads() << " intra-op threads, "
            << "tensor dim: " << FLAGS_tensor_dim << std::endl;

  for (auto bench_iter = 0; bench_iter < FLAGS_benchmark_iter; ++bench_iter) {
    start_time = clock::now();
    launch_tasks_and_wait(left, right, FLAGS_iter);
    duration = static_cast<float>(
        std::chrono::duration_cast<ms>(clock::now() - start_time).count());

    std::cout << "Time to run " << FLAGS_iter << " iterations "
              << (duration/1000.0) << " s." << std::endl;
  }

  return 0;
}
