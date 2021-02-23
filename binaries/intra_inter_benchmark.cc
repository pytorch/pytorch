#include "ATen/ATen.h"
#include "ATen/Parallel.h"

#include "c10/util/Flags.h"
#include "caffe2/core/init.h"

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <iostream>
#include <mutex>
#include <thread>

C10_DEFINE_int(iter_pow, 10, "Number of tasks, 2^N");
C10_DEFINE_int(sub_iter, 1024, "Number of subtasks");
C10_DEFINE_int(warmup_iter_pow, 3, "Number of warmup tasks, 2^N");
C10_DEFINE_int(inter_op_threads, 0, "Number of inter-op threads");
C10_DEFINE_int(intra_op_threads, 0, "Number of intra-op threads");
C10_DEFINE_int(tensor_dim, 50, "Tensor dim");
C10_DEFINE_int(benchmark_iter, 10, "Number of times to run benchmark")
C10_DEFINE_bool(extra_stats, false,
    "Collect extra stats; warning: skews results");
C10_DEFINE_string(task_type, "add", "Tensor operation: add or mm");

namespace {
std::atomic<int> counter{0};
int overall_tasks = 0;
std::condition_variable cv;
std::mutex tasks_mutex;
bool run_mm = false;

std::mutex stats_mutex;
std::unordered_set<std::thread::id> tids;
}

void wait() {
  std::unique_lock<std::mutex> lk(tasks_mutex);
  while (counter < overall_tasks) {
    cv.wait(lk);
  }
}

void _launch_tasks_tree(
    int level, int end_level, at::Tensor& left, at::Tensor& right) {
  if (level == end_level) {
    at::parallel_for(0, FLAGS_sub_iter, 1,
        [&left, &right](int64_t begin, int64_t end) {
      if (FLAGS_extra_stats) {
        std::unique_lock<std::mutex> lk(stats_mutex);
        tids.insert(std::this_thread::get_id());
      }
      for (auto k = begin; k < end; ++k) {
        if (run_mm) {
          left.mm(right);
        } else {
          left.add(right);
        }
        auto cur_ctr = ++counter;
        if (cur_ctr == overall_tasks) {
          std::unique_lock<std::mutex> lk(tasks_mutex);
          cv.notify_one();
        }
      }
    });
  } else {
    at::launch([&left, &right, level, end_level]() {
      _launch_tasks_tree(level + 1, end_level, left, right);
    });
    at::launch([&left, &right, level, end_level]() {
      _launch_tasks_tree(level + 1, end_level, left, right);
    });
  }
};

void launch_tasks_and_wait(at::Tensor& left, at::Tensor& right, int iter_pow) {
  overall_tasks = pow(2, iter_pow) * FLAGS_sub_iter;
  counter = 0;

  _launch_tasks_tree(0, iter_pow, left, right);
  wait();
}

void reset_extra_stats() {
  tids.clear();
}

void print_extra_stats() {
  std::cout << "# threads: " << tids.size() << std::endl;
}

void print_runtime_stats(const std::vector<float>& runtimes) {
  TORCH_INTERNAL_ASSERT(!runtimes.empty());
  float sum = 0.0;
  float sqr_sum = 0.0;
  size_t N = runtimes.size();
  for (size_t idx = 0; idx < N; ++idx) {
    sum += runtimes[idx];
    sqr_sum += runtimes[idx] * runtimes[idx];
  }
  float mean = sum / N;
  float sd = std::sqrt(sqr_sum / N - mean * mean);
  std::cout << "N = " << N << ", mean = " << mean << ", sd = " << sd
            << std::endl;
}

int main(int argc, char** argv) {
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cout << "Failed to parse command line flags" << std::endl;
    return -1;
  }
  caffe2::unsafeRunCaffe2InitFunction("registerThreadPools");
  at::init_num_threads();

  if (FLAGS_inter_op_threads > 0) {
    at::set_num_interop_threads(FLAGS_inter_op_threads);
  }
  if (FLAGS_intra_op_threads > 0) {
    at::set_num_threads(FLAGS_intra_op_threads);
  }

  TORCH_CHECK(FLAGS_task_type == "add" || FLAGS_task_type == "mm");
  run_mm = FLAGS_task_type == "mm";

  auto left = at::ones({FLAGS_tensor_dim, FLAGS_tensor_dim}, at::kFloat);
  auto right = at::ones({FLAGS_tensor_dim, FLAGS_tensor_dim}, at::kFloat);

  std::cout << "Launching " << pow(2, FLAGS_warmup_iter_pow)
            << " warmup tasks" << std::endl;

  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::milliseconds ms;

  std::chrono::time_point<clock> start_time = clock::now();
  launch_tasks_and_wait(left, right, FLAGS_warmup_iter_pow);
  auto duration = static_cast<float>(
      std::chrono::duration_cast<ms>(clock::now() - start_time).count());

  std::cout << "Warmup time: " << duration << " ms." << std::endl;

  std::cout << "Launching " << pow(2, FLAGS_iter_pow) << " tasks with "
            << FLAGS_sub_iter << " subtasks each, using "
            << at::get_num_interop_threads() << " inter-op threads and "
            << at::get_num_threads() << " intra-op threads, "
            << "tensor dim: " << FLAGS_tensor_dim
            << ", task type: " << FLAGS_task_type << std::endl;

  std::vector<float> runtimes;
  for (auto bench_iter = 0; bench_iter < FLAGS_benchmark_iter; ++bench_iter) {
    reset_extra_stats();
    start_time = clock::now();
    launch_tasks_and_wait(left, right, FLAGS_iter_pow);
    duration = static_cast<float>(
        std::chrono::duration_cast<ms>(clock::now() - start_time).count());
    runtimes.push_back(duration);

    if (FLAGS_extra_stats) {
      print_extra_stats();
    }

    std::cout << "Runtime: " << duration << " ms." << std::endl;
  }

  print_runtime_stats(runtimes);

  return 0;
}
