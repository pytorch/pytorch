#include "ATen/Parallel.h"

#include "c10/util/Flags.h"
#include "torch/torch.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <ctime>

C10_DEFINE_int(iter, 100, "Number of iterations");
C10_DEFINE_int(inter_op_threads, 0, "Number of inter-op iterations");
C10_DEFINE_int(intra_op_threads, 0, "Number of intra-op threads");
C10_DEFINE_int(tensor_dim, 2000, "Tensor dim");

namespace {
std::atomic<int> counter{0};
int iter = 0;
std::condition_variable cv;
std::mutex mutex;
}

void launch_task(at::Tensor& left, at::Tensor& right) {
  at::launch([&left, &right]() {
    auto result = left.sigmoid() + right;
    auto cur_ctr = ++counter;
    if (cur_ctr == iter) {
      std::unique_lock<std::mutex> lk(mutex);
      cv.notify_one();
    }
  });
}

void wait() {
  std::unique_lock<std::mutex> lk(mutex);
  while (counter < iter) {
    cv.wait(lk);
  }
}

int main(int argc, char** argv) {
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cout << "Failed to parse command line flags";
    return -1;
  }
  at::init_num_threads();

  if (FLAGS_inter_op_threads > 0) {
    at::set_num_interop_threads(FLAGS_inter_op_threads);
  }
  if (FLAGS_intra_op_threads > 0) {
    at::set_num_threads(FLAGS_intra_op_threads);
  }

  std::cout << "Launching " << FLAGS_iter << " tasks using "
            << at::get_num_interop_threads() << " inter-op threads and "
            << at::get_num_threads() << " intra-op threads, "
            << "tensor dim: " << FLAGS_tensor_dim << std::endl;

  auto left = torch::randn(
      {FLAGS_tensor_dim, FLAGS_tensor_dim}, torch::dtype(torch::kFloat32));
  auto right = torch::randn(
      {FLAGS_tensor_dim, FLAGS_tensor_dim}, torch::dtype(torch::kFloat32));

  // warmup
  iter = 10;
  for (auto idx = 0; idx < iter; ++idx) {
    launch_task(left, right);
  }
  wait();

  iter = FLAGS_iter;
  counter = 0;

  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::milliseconds ms;
  std::chrono::time_point<clock> start_time = clock::now();
  for (auto idx = 0; idx < iter; ++idx) {
    launch_task(left, right);
  }
  wait();

  auto duration = static_cast<float>(
      std::chrono::duration_cast<ms>(clock::now() - start_time).count());
  std::cout << "Time to run " << iter << " iterations "
            << (duration/1000.0) << " s." << std::endl;

  return 0;
}
