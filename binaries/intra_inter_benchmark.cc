#include "ATen/Parallel.h"

#include "torch/torch.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <ctime>

namespace {
constexpr int kDefaultIterations = 100;
constexpr int kDefaultTensorDim = 2000;
int iter = kDefaultIterations;
int inter_op_threads = at::get_num_interop_threads();
int intra_op_threads = at::get_num_threads();
int tensor_dim = kDefaultTensorDim;
std::atomic<int> counter{0};
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

void check_param(char* param_name, char* param_value) {
  int value = std::atoi(param_value);
  TORCH_CHECK(value > 0);
  if (std::string(param_name) == "--iter") {
    iter = value;
  } else if (std::string(param_name) == "--inter_op_threads") {
    inter_op_threads = value;
  } else if (std::string(param_name) == "--intra_op_threads") {
    intra_op_threads = value;
  } else if (std::string(param_name) == "--tensor_dim") {
    tensor_dim = value;
  } else {
    AT_ERROR("invalid argument, expected: --iter or --threads");
  }
}

int main(int argc, char** argv) {
  at::init_num_threads();

  if (argc > 1) {
    TORCH_CHECK(argc % 2 == 1);
    for (int idx = 1; idx < argc; idx+=2) {
      check_param(argv[idx], argv[idx + 1]);
    }
  }

  at::set_num_interop_threads(inter_op_threads);
  at::set_num_threads(intra_op_threads);

  std::cout << "Launching " << iter << " tasks using "
            << at::get_num_interop_threads() << " inter-op threads and "
            << at::get_num_threads() << " intra-op threads, "
            << "tensor dim: " << tensor_dim << std::endl;

  auto left = torch::randn(
      {tensor_dim, tensor_dim}, torch::dtype(torch::kFloat32));
  auto right = torch::randn(
      {tensor_dim, tensor_dim}, torch::dtype(torch::kFloat32));

  // warmup
  auto saved_iter = iter;
  iter = 10;
  for (auto idx = 0; idx < iter; ++idx) {
    launch_task(left, right);
  }
  wait();
  iter = saved_iter;
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
