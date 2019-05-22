#include "ATen/Parallel.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <ctime>

namespace {
constexpr int kDefaultIterations = 10e6;
int iter = kDefaultIterations;
int threads = at::get_num_interop_threads();
std::atomic<int> counter{0};
std::condition_variable cv;
std::mutex mutex;
}

void launch_tasks() {
  at::launch([]() {
    at::launch([](){
      at::launch([]() {
        auto cur_ctr = ++counter;
        if (cur_ctr == iter) {
          std::unique_lock<std::mutex> lk(mutex);
          cv.notify_one();
        }
      });
    });
  });
}

void check_param(char* param_name, char* param_value) {
  int value = std::atoi(param_value);
  TORCH_CHECK(value > 0);
  if (std::string(param_name) == "--iter") {
    iter = value;
  } else if (std::string(param_name) == "--threads") {
    threads = value;
  } else {
    AT_ERROR("invalid argument, expected: --iter or --threads");
  }
}

int main(int argc, char** argv) {
  at::init_num_threads();

  if (argc > 1) {
    TORCH_CHECK(argc == 3 || argc == 5);
    check_param(argv[1], argv[2]);
    if (argc > 3) {
      check_param(argv[3], argv[4]);
    }
  } else {
    TORCH_CHECK(argc == 1);
  }

  at::set_num_interop_threads(threads);

  std::cout << "Launching " << iter << " tasks using "
            << at::get_num_interop_threads() << " threads "
            << std::endl;

  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::milliseconds ms;
  std::chrono::time_point<clock> start_time = clock::now();
  for (auto idx = 0; idx < iter; ++idx) {
    launch_tasks();
  }
  std::unique_lock<std::mutex> lk(mutex);
  while (counter < iter) {
    cv.wait(lk);
  }
  auto duration = static_cast<float>(
      std::chrono::duration_cast<ms>(clock::now() - start_time).count());
  std::cout << "Time to run " << iter << " iterations "
            << (duration/1000.0) << " s." << std::endl;

  return 0;
}
