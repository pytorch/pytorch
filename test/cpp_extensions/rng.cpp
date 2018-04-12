#include <torch/torch.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include <thread>

void make_random_number() {
  cudaSetDevice(std::rand() % 2);
  auto x = at::CUDA(at::kFloat).randn({1000});
}

void cuda_rng_multithread() {
  auto threads = std::vector<std::thread>();
  for (auto i = 0; i < 1000; i++) {
    threads.emplace_back(make_random_number);
  }
  for (auto& t : threads) {
    t.join();
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_rng_multithread", &cuda_rng_multithread, "");
}
