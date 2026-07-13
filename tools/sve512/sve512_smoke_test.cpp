#include <ATen/Version.h>
#include <ATen/ATen.h>
#include <iostream>

int main() {
  std::cout << "cpu_capability=" << at::get_cpu_capability() << '\n';
  auto a = at::ones({512}, at::TensorOptions().dtype(at::kFloat));
  auto b = at::ones({512}, at::TensorOptions().dtype(at::kFloat));
  auto c = a + b;
  float sum = c.sum().item<float>();
  std::cout << "add_sum=" << sum << '\n';
  return (sum == 1024.0f) ? 0 : 1;
}
