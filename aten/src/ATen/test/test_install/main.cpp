#include "ATen/ATen.h"

int main() {
  std::cout << at::ones(at::CPU(at::kFloat), {3,4}) << "\n";
}
