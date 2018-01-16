#include "ATen/ATen.h"

int main() {
  std::cout << at::CPU(at::kFloat).ones({3,4}) << "\n";
}
