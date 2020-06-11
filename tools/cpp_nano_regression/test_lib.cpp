#include <atomic>

std::atomic<int> x{0};

void foo() {
  x++;
  x--;
}
