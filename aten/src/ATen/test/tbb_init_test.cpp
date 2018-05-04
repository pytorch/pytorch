#include "ATen/ATen.h"
#include "ATen/Parallel.h"
#include "test_assert.h"
#include "test_seed.h"
#include <thread>

using namespace at;

// This checks whether threads can see the global
// numbers of threads set and also whether the scheduler
// will throw an exception when multiple threads call 
// their first parallel construct.
void test(int given_num_threads) {
  auto t = ones(CPU(kFloat), {1000 * 1000});
  if (given_num_threads >= 0) {
    ASSERT(at::get_num_threads() == given_num_threads);
  } else {
    ASSERT(at::get_num_threads() == -1);
  }
  auto t_sum = t.sum();
  for (int i = 0; i < 1000; i ++) {
    t_sum = t_sum + t.sum();
  }
}

int main() {
  manual_seed(123, at::Backend::CPU);

  test(-1);
  std::thread t1(test, -1);
  t1.join();
  at::set_num_threads(4);
  std::thread t2(test, 4);
  std::thread t3(test, 4);
  std::thread t4(test, 4);
  t4.join();
  t3.join();
  t2.join();
  at::set_num_threads(5);
  test(5);

  return 0;
}
