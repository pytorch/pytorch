#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <test/cpp/tensorexpr/test_base.h>
#include <thread>


// This checks whether threads can see the global
// numbers of threads set and also whether the scheduler
// will throw an exception when multiple threads call
// their first parallel construct.
void test(int given_num_threads) {
  auto t = at::ones({1000 * 1000}, at::CPU(at::kFloat));
  ASSERT_TRUE(given_num_threads >= 0);
  ASSERT_EQ(at::get_num_threads(), given_num_threads);
  auto t_sum = t.sum();
  for (int i = 0; i < 1000; ++i) {
    t_sum = t_sum + t.sum();
  }
}

int main() {
  at::init_num_threads();

  at::set_num_threads(4);
  test(4);
  std::thread t1([](){
    at::init_num_threads();
    test(4);
  });
  t1.join();

  #if !AT_PARALLEL_NATIVE
  at::set_num_threads(5);
  ASSERT_TRUE(at::get_num_threads() == 5);
  #endif

  // test inter-op settings
  at::set_num_interop_threads(5);
  ASSERT_EQ(at::get_num_interop_threads(), 5);
  ASSERT_ANY_THROW(at::set_num_interop_threads(6));

  return 0;
}
