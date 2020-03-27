#include <c10/test/util/complex_test_common.h>

TEST(NonStaticTests, all) {
  run_all_host_tests();
}

// main
int main() {
  NonStaticTests_all();
}
