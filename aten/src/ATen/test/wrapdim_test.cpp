#include "ATen/ATen.h"
#include "test_assert.h"

using namespace at;

int main() {
  Type & T = CPU(kFloat);

  // test simple case
  {
    auto a = T.randn({2, 3, 4, 5});
    ASSERT(a.prod(-4).equal(a.prod(0)));
    ASSERT(a.prod(3).equal(a.prod(-1)));
  }

  // test case with expression specification
  {
    auto a = T.randn({2, 3, 4, 5});
    ASSERT(a.unsqueeze(-5).equal(a.unsqueeze(0)));
    ASSERT(a.unsqueeze(4).equal(a.unsqueeze(-1)));

    // can unsqueeze scalar
    auto b = T.randn(1);
    b.get()->maybeScalar(true);
    ASSERT(b.unsqueeze(0).equal(b.unsqueeze(-1)));
  }

  // test case with empty tensor
  {
    auto a = T.randn(0);
    try {
      a.prod(0);
      ASSERT(false);
    } catch (std::runtime_error &e) {}
  }

  // test case with scalar vs 1-dim, 1-size
  {
    auto a = T.randn(1);
    ASSERT(a.prod(0).equal(a.prod(-1)));
    a.get()->maybeScalar(true);
    ASSERT(a.get()->isScalar());
    try {
      a.prod(0);
      ASSERT(false);
    } catch (std::runtime_error &e) {}
  }
}
