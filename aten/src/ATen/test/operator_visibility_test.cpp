namespace x {
class A {};
}
void operator+(x::A, x::A) {}

#include <ATen/ATen.h>

namespace at { namespace native {

void f(x::A a) {
  a + a;
}

}}

int main() {}
