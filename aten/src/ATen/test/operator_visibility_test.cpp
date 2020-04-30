namespace x {
class A {};
}
void operator+(x::A, x::A) {}

#include <ATen/ATen.h>

namespace y{
class B {};
}
void operator+(y::B, y::B) {}

namespace at { namespace native {

void f(x::A a, y::B b) {
  a + a;
  b + b;
}

}}

int main() {}
