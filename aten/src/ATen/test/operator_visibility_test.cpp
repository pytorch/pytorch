class A {};
void operator+(A, A) {}

#include <ATen/ATen.h>

class B {};
void operator+(B, B) {}

namespace at { namespace native {

void f(A a, B b) {
  a + a;
  b + b;
}

}}

int main() {}
