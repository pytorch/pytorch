#include <ATen/ATen.h>

class A {};
void operator+(A, A) {}

namespace at { namespace native {

void f(A a) {
  a + a;
}

}}

int main() {}
