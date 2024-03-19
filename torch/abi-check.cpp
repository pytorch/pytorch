#include <iostream>

int main() {
#ifdef _GLIBCXX_USE_CXX11_ABI
  std::cout << _GLIBCXX_USE_CXX11_ABI;
#else
  std::cout << 0;
#endif
}
