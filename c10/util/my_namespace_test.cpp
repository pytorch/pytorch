#include <c10/util/my_namespace.h>
#include <iostream>

int main() {
  // Create a Half in the c10 namespace
  c10::Half c10_half(1.5f);
  
  // Create a Half in the my_namespace namespace
  my_namespace::Half my_half(1.5f);
  
  // Use the operator<< from c10 namespace
  std::cout << "c10::Half: " << c10_half << std::endl;
  
  // Use the operator<< from my_namespace
  std::cout << "my_namespace::Half: " << my_half << std::endl;
  
  return 0;
}