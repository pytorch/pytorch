// Simple test to show CHECK macro collision issue #159444
#include <iostream>

// First: Define CHECK like Catch2 would
#define CHECK(condition) \
    std::cout << "Catch2 CHECK: " #condition << std::endl

// Show what our CHECK does
void test_external_check() {
    std::cout << "=== Testing external library CHECK ===" << std::endl;
    CHECK(1 == 1);
    CHECK(true);
}

// Now let's see what happens when we include PyTorch's definition
// (We'll simulate it without including the full header)
#undef CHECK
#define CHECK(cond) \
    if (!(cond)) { \
        std::cout << "PyTorch CHECK FAILED: " #cond << std::endl; \
        abort(); \
    } else { \
        std::cout << "PyTorch CHECK passed: " #cond << std::endl; \
    }

void test_pytorch_check() {
    std::cout << "\n=== Testing PyTorch CHECK ===" << std::endl;
    CHECK(1 == 1);
    CHECK(true);
}

int main() {
    std::cout << "Demonstrating CHECK macro collision issue #159444\n" << std::endl;
    
    test_external_check();
    test_pytorch_check();
    
    std::cout << "\nProblem: PyTorch's CHECK overwrites external library's CHECK!" << std::endl;
    return 0;
}
