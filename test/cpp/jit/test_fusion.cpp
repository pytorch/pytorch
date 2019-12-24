#include <test/cpp/jit/test_base.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`
void testCPUFusion() {
    // ...
}

void testGPUFusion() {
    // ...
}

}} // torch::jit
