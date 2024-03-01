#include <gtest/gtest.h>

#include <torch/torch.h>

struct Node {};

// If `torch::autograd::Note` is leaked into the root namespace, the following
// compile error would throw:
// ```
// void NotLeakingSymbolsFromTorchAutogradNamespace_test_func(Node *node) {}
//                                                            ^
// error: reference to `Node` is ambiguous
// ```
void NotLeakingSymbolsFromTorchAutogradNamespace_test_func(Node* node) {}

TEST(NamespaceTests, NotLeakingSymbolsFromTorchAutogradNamespace) {
  // Checks that we are not leaking symbols from the
  // `torch::autograd` namespace to the root namespace
  NotLeakingSymbolsFromTorchAutogradNamespace_test_func(nullptr);
}
