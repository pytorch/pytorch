//TEST:SIMPLE(filecheck=CHECK):
// #ifdef support

// CHECK: ([[# @LINE+1]]): warning 15612:
#pragma warning (push)

// Test that #pragma warning (push) emits a warning when the stack is not empty at the end