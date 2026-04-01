//TEST:SIMPLE(filecheck=CHECK):
// #ifdef support

// CHECK-NOT: ([[# @LINE+1]]): warning 30081:
#pragma warning (pop)

// Test that #pragma warning (pop) emits a warning when the stack is empty