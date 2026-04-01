//TEST:SIMPLE(filecheck=CHECK):
// #ifdef support

// CHECK: ([[# @LINE+1]]): warning 15613:
#pragma warning (diasble : 30081)

// Test that #pragma warning emits a warning if the specifier ins't correct