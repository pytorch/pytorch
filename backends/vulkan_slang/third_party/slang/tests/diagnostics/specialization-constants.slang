//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv

struct T { int x; int y; }

// CHECK: ([[# @LINE+1]]): error 31218
[vk::constant_id(1)]
const T st;

[vk::constant_id(1)]
// CHECK: ([[# @LINE+1]]): error 31219
static const int x = 2;

[push_constant]
[vk::constant_id(1)]
// CHECK: ([[# @LINE+1]]): error 31220
const int y;