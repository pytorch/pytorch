//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

// CHECK-NOT: ([[# @LINE+1]]): error 31201
groupshared int b;

int test()
{
    // CHECK: ([[# @LINE+1]]): error 31201
    groupshared int a;
}
