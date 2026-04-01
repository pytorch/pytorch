//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv

struct Type
{
    // CHECK: ([[# @LINE+1]]): error 30102:
    namespace ns {}
}