//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

// Check that we can issue diagnostic on invalid left hand side of a type constraint.

// CHECK: ([[# @LINE+1]]): error 30402
void f<T>() where int : IInteger
{}