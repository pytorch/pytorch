// array-zero-size.slang

// Test that array size cannot be zero

//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    bar();
}
//CHECK: ([[# @LINE+1]]): error 30025
func bar() -> int[0];
