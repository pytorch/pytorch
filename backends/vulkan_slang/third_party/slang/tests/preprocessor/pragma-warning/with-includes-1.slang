//TEST:SIMPLE(filecheck=CHECK):
// #ifdef support
#include "helper-1.slang"

int64_t GetValue();

void f()
{
	int i;
	// CHECK-NOT: ([[# @LINE+1]]): warning 30081:
	i = GetValue();
}

// Test that the #pragma warning (disable) in helper-1.slang disables the warning on line 10