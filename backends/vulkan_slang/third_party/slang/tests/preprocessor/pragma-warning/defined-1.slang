//TEST:SIMPLE(filecheck=CHECK):
// #ifdef support

int64_t GetValue();
#define SPECIFIER error
#define IDs 30081 12 23 45

void f()
{
	int i;
#pragma warning (SPECIFIER : IDs)
	// CHECK: ([[# @LINE+1]]): error 30081:
	i = GetValue();
}

// Test that #pragma warning works with #defined specifiers and IDs