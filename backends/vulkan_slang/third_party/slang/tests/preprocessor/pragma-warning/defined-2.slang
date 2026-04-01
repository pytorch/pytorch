//TEST:SIMPLE(filecheck=CHECK):
// #ifdef support

int64_t GetValue();
#define SPECIFIER disable
#define IDs 30081

void f()
{
	int i;
#pragma warning (error : 30081)
	// CHECK: ([[# @LINE+1]]): error 30081:
	i = GetValue();
#pragma warning (SPECIFIER : IDs)
}

// Test that #pragma warning with #defined specifiers and IDs uses the correct SourceLoc
// (Use the macro invocation's SourceLoc, not the macro definition's SourceLoc)