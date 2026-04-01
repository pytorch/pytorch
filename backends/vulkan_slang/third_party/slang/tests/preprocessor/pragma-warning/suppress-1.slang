//TEST:SIMPLE(filecheck=CHECK):
// #ifdef support

int64_t GetValue();

void f()
{
	int i;
	// CHECK-NOT: ([[# @LINE+2]]): warning 30081:
#pragma warning (suppress : 30081)
	i = GetValue();
}

// Simple test of #pragma warning (suppress)