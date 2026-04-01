//TEST:SIMPLE(filecheck=CHECK):
// #ifdef support

int64_t GetValue();

void f()
{
	int i;
#pragma warning (disable : 30081)
	// CHECK-NOT: ([[# @LINE+1]]): warning 30081:
	i = GetValue();
}

// Simple test of #pragma warning (disable)