//TEST:SIMPLE(filecheck=CHECK):
// #ifdef support
#pragma warning (push)
#pragma warning (error : 30081)
int64_t GetValue();
#pragma warning (pop)
void f()
{
	int i;
	// CHECK: ([[# @LINE+1]]): warning 30081:
	i = GetValue();
}

// Test that #pragma warning (push) and (pop) work as expected