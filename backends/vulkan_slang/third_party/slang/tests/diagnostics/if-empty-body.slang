//DIAGNOSTIC_TEST(windows):SIMPLE:

// Test that use of empty statement after an `if` leads to an warning.

struct S {}

void test()
{
	if (1+1==2);
		return;
}