//DIAGNOSTIC_TEST:SIMPLE:

// "Obviously" unreachable code.

int foo()
{
	return 1;

	// The following statements can never be reached.
	int x = 0;
	return x;
}
