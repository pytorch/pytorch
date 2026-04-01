// void-function-returning-value.slang

// Confirm that we diagnose a `void` function that returns a value.

//DIAGNOSTIC_TEST:SIMPLE:-stage compute -entry main -target dxbc

void good()
{
	// Explicit cast to `void` is okay.
	return (void)1;
}

void bad()
{
	// Implicit cast to `void` is not allowed.
	return 1;
}

[numthreads(1, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    good();
    bad();
}