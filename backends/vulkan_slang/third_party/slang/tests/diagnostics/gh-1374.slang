// gh-1374.slang

// Test a `static` variable wwith a definition that refers to itself

//DIAGNOSTIC_TEST:REFLECTION:-stage compute -entry main -target hlsl

struct S
{
	static const int kVal = kVal;

	static const int kInf = kInf + 1;

	static const int kA = kB;
	static const int kB = kA;
}

[numthreads(1, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
}