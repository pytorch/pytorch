//DIAGNOSTIC_TEST:COMMAND_LINE_SIMPLE:-target hlsl -entry computeMain -stage compute -o no-type-conformance.hlsl
// no type conformance linked

interface IFoo
{
	float get();
}

void foo()
{
	IFoo obj;
	obj.get();
}

[numthreads(1, 1, 1)]
void computeMain(uint3       dispatchThreadID : SV_DispatchThreadID)
{
    foo();
}