//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* It appears the use of a generic defined within a function appears to trigger a crash


```
void Type::accept(ITypeVisitor* visitor, void* extra)
```

As this is nullptr.
*/

RWStructuredBuffer<float> outputBuffer;

int doThing()
{    
	int getValue<let N : int>() { return N; }
    
    return getValue<10>();
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	float inVal = float(tid);

	outputBuffer[tid] = doThing();
}
