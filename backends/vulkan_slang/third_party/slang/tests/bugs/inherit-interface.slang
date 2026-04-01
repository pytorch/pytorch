//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IA
{
    int doThing(int a);
};

interface IB : IA
{
    int anotherThing(int a);
};

struct Struct : IB
{
    int doThing(int a) { return a + 1; }
    int anotherThing(int a) { return a * a; }
};

int doThing<B : IB>(B b, int c)
{
    return b.doThing(c) + b.anotherThing(c);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

    Struct s;
    int v = doThing(s, int(tid));
    
	float inVal = float(tid + v);

	outputBuffer[tid] = inVal;
}