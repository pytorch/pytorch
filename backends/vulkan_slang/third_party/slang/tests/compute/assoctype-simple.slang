//TEST(smoke,compute):COMPARE_COMPUTE:-cpu -shaderobj
//TEST(smoke,compute):COMPARE_COMPUTE: -shaderobj

// Confirm that generics syntax can be used in user
// code and generates valid output.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface ISimple
{
    associatedtype U;
    U addt(U v0, U v1);
}

struct Simple : ISimple
{
    typedef float U;
    U addt(U v0, float v1)
    {
        return v0 + v1;
    }
};

__generic<T:ISimple>
T.U test(T simple, T.U v0, T.U v1)
{
    return simple.addt(v0, v1);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Simple s;
	float outVal = test<Simple>(s, 2.0, 1.0); // == 3.0
	outputBuffer[dispatchThreadID.x] = outVal;
}