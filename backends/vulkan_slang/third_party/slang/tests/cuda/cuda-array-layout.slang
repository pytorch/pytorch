//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer : register(u0);

struct PadLadenStruct
{
    double a;
    uint8_t b;
};

// This is to check if the last half can be inserted 'inside' the spare padding of a. It should not be
struct StructWithArray : IDefaultInitializable
{
    PadLadenStruct a[1];
    uint8_t b;
    
    matrix<half, 3, 3> c;
    uint8_t d;
};

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    StructWithArray s;
    outputBuffer[0] = __sizeOf(s);
    
    outputBuffer[1] = __offsetOf(s, s.a);
    outputBuffer[2] = __offsetOf(s, s.b);
    outputBuffer[3] = __offsetOf(s, s.c);
    outputBuffer[4] = __offsetOf(s, s.d);

    outputBuffer[5] = __sizeOf<int>();
}
