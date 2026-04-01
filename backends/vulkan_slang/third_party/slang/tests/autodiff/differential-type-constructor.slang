//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

struct MyStruct: IDifferentiable 
{
    float a;
    int b;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        var dp = diffPair(MyStruct(0.0, 0), MyStruct.Differential(0.0));

        outputBuffer[0] = dp.d.a;

        var dp2 = diffPair<MyStruct>(MyStruct(0.0, 0), MyStruct.Differential(1.f));

        outputBuffer[1] = dp2.d.a;
    }
} 
