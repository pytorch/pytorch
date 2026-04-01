//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type -xslang -fvk-use-gl-layout

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

//TEST_INPUT:set inBuffer = ubuffer(data=[1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0], stride=4)
RWByteAddressBuffer inBuffer;

struct MyStruct
{
    float v0;
    float2 v1;
    float2 v2;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var v = inBuffer.Load<MyStruct>(0);
    var g = inBuffer.Load<float4>(4);
    outputBuffer[dispatchThreadID.x] = v.v2.x + v.v1.x + g.x; // expect 10.0
}
