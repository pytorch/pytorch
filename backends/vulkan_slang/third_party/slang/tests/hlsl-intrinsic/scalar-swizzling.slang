//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-vk -compute -shaderobj -output-using-type

//TEST_INPUT: ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    bool result = true
        && (0.x is int)
        && 0 == 0.x
        && all(uint2(0) == 0.xx)
        && all(uint3(0) == 0.xxx)
        && all(uint4(0) == 0.xxxx)
        && (0.r is int)
        && 0 == 0.r
        && all(uint2(0) == 0.rr)
        && all(uint3(0) == 0.rrr)
        && all(uint4(0) == 0.rrrr)

        && (123.x is int)
        && 123 == 123.x
        && all(uint2(123) == 123.xx)
        && all(uint3(123) == 123.xxx)
        && all(uint4(123) == 123.xxxx)
        && (123.r is int)
        && 123 == 123.r
        && all(uint2(123) == 123.rr)
        && all(uint3(123) == 123.rrr)
        && all(uint4(123) == 123.rrrr)

        && (0.f.x is float)
        && 0.f == 0.f.x
        && all(float2(0.f) == 0.f.xx)
        && all(float3(0.f) == 0.f.xxx)
        && all(float4(0.f) == 0.f.xxxx)
        && (0.f.r is float)
        && 0.f == 0.f.r
        && all(float2(0.f) == 0.f.rr)
        && all(float3(0.f) == 0.f.rrr)
        && all(float4(0.f) == 0.f.rrrr)

        && (123.f.x is float)
        && 123.f == 123.f.x
        && all(float2(123.f) == 123.f.xx)
        && all(float3(123.f) == 123.f.xxx)
        && all(float4(123.f) == 123.f.xxxx)
        && (123.f.r is float)
        && 123.f == 123.f.r
        && all(float2(123.f) == 123.f.rr)
        && all(float3(123.f) == 123.f.rrr)
        && all(float4(123.f) == 123.f.rrrr)

        && (0..x is float)
        && 0.f == 0..x
        && all(float2(0.f) == 0..xx)
        && all(float3(0.f) == 0..xxx)
        && all(float4(0.f) == 0..xxxx)
        && (0..r is float)
        && 0.f == 0..r
        && all(float2(0.f) == 0..rr)
        && all(float3(0.f) == 0..rrr)
        && all(float4(0.f) == 0..rrrr)

        && (123..x is float)
        && 123.f == 123..x
        && all(float2(123.f) == 123..xx)
        && all(float3(123.f) == 123..xxx)
        && all(float4(123.f) == 123..xxxx)
        && (123..r is float)
        && 123.f == 123..r
        && all(float2(123.f) == 123..rr)
        && all(float3(123.f) == 123..rrr)
        && all(float4(123.f) == 123..rrrr)
        ;

    //CHK:1
    outputBuffer[0] = int(result);
}

