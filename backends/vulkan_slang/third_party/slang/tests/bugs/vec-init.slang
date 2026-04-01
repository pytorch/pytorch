//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4,1,1)]
void computeMain(uint2 dispatchID : SV_DispatchThreadID)
{
    float a = dispatchID.x * 0.5f;
    float b = dispatchID.x  + 1.0f;
    
    float2 c2_0 = float2( a, b);
    
    float3 c3_0 = float3( a, b, a);
    float3 c3_1 = float3( float2(a, b), a);
    float3 c3_2 = float3( b, float2(a, b));
    
    float4 c4_0 = float4(a, b, a, b);
    float4 c4_1 = float4(float2(a, b), a, b);
    float4 c4_2 = float4(a, float2(b, a), b); 
    float4 c4_3 = float4(a, b, float2(a, b));
    float4 c4_4 = float4(float2(a, b), float2(a, b));
    float4 c4_5 = float4(float3(a, b, a), a);
    float4 c4_6 = float4(a, float3(a, b, a));
    
    float r = c2_0.x + 
        c3_0.y + c3_1.z + c3_2.x +
        c4_0.x + c4_1.x + c4_2.x + c4_3.x + c4_4.x + c4_5.x + c4_6.x;
    
    outputBuffer[dispatchID.x] = r; 
}