//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -dx12 -use-dxil -shaderobj -xslang -matrix-layout-row-major
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -dx12 -use-dxil -shaderobj -xslang -matrix-layout-column-major
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -shaderobj -xslang -matrix-layout-row-major
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -shaderobj -xslang -matrix-layout-column-major

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float2x2 matrix2x2_1 = float2x2(1, 2, 3, 4);
    float4 vector4_1 = (float4)matrix2x2_1;

    float4 vector4_2 = float4(1, 2, 3, 4);
    float2x2 matrix2x2_2 = (float2x2)vector4_2;

    outputBuffer[0] = uint(true
            && all(vector4_1 == float4(1, 2, 3, 4))
            
            && all(matrix2x2_2[0] == float2(1,2))
            && all(matrix2x2_2[1] == float2(3,4))
        );
    //BUF: 1
}
