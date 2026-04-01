//TEST(smoke,compute):COMPARE_COMPUTE:-shaderobj
//TEST(smoke,compute):COMPARE_COMPUTE:-dx12 -use-dxil -shaderobj
//TEST(smoke,compute):COMPARE_COMPUTE:-cpu -shaderobj
//TEST(smoke,compute):COMPARE_COMPUTE:-vk -shaderobj
//TEST(smoke,compute):COMPARE_COMPUTE:-mtl -shaderobj
//TEST(smoke,compute):COMPARE_COMPUTE:-cuda -shaderobj
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -shaderobj

// CHECK:      0
// CHECK-NEXT: 3F800000
// CHECK-NEXT: 40000000
// CHECK-NEXT: 40400000

// This is a basic test for Slang compute shader.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadID.x] = float(dispatchThreadID.x);
}
