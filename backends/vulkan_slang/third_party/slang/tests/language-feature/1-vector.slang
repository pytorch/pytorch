//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -use-dxil -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK:      0
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    vector<int,1> i = vector<int,1>(dispatchThreadID.x);
    outputBuffer[i.x] = i.x;
}
