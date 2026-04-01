//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -dx12 -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -cpu -compute -shaderobj

// Test that char literals work as expected.

// CHECK: 48
// CHECK-NEXT: 65
// CHECK-NEXT: 6C
// CHECK-NEXT: 6C
// CHECK-NEXT: 6F
// CHECK-NEXT: 4E16
// CHECK-NEXT: 754C
// CHECK-NEXT: A

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    outputBuffer[0] = 'H';
    outputBuffer[1] = 'e';
    outputBuffer[2] = 'l';
    outputBuffer[3] = 'l';
    outputBuffer[4] = 'o';
    outputBuffer[5] = '世';
    outputBuffer[6] = '界';
    outputBuffer[7] = '\n';
}
