//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector

// CHECK: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4
// CHECK-NEXT: 5
// CHECK-NEXT: 6
// CHECK-NEXT: 7
// CHECK-NEXT: 8
// CHECK-NEXT: 9
// CHECK-NEXT: A
// CHECK-NEXT: B
// CHECK-NEXT: C
// CHECK-NEXT: D
// CHECK-NEXT: E
// CHECK-NEXT: F

//TEST_INPUT:ubuffer(data=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]):name=input
RWByteAddressBuffer input;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]):out,name=outputBuffer
RWStructuredBuffer<uint32_t> outputBuffer;

groupshared float3[5] temp;

[numthreads(1, 1, 1)]
void computeMain()
{
    let vec = coopVecLoad<15, uint32_t>(input);
    vec.storeAny(temp);
    let result = CoopVec<uint32_t, 15>.loadAny(temp);
    for(int i = 0; i < result.getCount(); ++i)
        outputBuffer[i] = result[i];;
}
