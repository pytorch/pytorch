//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -render-feature cooperative-vector -dx12-experimental -use-dxil -output-using-type -profile cs_6_8 -Xslang... -Xdxc -Vd -X.
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// Hilariously low precision because the different APIs don't agree (they're
// not specced to of course, so it's ok)
// CHECK: type: float
// CHECK-NEXT: 0.761
// CHECK-NEXT: 0.964
// CHECK-NEXT: 0.995
// CHECK-NEXT: 0.999

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

//TEST_INPUT:ubuffer(data=[1.0 2.0 3.0 4.0], stride=4),name=input
ByteAddressBuffer input;

[numthreads(1, 1, 1)]
void computeMain()
{
    CoopVec<float, 4> vec = coopVecLoad<4, float>(input);

    CoopVec<float, 4> result = tanh(vec);

    for(int i = 0; i < result.getCount(); ++i)
        outputBuffer[i] = result[i];;
}
