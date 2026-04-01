//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -render-feature cooperative-vector -dx12-experimental -use-dxil -output-using-type -profile cs_6_8 -Xslang... -Xdxc -Vd -X.
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// CHECK: type: float
// CHECK-NEXT: 2.0
// CHECK-NEXT: 2.0
// CHECK-NEXT: 3.0
// CHECK-NEXT: 4.0

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

//TEST_INPUT:ubuffer(data=[1.0 2.0 3.0 5.0], stride=4),name=input
ByteAddressBuffer input;

[numthreads(1, 1, 1)]
void computeMain()
{
    CoopVec<float, 4> vec = coopVecLoad<4, float>(input);
    CoopVec<float, 4> minVal = CoopVec<float, 4>(2.0);
    CoopVec<float, 4> maxVal = CoopVec<float, 4>(4.0);

    CoopVec<float, 4> result = clamp(vec, minVal, maxVal);

    for(int i = 0; i < result.getCount(); ++i)
        outputBuffer[i] = result[i];;
}
