//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -render-feature cooperative-vector -dx12-experimental -use-dxil -output-using-type -profile cs_6_8 -Xslang... -Xdxc -Vd -X.
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// CHECK: type: float
// CHECK-NEXT: 5.000000
// CHECK-NEXT: 10.000000
// CHECK-NEXT: 17.000000
// CHECK-NEXT: 26.000000

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

//TEST_INPUT:ubuffer(data=[1.0 2.0 3.0 4.0], stride=4),name=input1
ByteAddressBuffer input1;

//TEST_INPUT:ubuffer(data=[2.0 3.0 4.0 5.0], stride=4),name=input2
ByteAddressBuffer input2;

//TEST_INPUT:ubuffer(data=[3.0 4.0 5.0 6.0], stride=4),name=input3
ByteAddressBuffer input3;

[numthreads(1, 1, 1)]
void computeMain()
{
    CoopVec<float, 4> vec1 = coopVecLoad<4, float>(input1);
    CoopVec<float, 4> vec2 = coopVecLoad<4, float>(input2);
    CoopVec<float, 4> vec3 = coopVecLoad<4, float>(input3);

    CoopVec<float, 4> result = fma(vec1, vec2, vec3);

    for(int i = 0; i < result.getCount(); ++i)
        outputBuffer[i] = result[i];;
}
