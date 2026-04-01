//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -render-feature cooperative-vector -dx12-experimental -use-dxil -output-using-type -profile cs_6_8 -Xslang... -Xdxc -Vd -X.
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// CHECK: type: uint32_t
// CHECK-NEXT: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 1

//TEST_INPUT:ubuffer(data=[1.0 2.0 3.0 4.0], stride=4),name=input1
ByteAddressBuffer input1;

//TEST_INPUT:ubuffer(data=[1.0 3.0 2.0 4.0], stride=4),name=input2
ByteAddressBuffer input2;

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    CoopVec<float, 4> vec1 = coopVecLoad<4, float>(input1);
    CoopVec<float, 4> vec2 = coopVecLoad<4, float>(input2);

    uint32_t equals = vec1 == vec2 ? 1 : 0;
    uint32_t lessThan = vec1 < vec2 ? 1 : 0;
    uint32_t lessThanOrEquals = vec1 <= vec2 ? 1 : 0;

    outputBuffer[0] = equals;
    outputBuffer[1] = lessThan;
    outputBuffer[2] = lessThanOrEquals;
}
