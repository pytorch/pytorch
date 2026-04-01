//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type
///TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// HLSL doesn't support the training operations
//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -render-feature cooperative-vector -dx12-experimental -use-dxil -output-using-type -profile cs_6_8 -Xslang... -Xdxc -Vd -X.

// CHECK: type: half
// CHECK-NEXT: 112.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 3.000000
// CHECK-NEXT: 4.000000


//TEST_INPUT:ubuffer(data=[0 0 0 0 ], stride=2):out,name=outputBuffer
RWStructuredBuffer<half> outputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4),name=inputA
ByteAddressBuffer inputA;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4),name=output
RWByteAddressBuffer output;

[numthreads(1, 1, 1)]
void computeMain()
{
    CoopVec<half, 4> vecA;
    for(int i = 0; i < vecA.getCount(); ++i)
        vecA[i] = half(i+1);

    output.Store(0, half(111));

    coopVecReduceSumAccumulate(
        vecA,
        output,
        0,
    );

    for(int i = 0; i < vecA.getCount(); ++i)
        outputBuffer[i] = output.Load<half>(i * 2);
}

