//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type

// HLSL doesn't support the training operations
//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -render-feature cooperative-vector -dx12-experimental -use-dxil -output-using-type -profile cs_6_8 -Xslang... -Xdxc -Vd -X.

// Disabled because of some pecularities stemming from our lowering of half to float
//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// CHECK: type: half
// CHECK-NEXT: 112.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 3.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 5.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 8.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 8.000000
// CHECK-NEXT: 10.000000
// CHECK-NEXT: 12.000000
// CHECK-NEXT: 14.000000
// CHECK-NEXT: 16.000000
// CHECK-NEXT: 3.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 9.000000
// CHECK-NEXT: 12.000000
// CHECK-NEXT: 15.000000
// CHECK-NEXT: 18.000000
// CHECK-NEXT: 21.000000
// CHECK-NEXT: 24.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 8.000000
// CHECK-NEXT: 12.000000
// CHECK-NEXT: 16.000000
// CHECK-NEXT: 20.000000
// CHECK-NEXT: 24.000000
// CHECK-NEXT: 28.000000
// CHECK-NEXT: 32.000000

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=2):out,name=outputBuffer
RWStructuredBuffer<half> outputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4),name=output
RWByteAddressBuffer output;

[numthreads(1, 1, 1)]
void computeMain()
{
    CoopVec<half, 4> vecA;
    CoopVec<half, 8> vecB;
    for(int i = 0; i < vecA.getCount(); ++i)
        vecA[i] = half(i+1);
    for(int i = 0; i < vecB.getCount(); ++i)
        vecB[i] = half(i+1);

    output.Store<half>(0, half(111));

    coopVecOuterProductAccumulate(
        vecA,
        vecB,
        output,
        0,
        32,
        CoopVecMatrixLayout::RowMajor,
        CoopVecComponentType::Float16,
    );

    for(int i = 0; i < vecA.getCount() * vecB.getCount(); ++i)
        outputBuffer[i] = output.Load<half>(i * 2);
}
