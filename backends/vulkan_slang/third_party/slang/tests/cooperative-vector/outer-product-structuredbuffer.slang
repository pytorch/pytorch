//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type

// These platforms don't support these operations into structured buffers
//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -render-feature cooperative-vector -dx12-experimental -use-dxil -output-using-type -profile cs_6_8 -Xslang... -Xdxc -Vd -X.
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

//TEST_INPUT:ubuffer(data=[ 0x40003C00 0x44004200 ], stride=4),name=inputA
// [1,2,3,4]
StructuredBuffer<half> inputA;

//TEST_INPUT:ubuffer(data=[ 0x40003C00 0x44004200 0x46004500 0x48004700 ], stride=4),name=inputB
// [1,2,3,4,5,6,7,8]
StructuredBuffer<half> inputB;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=output
RWStructuredBuffer<half> output;

[numthreads(1, 1, 1)]
void computeMain()
{
    let vecA = coopVecLoad<4>(inputA);
    let vecB = coopVecLoad<8>(inputB);

    output[0] = half(111);

    coopVecOuterProductAccumulate(
        vecA,
        vecB,
        output,
        0,
        32,
        CoopVecMatrixLayout::RowMajor,
        CoopVecComponentType::Float16,
    );
}
