//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -render-feature cooperative-vector -dx12-experimental -use-dxil -output-using-type -profile cs_6_8 -Xslang... -Xdxc -Vd -X.
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu 

// CHECK: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4
// CHECK-NEXT: 5

//TEST_INPUT:ubuffer(data=[1 2 3 4 5]):name=input
RWByteAddressBuffer input;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0]):out,name=outputBuffer
RWByteAddressBuffer outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    let result = coopVecLoad<5, uint32_t>(input);
    result.store(outputBuffer);
}
