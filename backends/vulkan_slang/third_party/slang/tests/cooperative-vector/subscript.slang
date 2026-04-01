//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type -emit-spirv-directly
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -render-feature cooperative-vector -dx12-experimental -use-dxil -output-using-type -profile cs_6_8 -Xslang... -Xdxc -Vd -X.
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// CHECK: type: int32_t
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 8

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int32_t> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    CoopVec<int32_t, 4> vec;
    vec[0] = 2;
    vec[1] = vec[0]+2;
    vec[2] = vec[1]+2;
    vec[3] = vec[2]+2;

    for(int i = 0; i < vec.getCount(); ++i)
        outputBuffer[i] = vec[i];;
}
