//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -emit-spirv-directly -output-using-type

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK:      10
// CHECK-NEXT: 20
// CHECK-NEXT: 30
// CHECK-NEXT: 40

//
// This test checks that although we have a generic function and the spirv_asm
// block references the type parameter 'T', specialization resolves this before
// the SPIR-V is rendered.
//
func foo<T : __BuiltinFloatingPointType>(T x) -> T
{
    return spirv_asm
    {
        OpConstant $$int %int_10 10;
        OpConvertSToF $$T %float_10 %int_10;
        OpFMul $$T result $x %float_10;
    };
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];
    outputBuffer[i] = int(foo<float>(n));
}
