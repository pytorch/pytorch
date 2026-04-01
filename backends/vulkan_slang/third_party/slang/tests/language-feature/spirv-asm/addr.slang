//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -emit-spirv-directly -output-using-type

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK:      4
// CHECK-NEXT: 8
// CHECK-NEXT: 12
// CHECK-NEXT: 16

//
// This test tests that addresses of variables and inout parameters can be
// passed into a spirv_asm block
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];
    int r;
    spirv_asm
    {
        OpLoad $$int %n &n;
        OpConstant $$int %two 2;
        OpIMul $$int %r $n %two;
        OpStore &r %r;
    };
    foo(r);
    outputBuffer[i] = r;
}

func foo(inout int x)
{
    spirv_asm
    {
        OpLoad $$int %x &x;
        OpIAdd $$int %added %x %x;
        OpStore &x %added;
    };
}

