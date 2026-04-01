//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):-emit-spirv-directly -target spirv-asm -entry computeMain -stage compute

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK: ; Debug Information
// CHECK-NOT: ; Annotations
// CHECK: OpName %{{[a-zA-Z]+}} "INTEGERRRRRRRRRRRRRRRRRRRRR"
// CHECK: ; Annotations

//
// This test tests that OpName is put in the correct location in the generated spriv
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    spirv_asm
    {
        OpName $$int "INTEGERRRRRRRRRRRRRRRRRRRRR"
    };
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];
    int r = spirv_asm
    {
        OpConstant $$int %two 2;
        OpIMul $$int result $n %two
    };
    outputBuffer[i] = r;
}
