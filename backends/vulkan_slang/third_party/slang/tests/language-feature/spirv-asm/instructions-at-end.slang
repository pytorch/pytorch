//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):-emit-spirv-directly -target spirv-asm -entry computeMain -stage compute

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

//
// This test checks that we correctly diagnose using 'result' in an instruction
// not in the last position.
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];
    int r = spirv_asm
    {
        OpConstant $$int %two 2;
        OpIMul $$int result $n %two;
        // CHECK: instructions-at-end.slang([[#@LINE-1]]): error {{.*}}: the result-id marker must only be used in the last instruction of a spriv_asm expression
        // CHECK: note 29102: consider adding an OpCopyObject instruction to the end of the spirv_asm expression

        OpIAdd $$uint %something %two %two;
    };
    outputBuffer[i] = r;
}
