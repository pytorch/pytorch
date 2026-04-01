//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

RWStructuredBuffer<float> buf;

//
// This test checks that we correctly diagnose passing too many operands to a spriv inst
//
void foo(const int constParam)
{
    spirv_asm
    {
        // Make sure we diagnose the first offending operand
        OpLoad
            %a
            %b
            %c
            %d
        // CHECK: too-many-operands.slang([[#@LINE+1]]): warning {{.*}}: too many operands for OpLoad
        %r : $$int = OpIAdd %r %r
    };
}
