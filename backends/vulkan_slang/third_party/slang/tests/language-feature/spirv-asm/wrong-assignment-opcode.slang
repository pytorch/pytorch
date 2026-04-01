//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

RWStructuredBuffer<float> buf;

//
// This test checks that we correctly diagnose trying to use the assignment and
// typed assignment syntax with operands with no result type or id.
//
// By virtue of these tests being in the same file, we also check the error
// recovery for the asm block parser.
//
int foo(const int constParam)
{
    // Doesn't have a result type
    spirv_asm
    {
        %bar : %wrong = OpTypeInt 32 0
        // CHECK: wrong-assignment-opcode.slang([[#@LINE-1]]): error {{.*}}: cannot use this 'x : <type> = OpTypeInt...' syntax because OpTypeInt does not have a <result-type-id> operand

    };

    // Doesn't have a result value
    spirv_asm
    {
        %foo = OpStore 1 2;
        // CHECK: wrong-assignment-opcode.slang([[#@LINE-1]]): error {{.*}}: cannot use this 'x = OpStore...' syntax because OpStore does not have a <result-id> operand
    };

    // garbage
    spirv_asm
    {
        %foo =;
        // CHECK: wrong-assignment-opcode.slang([[#@LINE-1]]): error {{.*}}: unexpected ';'
        %foo :;
        // CHECK: wrong-assignment-opcode.slang([[#@LINE-1]]): error {{.*}}: unexpected ';'
        %foo : bar;
        // CHECK: wrong-assignment-opcode.slang([[#@LINE-1]]): error {{.*}}: unexpected ';', expected '='
        %foo : bar =;
        // CHECK: wrong-assignment-opcode.slang([[#@LINE-1]]): error {{.*}}: unexpected ';'
    };
}
