//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

RWStructuredBuffer<float> buf;

//
// This test checks that we correctly diagnose trying to use the address of
// something without one
//
int foo(const int constParam)
{
    return spirv_asm
    {
        OpLoad $$int %foo &buf;
        // CHECK: bad-addr.slang([[#@LINE-1]]): note {{.*}}: unable to take the address of this address-of asm operand
        OpLoad $$int result &constParam;
        // CHECK: bad-addr.slang([[#@LINE-1]]): note {{.*}}: unable to take the address of this address-of asm operand
    };
}
