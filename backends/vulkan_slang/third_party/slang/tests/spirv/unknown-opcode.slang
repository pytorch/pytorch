
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -stage compute -entry main -target spirv -emit-spirv-directly

void main()
{
    spirv_asm
    {
        // CHECK: ([[#@LINE+1]]): error
        unknownOpCode
    };
    spirv_asm
    {
        // CHECK-NOT: ([[#@LINE+1]]): error
        5
    };
}
