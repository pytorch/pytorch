//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -emit-spirv-directly -output-using-type

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK:      8
// CHECK-NEXT: 13
// CHECK-NEXT: 18
// CHECK-NEXT: 23

//
// This test tests the __truncate operator
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];

    int scalar = n;

    // 1-vectors are not valid in SPIR-V
    // vector<int, 1> vector1 = vector<int, 1>(n);

    vector<int, 4> vector4 = n + vector<int, 4>(0,1,2,3);
    //int expected = 0 + n + n + (n + (n+1) + (n+2));

    int r = 0;
    spirv_asm
    {
        // scalar to scalar
        __truncate $$int %a1 $$int $scalar;
        %r1 : $$int = OpIAdd %a1 $r;

        // scalar to 1-vector
        // __truncate $$vector<int,1> %a2 $$int $scalar;
        // %x1 : $$int = OpCompositeExtract %a2 0;
        // %r2 : $$int = OpIAdd %x1 %r1;
        %r2 : $$int = OpCopyObject %r1;

        // 1-vector to scalar
        // __truncate $$int %a3 $$vector<int,1> $vector1;
        // %r3 : $$int = OpIAdd %a3 %r2;
        %r3 : $$int = OpCopyObject %r2;

        // n-vector to scalar
        __truncate $$int %a4 $$vector<int,4> $vector4;
        %r4 : $$int = OpIAdd %a4 %r3;

        // n-vector to m-vector
        __truncate $$vector<int,3> %a5 $$vector<int,4> $vector4;
        %x2 : $$int = OpCompositeExtract %a5 0;
        %x3 : $$int = OpCompositeExtract %a5 1;
        %x4 : $$int = OpCompositeExtract %a5 2;
        %r5 : $$int = OpIAdd %x2 %r4;
        %r6 : $$int = OpIAdd %x3 %r5;
        %r7 : $$int = OpIAdd %x4 %r6;

        OpStore &r %r7
    };
    outputBuffer[i] = r;
}
