//TEST:SIMPLE(filecheck=CHECK_SPV):-target spirv -entry main -stage compute
//TEST:SIMPLE(filecheck=CHECK_GLSL_SPV):-target spirv -entry main -stage compute -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=CHECK_GLSL):-target glsl -entry main -stage compute
//TEST:SIMPLE(filecheck=CHECK_HLSL):-target hlsl -entry main -stage compute
RWStructuredBuffer<uint> globalBuffer[] : register(t0, space0);
RWStructuredBuffer<uint3> outputBuffer;

struct MyStruct
{
    uint a;
    uint b;
    uint c;
};


MyStruct func(RWStructuredBuffer<uint> buffer)
{
    MyStruct a;

    // CHECK_GLSL: globalBuffer_0[nonuniformEXT({{.*}})]
    // CHECK_GLSL: globalBuffer_0[nonuniformEXT({{.*}})]

    // For the last test case 3 that the callee passes globalBuffer[bufferIdx3] to the function,
    // we should not see nonuniformEXT here.

    // CHECK_GLSL: globalBuffer_0[_{{.*}})]
    // CHECK_GLSL: globalBuffer_0[_{{.*}})]
    a.a = buffer[0];
    a.b = a.a + 1;
    a.c = a.a + a.b + 1;

    return a;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint2 pixelIndex : SV_DispatchThreadID)
{

    // CHECK_SPV: OpDecorate %[[VAR1:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_SPV: OpDecorate %[[VAR2:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_SPV: OpDecorate %[[VAR3:[a-zA-Z0-9_]+]] NonUniform


    // CHECK_GLSL_SPV: OpDecorate %[[VAR1:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR2:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR3:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR4:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR5:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR6:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR7:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR8:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR9:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR10:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR11:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR12:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR13:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR14:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR15:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR16:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR17:[a-zA-Z0-9_]+]] NonUniform
    // CHECK_GLSL_SPV: OpDecorate %[[VAR18:[a-zA-Z0-9_]+]] NonUniform


    // Test case 1: slang will specialize the func call to 'MyStruct func(uint)'
    uint bufferIdx = pixelIndex.x;
    uint nonUniformIdx = NonUniformResourceIndex(bufferIdx);
    RWStructuredBuffer<uint> buffer = globalBuffer[nonUniformIdx];

    // CHECK_SPV: %[[VAR1]] = OpAccessChain %_ptr_StorageBuffer_RWStructuredBuffer{{.*}} %{{.*}} %bufferIdx

    // CHECK_GLSL_SPV: %[[VAR1]] = OpCopyObject %uint %{{.*}}

    // CHECK_GLSL_SPV: %[[VAR4]] = OpCopyObject %uint %[[VAR1]]
    // CHECK_GLSL_SPV: %[[VAR5]] = OpAccessChain %_ptr_Uniform_uint %globalBuffer_0 %[[VAR4]] %int_0 %int_0
    // CHECK_GLSL_SPV: %[[VAR6]] = OpLoad %uint %[[VAR5]]

    // CHECK_GLSL_SPV: %[[VAR7]] = OpCopyObject %uint %[[VAR1]]
    // CHECK_GLSL_SPV: %[[VAR8]] = OpAccessChain %_ptr_Uniform_uint %globalBuffer_0 %[[VAR7]] %int_0 %int_0
    // CHECK_GLSL_SPV: %[[VAR9]] = OpLoad %uint %[[VAR8]]

    // CHECK_GLSL: func_0({{.*}}nonuniformEXT({{.*}}))
    // CHECK_HLSL: func_0(globalBuffer_0[NonUniformResourceIndex({{.*}})])
    MyStruct myStruct = func(buffer);

    int bufferIdx2 = pixelIndex.y;

    // Test case 2: Make sure we cover the case for the different data type of the index.
    // In this case, slang will specialize the function to 'MyStruct func(int)'
    // CHECK_SPV: %[[VAR2]] = OpAccessChain %_ptr_StorageBuffer_RWStructuredBuffer{{.*}} %{{.*}} %bufferIdx2


    // CHECK_GLSL_SPV: %[[VAR2]] = OpCopyObject %int %{{.*}}

    // CHECK_GLSL-SPV: %[[VAR10]] = OpCopyObject %int %[[VAR2]]
    // CHECK_GLSL-SPV: %[[VAR11]] = OpAccessChain %_ptr_Uniform_uint %globalBuffer_0 %[[VAR10]] %int_0 %int_0
    // CHECK_GLSL-SPV: %[[VAR12]] = OpLoad %uint %[[VAR11]]

    // CHECK_GLSL-SPV: %[[VAR13]] = OpCopyObject %int %[[VAR2]]
    // CHECK_GLSL-SPV: %[[VAR14]] = OpAccessChain %_ptr_Uniform_uint %globalBuffer_0 %[[VAR13]] %int_0 %int_0
    // CHECK_GLSL-SPV: %[[VAR15]] = OpLoad %uint %[[VAR14]]
    RWStructuredBuffer<uint> buffer2 = globalBuffer[NonUniformResourceIndex(bufferIdx2)];

    // CHECK_GLSL: func_1({{.*}}nonuniformEXT({{.*}}))
    // CHECK_HLSL: func_0(globalBuffer_0[NonUniformResourceIndex({{.*}})])
    MyStruct myStruct2 = func(buffer2);

    // Test case 3: Test the case that we handle the uniformity correctly, the NonUniformResourceIndex will not propagate
    // to the function, so there should no NonUniform decoration appeared.
    int bufferIdx3 = pixelIndex.y;
    RWStructuredBuffer<uint> buffer3 = globalBuffer[bufferIdx3];

    // CHECK_SPV: %[[VAR4:[a-zA-Z0-9_]+]] = OpAccessChain %_ptr_StorageBuffer_RWStructuredBuffer{{.*}} %{{.*}} %bufferIdx2

    // Test to make sure this command is not decorated with NonUniform:
    // CHECK_SPV-NOT: OpDecorate %[[VAR4]] NonUniform
    MyStruct myStruct3 = func(buffer3);


    // Test case 4: Test to make sure we correctly cover the case that intCast or uintCast of a NonUniformResourceIndex
    // is still a NonUniformResourceIndex.

    // CHECK_SPV: %[[VAR5:[a-zA-Z0-9_]+]] = OpBitcast %uint %{{.*}}
    // CHECK_SPV: %[[VAR3]] = OpAccessChain %_ptr_StorageBuffer_RWStructuredBuffer{{.*}} %{{.*}} %[[VAR5]]

    // CHECK_GLSL-SPV: %[[VAR19:[a-zA-Z0-9_]+]]  = OpBitcast %int %[[VAR3]]
    // CHECK_GLSL-SPV: %[[VAR16]] = OpCopyObject %int %[[VAR19]]
    // CHECK_GLSL-SPV: %[[VAR17]] = OpAccessChain %_ptr_Uniform_uint %globalBuffer_0 %[[VAR16]] %int_0 %int_0
    // CHECK_GLSL-SPV: %[[VAR18]] = OpLoad %uint %[[VAR17]]
    //
    // Since after the nested cast, the index data type is 'uint' now, make sure it calls the same function as the test case 1.
    // CHECK_GLSL: func_0({{.*}}nonuniformEXT({{.*}}))
    RWStructuredBuffer<uint> buffer4 = globalBuffer[(uint)((int)NonUniformResourceIndex(bufferIdx))];
    MyStruct myStruct4 = func(buffer4);

    outputBuffer[0] = uint3(myStruct.a, myStruct.b, myStruct.c);
    outputBuffer[1] = uint3(myStruct2.a, myStruct2.b, myStruct2.c);
    outputBuffer[2] = uint3(myStruct3.a, myStruct3.b, myStruct3.c);
    outputBuffer[3] = uint3(myStruct4.a, myStruct4.b, myStruct4.c);
}

