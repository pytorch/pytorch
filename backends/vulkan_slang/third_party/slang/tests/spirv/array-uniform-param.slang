// array-uniform-param.slang

//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry computeMain -stage compute -emit-spirv-directly -force-glsl-scalar-layout
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER):-vk -compute -output-using-type

// CHECK: OpEntryPoint

// Test direct SPIR-V emit on arrays in uniforms.

//TEST_INPUT:set resultBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<uint> resultBuffer;

struct Data
{
    uint arr[4];
}

struct Data2
{
    uint arr[4u];
}

//TEST_INPUT: set Uniforms.d = new Data{{1,2,3,4}};
//TEST_INPUT: set Uniforms.d2 = new Data2{{1,2,3,4}};

cbuffer Uniforms
{
    Data d;
    Data2 d2;
}

// BUFFER: 3

[numthreads(4,1,1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadId = dispatchThreadID.x;
    uint result = d.arr[1];
    var u2 = reinterpret<Data>(d2);
    result += u2.arr[0];
    resultBuffer[threadId] = result;
}
