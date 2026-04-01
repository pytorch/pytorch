//TEST:SIMPLE(filecheck=SPIRV): -entry computeMain -stage compute -target spirv
//TEST:SIMPLE(filecheck=HLSL): -entry computeMain -stage compute -target spirv
//TEST:SIMPLE(filecheck=GLSL): -entry computeMain -stage compute -target spirv

// SPIRV: OpEntryPoint
// GLSL: main
// HLSL: computeMain


[[vk::binding(0, 0)]] uniform ConstantBuffer<uint64_t[1000]> addresses;
RWStructuredBuffer<uint> buffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID, uint groupIndex: SV_GroupIndex)
{
    printf("\nfrom gpu: %llu", addresses[0]);
}