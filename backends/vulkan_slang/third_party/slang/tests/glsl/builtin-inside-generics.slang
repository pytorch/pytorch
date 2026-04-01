//TEST:SIMPLE(filecheck=CHECK_SPIRV): -entry main -stage compute -target spirv -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_GLSL): -entry main -stage compute -target glsl -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_HLSL): -entry main -stage compute -target hlsl -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_METAL): -entry main -stage compute -target metal -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_WGSL): -entry main -stage compute -target wgsl -allow-glsl

RWStructuredBuffer<uint> outputBuffer;

T getGlobalInvocationID<T: IInteger>(T value)
{
    return T(gl_GlobalInvocationID.x) + value;
}

T getWaveLaneIndex<T: IInteger>(T value)
{
    return T(WaveGetLaneIndex()) + value;
}

// CHECK_SPIRV: GlobalInvocationId
// CHECK_SPIRV: SubgroupLocalInvocationId

// CHECK_GLSL: gl_GlobalInvocationID
// CHECK_GLSL: gl_SubgroupInvocationID

// CHECK_HLSL: WaveGetLaneIndex()
// CHECK_HLSL: SV_DispatchThreadID

// CHECK_METAL: thread_position_in_grid
// CHECK_METAL: thread_index_in_simdgroup

// CHECK_WGSL: global_invocation_id
// CHECK_WGSL: subgroup_invocation_id

[shader("compute")]
void main()
{
    outputBuffer[0U] = getGlobalInvocationID(0U);
    outputBuffer[1U] = getWaveLaneIndex(0U);
}


