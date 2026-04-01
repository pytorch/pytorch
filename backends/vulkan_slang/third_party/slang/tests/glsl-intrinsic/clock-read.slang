//TEST:SIMPLE(filecheck=GLSL): -target glsl
//TEST:SIMPLE(filecheck=SPIRV): -target spirv
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -emit-spirv-via-glsl

// GLSL: GL_ARB_shader_clock : require
// GLSL: GL_ARB_gpu_shader_int64 : require

//SPIRV: OpCapability ShaderClockKHR
//SPIRV: OpExtension "SPV_KHR_shader_clock"
RWStructuredBuffer<float> output;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    //GLSL: clock2x32ARB
    //SPIRV: OpReadClockKHR %v2uint %uint_3
    output[0] = clock2x32ARB().x;

    //GLSL: clockARB
    //SPIRV: OpReadClockKHR %ulong %uint_3
    uint64_t c = clockARB();
    output[1] = c;
}
