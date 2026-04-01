//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_SPIRV): -stage fragment -entry fragmentMain -target spirv -profile spirv_1_3

// We currently do not have GLSL profiles, pass in SPIRV profile instead. We need
// to pass in a profile parameter to emit the warning.
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_GLSL): -stage fragment -entry fragmentMain -target glsl -profile spirv_1_3

// CHECK_SPIRV: warning 41012{{.*}}'spvImageGatherExtended'
// CHECK_GLSL: warning 41012{{.*}}'GLSL_150 + GL_ARB_texture_gather'
float4 fragmentMain(float2 pos : SV_Position, Sampler2D<float> foo) : SV_Target
{
    return foo.Gather(uint2(pos));
}

