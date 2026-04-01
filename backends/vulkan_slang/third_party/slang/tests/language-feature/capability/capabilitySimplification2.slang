//TEST:SIMPLE(filecheck=SPIRV): -target spirv -emit-spirv-directly -entry computeMain -stage compute -profile sm_5_0
//TEST:SIMPLE(filecheck=GLSL): -target glsl -entry computeMain -stage compute -profile sm_5_0
//TEST:SIMPLE(filecheck=HLSL): -target hlsl  -entry computeMain -stage compute -profile sm_5_0
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target spirv -emit-spirv-directly -entry computeMain -stage compute  -profile sm_5_0 -ignore-capabilities


// CHECK_IGNORE_CAPS-NOT: warning 41012

// SPIRV: warning 41012
// SPIRV-NOT: spirv_1_2
// SPIRV-NOT: spirv_1_3
// SPIRV-SAME: spvGroupNonUniformBallot

// GLSL: warning 41012
// GLSL-NOT: GLSL_400
// GLSL-NOT: GLSL_430
// GLSL-SAME: GL_KHR_shader_subgroup_ballot

// HLSL: warning 41012
// HLSL-NOT: sm_5_1
// HLSL-SAME: sm_6_0

[require(sm_6_0)]
[numthreads(1,1,1)]
void computeMain()
{
}
