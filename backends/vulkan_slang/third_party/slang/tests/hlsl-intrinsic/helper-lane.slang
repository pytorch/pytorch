//TEST:SIMPLE(filecheck=HLSL): -entry main -target hlsl -stage fragment
//TEST:SIMPLE(filecheck=GLSL): -entry main -target glsl -stage fragment
//TEST:SIMPLE(filecheck=METAL): -entry main -target metal -stage fragment
//TEST:SIMPLE(filecheck=SPIRV): -entry main -target spirv -stage fragment
//TEST:SIMPLE(filecheck=SPIRV): -entry main -target spirv-asm -stage fragment

float4 main() : SV_Target
{
    //HLSL: IsHelperLane()

    //GLSL: GL_EXT_demote_to_helper_invocation
    //GLSL: gl_HelperInvocation()

    //METAL: simd_is_helper_thread()

    //SPIRV: DemoteToHelperInvocation
    //SPIRV: SPV_EXT_demote_to_helper_invocation
    //SPIRV: OpIsHelperInvocationEXT
    return float4(IsHelperLane());
}
