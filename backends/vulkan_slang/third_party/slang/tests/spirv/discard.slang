//TEST:SIMPLE(filecheck=SPV16): -target spirv -profile spirv_1_6
//TEST:SIMPLE(filecheck=SPV15): -target spirv -profile spirv_1_5
//TEST:SIMPLE(filecheck=SPV14): -target spirv -profile spirv_1_4+SPV_EXT_demote_to_helper_invocation

uniform int a;

// Test that we translate `discard` to OpDemoateToHelperInvocation
// when targeting SPIRV 1.6, and to OpKill when targeting SPIRV 1.5.
// If the user explicitly specified SPV_EXT_demote_to_helper_invocation
// in the profile, we will also emit OpDemoteToHelperInvocation.

//SPV16: OpDemoteToHelperInvocation
//SPV15: OpKill
//SPV14: OpDemoteToHelperInvocation

[shader("fragment")]
float4 frag() : SV_Target
{
    if (a == 0)
        discard;
    return 0;
}