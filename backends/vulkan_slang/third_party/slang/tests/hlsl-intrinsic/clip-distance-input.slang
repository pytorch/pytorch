//TEST:SIMPLE(filecheck=CHECK): -emit-spirv-via-glsl -stage fragment -entry fragMain -target spirv
//TEST:SIMPLE(filecheck=CHECK): -emit-spirv-directly -stage fragment -entry fragMain -target spirv

// CHECK: OpDecorate %[[VARNAME:[A-Za-z_0-9]+]] BuiltIn ClipDistance
// CHECK: %[[VARNAME]] = OpVariable %_ptr_Input__arr_float_{{u?}}int_1 Input

float4 fragMain(float flClipDist : SV_ClipDistance0 )
{
    return flClipDist;
}