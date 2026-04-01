// acceleration-structure-in-compute.slang

// Test for using ray-tracing acceleration structures as a shader input
// in non-ray-tracing shader code (that never actually traces rays)

//TEST:CROSS_COMPILE:-target spirv-asm -stage compute -profile glsl_460+spirv_1_4 -entry main

uniform RaytracingAccelerationStructure gScene;

int helper(RaytracingAccelerationStructure a, int b)
{
    return b;
}

[shader("compute")]
void main(
    uniform RaytracingAccelerationStructure x)
{
    helper(x, 1);
}