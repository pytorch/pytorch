//TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry main -allow-glsl -profile sm_5_0
// CHECK: error 36116
struct RayPayload
{
    uint data;
};

uniform RaytracingAccelerationStructure sceneBVH;
RWStructuredBuffer<uint> outputBuffer;
[shader("compute_fragment")]
void main()
{
    outputBuffer[0] = 0;
}
