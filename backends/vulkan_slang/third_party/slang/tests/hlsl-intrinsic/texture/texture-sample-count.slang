//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile sm_6_6 -entry main -stage compute
//TEST:SIMPLE(filecheck=DXIL): -target dxil -profile sm_6_6 -entry main -stage compute

// Test that Texture2DMS accepts an optional sampleCount argument
// and the argument correctly appears in the output hlsl.

// CHECK: Texture2DMS<uint > t_0 : register(t0);
// CHECK: Texture2DMS<uint, 4 > tExplicit_0 : register(t1);
// DXIL: main

Texture2DMS<uint> t;
Texture2DMS<uint, 4> tExplicit;

SamplerState s;
RWBuffer<uint4> b;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    let v = t.Load(int2(0, 0), 1) + tExplicit.Load(int2(1,1),2);
    b[tid.x] = v;
}
