// gather-texture2darray.slang

//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry main -stage compute -emit-spirv-directly
//TEST:SIMPLE(filecheck=DXIL): -target dxil -profile sm_6_1 -entry main -stage compute
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry main -stage compute

// Test gathering from a `Texture2DArray`

Texture2DArray<uint> t;
Texture2DArray<float4> t1;
SamplerState s;
RWBuffer<uint4> b;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    // DXIL: @dx.op.textureGather.i32
    // DXIL: @dx.op.textureGather.f32

    // SPIRV: OpImageGather
    // SPIRV: OpImageGather

    b[tid.x] = t.Gather(s, tid);

    b[tid.x] += uint4(t1.Gather(s, tid));
}
