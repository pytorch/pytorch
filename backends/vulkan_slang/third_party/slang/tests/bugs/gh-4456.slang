//TEST:SIMPLE(filecheck=CHECK): -target spirv -lang slang -D__spirv__ -emit-spirv-directly -profile ds_6_5 -entry main

// CHECK: OpEntryPoint

struct VSSceneIn {
  float3 pos : POSITION;
};

struct PSSceneIn {
  float4 pos : SV_Position;
};

struct HSPerVertexData {
  PSSceneIn v;
};

struct HSPerPatchData {
  float edges[3] : SV_TessFactor;
  float inside : SV_InsideTessFactor;
};

RaytracingAccelerationStructure AccelerationStructure : register(t0);
RayDesc MakeRayDesc()
{
    RayDesc desc;
    desc.Origin = float3(0,0,0);
    desc.Direction = float3(1,0,0);
    desc.TMin = 0.0f;
    desc.TMax = 9999.0;
    return desc;
}
void doInitialize(out RayQuery<RAY_FLAG_FORCE_OPAQUE> query, RayDesc ray)
{
    query.TraceRayInline(AccelerationStructure,RAY_FLAG_FORCE_NON_OPAQUE,0xFF,ray);
}

[domain("tri")] PSSceneIn main(
    const float3 bary : SV_DomainLocation,
    const OutputPatch<HSPerVertexData, 3> patch,
    const HSPerPatchData perPatchData) 
{
    PSSceneIn v;
    v.pos = patch[0].v.pos * bary.x + patch[1].v.pos * bary.y + patch[2].v.pos * bary.z + perPatchData.edges[1];

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    RayDesc ray = MakeRayDesc();

    q.TraceRayInline(AccelerationStructure,RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    doInitialize(q, ray);

 	return v;
}
