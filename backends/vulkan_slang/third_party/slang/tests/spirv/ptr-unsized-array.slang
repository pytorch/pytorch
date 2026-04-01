//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: OpPtrAccessChain

struct MeshVertex {
    float3 Pos;
    float2 TexCoord;
};
struct MeshData { 
    float4x4 ModelMat;
    MeshVertex Vertices[];
};
struct DispatchParams {
    MeshData* Mesh;
    float3* Dest;
};

[vk::push_constant] DispatchParams pc;

[numthreads(64)]
void ComputeMain(uint tid: SV_DispatchThreadID) {
    pc.Dest[tid] = pc.Mesh->Vertices[tid].Pos;
}