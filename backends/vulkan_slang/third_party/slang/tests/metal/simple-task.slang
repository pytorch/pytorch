//TEST:SIMPLE(filecheck=METAL): -entry taskMain -stage amplification -target metal

cbuffer Uniforms
{
    float4x4 modelViewProjection;
}

//
// Task shader
//

struct MeshPayload
{
    int exponent;
};

// METAL: MeshPayload_0 object_data* _slang_mesh_payload
// METAL: mesh_grid_properties  _slang_mgp
[numthreads(1,1,1)]
void taskMain()
{
    // METAL: _slang_mesh_payload
    // METAL: _slang_mgp.set_threadgroups_per_grid
    MeshPayload p;
    p.exponent = 3;
    DispatchMesh(1, 1, 1, p);
}
