//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly

// CHECK: OpEntryPoint

struct Vertex
{
  float4 pos : SV_Position;
  [[vk::location(0)]] float3 color;
};

const static uint MAX_VERTS = 3;
const static uint MAX_PRIMS = 1;

[outputtopology("triangle")]
[numthreads(3, 1, 1)]
[shader("mesh")]
void entry_mesh(
    in uint tig : SV_GroupIndex,
    out vertices Vertex verts[MAX_VERTS],
    out indices uint3 triangles[MAX_PRIMS])
{
}