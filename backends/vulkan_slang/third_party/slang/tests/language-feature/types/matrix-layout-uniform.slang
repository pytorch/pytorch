//TEST:SIMPLE(filecheck=HLSL): -target hlsl -profile vs_6_0 -entry Main

// HLSL: #pragma pack_matrix(column_major)
// HLSL-NOT: row_major float4x4 worldToClip
// HLSL: row_major float4x4 instanceMeshToWorld

[shader("vertex")]
void Main(
    uniform column_major float4x4 worldToClip,
    out float4 SV_Position : SV_Position,
    in row_major float4x4 instanceMeshToWorld : instanceMeshToWorld,
    in float3 vertexPositionInMesh : vertexPositionInMesh
)
{
    SV_Position = mul(
        mul(worldToClip, instanceMeshToWorld),
        float4(vertexPositionInMesh, 1.f),
    );
}