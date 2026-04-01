// component-write.slang

// This tests that writing to individual components of the output struct works

//TEST:CROSS_COMPILE:-target spirv -profile sm_6_5 -entry main -stage mesh

const static uint MAX_VERTS = 3;
const static uint MAX_PRIMS = 1;

[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void main(
    in uint tig : SV_GroupIndex,
    OutputVertices<float4, MAX_VERTS> verts : SV_Position,
    OutputIndices<uint3, MAX_PRIMS> triangles
    )
{
    const uint numVertices = 3;
    const uint numPrimitives = 1;
    SetMeshOutputCounts(numVertices, numPrimitives);

    if(tig < numVertices) {
        verts[tig].wz = float2(1, 0);
        verts[tig].x = 0;
        verts[tig].y = 0;
    }

    if(tig < numPrimitives) {
        triangles[tig] = uint3(0,1,2);
    }
}

