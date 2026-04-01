// geometry-shader-sub-func.slang

//TEST:SIMPLE(filecheck=SPIRV): -target spirv -emit-spirv-directly -O0

struct CoarseVertex
{
    float4 position     : POSITION;
    float3 color        : COLOR;
    uint   id           : ID;
}

struct RasterVertex
{
    float4 position : POSITION;
    float3 color    : COLOR;
    uint   id       : SV_RenderTargetArrayIndex;
}

// SPIRV: OpEntryPoint
// SPIRV: OpStore %outputStream_position
// SPIRV: OpEmitVertex
void appendVertex(TriangleStream<RasterVertex> outputStream, RasterVertex vert)
{
    outputStream.Append(vert);
}

[shader("geometry")]
[maxvertexcount(3)]
void main(
    triangle CoarseVertex coarseVertices[3],
    inout TriangleStream<RasterVertex> outputStream,
    uint primitiveID : SV_PrimitiveID)
{
    [ForceUnroll]
    for(int ii = 0; ii < 3; ++ii)
    {
        CoarseVertex coarseVertex = coarseVertices[ii];
        RasterVertex rasterVertex;
        rasterVertex.position   = coarseVertex.position;
        rasterVertex.color      = coarseVertex.color;
        rasterVertex.id         = coarseVertex.id + primitiveID;
        appendVertex(outputStream, rasterVertex);
    }
}
