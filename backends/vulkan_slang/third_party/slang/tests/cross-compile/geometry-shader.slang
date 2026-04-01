// geometry-shader.slang

//TEST:SIMPLE(filecheck=CHECK): -profile sm_5_0 -stage geometry -entry main -target glsl
//TEST:SIMPLE(filecheck=SPIRV): -profile sm_5_0 -stage geometry -entry main -target spirv

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
// CHECK: void main()
// CHECK: gl_Layer =
// CHECK: EmitVertex();

[maxvertexcount(3)]
void main(
    triangle CoarseVertex coarseVertices[3],
    inout TriangleStream<RasterVertex> outputStream,
    uint primitiveID : SV_PrimitiveID)
{
    for(int ii = 0; ii < 3; ++ii)
    {
        CoarseVertex coarseVertex = coarseVertices[ii];
        RasterVertex rasterVertex;
        rasterVertex.position   = coarseVertex.position;
        rasterVertex.color      = coarseVertex.color;
        rasterVertex.id         = coarseVertex.id + primitiveID;
        outputStream.Append(rasterVertex);
    }
}
