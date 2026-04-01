//TEST:SIMPLE(filecheck=CHECK):-entry gsMain -stage geometry -target spirv -emit-spirv-directly

struct GsOut
{
    float2 uv                               : TexCoord;
    float2 texelPos                         : TexelPosition;
    float4 posH                             : SV_Position;
};

// CHECK-DAG: OpEntryPoint Geometry %gsMain "main" %[[OUT1:[A-Za-z0-9_]+]] %[[OUT2:[A-Za-z0-9_]+]] %gl_Position
// CHECK-DAG:          %[[OUT1]] = OpVariable %_ptr_Output_v2float Output
// CHECK-DAG:          %[[OUT2]] = OpVariable %_ptr_Output_v2float Output
// CHECK-DAG: %gl_Position = OpVariable %_ptr_Output_v4float Output

// CHECK-DAG: OpExecutionMode %gsMain Invocations 1
// CHECK-DAG: OpExecutionMode %gsMain OutputTriangleStrip
// CHECK-DAG: OpExecutionMode %gsMain Triangles
// CHECK-DAG: OpExecutionMode %gsMain OutputVertices 15

// CHECK: OpStore %gl_Position
// CHECK: OpEmitVertex
// CHECK: OpEndPrimitive

[maxvertexcount(15)]
void gsMain(uint triIdx : SV_PrimitiveID, inout TriangleStream<GsOut> outStream)
{
    for (int j = 0; j < 5; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            GsOut v;
            v.uv = float2(j,i);
            v.texelPos = float2(0,0);
            v.posH = float4(j, i, 1, 1);
            outStream.Append(v);
        }
        outStream.RestartStrip();
    }
}