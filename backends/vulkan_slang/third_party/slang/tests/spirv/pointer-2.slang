//TEST:SIMPLE(filecheck=CHECK_SPV): -entry vertexMain -stage vertex -emit-spirv-directly -target spirv
//TEST:SIMPLE(filecheck=CHECK_SPV_VIA_GLSL): -entry vertexMain -stage vertex -emit-spirv-via-glsl -target spirv
//TEST:SIMPLE(filecheck=CHECK_GLSL): -entry vertexMain -stage vertex -target glsl

struct Inner1
{
    float4 position;
    Inner2* inner;
}

struct Inner2
{
    float4 position;
}

struct Vertex
{
    float4 position;
    Inner1* inner;
}

struct VSOutput
{
    float4 position : SV_Position;
}

struct PushConstants
{
    float a;
    Vertex* verts;
};

[[vk::push_constant]]
PushConstants pushConstants;

ConstantBuffer<PushConstants> constantBuffer;

// CHECK_SPV: OpEntryPoint
// CHECK_SPV: OpTypePointer PhysicalStorageBuffer
// CHECK_SPV: OpPtrAccessChain

// CHECK_SPV_VIA_GLSL: Glslang
// CHECK_SPV_VIA_GLSL: OpEntryPoint
// CHECK_SPV_VIA_GLSL: OpTypePointer PhysicalStorageBuffer

[shader("vertex")]
VSOutput vertexMain(int vertexIndex: SV_VertexID)
{
    //
    // Test field access chains.
    //

    // CHECK_GLSL: (((((pushConstants_0.verts_0 + gl_VertexIndex)._data.inner_1) + 1)._data.inner_0) + 5)._data.position_0
    let position1 = pushConstants.verts[vertexIndex].inner[1].inner[5].position;

    // CHECK_GLSL: (((((constantBuffer_0.verts_0 + 7)._data.inner_1) + 3)._data.inner_0) + 4)._data.position_0
    let position2 = constantBuffer.verts[7].inner[3].inner[4].position;

    // CHECK_GLSL: (((constantBuffer_0.verts_0 + gl_VertexIndex)._data.inner_1) + 12)._data.position_1
    let position3 = constantBuffer.verts[vertexIndex].inner[12].position;

    //
    // Test accesing from a variable.
    //

    // CHECK_GLSL: ((((pushConstants_0.verts_0 + 8)._data.inner_1 + 6)._data.inner_0))._data.position_0
    let vertex = pushConstants.verts[8];
    let position4 = vertex.inner[6].inner.position;

    VSOutput output;
    output.position = position1 + position2 + position3 + position4;
    output.position *= pushConstants.a;

    return output;
}

