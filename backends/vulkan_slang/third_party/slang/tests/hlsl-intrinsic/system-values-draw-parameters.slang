//TEST:SIMPLE(filecheck=CHECK_SPIRV): -entry main -stage vertex -target spirv
//TEST:SIMPLE(filecheck=CHECK_GLSL): -entry main -stage vertex -target glsl
//TEST:SIMPLE(filecheck=CHECK_HLSL): -entry main -stage vertex -target hlsl
//TEST:SIMPLE(filecheck=CHECK_METAL): -entry main -stage vertex -target metal

StructuredBuffer<float> scales;

struct VSInput
{
    uint vertexID : SV_VertexID;
    uint instanceID : SV_InstanceID;
};

struct VSOutput
{
    float4 position : SV_POSITION;
};

VSOutput main(VSInput input, 
              uint startVertexLocation : SV_StartVertexLocation, 
              uint startInstanceLocation : SV_StartInstanceLocation,
              uint viewIndex : SV_ViewID)
{
    VSOutput output;

    float x = (float)(input.vertexID + startVertexLocation) * 0.1f;
    float y = (float)(input.instanceID + startInstanceLocation) * 0.2f;
    output.position = float4(x, y, 0.0f, 1.0f) * scales[viewIndex];

    // CHECK_SPIRV: BuiltIn BaseVertex
    // CHECK_GLSL: gl_BaseVertex
    // CHECK_HLSL: SV_StartVertexLocation
    // CHECK_METAL: base_vertex

    // CHECK_SPIRV: BuiltIn InstanceIndex
    // CHECK_GLSL: gl_BaseInstance
    // CHECK_HLSL: SV_StartInstanceLocation
    // CHECK_METAL: base_instance

    // CHECK_SPIRV: BuiltIn ViewIndex
    // CHECK_GLSL: gl_ViewIndex
    // CHECK_HLSL: SV_ViewID
    // CHECK_METAL: amplification_id

    return output;
}

