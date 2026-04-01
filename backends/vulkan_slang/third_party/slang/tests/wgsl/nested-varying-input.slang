//TEST:SIMPLE(filecheck=VERT): -target wgsl -stage vertex -entry vertexMain
//TEST:SIMPLE(filecheck=FRAG): -target wgsl -stage fragment -entry fragmentMain

// Tests three aspects:
//  1. Flatten the nested struct for the return type of the entry-point
//  2. For fragment shader, SV_TARGET index must be emitted as @location(index)
//  3. For non-fragment shader, the user defined semantics should be emitted as @location(index)

struct FragmentOutput
{
    //FRAG: @location(1) color1
    float4 color1 : SV_TARGET1;

    //FRAG: @location(0) color0
    float4 color0 : SV_TARGET0;
};

struct NestedVertexOutput
{
    float4 color : COLOR0;
};

struct VertexOutput
{
    //VERT: @location(0) uv
    //FRAG-DAG: @location(0) uv
    float2 uv : TEXCOORD0;

    //VERT: @location(1) color
    //FRAG-DAG: @location(1) color
    NestedVertexOutput nested;

    //VERT: @builtin(position) position
    //FRAG-DAG: @builtin(position) position
    float4 position : SV_Position;
};

VertexOutput vertexMain()
{
    VertexOutput out;
    out.position = float4(1.0, 1.0, 1.0, 1.0);
    out.uv = float2(0.5, 0.5);
    out.nested.color = float4(0.0, 0.0, 0.0, 0.0);
    return out;
}

//FRAG-DAG: @location(3) color3
//FRAG-DAG: @location(6) color6
FragmentOutput fragmentMain(VertexOutput input, float4 color3: COLOR3, float4 color6: COLOR6)
{
    FragmentOutput out;
    out.color0 = input.nested.color + color3;
    out.color1 = input.nested.color + color6;
    return out;
}
