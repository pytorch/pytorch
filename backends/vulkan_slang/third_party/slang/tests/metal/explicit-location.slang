//TEST:SIMPLE(filecheck=MTL): -target metal
//TEST:SIMPLE(filecheck=WGSL): -target wgsl -entry vertMain -stage vertex

// MTL: attribute(3)
// WGSL: @location(3)

struct Vertex
{
    [vk::location(0)] float4 pos;
    [vk::location(3)] float2 uv;
}

[shader("vertex")]
float4 vertMain(Vertex vin) : SV_Position
{
    return vin.pos + vin.uv.x;
}