//TEST:SIMPLE: -profile vs_5_0

[[vk::binding(UNDEFINED_VK_BINDING, UNDEFINED_VK_SET)]]
Texture2DArray<float4> Float4Texture2DArrays[] : register(t0, space100);

float4 main() : SV_Position
{
    return 0;
}
