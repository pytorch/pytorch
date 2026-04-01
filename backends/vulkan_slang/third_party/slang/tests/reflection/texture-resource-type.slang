// texture-resource-type.slang

// Tests reflection of inner texture type and formats.

//TEST:REFLECTION:-stage compute -entry main -target hlsl

Texture2D NoParameters;
Texture2D<float> FloatTexture;
Texture2D<float2> FloatTwoTexture;
Texture2D<float3> FloatThreeTexture;
Texture2D<float4> FloatFourTexture;
Texture2D<int> IntTexture;
Texture2D<int2> IntTwoTexture;
Texture2D<int4> IntFourTexture;
Texture2D<uint> UintTexture;
Texture2D<uint2> UintTwoTexture;
Texture2D<uint4> UintFourTexture;
[format("r11f_g11f_b10f")]
Texture2D R11G11B10Texture;
[format("rgb10_a2ui")]
Texture2D<uint3> RGB10_A2_UI_Texture;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
}
