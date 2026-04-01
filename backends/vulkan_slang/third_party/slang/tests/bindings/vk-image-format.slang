// vk-image-format.slang

//TEST:SIMPLE(filecheck=CHECK):-target glsl -profile ps_4_0 -entry main -line-directive-mode none

//CHECK: layout(r32f)
//CHECK: layout(r16_snorm)
//CHECK: layout(r11f_g11f_b10f)

// Something typical
[vk::image_format("r32f")]
RWTexture2D<float> typicalTexture;

// snorm
[vk::image_format("r16snorm")]
RWTexture2D<float> snormTexture;

// Special case
[vk::image_format("r11g11b10f")]
RWTexture2D<float4> specialTexture;

cbuffer C
{
	uint2 index;
}

float4 main(): SV_Target
{
    float4 result = 0;

    result += typicalTexture[index];
    result += snormTexture[index];
    result += specialTexture[index];
    
    return result;
}
