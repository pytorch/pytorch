//TEST:SIMPLE: -profile vs_5_0

layout(
	binding = UNDEFINED_VK_BINDING,
	set = UNDEFINED_VK_SET) 
Texture2DArray<float4> Float4Texture2DArrays[] : register(t0, space100);

