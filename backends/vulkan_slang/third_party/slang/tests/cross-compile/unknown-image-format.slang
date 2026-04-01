// unknown-image-format.slang

//TEST:SIMPLE(filecheck=CHECK):-target spirv-assembly -entry main -stage fragment -default-image-format-unknown

// Ensure that we can emit R/W images with an unknown format, when required.

// CHECK: OpTypeImage %float 2D 2 0 0 2 Unknown
// CHECK: OpTypeImage %float 2D 2 0 0 2 R32f
// CHECK: OpTypeImage %float 2D 2 0 0 2 Rgba8
// CHECK: OpTypeImage %float 2D 2 0 0 2 Rgba16f

// Global Scope:

RWTexture2D<float> gNoFormat;

[format("r32f")]
RWTexture2D<float> gExplicitFormat;

// Nested in a Parameter Block

struct PB
{
	RWTexture2D<float4> noFormat;

	[format("rgba8")]
	RWTexture2D<float4> explicitFormat;
}
ParameterBlock<PB> gBlock;

cbuffer C
{
	uint2 index;
}

float4 main(

	// In entry-point parameter list

	uniform RWTexture2D<float4> noFormat,

	[format("rgba16f")]
	uniform RWTexture2D<float4> explicitFormat
	) : SV_Target
{
    float4 result = 0;

#define USE(NAME) result += NAME[index]

	USE(gNoFormat);
	USE(gExplicitFormat);

	USE(gBlock.noFormat);
	USE(gBlock.explicitFormat);

	USE(noFormat);
	USE(explicitFormat);

#undef USE

    return result;
}
