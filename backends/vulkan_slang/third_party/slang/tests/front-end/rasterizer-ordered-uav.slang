// rasterizer-ordered-uav.slang
//TEST:SIMPLE:

// Confirm that the `RasterizerOrdered*` types are supported

struct S { float4 a; }

RasterizerOrderedBuffer<float4> 		b0;
RasterizerOrderedByteAddressBuffer 		b1;
RasterizerOrderedStructuredBuffer<S>	b2;
RasterizerOrderedTexture1D<float4>		b3;
RasterizerOrderedTexture1DArray			b4;
RasterizerOrderedTexture2D 				b5;
RasterizerOrderedTexture2DArray<float4>	b6;
RasterizerOrderedTexture3D<int>			b7;


float4 test(uint4 coord, float4 value)
{
	value += b0[coord.x];

	// TODO: use the rest of the `b*`

	return value;
}
