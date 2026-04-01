// implicit-conversion-binary-op.hlsl
//TEST:COMPARE_HLSL: -profile ps_5_0

// Make sure that we can pick resolve the right overload
// to call when applying a binary operator to vectors
// with different element types. We should pick
// the "better" of the two element types, and not
// get an ambiguity error.

float4 main(
	float4 	a : A,
	uint4	b : B
	) : SV_TARGET
{
	return a * b;
}
