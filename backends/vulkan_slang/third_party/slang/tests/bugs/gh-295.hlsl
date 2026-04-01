// Disabled because Slang IR path is missing support for [fastopt]
//TEST_IGNORE_FILE

//TEST:COMPARE_HLSL: -profile vs_4_0

// Confirm that we pass through `[fastopt]` attributes
//
// This shader does indexing into the elements of
// a vector, fetched from a `cbuffer`, based on
// a loop counter (or a loop with a small trip
// count), so `fxc` seems to want to unroll the
// loop. The `[fastopt]` attribute changes this
// behavior and results in a `loop` instruction
// in the DX bytecode, so we can use this to
// test whether Slang is passing through the
// attribute or not.

// Import Slang code so that we aren't just in
// the 100% pass-through mode.
#ifdef __SLANG__
__import empty;
#endif

cbuffer C
{
	float4 b[4];
}
float test(float x, float c)
{
	[fastopt]
	for(int ii = 0; ii < 2; ++ii)
	{
		x = x*x + c + b[ii][ii];
	}
	return x;
}

float4 main(float4 a : A) : SV_Position
{
	a.x = test(a.x, a.y);

	return a;
}
