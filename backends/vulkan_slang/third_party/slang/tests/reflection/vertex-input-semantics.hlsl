//TEST:REFLECTION:-profile vs_4_0 -target hlsl -no-codegen

// Confirm that we can generate reflection info for
// vertex shader input parameters, including those
// that have semantics, and including nesting
// via struct types.

struct X
{
	float4 x0;
	float4 x1;
};

struct B
{
	int4 b0;
	X b1;
};

struct C
{
	X c0 : CX;
	int4 c1 : CY;
};

float4 main(
	float4 a : A,
	B b : B,
	C c)
	: SV_Position
{
	return 0.0;
}