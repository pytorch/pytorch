//TEST_DISABLED:COMPARE_HLSL: -profile cs_5_0 -entry main

// Test various cases of implicit type conversion and preference
// for overload resolution.

cbuffer U
{
	int 	ii;
	uint 	uu;
	float 	ff;	
};

Buffer<int> ib;
RWBuffer<int> ob;


int pick(int   x) 	{ return 1; }
int pick(uint  x)	{ return 2; }
int pick(float x) 	{ return 3; }


int test0(int x) { return x; }
uint test0(uint x) { return x; }

// Test: is integer-to-float conversion preferred
// over scalar-to-vector conversion?
int test1(uint3 v) { return 0; }
float test1(float v) { return 0; }

// Is rank of signed-int-to-float the same
// as unsigned-init-to-float?
int test2(float f, uint u) { return 0; }
float test2(int i, float f) { return 0; }

// Is just having "more" implicit conversions
// enough to rank overloads?
int test3(float f, uint u, uint u2) { return 0; }
float test3(int i, float f, float f2) { return 0; }

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint idx = tid.x;

	bool bb = (ii + uu) != 0;

#define CASE(exp) ob[ib[idx++]] = pick(exp)

	CASE(ii + uu);
	CASE(uu + ii);
	CASE(ii + ff);
	CASE(uu + ff);

	// Should be ambiguous, but currently isn't:
//	CASE(test0(bb));

	CASE(test1(uu));

	// Ambiguous, and it should be
//	CASE(test2(ii, uu));

	// Prefer overload with lower overall converion cost
	// (not necessarily one that is unambiguously "better"
	// at every argument position).
	//
	CASE(test3(ii, uu, uu));
}