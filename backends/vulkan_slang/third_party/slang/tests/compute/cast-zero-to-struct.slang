// cast-zero-to-struct.slang

// Test that HLSL legacy syntax for casting from literal zero
// to a `struct` type works.

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

struct S
{
	int2 a;
	int2 b;
	int2 c;
}

int test(int val)
{
	S s = (S) 0;

	s.a.x = val;
	s.b.y = val;
	s.c.x = val;

	int2 t = s.a + s.b*256 + s.c*65536;
	return t.x + t.y*16;
}

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<int> gOutputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 tid : SV_DispatchThreadID)
{
    int inVal = tid.x;
    int outVal = test(inVal);
    gOutputBuffer[tid.x] = outVal;
}
