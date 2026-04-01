//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

// test specialization of nested generic functions

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IBRDF
{
    float getF();
}

interface ILight
{
    float illum<B:IBRDF>(B b);
};

struct B0 : IBRDF
{
    float getF() { return 1.0; }
};

struct L0 : ILight
{
    float illum<B:IBRDF>(B b) { return b.getF(); }
};

struct L1<L:ILight> : ILight
{
    L l;
    float illum<BXX:IBRDF>(BXX bxx) { return l.illum<BXX>(bxx); }
};


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
    L1<L0> light;
    B0 b0;
	float outVal = light.illum<B0>(b0);
	outputBuffer[tid] = int(outVal);
}