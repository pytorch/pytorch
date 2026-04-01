// Tests specializing a function with existential-struct-typed param.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cuda -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cpu -shaderobj

[anyValueSize(8)]
interface IInterface
{
    uint eval();
}

export struct Impl : IInterface
{
    uint val;
    uint eval()
    {
        return val;
    }
};

struct Params
{
    StructuredBuffer<IInterface> obj;
};

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<uint> gOutputBuffer;

void compute(uint tid, Params p)
{
    gOutputBuffer[tid] = p.obj[0].eval();
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID,
//TEST_INPUT:set params.obj = new StructuredBuffer<IInterface>{new Impl{1}}
    uniform Params params)
{
	uint tid = dispatchThreadID.x;
	compute(tid, params);
}
