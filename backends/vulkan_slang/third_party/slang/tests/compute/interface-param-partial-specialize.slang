// Tests generating dynamic dispatch code for a function
// with existential-struct-typed param by specializing it
// with __Dynamic. This verifies that the handling of
// "partially" specializing an existential type is correct.

//DISABLED_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cuda -shaderobj
//DISABLED_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cpu -shaderobj

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

//TEST_INPUT: entryPointExistentialType __Dynamic

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID,

//TEST_INPUT:begin_buffer(stride=8):name=params.obj
//TEST_INPUT:begin_object(type=Impl)
//TEST_INPUT:uniform(data=[1]):name=val
//TEST_INPUT:end
//TEST_INPUT:end
    uniform Params params)
{
	uint tid = dispatchThreadID.x;
	compute(tid, params);
}
