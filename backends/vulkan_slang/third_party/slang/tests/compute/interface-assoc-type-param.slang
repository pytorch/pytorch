// Tests using associated types through an existential-struct-typed param.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cuda -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cpu -shaderobj

[anyValueSize(8)]
interface IInterface
{
    associatedtype TEval:IEval;
    TEval getEval();
}

[anyValueSize(8)]
interface IEval
{
    uint eval();
}

//TEST_INPUT: type_conformance Impl:IInterface = 0
struct Impl : IInterface
{
    uint val;
    struct TEval : IEval
    {
        uint val;
        uint eval()
        {
            return val;
        }
    };
    TEval getEval()
    {
        TEval rs;
        rs.val = val;
        return rs;
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
    gOutputBuffer[tid] = p.obj[0].getEval().eval();
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID,
//TEST_INPUT:set params.obj = new StructuredBuffer<IInterface>{ new Impl{1}}
    uniform Params params)
{
	uint tid = dispatchThreadID.x;
	compute(tid, params);
}
