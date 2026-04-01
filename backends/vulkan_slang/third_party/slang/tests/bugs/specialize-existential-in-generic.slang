//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

[Specialize]
interface IAssoc
{
    int getInner();
}

interface IFoo
{
    associatedtype Assoc : IAssoc;
    Assoc getValue();
}

struct Impl : IFoo
{
    struct Assoc : IAssoc { int getInner() { return 1; } }
    Assoc getValue() { Assoc r; return r; }
}

struct GenType<T : IFoo> : IDefaultInitializable
{
    T obj;
    int doThing()
    {
        IAssoc soc = obj.getValue(); // "boxing" into an existential

        // a specialized version of this function should call specialized method instead of going through dynamic dispatch.
        return soc.getInner();
    }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<int> gOutputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    GenType<Impl> val;
    gOutputBuffer[tid] = val.doThing();
}
