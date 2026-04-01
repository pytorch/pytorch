//TEST(compute):COMPARE_COMPUTE: -shaderobj


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IAdd
{
    int addf(int u, int v);
}

interface ISub
{
    int subf(int u, int v);
}

interface IAddAndSub : IAdd, ISub
{
}

struct Simple : IAddAndSub
{
    int addf(int u, int v)
    {
        return u+v;
    }
    int subf(int u, int v)
    {
        return u-v;
    }
};

int testAdd<T:IAdd>(T t)
{
    return t.addf(1, 1);
}

interface IAssoc
{
    associatedtype AT : IAdd;
}

struct AssocImpl : IAssoc
{
    typedef Simple AT;
};

int testAdd2<T:IAssoc>(T assoc)
{
    T.AT obj = T.AT();
    return obj.addf(1, 1);
}

int testSub<T:ISub>(T t, int base)
{
    return t.subf(base, 1);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    AssocImpl s;

    // expected value: 1 + 1 = 2
	int outVal = testAdd2(s);
    Simple s1;

    // expected value: 2 + (2-1) = 3
    outVal += testSub(s1, outVal);

	outputBuffer[dispatchThreadID.x] = outVal;
}
