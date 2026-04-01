//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Test type checking of associatedtype and typedef

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IBase
{
    associatedtype SubTypeT;
    associatedtype RetT;
    RetT getVal(SubTypeT t);
    SubTypeT setVal(RetT v);
}

struct SubType<T>
{
    T x;
};

struct GenStruct<T> : IBase
{
    typedef T RetT;
    typedef SubType<RetT> SubTypeT;
    SubTypeT setVal(T val)
    {
        SubTypeT rs;
        rs.x = val;
        return rs;
    }
    T getVal(SubTypeT v)
    {
        return v.x;
    }
};

U.RetT test<U:IBase>(U.RetT val)
{
    U obj = U();
    U.SubTypeT sb = obj.setVal(val);
    return obj.getVal(sb);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    float inVal = float(tid);
    /* wrong error message for the following line */
    float outVal = test<GenStruct<float> >(inVal);
    outputBuffer[tid] = outVal.x;
}
