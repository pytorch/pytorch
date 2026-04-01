//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Confirm that a struct type defined in a generic parent works

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct SubType<T>
{
    T x;
};

struct GenStruct<T>
{
    typedef SubType<T> SubTypeT;
    T getVal(SubTypeT v)
    {
        return v.x;
    }
};

float test(float val)
{
    GenStruct<float>.SubTypeT sb;
    sb.x = val;
    GenStruct<float> obj;
    return obj.getVal(sb);
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    float inVal = float(tid);
    float outVal = test(inVal);
    outputBuffer[tid] = outVal.x;
}