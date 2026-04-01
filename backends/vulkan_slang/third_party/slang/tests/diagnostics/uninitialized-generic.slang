//TEST:SIMPLE(filecheck=CHK): -target spirv -entry computeMain

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

groupshared float4 gsVar;

__generic<TYPE1 : __BuiltinArithmeticType>
struct MyContainer
{
    __generic<TYPE2 : __BuiltinArithmeticType>
    void store(__ref vector<TYPE2,4> v)
    {
        v[0] = TYPE2(0);
        v[1] = TYPE2(1);
        v[2] = TYPE2(2);
        v[3] = TYPE2(3);
    }
};

[Shader("compute")]
[NumThreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    MyContainer<float> obj;
    obj.store(gsVar);

    // CHK-NOT:warning 41017:
    outputBuffer[0] = gsVar.x;
}
