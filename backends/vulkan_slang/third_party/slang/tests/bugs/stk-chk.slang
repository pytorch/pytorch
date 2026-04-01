//TEST(compute):COMPARE_COMPUTE:-cpu -g0

// Tests slang-llvm can deal with large items on stack. 
// On some targets this requires special handling (_chkstk on windows for example) 

struct LargeStruct
{
    int values[4096];
};

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT:ubuffer(data=[9 1000 27 34], stride=4):name inputBuffer
StructuredBuffer<int> inputBuffer;

LargeStruct _calcStruct(int a)
{
    LargeStruct s;
    
    int t = a;
    
    for (int i = 0; i < 4096; ++i)
    {
        s.values[i] = t;
        
        // Munge
        t = t ^ (i * t) + (i - a) * 17 + inputBuffer[t & 3];
    }
    
    return s;
}

int _calcValue(LargeStruct s)
{
    int v = 0;
    for (int i = 0; i < 4096; ++i)
    {
        v ^= s.values[i];
    }
    return v + 1;
}

int _calc1(int a)
{    
    return _calcValue(_calcStruct(_calcValue(_calcStruct(_calcValue(_calcStruct(a ^ 19))))));
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = int(dispatchThreadID.x);
    
    outputBuffer[index] = _calc1(index) + 1;
}