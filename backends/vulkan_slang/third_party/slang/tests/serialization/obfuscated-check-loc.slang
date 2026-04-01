//TEST:SIMPLE:-target cpp -stage compute -entry computeMain -obfuscate -line-directive-mode none

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

#if 0
[DllExport]
String getString()
{
    return "Hello!";
}
#endif

int silly(int a)
{
    int t = 0;
    [ForceUnroll(10)]
    while ( a > 0)
    {
        t = t + t + a;
    }
    return t;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int x = int(dispatchThreadID.x);
    
    //String v = getString();
    int v = silly(x);
    
    outputBuffer[x] = v;
    
}

