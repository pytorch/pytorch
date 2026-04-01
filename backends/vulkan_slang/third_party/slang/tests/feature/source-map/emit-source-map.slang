//TEST:SIMPLE:-target dxil -entry computeMain -profile cs_6_2 -line-directive-mode source-map -o tests/feature/source-map/emit-source-map.zip

RWStructuredBuffer<int> outputBuffer;

int doThing(int a)
{
    return a + a - 1;
}

[numthreads(4, 4, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int x = (int)dispatchThreadID.x;
    
    outputBuffer[x] = doThing(x);
}