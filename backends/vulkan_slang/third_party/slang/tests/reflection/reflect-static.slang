//TEST:REFLECTION:-profile cs_6_0  -target hlsl -entry computeMain -no-codegen

struct Thing
{
    static int value;
};

struct AnotherThing
{
    int a;
    int b;
    static Texture2D t;
};


ConstantBuffer<Thing> cbThing;
ConstantBuffer<AnotherThing> cbAnotherThing;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Texture2D t = AnotherThing::t;
    
    float4 v = t.Load(int3(dispatchThreadID.x, dispatchThreadID.y, 0));
    
    int m = cbAnotherThing.a + cbAnotherThing.b + Thing::value;
    
    m += int(v.x);
    
    outputBuffer[dispatchThreadID.x] = m;
}
