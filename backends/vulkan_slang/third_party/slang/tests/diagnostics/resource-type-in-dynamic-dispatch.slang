//TEST:SIMPLE(filecheck=CHECK): -target spirv
// CHECK: type 'FooImpl' contains fields that cannot be packed into ordinary bytes for dynamic dispatch.
interface IFoo
{
    float get();
}

export struct FooImpl : IFoo
{
    Texture2D t;
    float get() { return 1.0; }
}

export struct FooImpl2 : IFoo
{
    float v;
    float get() { return v; }
}


RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    IFoo f = createDynamicObject<IFoo>(0, 5);
    outputBuffer[0] = f.get();
}