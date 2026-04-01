//TEST:SIMPLE(filecheck=CHECK): -target wgsl

RWStructuredBuffer<float> outputBuffer;

// CHECK: fn inner{{.*}}( x{{.*}} : ptr<function, f32>)
// CHECK: (*x{{.*}}) = (*x{{.*}}) + 1.0
void inner(inout float x)
{
    x = x + 1;
}

// CHECK: fn test{{.*}}( x{{.*}} : ptr<function, f32>)
void test(inout float x)
{
    inner(x);
}

struct MyType
{
    float myField[3];
}

[numthreads(1,1,1)]
void computeMain(int id : SV_DispatchThreadID)
{
    MyType v;
    v.myField[id] = 0.0f;
    // CHECK: test{{.*}}(&({{.*}}));
    test(v.myField[id]);
    v.myField[1] = 2.0;
    outputBuffer[0] = v.myField[id];
}
