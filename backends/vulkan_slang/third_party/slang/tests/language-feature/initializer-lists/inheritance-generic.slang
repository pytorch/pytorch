//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj -vk
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj

struct Base<let ND:int>
{
    int a = 1;
}

struct Derived<let ND:int>: Base<ND>
{
    bool x;
    bool y;
}

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=result
RWStructuredBuffer<int> result;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    // Previously, this test is just test we can handle the base constructor invoke correctly,
    // so we don't construct the Derived object, since #6058, there will not be implicit constructor
    // to construct the struct, we will have to invoke the constructor explicitly.
    Derived<3> d = {1,1,1};

    // BUFFER: 1
    result[0] = d.a;
}
