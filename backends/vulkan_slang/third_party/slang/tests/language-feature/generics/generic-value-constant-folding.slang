//TEST(compute):COMPARE_COMPUTE: -shaderobj -output-using-type

struct PlusOne<let v : int>
{
    static const int value = v > 0? v + 1 : v - 1;
}

struct GetConst<let v : int, let u : int>
{
    static const int value = (u/2+v)*(u+v) / PlusOne<u-1>.value;
    int arr[value];
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int arr[GetConst<5, 2>.value + 1];
    GetConst<5, 3> obj;
    obj.arr[0] = GetConst<5, 3>.value + 1;
    outputBuffer[tid] = obj.arr[0];
}
