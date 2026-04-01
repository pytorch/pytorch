//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj
interface IFoo<T>
{
    T getFirst();    
}

__generic<S, T : IArray<S>>
extension T : IFoo<S>
{ 
    S getFirst()
    {
        return this[0];
    }
}

T getFirstOuter<T>(IFoo<T> arr)
{
    return arr.getFirst();
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    int arr[] = {1,2,3};
    // CHECK: 1
    outputBuffer[0] = getFirstOuter(arr);
}