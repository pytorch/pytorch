// array-size-static-const-2.hlsl
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

interface IA
{
    static const int M;
}
struct B : IA
{
    static const int M = 2;
}
struct GenType<T : __BuiltinIntegerType, A: IA, let N : int>
{
    static const int HalfN = N > 1? N / A.M : 1;
    T sum(T arr[HalfN])
    {
        T rs = T(0);
        for (int i = 0; i < HalfN; i++)
            rs += arr[i];
        return rs;
    }
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    int arr[2] = { 1, 2 };
    GenType<int, B, 4> obj;
    outputBuffer[0] = obj.sum(arr);
}