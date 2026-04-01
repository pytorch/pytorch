//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

// Test that we can disambiguiate subscript decls by prefering the candidate that contains a super set of
// accessors than the other candidates.
interface IBuf<T>
{
    T read(int x);
}
interface IRWBuf<T> : IBuf<T>
{
    [mutating]
    void write(int x, T v);
}

extension<T, U : IBuf<T>> U
{
    __subscript(int x) -> T { get { return read(x); } }
}

extension<T, U : IRWBuf<T>> U
{
    __subscript(int x)->T { get { return read(x); } set { write(x, newValue); } }
}

struct MyArray<T> : IRWBuf<T>
{
    T data[4];
    T read(int x) { return data[x]; }
    [mutating]
    void write(int x, T v) { data[x] = v; }
}


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    MyArray<int> arr = {{1, 2, 3, 4}};
    arr[0] = 1;
    arr[1] = 2;
    // CHECK: 1
    // CHECK: 2
    outputBuffer[0] = arr[0];
    outputBuffer[1] = arr[1];
}
