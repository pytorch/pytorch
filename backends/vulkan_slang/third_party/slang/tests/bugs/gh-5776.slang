//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cuda -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cpu -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -wgpu -output-using-type


interface IFoo
{
    associatedtype FooType : IFoo;
}

extension float : IFoo
{
    typedef float FooType;
}

__generic<T:IFoo, let N:int>
extension Array<T, N> : IFoo
{
    typedef Array<T.FooType, N> FooType;
}

__generic<T:IFoo, let N:int>
extension vector<T, N> : IFoo
{
    typedef vector<T.FooType, N> FooType;
}

__generic<T:IFoo, let N:int, let M:int>
extension matrix<T, N, M> : IFoo
{
    typedef matrix<T.FooType, N, M> FooType;
}

struct WrappedBuffer<T : IFoo>
{
    StructuredBuffer<T> buffer;
    int shape;

    T get(int idx) { return buffer[idx]; }
}


struct GradInBuffer<T : IFoo>
{
    WrappedBuffer<T.FooType> wrapBuffer;
}

struct CallData
{
    GradInBuffer<float[2]> grad_in1;
    GradInBuffer<vector<float, 2>> grad_in2;
    GradInBuffer<float2x2> grad_in3;
}


//TEST_INPUT: set call_data.grad_in1.wrapBuffer.buffer = ubuffer(data=[1.0 2.0 3.0 4.0], stride=4);
//TEST_INPUT: set call_data.grad_in2.wrapBuffer.buffer = ubuffer(data=[5.0 6.0 7.0 8.0], stride=4);
//TEST_INPUT: set call_data.grad_in3.wrapBuffer.buffer = ubuffer(data=[1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0], stride=4);
ParameterBlock<CallData> call_data;


//TEST_INPUT:ubuffer(data=[0.0 0.0 0.0 0.0 0.0 0.0 0.0], stride=4):out, name outputBuffer
RWStructuredBuffer<float> outputBuffer;


[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    float[2] data1 = call_data.grad_in1.wrapBuffer.buffer[0];
    float[2] data2 = call_data.grad_in1.wrapBuffer.get(1);
    outputBuffer[0] = data1[0];
    outputBuffer[1] = data2[0];

    vector<float, 2> data3 = call_data.grad_in2.wrapBuffer.buffer[0];
    vector<float, 2> data4 = call_data.grad_in2.wrapBuffer.get(1);
    outputBuffer[2] = data3[0];
    outputBuffer[3] = data4[0];

    float2x2 data5 = call_data.grad_in3.wrapBuffer.buffer[0];
    float2x2 data6 = call_data.grad_in3.wrapBuffer.get(1);
    outputBuffer[4] = data5[0][0];
    outputBuffer[5] = data6[0][0];
}
