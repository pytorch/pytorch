//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -profile sm_5_0 -output-using-type

// Test that we can specialize a generic method called through a dynamic interface.

interface IValue
{
    float getVal();
}

struct SimpleVal : IValue
{
    float val;
    float getVal() { return val; }
}

[anyValueSize(16)]
interface IInterface
{
    associatedtype V : IValue;
    V run<let N : int>(float arr[N]);
}

struct Add : IInterface
{
    float base;
    typealias V = SimpleVal;
    V run<let N : int>(float arr[N])
    {
        float sum = base;
        for (int i = 0; i < N; i++)
            sum += arr[i];
        V rs;
        rs.val = sum;
        return rs;
    }
}

struct Mul : IInterface
{
    float base;
    typealias V = SimpleVal;
    V run<let N : int>(float arr[N])
    {
        float sum = base;
        for (int i = 0; i < N; i++)
            sum *= arr[i];
         V rs;
        rs.val = sum;
        return rs;
    }
}

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<float> gOutputBuffer;

//TEST_INPUT:type_conformance Add:IInterface=1
//TEST_INPUT:type_conformance Mul:IInterface=2

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    var obj = createDynamicObject<IInterface>(1, 1.0); // Add.
    float arr[3] = { 2, 3, 4 };
    gOutputBuffer[0] = obj.run(arr).getVal();
}