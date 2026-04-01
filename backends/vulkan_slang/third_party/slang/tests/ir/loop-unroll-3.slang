//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -output-using-type

//TEST_INPUT:set buffer = ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> buffer;

struct Foo<let D : int>
{
    float load()
    {
        int idx[D];
        [ForceUnroll]
        for (int i = 0; i < D; ++i)
            idx[i] = 0;
        [ForceUnroll]
        for (int i = 0; i < D-2; ++i)
            idx[i] = 0;
        float result = 100.0;
        [ForceUnroll]
        for (int i = 0; i < 1; i++)
            result += buffer[idx[0]];

        return result;
    }
}

//TEST_INPUT:set result = out ubuffer(data=[0 0 0 0], stride=4)
uniform RWStructuredBuffer<float> result;

[shader("compute")]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    Foo<2> biases = {};

    // Test that we can unroll loops of 0 iterations.
    result[0] = biases.load();

    // CHECK: 100.0
}