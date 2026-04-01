//DIAGNOSTIC_TEST:SIMPLE: -target hlsl -DTEST_1
//DIAGNOSTIC_TEST:SIMPLE: -target hlsl -DTEST_2

SamplerState sampler;

struct Foo
{
    bool sample_bar = false;
#ifdef TEST_1
    Texture2D<float4> bar = {};
#elif defined(TEST_2)
    Texture2D<float4> bar[2];
#endif
};

struct Result
{
    float4 color = float4(0.0);
};

Result process(in Foo foo)
{
    Result result = {};

    if (foo.sample_bar) {
#ifdef TEST_1
        result.color = foo.bar.Sample(sampler, float2(0.0, 0.0));
#elif defined(TEST_2)
        result.color = foo.bar[0].Sample(sampler, float2(0.0, 0.0));
#endif
    }

    return result;
}

[shader("compute")]
[numthreads(8, 8, 1)]
float4 cs_main(uint3 thread_id : SV_DispatchThreadID) {
    Foo foo;
    const let result = process(foo);
    return result.color;
}
