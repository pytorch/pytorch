module environment;

uint lcg(inout uint prev)
{
    const uint LCG_A = 1664525u;
    const uint LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

public float rnd(inout uint prev)
{
    return ((float) lcg(prev) / (float) 0x01000000);
}

public struct Environment_sample_data
{
    uint alias;
    float q;
};

public float3 environment_sample(StructuredBuffer <Environment_sample_data> sample_buffer, inout int seed)
{
    float3 xi;
    xi.x = rnd(seed);
    xi.y = rnd(seed);
    xi.z = rnd(seed);
    return xi.z;
}
