//TEST:SIMPLE(filecheck=CHECK): -target glsl -entry computeMain -stage compute
// CHECK: main(
struct DiffMaterialData : IDifferentiable
{
    static const uint kMaterialParamCount = 20;

    // Material parameters.
    float data[kMaterialParamCount];

    [Differentiable]
    __init()
    {
        [ForceUnroll]
        for (uint i = 0; i < kMaterialParamCount; i++)
            data[i] = 0.f;
    }

    [Differentiable]
    float read(inout uint offset) { return data[offset++]; }

    [Differentiable]
    void read<let N : int>(out vector<float, N> value, inout uint offset)
    {
        [ForceUnroll]
        for (uint i = 0; i < N; i++)
            value[i] = read(offset);
    }

    [Differentiable]
    vector<float, N> read<let N : int>(inout uint offset)
    {
        vector<float, N> value;
        this.read(value, offset);
        return value;
    }

    [mutating]
    [Differentiable]
    void write(float value, inout uint offset) { data[offset++] = value; }

    [mutating]
    [Differentiable]
    void write<let N : int>(vector<float, N> value, inout uint offset)
    {
        [ForceUnroll]
        for (uint i = 0; i < N; i++)
            this.write(value[i], offset);
    }
};

RWStructuredBuffer<float3> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    DiffMaterialData diffData;
    uint offset = 0;
    outputBuffer[0] = diffData.read<3>(offset);
}