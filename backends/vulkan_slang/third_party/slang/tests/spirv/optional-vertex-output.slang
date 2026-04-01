//TEST:SIMPLE(filecheck=CHECK): -target spirv

// Test that we can use Optional<T> or bool types in varying input or outputs.

// CHECK: OpDecorate %i_inA_value Location 0
// CHECK: OpDecorate %i_inA_hasValue Location 1
// CHECK: OpDecorate %entryPointParam_vertMain_a_value Location 0
// CHECK: OpDecorate %entryPointParam_vertMain_a_hasValue Location 1

struct VIn {
    Optional<float> inA;
}

struct VSOut {
    Optional<float> a;
    bool outputValues[3];
};

[shader("vertex")]
VSOut vertMain(VIn i)
{
    VSOut o;
    if (i.inA.hasValue)
        o.a = i.inA;
    else
        o.a = 0.0;
    o.outputValues = { true, false, true };
    return o;
}