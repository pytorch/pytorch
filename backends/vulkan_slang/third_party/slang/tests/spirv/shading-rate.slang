//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry main -emit-spirv-directly

// CHECK: BuiltIn ShadingRateKHR
[shader("fragment")]
uint main(uint shading_rate : SV_ShadingRate) : SV_Target {
    return shading_rate;
}
